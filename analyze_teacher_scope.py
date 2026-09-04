"""Zero-training diagnostic for teacher-defined distillation scope.

The primary, pre-specified teacher-only signal is the fraction of the
teacher's top-101 score decline that has already happened by rank 20:

    head_drop_20 = (score_1 - score_20) / (score_1 - score_101 + eps)

It is invariant to per-user affine score scaling.  Users are split into
equal-count flat/middle/front-loaded groups using this signal.  Validation
selects a fixed K per group; only that frozen mapping is evaluated on test.
All student checkpoints already exist, so this script performs no training.
"""

import argparse
import gc
import json
import os

import torch

from analyze_per_user_k import (
    balanced_overlap_groups,
    per_user_recall_ndcg,
    select_and_route,
    topk_excluding_train_batched,
)
from dataset import implicit_CF_dataset, load_cf_data
from diagnose import build_score_mat
from utils import load_yaml
from utils.parse_utils import parse_cfg


def teacher_scope_features(top_values, boundary_ranks=(20, 50)):
    """Compute affine-scale-invariant shape features from sorted top scores."""
    if top_values.ndim != 2 or top_values.shape[1] < max(boundary_ranks) + 1:
        raise ValueError("top_values must contain one score beyond every boundary")
    values = top_values.double()
    eps = torch.finfo(values.dtype).eps
    score_range = (values[:, 0] - values[:, -1]).clamp_min(eps)
    gaps = values[:, :-1] - values[:, 1:]
    normalized_gaps = gaps / score_range.unsqueeze(1)
    max_gap, max_index = normalized_gaps.max(dim=1)
    median_gap = normalized_gaps.median(dim=1).values

    result = {
        "head_drop_20": (values[:, 0] - values[:, 19]) / score_range,
        "max_gap_rank": max_index + 1,  # gap between ranks k and k+1
        "max_normalized_gap": max_gap,
        "gap_prominence": max_gap / median_gap.clamp_min(eps),
    }
    for rank in boundary_ranks:
        result[f"normalized_gap_at_{rank}"] = normalized_gaps[:, rank - 1]

    # Standardizing first removes user-specific location and scale. Effective
    # support remains a descriptive curve-shape statistic, not a probability.
    first_hundred = values[:, :100]
    standardized = (
        first_hundred - first_hundred.mean(dim=1, keepdim=True)
    ) / first_hundred.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    probabilities = torch.softmax(standardized, dim=1)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=1)
    result["standardized_effective_support"] = entropy.exp()
    return result


def summarize(values):
    values = values.double()
    quantiles = torch.quantile(values, torch.tensor([0., .25, .5, .75, 1.], dtype=torch.float64))
    return {
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "min": quantiles[0].item(),
        "q25": quantiles[1].item(),
        "median": quantiles[2].item(),
        "q75": quantiles[3].item(),
        "max": quantiles[4].item(),
    }


def cutoff_histogram(max_gap_rank):
    total = max_gap_rank.numel()
    masks = {
        "1-20": max_gap_rank <= 20,
        "21-50": (max_gap_rank > 20) & (max_gap_rank <= 50),
        "51-100": max_gap_rank > 50,
    }
    return {
        name: {"count": mask.sum().item(), "fraction": mask.double().mean().item()}
        for name, mask in masks.items()
    }


def resolve_student_checkpoint(args, dimension, k):
    suffix = f"capacity_d{dimension}_k{k}_l{args.L}_seed{args.seed}"
    candidates = [
        os.path.join(
            "checkpoints", args.dataset, args.S_backbone,
            f"{args.model.lower()}-{dimension}_{suffix}", "BEST_EPOCH.pt",
        )
    ]
    # The original reproduced baseline is the missing center of the capacity
    # grid: d=20, K=50, L=50, seed=0.
    if dimension == 20 and k == 50 and args.L == 50 and args.seed == 0:
        candidates.append(os.path.join(
            "checkpoints", args.dataset, args.S_backbone,
            f"{args.model.lower()}-{dimension}", "BEST_EPOCH.pt",
        ))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("none of the expected checkpoints exists: " + ", ".join(candidates))


def winner_diagnostic(metrics_by_k, k_values):
    validation = torch.stack([
        metrics_by_k[k]["valid"]["NDCG"] for k in k_values
    ], dim=1)
    maxima = validation.max(dim=1).values
    tied = validation == maxima.unsqueeze(1)
    informative = maxima > 0
    unique = informative & (tied.sum(dim=1) == 1)
    return {
        "all_models_zero_rate": (~informative).double().mean().item(),
        "positive_but_tied_rate": (informative & ~unique).double().mean().item(),
        "unique_winner_rate": unique.double().mean().item(),
        "unique_winner_counts": {
            str(k): (unique & tied[:, index]).sum().item()
            for index, k in enumerate(k_values)
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="citeulike")
    parser.add_argument("--T_backbone", default="bpr")
    parser.add_argument("--S_backbone", default="bpr")
    parser.add_argument("--model", default="rcekd")
    parser.add_argument("--student_dims", type=int, nargs="+", default=[5, 20, 80])
    parser.add_argument("--K_values", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--L", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--anchor_K", type=int, default=100)
    parser.add_argument("--eval_K", type=int, default=20)
    parser.add_argument("--num_groups", type=int, default=3)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    args.student_dims = sorted(set(args.student_dims))
    args.K_values = sorted(set(args.K_values))
    if args.anchor_K != 100 or args.eval_K != 20 or args.num_groups != 3:
        raise ValueError("the pre-specified primary analysis requires anchor_K=100, eval_K=20, num_groups=3")
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint inference requires a CUDA-enabled environment for this repository")
    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda")

    (num_users, num_items, train_pairs, _valid_pairs, _test_pairs,
     train_dict, valid_dict, test_dict, train_matrix, user_pop, item_pop) = load_cf_data(args.dataset)
    trainset = implicit_CF_dataset(
        args.dataset, num_users, num_items, train_pairs, train_matrix, train_dict,
        user_pop, item_pop, num_ns=1, no_neg_sampling=True,
    )
    backbone_cfg = load_yaml(os.path.join(
        "configs", args.dataset, args.S_backbone, "base_config.yaml",
    ))
    defaults = {"dataset": args.dataset, "DATA_DIR": "data/"}
    teacher_cfg = parse_cfg(
        argparse.Namespace(**defaults),
        backbone_cfg["teacher"][args.T_backbone.lower()],
    )

    teacher_checkpoint = os.path.join(
        "checkpoints", args.dataset, args.T_backbone,
        f"scratch-{teacher_cfg.embedding_dim}", "BEST_EPOCH.pt",
    )
    print(f"[teacher] loading {teacher_checkpoint}")
    teacher_scores = build_score_mat(
        teacher_checkpoint, args.T_backbone, teacher_cfg, trainset, device,
    )
    teacher_top = torch.topk(teacher_scores, args.anchor_K + 1, dim=1)
    top_values = teacher_top.values.cpu()
    del teacher_scores, teacher_top
    gc.collect()
    torch.cuda.empty_cache()

    features = teacher_scope_features(top_values)
    raw_groups = balanced_overlap_groups(features["head_drop_20"], args.num_groups)
    group_names = ["flat", "middle", "front_loaded"]
    groups = [(group_names[index], users) for index, (_, users) in enumerate(raw_groups)]
    feature_report = {
        name: summarize(values) if name != "max_gap_rank" else summarize(values.double())
        for name, values in features.items()
    }
    feature_report["max_gap_rank_buckets"] = cutoff_histogram(features["max_gap_rank"])

    capacities = {}
    for dimension in args.student_dims:
        student_cfg = parse_cfg(
            argparse.Namespace(**defaults), backbone_cfg["student"],
            {"embedding_dim": dimension},
        )
        metrics_by_k = {}
        checkpoint_paths = {}
        print(f"\n[student dimension={dimension}]")
        for k in args.K_values:
            checkpoint = resolve_student_checkpoint(args, dimension, k)
            checkpoint_paths[str(k)] = checkpoint
            print(f"  [K={k}] loading {checkpoint}")
            scores = build_score_mat(
                checkpoint, args.S_backbone, student_cfg, trainset, device,
            )
            recommendations = topk_excluding_train_batched(
                scores, train_dict, args.eval_K,
            )
            metrics_by_k[k] = {
                "valid": per_user_recall_ndcg(recommendations, valid_dict, num_users, args.eval_K),
                "test": per_user_recall_ndcg(recommendations, test_dict, num_users, args.eval_K),
            }
            print(
                f"        valid NDCG@20={metrics_by_k[k]['valid']['NDCG'].mean():.5f}  "
                f"test NDCG@20={metrics_by_k[k]['test']['NDCG'].mean():.5f}"
            )
            del scores, recommendations
            gc.collect()
            torch.cuda.empty_cache()

        group_reports, _selected, routing = select_and_route(
            metrics_by_k, groups, args.K_values, args.bootstrap_samples,
        )
        for group_report, (_, users) in zip(group_reports, groups):
            group_report["teacher_scope"] = {
                name: summarize(values[users]) for name, values in features.items()
            }
        aggregate = {
            str(k): {
                split: {
                    "Recall@20": metrics_by_k[k][split]["Recall"].mean().item(),
                    "NDCG@20": metrics_by_k[k][split]["NDCG"].mean().item(),
                } for split in ("valid", "test")
            } for k in args.K_values
        }
        capacities[str(dimension)] = {
            "checkpoints": checkpoint_paths,
            "aggregate_metrics": aggregate,
            "per_user_validation_winners": winner_diagnostic(metrics_by_k, args.K_values),
            "groups": group_reports,
            "validation_selected_routing": routing,
            "K_sensitivity_range": {
                split: {
                    "Recall@20": max(aggregate[str(k)][split]["Recall@20"] for k in args.K_values)
                                 - min(aggregate[str(k)][split]["Recall@20"] for k in args.K_values),
                    "NDCG@20": max(aggregate[str(k)][split]["NDCG@20"] for k in args.K_values)
                               - min(aggregate[str(k)][split]["NDCG@20"] for k in args.K_values),
                } for split in ("valid", "test")
            },
        }

    report = {
        "experiment": {
            "dataset": args.dataset,
            "teacher": args.T_backbone,
            "student": args.S_backbone,
            "student_dims": args.student_dims,
            "K_values": args.K_values,
            "L": args.L,
            "seed": args.seed,
            "anchor_K": args.anchor_K,
            "primary_signal": "head_drop_20=(score_1-score_20)/(score_1-score_101+eps)",
            "primary_expected_direction": "front_loaded prefers smaller K; flat prefers larger K",
            "teacher_checkpoint": teacher_checkpoint,
        },
        "teacher_scope_features": feature_report,
        "capacities": capacities,
        "limitations": [
            "BPR scores are uncalibrated logits; only affine-scale-invariant curve-shape features are compared across users.",
            "The primary head-drop@20 signal and its direction were specified before inspecting this report.",
            "Each fixed-K checkpoint was already selected using validation; group routing reuses validation.",
            "A routed mixture of fixed-K checkpoints does not prove a dynamically trained single model will improve.",
            "Seed 0 is exploratory; any surviving association requires confirmation on additional seeds.",
        ],
    }
    output = args.output or os.path.join(
        "logs", f"teacher_scope_{args.dataset}_{args.S_backbone}_dims-"
        f"{'-'.join(map(str, args.student_dims))}_seed{args.seed}.json",
    )
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\nTeacher max-gap cutoff distribution:")
    for bucket, values in feature_report["max_gap_rank_buckets"].items():
        print(f"  {bucket}: {values['count']} users ({values['fraction']:.1%})")
    print("\nValidation-selected teacher-scope routing:")
    for dimension, capacity in capacities.items():
        mapping = ", ".join(
            f"{group['name']}->K{group['selected_K_from_validation']}"
            for group in capacity["groups"]
        )
        routing = capacity["validation_selected_routing"]
        print(
            f"  d={dimension}: {mapping}; test NDCG@20="
            f"{routing['group_routed']['test']['NDCG@20']:.5f}, gain vs valid-selected fixed="
            f"{routing['test_gain_over_best_fixed']['NDCG@20']:+.6f}"
        )
    print(f"report saved to {output}")


if __name__ == "__main__":
    main()

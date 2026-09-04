"""Post-hoc, zero-training analysis of user-specific distillation scope.

This script compares already-trained checkpoints with different fixed K.  It
uses validation interactions to select one K for each difficulty group and
then evaluates that fixed group-to-K mapping on the untouched test split.

The difficulty proxy is the mean teacher/student top-anchor overlap across
all compared checkpoints.  Because it is computed after training, this is a
descriptive diagnostic, not evidence that an online dynamic-K policy works.
"""

import argparse
import gc
import json
import os

import torch

from dataset import load_cf_data, implicit_CF_dataset
from diagnose import build_score_mat
from utils import load_yaml
from utils.parse_utils import parse_cfg


def row_overlap_ratio(left, right, chunk_size=256):
    """Per-row set overlap divided by width, computed without a huge cube."""
    assert left.shape == right.shape
    result = torch.empty(left.shape[0], dtype=torch.float64)
    for start in range(0, left.shape[0], chunk_size):
        end = min(start + chunk_size, left.shape[0])
        matches = left[start:end].unsqueeze(2) == right[start:end].unsqueeze(1)
        result[start:end] = matches.any(dim=2).sum(dim=1).double() / left.shape[1]
    return result


def topk_excluding_train_batched(score_mat, train_dict, k, batch_size=1024):
    """Match evaluation.py's -1e10 training-item mask and top-k operation."""
    result = torch.empty((score_mat.shape[0], k), dtype=torch.long)
    for start in range(0, score_mat.shape[0], batch_size):
        end = min(start + batch_size, score_mat.shape[0])
        scores = score_mat[start:end].clone()
        positives = [train_dict[user] for user in range(start, end)]
        lengths = torch.tensor([items.numel() for items in positives], dtype=torch.long)
        if lengths.sum().item() > 0:
            rows = torch.repeat_interleave(torch.arange(end - start), lengths)
            columns = torch.cat(positives).long()
            scores[rows.to(scores.device), columns.to(scores.device)] = -1e10
        result[start:end] = torch.topk(scores, k, dim=1).indices.cpu()
    return result


def per_user_recall_ndcg(topk_items, ground_truth, num_users, k):
    """Return per-user metrics with the same zero-target treatment as evaluation.py."""
    recommendations = topk_items[:, :k].cpu()
    labels = torch.zeros((num_users, k), dtype=torch.float64)
    num_targets = torch.zeros(num_users, dtype=torch.long)
    for user, truth in ground_truth.items():
        truth = truth.cpu().long()
        num_targets[user] = truth.numel()
        if truth.numel() > 0:
            labels[user] = torch.isin(recommendations[user], truth).double()

    hits = labels.sum(dim=1)
    recall = hits / num_targets.clamp_min(1).double()
    discounts = 1.0 / torch.log2(torch.arange(2, k + 2, dtype=torch.float64))
    dcg = (labels * discounts).sum(dim=1)
    ideal_lengths = num_targets.clamp(min=1, max=k)
    cumulative_discount = discounts.cumsum(dim=0)
    idcg = cumulative_discount[ideal_lengths - 1]
    ndcg = dcg / idcg
    return {"Recall": recall, "NDCG": ndcg, "has_target": num_targets > 0}


def balanced_overlap_groups(overlap, num_groups=3):
    """Equal-count groups avoid empty quantile bins when overlap has many ties."""
    if num_groups < 2:
        raise ValueError("num_groups must be at least 2")
    order = torch.argsort(overlap, stable=True)
    chunks = torch.tensor_split(order, num_groups)
    if num_groups == 3:
        names = ["low", "middle", "high"]
    else:
        names = [f"group_{index + 1}" for index in range(num_groups)]
    return list(zip(names, chunks))


def paired_bootstrap_delta(first, second, users, samples=2000, seed=0):
    """Paired user-bootstrap CI for mean(first - second)."""
    delta = (first[users] - second[users]).double()
    mean = delta.mean().item()
    if samples <= 0 or delta.numel() == 0:
        return {"mean": mean, "ci95": None}
    generator = torch.Generator().manual_seed(seed)
    estimates = []
    chunk_size = 100
    for start in range(0, samples, chunk_size):
        count = min(chunk_size, samples - start)
        draws = torch.randint(delta.numel(), (count, delta.numel()), generator=generator)
        estimates.append(delta[draws].mean(dim=1))
    estimates = torch.cat(estimates)
    low, high = torch.quantile(estimates, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {"mean": mean, "ci95": [low.item(), high.item()]}


def select_and_route(metrics_by_k, groups, k_values, bootstrap_samples=2000):
    """Select group K on validation only, then apply that mapping to test."""
    report_groups = []
    selected_for_user = torch.empty_like(metrics_by_k[k_values[0]]["valid"]["NDCG"],
                                         dtype=torch.long)
    for group_index, (name, users) in enumerate(groups):
        valid_scores = {
            k: metrics_by_k[k]["valid"]["NDCG"][users].mean().item() for k in k_values
        }
        # Stable, conservative tie-break: use the smaller scope.
        selected_k = max(sorted(k_values), key=lambda k: valid_scores[k])
        selected_for_user[users] = selected_k
        pairwise = {}
        for index, lower_k in enumerate(k_values):
            for upper_k in k_values[index + 1:]:
                key = f"K{upper_k}-K{lower_k}"
                pairwise[key] = paired_bootstrap_delta(
                    metrics_by_k[upper_k]["valid"]["NDCG"],
                    metrics_by_k[lower_k]["valid"]["NDCG"],
                    users,
                    samples=bootstrap_samples,
                    seed=1000 + group_index * 100 + lower_k + upper_k,
                )
        report_groups.append({
            "name": name,
            "num_users": users.numel(),
            "selected_K_from_validation": selected_k,
            "validation": {
                str(k): {
                    "Recall@20": metrics_by_k[k]["valid"]["Recall"][users].mean().item(),
                    "NDCG@20": valid_scores[k],
                } for k in k_values
            },
            "test": {
                str(k): {
                    "Recall@20": metrics_by_k[k]["test"]["Recall"][users].mean().item(),
                    "NDCG@20": metrics_by_k[k]["test"]["NDCG"][users].mean().item(),
                } for k in k_values
            },
            "validation_pairwise_NDCG_deltas": pairwise,
        })

    routed = {}
    for split in ("valid", "test"):
        routed[split] = {}
        for metric in ("Recall", "NDCG"):
            values = torch.empty_like(metrics_by_k[k_values[0]][split][metric])
            for k in k_values:
                users = selected_for_user == k
                values[users] = metrics_by_k[k][split][metric][users]
            routed[split][f"{metric}@20"] = values.mean().item()

    global_valid = {
        k: metrics_by_k[k]["valid"]["NDCG"].mean().item() for k in k_values
    }
    best_fixed_k = max(sorted(k_values), key=lambda k: global_valid[k])
    fixed = {
        split: {
            f"{metric}@20": metrics_by_k[best_fixed_k][split][metric].mean().item()
            for metric in ("Recall", "NDCG")
        } for split in ("valid", "test")
    }
    return report_groups, selected_for_user, {
        "best_fixed_K_selected_from_validation": best_fixed_k,
        "best_fixed": fixed,
        "group_routed": routed,
        "test_gain_over_best_fixed": {
            metric: routed["test"][metric] - fixed["test"][metric]
            for metric in ("Recall@20", "NDCG@20")
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="citeulike")
    parser.add_argument("--T_backbone", default="bpr")
    parser.add_argument("--S_backbone", default="bpr")
    parser.add_argument("--model", default="rcekd")
    parser.add_argument("--student_dim", type=int, default=5)
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
    args.K_values = sorted(set(args.K_values))
    if args.eval_K != 20:
        raise ValueError("this report currently labels and routes @20 metrics; use --eval_K=20")
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
        "configs", args.dataset, args.S_backbone, "base_config.yaml"
    ))
    defaults = {"dataset": args.dataset, "DATA_DIR": "data/"}
    teacher_cfg = parse_cfg(
        argparse.Namespace(**defaults),
        backbone_cfg["teacher"][args.T_backbone.lower()],
    )
    student_cfg = parse_cfg(
        argparse.Namespace(**defaults),
        backbone_cfg["student"],
        {"embedding_dim": args.student_dim},
    )

    teacher_dir = os.path.join(
        "checkpoints", args.dataset, args.T_backbone,
        f"scratch-{teacher_cfg.embedding_dim}",
    )
    teacher_ckpt = os.path.join(teacher_dir, "BEST_EPOCH.pt")
    teacher_scores = build_score_mat(
        teacher_ckpt, args.T_backbone, teacher_cfg, trainset, device,
    )
    teacher_anchor = torch.topk(teacher_scores, args.anchor_K, dim=1).indices.cpu()
    del teacher_scores
    torch.cuda.empty_cache()

    metrics_by_k = {}
    overlap_by_k = {}
    checkpoint_paths = {}
    for k in args.K_values:
        suffix = f"capacity_d{args.student_dim}_k{k}_l{args.L}_seed{args.seed}"
        checkpoint = os.path.join(
            "checkpoints", args.dataset, args.S_backbone,
            f"{args.model.lower()}-{args.student_dim}_{suffix}", "BEST_EPOCH.pt",
        )
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(checkpoint)
        print(f"[K={k}] loading {checkpoint}")
        scores = build_score_mat(
            checkpoint, args.S_backbone, student_cfg, trainset, device,
        )
        student_anchor = torch.topk(scores, args.anchor_K, dim=1).indices.cpu()
        overlap_by_k[k] = row_overlap_ratio(teacher_anchor, student_anchor)
        recommendations = topk_excluding_train_batched(
            scores, train_dict, args.eval_K,
        )
        metrics_by_k[k] = {
            "valid": per_user_recall_ndcg(recommendations, valid_dict, num_users, args.eval_K),
            "test": per_user_recall_ndcg(recommendations, test_dict, num_users, args.eval_K),
        }
        checkpoint_paths[str(k)] = checkpoint
        print(
            f"      overlap={overlap_by_k[k].mean():.4f}  "
            f"valid NDCG@20={metrics_by_k[k]['valid']['NDCG'].mean():.5f}  "
            f"test NDCG@20={metrics_by_k[k]['test']['NDCG'].mean():.5f}"
        )
        del scores, student_anchor, recommendations
        gc.collect()
        torch.cuda.empty_cache()

    mean_overlap = torch.stack([overlap_by_k[k] for k in args.K_values]).mean(dim=0)
    groups = balanced_overlap_groups(mean_overlap, args.num_groups)
    group_reports, _selected_for_user, routing = select_and_route(
        metrics_by_k, groups, args.K_values, args.bootstrap_samples,
    )
    for group_report, (_, users) in zip(group_reports, groups):
        group_report["overlap"] = {
            "mean": mean_overlap[users].mean().item(),
            "min": mean_overlap[users].min().item(),
            "max": mean_overlap[users].max().item(),
        }

    valid_ndcg = torch.stack([
        metrics_by_k[k]["valid"]["NDCG"] for k in args.K_values
    ], dim=1)
    maxima = valid_ndcg.max(dim=1).values
    tied_for_best = valid_ndcg == maxima.unsqueeze(1)
    informative = maxima > 0
    unique = informative & (tied_for_best.sum(dim=1) == 1)
    winner_diagnostic = {
        "all_models_zero_rate": (~informative).double().mean().item(),
        "positive_but_tied_rate": (informative & ~unique).double().mean().item(),
        "unique_winner_rate": unique.double().mean().item(),
        "unique_winner_counts": {
            str(k): (unique & tied_for_best[:, index]).sum().item()
            for index, k in enumerate(args.K_values)
        },
    }

    aggregate = {
        str(k): {
            split: {
                "Recall@20": metrics_by_k[k][split]["Recall"].mean().item(),
                "NDCG@20": metrics_by_k[k][split]["NDCG"].mean().item(),
            } for split in ("valid", "test")
        } for k in args.K_values
    }
    report = {
        "experiment": {
            "dataset": args.dataset,
            "teacher": args.T_backbone,
            "student": args.S_backbone,
            "student_dim": args.student_dim,
            "K_values": args.K_values,
            "L": args.L,
            "seed": args.seed,
            "anchor_K": args.anchor_K,
            "eval_K": args.eval_K,
            "difficulty": "mean final top-anchor overlap across compared K checkpoints",
            "checkpoints": checkpoint_paths,
        },
        "aggregate_metrics": aggregate,
        "per_user_validation_winners": winner_diagnostic,
        "groups": group_reports,
        "validation_selected_routing": routing,
        "limitations": [
            "Difficulty is measured after training and is descriptive, not an online policy state.",
            "Each checkpoint was already selected using validation; routing reuses validation.",
            "Per-user held-out feedback is sparse; use group aggregates and paired intervals, not raw argmax counts alone.",
            "A routed mixture of fixed-K models does not prove a single dynamically trained model will improve.",
        ],
    }

    output = args.output or os.path.join(
        "logs", f"per_user_k_{args.dataset}_{args.S_backbone}_d{args.student_dim}_seed{args.seed}.json",
    )
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\nGroup selection (validation only):")
    for group in group_reports:
        scores = ", ".join(
            f"K={k}: {group['validation'][str(k)]['NDCG@20']:.5f}"
            for k in args.K_values
        )
        print(
            f"  {group['name']:>6s} overlap={group['overlap']['mean']:.4f}: "
            f"{scores} -> K={group['selected_K_from_validation']}"
        )
    print(
        "Routing test: "
        f"NDCG@20={routing['group_routed']['test']['NDCG@20']:.5f}, "
        f"Recall@20={routing['group_routed']['test']['Recall@20']:.5f}; "
        f"best fixed validation-selected K={routing['best_fixed_K_selected_from_validation']}"
    )
    print(f"report saved to {output}")


if __name__ == "__main__":
    main()

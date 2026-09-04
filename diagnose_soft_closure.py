"""Zero-training validation of the soft-closure NDCG correction.

The script reconstructs RCE-KD's L2 item set from an existing checkpoint,
computes the exact full-catalog closure-missing mass, and compares it with the
mass visible inside Student top-M.  It also verifies the corrected lower bound
numerically before the term is introduced into training.
"""

import argparse
import gc
import json
import os

import torch

from dataset import implicit_CF_dataset, load_cf_data
from diagnose import build_score_mat
from soft_closure import (
    exact_partial_log_ndcg,
    exact_soft_closure_rho,
    pearson_correlation,
    soft_closure_bound_terms,
    summarize_tensor,
    topm_soft_closure_rho,
)
from utils import load_yaml
from utils.parse_utils import parse_cfg


def rowwise_isin(left, right):
    return (left.unsqueeze(2) == right.unsqueeze(1)).any(dim=2)


def reconstruct_rcekd_l2_sets(teacher_scores, student_scores, k, length,
                              max_k, sampling_temperature, seed):
    """Reproduce RCE-KD's score-count sampling and deduplication masks."""
    teacher_topk = torch.topk(teacher_scores, k, dim=1).indices
    student_topm = torch.topk(student_scores, max_k, dim=1).indices
    student_topk = student_topm[:, :k]

    weight = torch.zeros_like(student_topm, dtype=student_scores.dtype)
    matches = (teacher_topk.unsqueeze(2) == student_topm.unsqueeze(1)).nonzero()
    if matches.numel() > 0:
        weight[matches[:, 0], matches[:, 2]] += 1
    weight = torch.minimum(
        torch.cumsum(weight.flip(1), dim=1).flip(1),
        weight.new_tensor(50.),
    )
    weight = torch.exp((weight + 1.) / sampling_temperature)
    generator = torch.Generator(device=student_scores.device).manual_seed(seed)
    sampled_positions = torch.multinomial(
        weight, length, replacement=False, generator=generator,
    )
    interesting = torch.gather(student_topm, 1, sampled_positions)

    # This matches RCEKD.get_loss: every sampled item is active; a teacher
    # item is dropped from L2 when it is already sampled or lies in Q_S.
    active_teacher = ~(
        rowwise_isin(teacher_topk, interesting)
        | rowwise_isin(teacher_topk, student_topk)
    )
    items = torch.cat([interesting, teacher_topk], dim=1)
    active = torch.cat([
        torch.ones_like(interesting, dtype=torch.bool), active_teacher,
    ], dim=1)
    return {
        "items": items,
        "active": active,
        "teacher_topk": teacher_topk,
        "student_topm": student_topm,
        "interesting": interesting,
    }


def resolve_student_checkpoint(args):
    directory = f"{args.model.lower()}-{args.student_dim}"
    if args.suffix:
        directory += f"_{args.suffix}"
    checkpoint = os.path.join(
        "checkpoints", args.dataset, args.S_backbone, directory, "BEST_EPOCH.pt",
    )
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(checkpoint)
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="citeulike")
    parser.add_argument("--T_backbone", default="bpr")
    parser.add_argument("--S_backbone", default="bpr")
    parser.add_argument("--model", default="rcekd")
    parser.add_argument("--student_dim", type=int, default=20)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--L", type=int, default=None)
    parser.add_argument("--mxK", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--user_batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("full-catalog checkpoint diagnostics require CUDA")
    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda")

    (num_users, num_items, train_pairs, _valid_pairs, _test_pairs,
     train_dict, _valid_dict, _test_dict, train_matrix, user_pop, item_pop) = load_cf_data(
        args.dataset
    )
    trainset = implicit_CF_dataset(
        args.dataset, num_users, num_items, train_pairs, train_matrix,
        train_dict, user_pop, item_pop, num_ns=1, no_neg_sampling=True,
    )
    defaults = {"dataset": args.dataset, "DATA_DIR": "data/"}
    backbone_cfg = load_yaml(os.path.join(
        "configs", args.dataset, args.S_backbone, "base_config.yaml",
    ))
    teacher_cfg = parse_cfg(
        argparse.Namespace(**defaults),
        backbone_cfg["teacher"][args.T_backbone.lower()],
    )
    student_cfg = parse_cfg(
        argparse.Namespace(**defaults), backbone_cfg["student"],
        {"embedding_dim": args.student_dim},
    )
    rce_cfg = load_yaml(os.path.join(
        "configs", args.dataset, args.S_backbone, "rcekd.yaml",
    ))[args.T_backbone.lower()]
    k = args.K if args.K is not None else int(rce_cfg["mkd_K"])
    length = args.L if args.L is not None else int(rce_cfg["mkd_L"])
    max_k = args.mxK if args.mxK is not None else int(rce_cfg["mkd_mxK"])
    tau = float(rce_cfg["mkd_tau"])
    sampling_temperature = float(rce_cfg["mkd_T"])
    if not (0 < k <= max_k <= num_items):
        raise ValueError("expected 0 < K <= mxK <= num_items")
    if not (0 < length <= max_k):
        raise ValueError("expected 0 < L <= mxK")

    teacher_checkpoint = os.path.join(
        "checkpoints", args.dataset, args.T_backbone,
        f"scratch-{teacher_cfg.embedding_dim}", "BEST_EPOCH.pt",
    )
    student_checkpoint = resolve_student_checkpoint(args)
    print(f"[teacher] {teacher_checkpoint}")
    teacher_scores = build_score_mat(
        teacher_checkpoint, args.T_backbone, teacher_cfg, trainset, device,
    ) / tau
    print(f"[student] {student_checkpoint}")
    student_scores = build_score_mat(
        student_checkpoint, args.S_backbone, student_cfg, trainset, device,
    ) / tau

    print(f"[sets] reconstructing RCE-KD L2 with K={k}, L={length}, mxK={max_k}")
    sets = reconstruct_rcekd_l2_sets(
        teacher_scores, student_scores, k, length, max_k,
        sampling_temperature, args.seed,
    )

    collected = {
        "exact_penalty": [],
        "topm_penalty": [],
        "ce": [],
        "log_c_j": [],
        "exact_bound": [],
        "topm_bound": [],
        "ce_only_bound": [],
        "log_ndcg": [],
        "exact_weighted_rho": [],
        "topm_weighted_rho": [],
        "active_size": [],
    }
    exact_rho_active = []
    topm_rho_active = []
    print("[diagnostic] computing exact full-catalog and top-M closure mass")
    for start in range(0, num_users, args.user_batch_size):
        end = min(start + args.user_batch_size, num_users)
        scores_s = student_scores[start:end]
        scores_t = teacher_scores[start:end]
        items = sets["items"][start:end]
        active = sets["active"][start:end]
        topm = sets["student_topm"][start:end]

        exact_rho = exact_soft_closure_rho(scores_s, items, active)
        topm_rho = topm_soft_closure_rho(scores_s, items, topm, active)
        exact = soft_closure_bound_terms(
            scores_s, scores_t, items, active, exact_rho,
        )
        approximate = soft_closure_bound_terms(
            scores_s, scores_t, items, active, topm_rho,
        )
        log_ndcg = exact_partial_log_ndcg(scores_s, scores_t, items, active)
        p_t = exact["teacher_prob_j"]

        collected["exact_penalty"].append(exact["penalty"].cpu())
        collected["topm_penalty"].append(approximate["penalty"].cpu())
        collected["ce"].append(exact["ce"].cpu())
        collected["log_c_j"].append(exact["log_c_j"].cpu())
        collected["exact_bound"].append(exact["lower_bound"].cpu())
        collected["topm_bound"].append(approximate["lower_bound"].cpu())
        collected["ce_only_bound"].append((-exact["ce"] + exact["log_c_j"]).cpu())
        collected["log_ndcg"].append(log_ndcg.cpu())
        collected["exact_weighted_rho"].append((p_t * exact_rho).sum(1).cpu())
        collected["topm_weighted_rho"].append((p_t * topm_rho).sum(1).cpu())
        collected["active_size"].append(active.sum(1).cpu())
        exact_rho_active.append(exact_rho[active].cpu())
        topm_rho_active.append(topm_rho[active].cpu())

        if start == 0 or end == num_users or (start // args.user_batch_size) % 25 == 0:
            print(f"  users {start}:{end}/{num_users}")
        del exact_rho, topm_rho, exact, approximate, log_ndcg, p_t

    values = {name: torch.cat(parts).double() for name, parts in collected.items()}
    exact_rho_active = torch.cat(exact_rho_active).double()
    topm_rho_active = torch.cat(topm_rho_active).double()
    tolerance = 1e-6
    nonzero = values["exact_penalty"] > 1e-12
    penalty_coverage = values["topm_penalty"][nonzero] / values["exact_penalty"][nonzero]
    rho_coverage = torch.where(
        exact_rho_active > 1e-12,
        topm_rho_active / exact_rho_active.clamp_min(1e-12),
        torch.ones_like(exact_rho_active),
    )

    report = {
        "experiment": {
            "dataset": args.dataset,
            "teacher": args.T_backbone,
            "student": args.S_backbone,
            "student_dim": args.student_dim,
            "student_checkpoint": student_checkpoint,
            "teacher_checkpoint": teacher_checkpoint,
            "K": k,
            "L": length,
            "mxK": max_k,
            "tau": tau,
            "sampling_temperature": sampling_temperature,
            "seed": args.seed,
            "rho_definition": "sum_{j outside J, s_j>=s_i} exp(s_j) / sum_{j in J} exp(s_j)",
        },
        "per_user": {
            name: summarize_tensor(tensor) for name, tensor in values.items()
        },
        "per_active_item": {
            "exact_rho": summarize_tensor(exact_rho_active),
            "topm_rho": summarize_tensor(topm_rho_active),
            "topm_over_exact_rho": summarize_tensor(rho_coverage),
        },
        "approximation": {
            "topm_over_exact_penalty": summarize_tensor(penalty_coverage),
            "pearson_exact_vs_topm_penalty": pearson_correlation(
                values["exact_penalty"], values["topm_penalty"],
            ),
        },
        "bound_checks": {
            "exact_bound_violation_count": (
                values["exact_bound"] > values["log_ndcg"] + tolerance
            ).sum().item(),
            "exact_bound_max_violation": (
                values["exact_bound"] - values["log_ndcg"]
            ).max().item(),
            "ce_only_violation_count": (
                values["ce_only_bound"] > values["log_ndcg"] + tolerance
            ).sum().item(),
            "topm_approx_bound_violation_count": (
                values["topm_bound"] > values["log_ndcg"] + tolerance
            ).sum().item(),
            "tolerance": tolerance,
        },
        "interpretation_guardrails": [
            "The exact penalty coefficient is one in the derived lower bound; a tunable coefficient is heuristic.",
            "Top-M rho is a lower estimate of missing mass, so its induced bound is not guaranteed.",
            "Rank-indicator derivatives are zero almost everywhere; score-mass derivatives remain available.",
            "This diagnostic validates the objective and approximation only; it is not a recommendation-performance result.",
        ],
    }
    output = args.output or os.path.join(
        "logs", f"soft_closure_{args.dataset}_{args.T_backbone}-{args.S_backbone}_"
        f"d{args.student_dim}_k{k}_l{length}_m{max_k}_seed{args.seed}.json",
    )
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    checks = report["bound_checks"]
    coverage = report["approximation"]["topm_over_exact_penalty"]
    print("\n[summary]")
    print(f"  exact-bound violations: {checks['exact_bound_violation_count']}")
    print(f"  CE-only violations: {checks['ce_only_violation_count']}")
    print(f"  top-M approximate-bound violations: {checks['topm_approx_bound_violation_count']}")
    print(f"  top-M/exact penalty median: {coverage['median']:.4f}")
    print(f"  top-M/exact penalty mean: {coverage['mean']:.4f}")
    print(f"  report saved to {output}")

    del teacher_scores, student_scores, sets
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

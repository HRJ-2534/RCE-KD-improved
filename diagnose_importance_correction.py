"""Diagnose sampled-CE gradient bias and log-inclusion corrections.

No model is trained.  The exact reference is CE on the unique union of the
student Top-M and teacher Top-K.  Every sampled estimator uses the same ARCE
Marginal proposal and the same sampled sets, then embeds its analytical logit
gradient back into the reference coordinates.
"""

import argparse
import gc
import json
import os

import torch

from compare_soft_closure_samplers import resolve_student_checkpoint
from dataset import implicit_CF_dataset, load_cf_data
from diagnose import build_score_mat
from importance_correction import (
    embed_sample_gradient,
    estimate_inclusion_probabilities,
    masked_softmax,
    sample_weighted_prefix,
    sampled_ce_logit_gradient,
)
from modeling.KD.anchor_rce import (
    rowwise_isin,
    topm_closure_statistics,
    topm_marginal_blocker_probabilities,
)
from soft_closure import summarize_tensor
from utils import load_yaml
from utils.parse_utils import parse_cfg


METHODS = (
    "sampled_uncorrected",
    "log_bq_student",
    "log_bq_both",
    "log_wr_inclusion_both",
    "log_mc_inclusion_student",
    "log_mc_inclusion_both",
)


def reference_gradient(student_logits, teacher_logits, active):
    return (
        masked_softmax(student_logits, active)
        - masked_softmax(teacher_logits, active)
    ) * active


def sample_items_and_logits(scores_s, scores_t, teacher_topk, student_topm,
                            positions, blocker_active):
    blocker_items = student_topm.gather(1, positions)
    items = torch.cat([teacher_topk, blocker_items], dim=1)
    active = torch.cat([
        torch.ones_like(teacher_topk, dtype=torch.bool), blocker_active,
    ], dim=1)
    return (
        items, active, scores_s.gather(1, items), scores_t.gather(1, items),
    )


def sample_repeated_trial_items_and_logits(
        scores_s, scores_t, teacher_topk, student_topm, positions,
        blocker_active, trial_chunk):
    repeated_teacher = teacher_topk.repeat_interleave(trial_chunk, dim=0)
    repeated_topm = student_topm.repeat_interleave(trial_chunk, dim=0)
    blocker_items = repeated_topm.gather(1, positions)
    items = torch.cat([repeated_teacher, blocker_items], dim=1)
    active = torch.cat([
        torch.ones_like(repeated_teacher, dtype=torch.bool), blocker_active,
    ], dim=1)
    items_by_trial = items.reshape(
        scores_s.shape[0], trial_chunk, items.shape[1],
    )
    sample_s = torch.gather(
        scores_s.unsqueeze(1).expand(-1, trial_chunk, -1),
        2, items_by_trial,
    ).reshape_as(items)
    sample_t = torch.gather(
        scores_t.unsqueeze(1).expand(-1, trial_chunk, -1),
        2, items_by_trial,
    ).reshape_as(items)
    return items, active, sample_s, sample_t


def compose_inclusion(selected_inclusion, blocker_active, teacher_width):
    blocker_inclusion = torch.where(
        blocker_active,
        selected_inclusion.clamp_min(
            torch.finfo(selected_inclusion.dtype).tiny
        ),
        torch.ones_like(selected_inclusion),
    )
    return torch.cat([
        torch.ones(
            (selected_inclusion.shape[0], teacher_width),
            device=selected_inclusion.device, dtype=selected_inclusion.dtype,
        ),
        blocker_inclusion,
    ], dim=1)


def initialize_accumulators(batch_size, reference_width, device, dtype):
    return {
        method: {
            "gradient_sum": torch.zeros(
                (batch_size, reference_width), device=device, dtype=dtype,
            ),
            "gradient_square_norm_sum": torch.zeros(
                batch_size, device=device, dtype=dtype,
            ),
            "draw_cosine_sum": torch.zeros(
                batch_size, device=device, dtype=dtype,
            ),
        }
        for method in METHODS
    }


def add_gradient_chunk(accumulator, embedded, reference, reference_norm,
                       chunk_size):
    gradients = embedded.reshape(
        reference.shape[0], chunk_size, reference.shape[1],
    )
    accumulator["gradient_sum"] += gradients.sum(dim=1)
    accumulator["gradient_square_norm_sum"] += (
        gradients.square().sum(dim=2).sum(dim=1)
    )
    accumulator["draw_cosine_sum"] += (
        (gradients * reference.unsqueeze(1)).sum(dim=2)
        / gradients.norm(dim=2).clamp_min(1e-12)
        / reference_norm.unsqueeze(1).clamp_min(1e-12)
    ).sum(dim=1)


def finalize_batch(accumulator, reference, reference_norm, trials):
    mean_gradient = accumulator["gradient_sum"] / trials
    bias = mean_gradient - reference
    reference_norm_square = reference_norm.square().clamp_min(1e-12)
    variance = (
        accumulator["gradient_square_norm_sum"] / trials
        - mean_gradient.square().sum(dim=1)
    ).clamp_min(0.)
    return {
        "mean_gradient_cosine": (
            (mean_gradient * reference).sum(dim=1)
            / mean_gradient.norm(dim=1).clamp_min(1e-12)
            / reference_norm.clamp_min(1e-12)
        ),
        "relative_l2_bias": bias.norm(dim=1) / reference_norm.clamp_min(1e-12),
        "mean_gradient_norm_ratio": (
            mean_gradient.norm(dim=1) / reference_norm.clamp_min(1e-12)
        ),
        "variance_over_reference_norm2": variance / reference_norm_square,
        "mean_draw_cosine": accumulator["draw_cosine_sum"] / trials,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="citeulike")
    parser.add_argument("--T_backbone", default="bpr")
    parser.add_argument("--S_backbone", default="bpr")
    parser.add_argument("--model", default="arcekd")
    parser.add_argument("--student_dim", type=int, default=20)
    parser.add_argument("--suffix", default="anchor_marginal_es20")
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--L", type=int, default=None)
    parser.add_argument("--mxK", type=int, default=None)
    parser.add_argument("--mix_alpha", type=float, default=.9)
    parser.add_argument("--calibration_trials", type=int, default=1024)
    parser.add_argument("--calibration_chunk", type=int, default=64)
    parser.add_argument("--evaluation_trials", type=int, default=128)
    parser.add_argument("--evaluation_chunk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--user_batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint gradient diagnostics require CUDA")
    if (
            args.calibration_trials < 1 or args.evaluation_trials < 1
            or args.calibration_chunk < 1 or args.evaluation_chunk < 1):
        raise ValueError("trial counts must be positive")
    if not 0. <= args.mix_alpha <= 1.:
        raise ValueError("mix_alpha must be between zero and one")
    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda")

    (num_users, num_items, train_pairs, _valid_pairs, _test_pairs,
     train_dict, _valid_dict, _test_dict, train_matrix, user_pop, item_pop) = (
        load_cf_data(args.dataset)
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
    if not (0 < k <= max_k <= num_items and 0 < length <= max_k - k):
        raise ValueError("expected 0 < K <= mxK and L <= mxK-K")

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

    teacher_topk = torch.topk(teacher_scores, k, dim=1).indices
    student_scores_m, student_topm = torch.topk(
        student_scores, max_k, dim=1,
    )
    student_topk = student_topm[:, :k]
    q1_active = rowwise_isin(teacher_topk, student_topk)
    q2_active = ~q1_active
    budgets_all = length - q1_active.sum(dim=1)
    eligible_users = budgets_all > 0
    print(
        f"[users] {eligible_users.sum().item()}/{num_users} require blockers; "
        f"mean budget={budgets_all[eligible_users].float().mean().item():.4f}"
    )

    metric_parts = {
        method: {
            metric: [] for metric in (
                "mean_gradient_cosine", "relative_l2_bias",
                "mean_gradient_norm_ratio",
                "variance_over_reference_norm2", "mean_draw_cosine",
            )
        }
        for method in METHODS
    }
    mc_standard_error_parts = []
    mc_sum_error_parts = []
    wr_mc_difference_parts = []
    calibration_generator = torch.Generator(device=device).manual_seed(
        args.seed + 17041,
    )
    evaluation_generator = torch.Generator(device=device).manual_seed(
        args.seed + 65537,
    )

    print(
        "[diagnostic] exact Top-M union reference; independent calibration "
        "and evaluation draws"
    )
    for start in range(0, num_users, args.user_batch_size):
        end = min(start + args.user_batch_size, num_users)
        keep = eligible_users[start:end]
        if not keep.any():
            continue
        scores_s = student_scores[start:end][keep]
        scores_t = teacher_scores[start:end][keep]
        items_t = teacher_topk[start:end][keep]
        items_m = student_topm[start:end][keep]
        scores_m = student_scores_m[start:end][keep]
        active_q2 = q2_active[start:end][keep]
        budgets = budgets_all[start:end][keep]
        scores_s_t = scores_s.gather(1, items_t)
        scores_t_t = scores_t.gather(1, items_t)
        closure_statistics = topm_closure_statistics(
            scores_m, scores_s_t, items_t, items_m, active_q2,
        )
        proposal = topm_marginal_blocker_probabilities(
            scores_m, scores_s_t, scores_t_t, items_t, items_m,
            active_q2, args.mix_alpha,
            closure_statistics=closure_statistics,
        )

        mc_inclusion, mc_standard_error = estimate_inclusion_probabilities(
            proposal, budgets, length, args.calibration_trials,
            args.calibration_chunk, calibration_generator,
        )
        support = proposal > 0.
        wr_inclusion = 1. - torch.pow(
            1. - proposal, budgets.to(proposal.dtype).unsqueeze(1),
        )
        mc_standard_error_parts.append(
            (mc_standard_error * support).sum(dim=1)
            .div(support.sum(dim=1).clamp_min(1)).cpu()
        )
        mc_sum_error_parts.append(
            (mc_inclusion.sum(dim=1) - budgets).abs().cpu()
        )
        wr_mc_difference_parts.append(
            ((wr_inclusion - mc_inclusion).abs() * support).sum(dim=1)
            .div(support.sum(dim=1).clamp_min(1)).cpu()
        )

        teacher_in_topm = rowwise_isin(items_t, items_m)
        reference_items = torch.cat([items_m, items_t], dim=1)
        reference_active = torch.cat([
            torch.ones_like(items_m, dtype=torch.bool), ~teacher_in_topm,
        ], dim=1)
        reference_student = scores_s.gather(1, reference_items)
        reference_teacher = scores_t.gather(1, reference_items)
        reference = reference_gradient(
            reference_student, reference_teacher, reference_active,
        )
        reference_norm = reference.norm(dim=1)
        accumulators = initialize_accumulators(
            scores_s.shape[0], reference_items.shape[1], device,
            scores_s.dtype,
        )

        completed_trials = 0
        while completed_trials < args.evaluation_trials:
            trial_chunk = min(
                args.evaluation_chunk,
                args.evaluation_trials - completed_trials,
            )
            repeated_proposal = proposal.repeat_interleave(trial_chunk, dim=0)
            repeated_budgets = budgets.repeat_interleave(trial_chunk)
            positions, blocker_active = sample_weighted_prefix(
                repeated_proposal, repeated_budgets, length,
                evaluation_generator,
            )
            items, sample_active, sample_s, sample_t = (
                sample_repeated_trial_items_and_logits(
                    scores_s, scores_t, items_t, items_m,
                    positions, blocker_active, trial_chunk,
                )
            )
            selected_q = repeated_proposal.gather(1, positions)
            budget_float = repeated_budgets.to(proposal.dtype).unsqueeze(1)
            inclusion_bq = compose_inclusion(
                (budget_float * selected_q).clamp(max=1.),
                blocker_active, k,
            )
            inclusion_wr = compose_inclusion(
                wr_inclusion.repeat_interleave(
                    trial_chunk, dim=0,
                ).gather(1, positions), blocker_active, k,
            )
            inclusion_mc = compose_inclusion(
                mc_inclusion.repeat_interleave(
                    trial_chunk, dim=0,
                ).gather(1, positions), blocker_active, k,
            )
            method_gradients = {
                "sampled_uncorrected": sampled_ce_logit_gradient(
                    sample_s, sample_t, sample_active,
                ),
                "log_bq_student": sampled_ce_logit_gradient(
                    sample_s, sample_t, sample_active, inclusion_bq,
                    correct_teacher=False,
                ),
                "log_bq_both": sampled_ce_logit_gradient(
                    sample_s, sample_t, sample_active, inclusion_bq,
                    correct_teacher=True,
                ),
                "log_wr_inclusion_both": sampled_ce_logit_gradient(
                    sample_s, sample_t, sample_active, inclusion_wr,
                    correct_teacher=True,
                ),
                "log_mc_inclusion_student": sampled_ce_logit_gradient(
                    sample_s, sample_t, sample_active, inclusion_mc,
                    correct_teacher=False,
                ),
                "log_mc_inclusion_both": sampled_ce_logit_gradient(
                    sample_s, sample_t, sample_active, inclusion_mc,
                    correct_teacher=True,
                ),
            }
            repeated_reference_items = reference_items.repeat_interleave(
                trial_chunk, dim=0,
            )
            repeated_reference_active = reference_active.repeat_interleave(
                trial_chunk, dim=0,
            )
            for method, gradient in method_gradients.items():
                embedded = embed_sample_gradient(
                    gradient, items, sample_active,
                    repeated_reference_items, repeated_reference_active,
                )
                add_gradient_chunk(
                    accumulators[method], embedded, reference, reference_norm,
                    trial_chunk,
                )
            completed_trials += trial_chunk

        for method in METHODS:
            batch_metrics = finalize_batch(
                accumulators[method], reference, reference_norm,
                args.evaluation_trials,
            )
            for metric, values in batch_metrics.items():
                metric_parts[method][metric].append(values.detach().cpu())

        if start == 0 or end == num_users or (
                start // args.user_batch_size) % 25 == 0:
            print(f"  users {start}:{end}/{num_users}")

    methods = {
        method: {
            metric: summarize_tensor(torch.cat(parts))
            for metric, parts in metrics.items()
        }
        for method, metrics in metric_parts.items()
    }
    report = {
        "experiment": {
            "dataset": args.dataset,
            "teacher": args.T_backbone,
            "student": args.S_backbone,
            "model": args.model,
            "student_dim": args.student_dim,
            "suffix": args.suffix,
            "teacher_checkpoint": teacher_checkpoint,
            "student_checkpoint": student_checkpoint,
            "K": k,
            "L": length,
            "mxK": max_k,
            "mix_alpha": args.mix_alpha,
            "calibration_trials": args.calibration_trials,
            "evaluation_trials": args.evaluation_trials,
            "evaluation_chunk": args.evaluation_chunk,
            "seed": args.seed,
            "reference_objective": (
                "CE on the unique union of student Top-M and teacher Top-K"
            ),
        },
        "eligible_users": eligible_users.sum().item(),
        "blocker_budget": summarize_tensor(budgets_all[eligible_users]),
        "inclusion_estimation": {
            "mean_mc_standard_error_per_user": summarize_tensor(
                torch.cat(mc_standard_error_parts)
            ),
            "absolute_sum_pi_minus_budget": summarize_tensor(
                torch.cat(mc_sum_error_parts)
            ),
            "mean_absolute_wr_approx_minus_mc_pi": summarize_tensor(
                torch.cat(wr_mc_difference_parts)
            ),
        },
        "methods": methods,
        "guardrails": [
            "This is a checkpoint-level logit-gradient diagnostic, not a training result.",
            "Every estimator is compared with exact CE on the same Top-M plus teacher-Top-K reference set.",
            "All estimators use identical evaluation samples; Monte Carlo inclusion calibration uses an independent random stream.",
            "Bq is the conventional with-replacement log-Q approximation and is not exact for the repository's weighted sampling without replacement.",
            "The with-replacement inclusion approximation is 1-(1-q)^B.",
            "MC inclusion probabilities use Jeffreys smoothing followed by the fixed-size identity sum(pi)=B; budget-one rows use pi=q exactly.",
            "Self-normalization means even exact inclusion probabilities do not make the ratio estimator strictly unbiased; empirical gradient bias is therefore the decision endpoint.",
        ],
    }
    output = args.output or os.path.join(
        "logs", f"importance_correction_{args.dataset}_"
        f"{args.T_backbone}-{args.S_backbone}_{args.model}_d{args.student_dim}_"
        f"k{k}_l{length}_m{max_k}_seed{args.seed}.json",
    )
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\n[summary: gradient against exact Top-M union CE]")
    for method in METHODS:
        result = methods[method]
        print(
            f"  {method:28s} "
            f"mean-grad-cos={result['mean_gradient_cosine']['mean']:.6f} "
            f"rel-bias={result['relative_l2_bias']['mean']:.6f} "
            f"variance/ref2={result['variance_over_reference_norm2']['mean']:.6f} "
            f"draw-cos={result['mean_draw_cosine']['mean']:.6f}"
        )
    print(f"  report saved to {output}")

    del teacher_scores, student_scores
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

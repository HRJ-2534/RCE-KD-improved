"""Select and evaluate bias-variance shrinkage for sampled CE gradients.

No model is trained.  A first independent set of paired samples estimates the
global MSE-optimal coefficient between current sampled CE and bilateral log-Q
corrected CE.  A second independent set evaluates that coefficient and a
fixed grid against exact CE on the student Top-M plus teacher Top-K union.
"""

import argparse
import gc
import json
import os

import torch

from compare_soft_closure_samplers import resolve_student_checkpoint
from dataset import implicit_CF_dataset, load_cf_data
from diagnose import build_score_mat
from diagnose_importance_correction import (
    compose_inclusion,
    reference_gradient,
    sample_repeated_trial_items_and_logits,
)
from importance_correction import (
    embed_sample_gradient,
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


DEFAULT_LAMBDAS = (0., .1, .25, .5, .75, 1.)


def parse_lambdas(value):
    values = tuple(float(part.strip()) for part in value.split(","))
    if not values or any(value < 0. or value > 1. for value in values):
        raise argparse.ArgumentTypeError(
            "lambdas must be a comma-separated, non-empty list in [0, 1]"
        )
    return values


def build_batch_state(student_scores, teacher_scores, teacher_topk,
                      student_topm, student_scores_m, q2_active, budgets_all,
                      start, end, mix_alpha):
    keep = budgets_all[start:end] > 0
    if not keep.any():
        return None
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
        active_q2, mix_alpha, closure_statistics=closure_statistics,
    )
    teacher_in_topm = rowwise_isin(items_t, items_m)
    reference_items = torch.cat([items_m, items_t], dim=1)
    reference_active = torch.cat([
        torch.ones_like(items_m, dtype=torch.bool), ~teacher_in_topm,
    ], dim=1)
    reference = reference_gradient(
        scores_s.gather(1, reference_items),
        scores_t.gather(1, reference_items),
        reference_active,
    )
    return {
        "scores_s": scores_s,
        "scores_t": scores_t,
        "items_t": items_t,
        "items_m": items_m,
        "budgets": budgets,
        "proposal": proposal,
        "reference_items": reference_items,
        "reference_active": reference_active,
        "reference": reference,
    }


def draw_paired_gradients(state, k, length, trials, trial_chunk, generator):
    completed = 0
    while completed < trials:
        current = min(trial_chunk, trials - completed)
        proposal = state["proposal"]
        repeated_proposal = proposal.repeat_interleave(current, dim=0)
        repeated_budgets = state["budgets"].repeat_interleave(current)
        positions, blocker_active = sample_weighted_prefix(
            repeated_proposal, repeated_budgets, length, generator,
        )
        items, sample_active, sample_s, sample_t = (
            sample_repeated_trial_items_and_logits(
                state["scores_s"], state["scores_t"], state["items_t"],
                state["items_m"], positions, blocker_active, current,
            )
        )
        selected_q = repeated_proposal.gather(1, positions)
        inclusion_bq = compose_inclusion(
            (repeated_budgets.to(proposal.dtype).unsqueeze(1) * selected_q)
            .clamp(max=1.),
            blocker_active, k,
        )
        sampled = sampled_ce_logit_gradient(
            sample_s, sample_t, sample_active,
        )
        corrected = sampled_ce_logit_gradient(
            sample_s, sample_t, sample_active, inclusion_bq,
            correct_teacher=True,
        )
        repeated_reference_items = state["reference_items"].repeat_interleave(
            current, dim=0,
        )
        repeated_reference_active = (
            state["reference_active"].repeat_interleave(current, dim=0)
        )
        embedded_sampled = embed_sample_gradient(
            sampled, items, sample_active, repeated_reference_items,
            repeated_reference_active,
        ).reshape(state["reference"].shape[0], current, -1)
        embedded_corrected = embed_sample_gradient(
            corrected, items, sample_active, repeated_reference_items,
            repeated_reference_active,
        ).reshape(state["reference"].shape[0], current, -1)
        yield embedded_sampled, embedded_corrected
        completed += current


def add_fit_moments(moments, sampled, corrected, reference):
    reference_draws = reference.unsqueeze(1).expand_as(sampled)
    direction = corrected - sampled
    error = sampled - reference_draws
    numerator = (error * direction).sum(dim=2)
    denominator = direction.square().sum(dim=2)
    reference_norm_square = reference.square().sum(dim=1).clamp_min(1e-12)
    moments["normalized_numerator"] += (
        numerator / reference_norm_square.unsqueeze(1)
    ).sum().double().cpu()
    moments["normalized_denominator"] += (
        denominator / reference_norm_square.unsqueeze(1)
    ).sum().double().cpu()
    moments["raw_numerator"] += numerator.sum().double().cpu()
    moments["raw_denominator"] += denominator.sum().double().cpu()


def coefficient_from_moments(moments, prefix):
    denominator = moments[f"{prefix}_denominator"]
    if denominator <= 0.:
        return 0.
    return float((-moments[f"{prefix}_numerator"] / denominator).clamp(0., 1.))


def initialize_evaluation_accumulators(lambdas, batch_size, width, device,
                                       dtype):
    return {
        label: {
            "lambda": value,
            "gradient_sum": torch.zeros(
                (batch_size, width), device=device, dtype=dtype,
            ),
            "gradient_square_norm_sum": torch.zeros(
                batch_size, device=device, dtype=dtype,
            ),
            "squared_error_sum": torch.zeros(
                batch_size, device=device, dtype=dtype,
            ),
            "draw_cosine_sum": torch.zeros(
                batch_size, device=device, dtype=dtype,
            ),
        }
        for label, value in lambdas.items()
    }


def add_evaluation_chunk(accumulators, sampled, corrected, reference):
    reference_draws = reference.unsqueeze(1)
    reference_norm = reference.norm(dim=1).clamp_min(1e-12)
    for accumulator in accumulators.values():
        value = accumulator["lambda"]
        gradient = sampled + value * (corrected - sampled)
        accumulator["gradient_sum"] += gradient.sum(dim=1)
        accumulator["gradient_square_norm_sum"] += (
            gradient.square().sum(dim=2).sum(dim=1)
        )
        accumulator["squared_error_sum"] += (
            (gradient - reference_draws).square().sum(dim=2).sum(dim=1)
        )
        accumulator["draw_cosine_sum"] += (
            (gradient * reference_draws).sum(dim=2)
            / gradient.norm(dim=2).clamp_min(1e-12)
            / reference_norm.unsqueeze(1)
        ).sum(dim=1)


def finalize_evaluation(accumulator, reference, trials):
    mean_gradient = accumulator["gradient_sum"] / trials
    bias = mean_gradient - reference
    reference_norm = reference.norm(dim=1).clamp_min(1e-12)
    reference_norm_square = reference_norm.square()
    variance = (
        accumulator["gradient_square_norm_sum"] / trials
        - mean_gradient.square().sum(dim=1)
    ).clamp_min(0.)
    squared_bias = bias.square().sum(dim=1)
    direct_mse = accumulator["squared_error_sum"] / trials
    return {
        "mean_gradient_cosine": (
            (mean_gradient * reference).sum(dim=1)
            / mean_gradient.norm(dim=1).clamp_min(1e-12)
            / reference_norm
        ),
        "relative_l2_bias": bias.norm(dim=1) / reference_norm,
        "mean_gradient_norm_ratio": mean_gradient.norm(dim=1) / reference_norm,
        "squared_bias_over_reference_norm2": squared_bias / reference_norm_square,
        "variance_over_reference_norm2": variance / reference_norm_square,
        "mse_over_reference_norm2": direct_mse / reference_norm_square,
        "bias_plus_variance_over_reference_norm2": (
            squared_bias + variance
        ) / reference_norm_square,
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
    parser.add_argument("--fit_trials", type=int, default=64)
    parser.add_argument("--evaluation_trials", type=int, default=128)
    parser.add_argument("--trial_chunk", type=int, default=32)
    parser.add_argument("--lambdas", type=parse_lambdas,
                        default=DEFAULT_LAMBDAS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--user_batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint gradient diagnostics require CUDA")
    if args.fit_trials < 1 or args.evaluation_trials < 1 or args.trial_chunk < 1:
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
    q2_active = ~rowwise_isin(teacher_topk, student_topk)
    budgets_all = length - (~q2_active).sum(dim=1)
    eligible_users = budgets_all > 0
    print(
        f"[users] {eligible_users.sum().item()}/{num_users} require blockers; "
        f"mean budget={budgets_all[eligible_users].float().mean().item():.4f}"
    )

    fit_moments = {
        "normalized_numerator": torch.zeros((), dtype=torch.float64),
        "normalized_denominator": torch.zeros((), dtype=torch.float64),
        "raw_numerator": torch.zeros((), dtype=torch.float64),
        "raw_denominator": torch.zeros((), dtype=torch.float64),
    }
    fit_generator = torch.Generator(device=device).manual_seed(args.seed + 31013)
    print("[fit] independent paired draws for global shrinkage coefficient")
    for start in range(0, num_users, args.user_batch_size):
        end = min(start + args.user_batch_size, num_users)
        state = build_batch_state(
            student_scores, teacher_scores, teacher_topk, student_topm,
            student_scores_m, q2_active, budgets_all, start, end,
            args.mix_alpha,
        )
        if state is None:
            continue
        for sampled, corrected in draw_paired_gradients(
                state, k, length, args.fit_trials, args.trial_chunk,
                fit_generator):
            add_fit_moments(
                fit_moments, sampled, corrected, state["reference"],
            )
        if start == 0 or end == num_users or (
                start // args.user_batch_size) % 25 == 0:
            print(f"  users {start}:{end}/{num_users}")

    lambda_user_normalized = coefficient_from_moments(
        fit_moments, "normalized",
    )
    lambda_raw = coefficient_from_moments(fit_moments, "raw")
    print(
        f"[fit] lambda_user_normalized={lambda_user_normalized:.6f}; "
        f"lambda_raw={lambda_raw:.6f}"
    )

    lambdas = {
        f"fixed_{value:g}": float(value) for value in args.lambdas
    }
    lambdas["fitted_user_normalized"] = lambda_user_normalized
    lambdas["fitted_raw"] = lambda_raw
    metric_parts = {label: {} for label in lambdas}
    evaluation_generator = torch.Generator(device=device).manual_seed(
        args.seed + 91009,
    )
    print("[evaluate] fresh held-out paired draws")
    for start in range(0, num_users, args.user_batch_size):
        end = min(start + args.user_batch_size, num_users)
        state = build_batch_state(
            student_scores, teacher_scores, teacher_topk, student_topm,
            student_scores_m, q2_active, budgets_all, start, end,
            args.mix_alpha,
        )
        if state is None:
            continue
        accumulators = initialize_evaluation_accumulators(
            lambdas, state["reference"].shape[0],
            state["reference"].shape[1], device, state["reference"].dtype,
        )
        for sampled, corrected in draw_paired_gradients(
                state, k, length, args.evaluation_trials, args.trial_chunk,
                evaluation_generator):
            add_evaluation_chunk(
                accumulators, sampled, corrected, state["reference"],
            )
        for label, accumulator in accumulators.items():
            metrics = finalize_evaluation(
                accumulator, state["reference"], args.evaluation_trials,
            )
            for metric, values in metrics.items():
                metric_parts[label].setdefault(metric, []).append(
                    values.detach().cpu()
                )
        if start == 0 or end == num_users or (
                start // args.user_batch_size) % 25 == 0:
            print(f"  users {start}:{end}/{num_users}")

    methods = {
        label: {
            "lambda": lambdas[label],
            **{
                metric: summarize_tensor(torch.cat(parts))
                for metric, parts in metrics.items()
            },
        }
        for label, metrics in metric_parts.items()
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
            "fit_trials": args.fit_trials,
            "evaluation_trials": args.evaluation_trials,
            "seed": args.seed,
            "reference_objective": (
                "CE on the unique union of student Top-M and teacher Top-K"
            ),
        },
        "eligible_users": eligible_users.sum().item(),
        "blocker_budget": summarize_tensor(budgets_all[eligible_users]),
        "fitted_coefficients": {
            "user_normalized": lambda_user_normalized,
            "raw_gradient_weighted": lambda_raw,
        },
        "methods": methods,
        "guardrails": [
            "This is a checkpoint-level logit-gradient diagnostic, not a training result.",
            "The shrinkage coefficient is fitted on random draws independent of all reported evaluation draws.",
            "Current sampled CE and bilateral Bq-corrected CE use exactly the same sampled set in every paired draw.",
            "The user-normalized coefficient minimizes average per-user squared gradient error; the raw coefficient gives larger-gradient users more weight.",
            "MSE includes both squared bias and single-draw variance and is the primary trainability endpoint.",
            "Finite trials estimate the reported mean-gradient bias and variance; bias_plus_variance and direct MSE are both emitted as a numerical consistency check.",
        ],
    }
    output = args.output or os.path.join(
        "logs", f"importance_shrinkage_{args.dataset}_"
        f"{args.T_backbone}-{args.S_backbone}_{args.model}_d{args.student_dim}_"
        f"k{k}_l{length}_m{max_k}_seed{args.seed}.json",
    )
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\n[summary: held-out gradient shrinkage evaluation]")
    for label, result in methods.items():
        print(
            f"  {label:25s} lambda={result['lambda']:.6f} "
            f"mean-grad-cos={result['mean_gradient_cosine']['mean']:.6f} "
            f"rel-bias={result['relative_l2_bias']['mean']:.6f} "
            f"variance/ref2={result['variance_over_reference_norm2']['mean']:.6f} "
            f"mse/ref2={result['mse_over_reference_norm2']['mean']:.6f}"
        )
    print(f"  report saved to {output}")

    del teacher_scores, student_scores
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

"""Compare closure samplers by exact post-sampling missing-mass penalty.

No model is trained.  Every sampler receives the same checkpoint, K, L and
top-M candidate pool.  The primary endpoint is the reduction of the exact
full-catalog soft-closure penalty after adding L sampled items to the
teacher-only portion of top-K.
"""

import argparse
import gc
import json
import os

import torch

from dataset import implicit_CF_dataset, load_cf_data
from diagnose import build_score_mat
from soft_closure import (
    exact_soft_closure_rho,
    prepare_soft_closure_catalog,
    soft_closure_bound_terms,
    summarize_tensor,
)
from utils import load_yaml
from utils.parse_utils import parse_cfg


def rowwise_isin(left, right):
    return (left.unsqueeze(2) == right.unsqueeze(1)).any(dim=2)


def normalize_with_uniform_mixture(raw_weight, eligible, mix_alpha):
    """Normalize a proposal and guarantee support with an eligible uniform mix."""
    if not 0. <= mix_alpha <= 1.:
        raise ValueError("mix_alpha must be between zero and one")
    eligible = eligible.bool()
    uniform = eligible.to(raw_weight.dtype)
    uniform = uniform / uniform.sum(dim=1, keepdim=True).clamp_min(1.)
    closure = raw_weight.clamp_min(0.) * eligible
    closure_sum = closure.sum(dim=1, keepdim=True)
    closure = torch.where(
        closure_sum > 0., closure / closure_sum.clamp_min(torch.finfo(raw_weight.dtype).tiny),
        uniform,
    )
    return mix_alpha * closure + (1. - mix_alpha) * uniform


def build_sampler_probabilities(student_scores, teacher_scores, teacher_topk,
                                student_topm, q2_active, base_rho,
                                catalog_cache, count_temperature=10.,
                                score_temperature=1., mix_alpha=.9):
    """Build original, Q2-count, mass and marginal-mass proposals."""
    scores_m = student_scores.gather(1, student_topm)
    scores_i = student_scores.gather(1, teacher_topk)
    teacher_i = teacher_scores.gather(1, teacher_topk)
    eligible = ~rowwise_isin(student_topm, teacher_topk)

    # Exact current-code proposal: only teacher items found in top-M increment
    # a position, followed by the reverse cumulative count and exp(z / T).
    code_count = torch.zeros_like(scores_m)
    matches = (teacher_topk.unsqueeze(2) == student_topm.unsqueeze(1)).nonzero()
    if matches.numel() > 0:
        code_count[matches[:, 0], matches[:, 2]] += 1
    code_count = torch.minimum(
        torch.cumsum(code_count.flip(1), dim=1).flip(1),
        code_count.new_tensor(50.),
    )
    original = torch.softmax((code_count + 1.) / count_temperature, dim=1)

    block = (scores_m.unsqueeze(2) >= scores_i.unsqueeze(1)) & q2_active.unsqueeze(1)
    q2_count_raw = torch.exp(
        (block.sum(dim=2).to(scores_m.dtype) + 1.) / count_temperature
    )

    neg_inf = torch.full_like(teacher_i, -float("inf"))
    p_teacher_q2 = torch.softmax(torch.where(q2_active, teacher_i, neg_inf), dim=1)
    p_teacher_q2 = torch.where(q2_active, p_teacher_q2, torch.zeros_like(p_teacher_q2))
    exp_m = torch.exp(
        (scores_m - scores_m.max(dim=1, keepdim=True).values) / score_temperature
    )
    teacher_block_mass = (block * p_teacher_q2.unsqueeze(1)).sum(dim=2)
    mass_raw = exp_m * teacher_block_mass

    # First-order removal value of a blocker.  A+B_i is represented in the
    # same numerically stable score scale as the candidate exponential.
    shift = catalog_cache["shift"]
    exp_i = torch.exp(scores_i - shift) * q2_active
    a_mass = exp_i.sum(dim=1, keepdim=True).clamp_min(torch.finfo(scores_m.dtype).tiny)
    denominator_i = a_mass * (1. + base_rho)
    marginal_coefficient = torch.where(
        q2_active,
        p_teacher_q2 / denominator_i.clamp_min(torch.finfo(scores_m.dtype).tiny),
        torch.zeros_like(p_teacher_q2),
    )
    exp_m_full_scale = torch.exp(scores_m - shift)
    marginal_raw = exp_m_full_scale * (
        block * marginal_coefficient.unsqueeze(1)
    ).sum(dim=2)

    return {
        "original_code_count": original,
        "q2_count": normalize_with_uniform_mixture(q2_count_raw, eligible, mix_alpha),
        "teacher_student_mass": normalize_with_uniform_mixture(mass_raw, eligible, mix_alpha),
        "marginal_mass": normalize_with_uniform_mixture(marginal_raw, eligible, mix_alpha),
        "uniform": normalize_with_uniform_mixture(torch.zeros_like(scores_m), eligible, 0.),
    }


def active_l2_set(sampled, teacher_topk, q2_active):
    active_teacher = q2_active & ~rowwise_isin(teacher_topk, sampled)
    return (
        torch.cat([sampled, teacher_topk], dim=1),
        torch.cat([torch.ones_like(sampled, dtype=torch.bool), active_teacher], dim=1),
    )


def proposal_entropy(probabilities):
    safe_log = torch.where(
        probabilities > 0., probabilities.clamp_min(1e-30).log(),
        torch.zeros_like(probabilities),
    )
    return -(probabilities * safe_log).sum(dim=1)


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
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--mix_alpha", type=float, default=.9)
    parser.add_argument("--score_temperature", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--user_batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("full-catalog checkpoint diagnostics require CUDA")
    if args.trials < 1:
        raise ValueError("trials must be positive")
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
    kd_tau = float(rce_cfg["mkd_tau"])
    count_temperature = float(rce_cfg["mkd_T"])
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
    ) / kd_tau
    print(f"[student] {student_checkpoint}")
    student_scores = build_score_mat(
        student_checkpoint, args.S_backbone, student_cfg, trainset, device,
    ) / kd_tau
    teacher_topk = torch.topk(teacher_scores, k, dim=1).indices
    student_topm = torch.topk(student_scores, max_k, dim=1).indices
    student_topk = student_topm[:, :k]
    q2_active = ~rowwise_isin(teacher_topk, student_topk)
    eligible_users = q2_active.any(dim=1)
    if not eligible_users.any():
        raise RuntimeError(
            "no eligible users: every teacher top-K set is already contained "
            "in the corresponding student top-K set"
        )
    print(
        f"[users] {eligible_users.sum().item()}/{num_users} have at least one "
        "teacher top-K item outside student top-K"
    )

    sampler_names = [
        "original_code_count", "q2_count", "teacher_student_mass",
        "marginal_mass", "uniform",
    ]
    reductions = {name: [] for name in sampler_names}
    reduction_ratios = {name: [] for name in sampler_names}
    after_penalties = {name: [] for name in sampler_names}
    entropies = {name: [] for name in sampler_names}
    trial_means = {name: [[] for _ in range(args.trials)] for name in sampler_names}
    base_penalties = []
    generators = {
        (name, trial): torch.Generator(device=device).manual_seed(
            args.seed + 1009 * trial + 7919 * (index + 1)
        )
        for index, name in enumerate(sampler_names)
        for trial in range(args.trials)
    }

    print("[comparison] exact post-sampling penalty, shared full-catalog sort per batch")
    for start in range(0, num_users, args.user_batch_size):
        end = min(start + args.user_batch_size, num_users)
        local_keep = eligible_users[start:end]
        if not local_keep.any():
            continue
        scores_s = student_scores[start:end][local_keep]
        scores_t = teacher_scores[start:end][local_keep]
        items_t = teacher_topk[start:end][local_keep]
        items_m = student_topm[start:end][local_keep]
        active_q2 = q2_active[start:end][local_keep]
        cache = prepare_soft_closure_catalog(scores_s)

        base_rho = exact_soft_closure_rho(
            scores_s, items_t, active_q2, catalog_cache=cache,
        )
        base = soft_closure_bound_terms(
            scores_s, scores_t, items_t, active_q2, base_rho,
        )["penalty"]
        base_penalties.append(base.cpu())
        proposals = build_sampler_probabilities(
            scores_s, scores_t, items_t, items_m, active_q2, base_rho, cache,
            count_temperature=count_temperature,
            score_temperature=args.score_temperature,
            mix_alpha=args.mix_alpha,
        )

        for name, probabilities in proposals.items():
            entropy = proposal_entropy(probabilities).cpu()
            entropies[name].append(entropy)
            for trial in range(args.trials):
                positions = torch.multinomial(
                    probabilities, length, replacement=False,
                    generator=generators[(name, trial)],
                )
                sampled = torch.gather(items_m, 1, positions)
                items_j, active_j = active_l2_set(sampled, items_t, active_q2)
                rho_after = exact_soft_closure_rho(
                    scores_s, items_j, active_j, catalog_cache=cache,
                )
                after = soft_closure_bound_terms(
                    scores_s, scores_t, items_j, active_j, rho_after,
                )["penalty"]
                reduction = base - after
                after_penalties[name].append(after.cpu())
                reductions[name].append(reduction.cpu())
                reduction_ratios[name].append((
                    reduction / base.clamp_min(1e-12)
                ).cpu())
                trial_means[name][trial].append(reduction.detach().cpu())

        if start == 0 or end == num_users or (start // args.user_batch_size) % 25 == 0:
            print(f"  users {start}:{end}/{num_users}")

    base_penalties = torch.cat(base_penalties).double()
    samplers = {}
    for name in sampler_names:
        reduction = torch.cat(reductions[name]).double()
        ratio = torch.cat(reduction_ratios[name]).double()
        after = torch.cat(after_penalties[name]).double()
        entropy = torch.cat(entropies[name]).double()
        per_trial = torch.tensor([
            torch.cat(parts).double().mean().item() for parts in trial_means[name]
        ])
        samplers[name] = {
            "exact_penalty_after": summarize_tensor(after),
            "absolute_penalty_reduction": summarize_tensor(reduction),
            "relative_penalty_reduction": summarize_tensor(ratio),
            "fraction_reducing_penalty": (reduction > 0.).double().mean().item(),
            "proposal_entropy": summarize_tensor(entropy),
            "effective_support_from_mean_entropy": float(torch.exp(entropy.mean()).item()),
            "trial_mean_absolute_reduction": per_trial.tolist(),
            "trial_mean_reduction_summary": summarize_tensor(per_trial),
        }

    original_reduction = torch.cat(reductions["original_code_count"]).double()
    for name in sampler_names:
        candidate = torch.cat(reductions[name]).double()
        paired_gain = candidate - original_reduction
        samplers[name]["paired_gain_over_original"] = summarize_tensor(paired_gain)
        samplers[name]["paired_win_fraction_over_original"] = (
            paired_gain > 0.
        ).double().mean().item()
        samplers[name]["paired_tie_fraction_with_original"] = (
            paired_gain == 0.
        ).double().mean().item()

    report = {
        "experiment": {
            "dataset": args.dataset,
            "teacher": args.T_backbone,
            "student": args.S_backbone,
            "student_dim": args.student_dim,
            "teacher_checkpoint": teacher_checkpoint,
            "student_checkpoint": student_checkpoint,
            "K": k,
            "L": length,
            "mxK": max_k,
            "trials": args.trials,
            "mix_alpha": args.mix_alpha,
            "score_temperature": args.score_temperature,
            "seed": args.seed,
            "primary_endpoint": "exact soft-closure penalty reduction at fixed L",
        },
        "eligible_users": eligible_users.sum().item(),
        "base_q2_exact_penalty": summarize_tensor(base_penalties),
        "samplers": samplers,
        "guardrails": [
            "This is a checkpoint-level sampler test, not a training-performance result.",
            "Original-code sampling intentionally retains its current support; proposed samplers exclude teacher top-K duplicates.",
            "Mass samplers use a uniform mixture so every eligible top-M candidate has nonzero probability.",
            "The marginal score is a first-order removal approximation, not the exact discrete set-addition gain.",
        ],
    }
    output = args.output or os.path.join(
        "logs", f"soft_closure_sampler_compare_{args.dataset}_"
        f"{args.T_backbone}-{args.S_backbone}_d{args.student_dim}_k{k}_l{length}_"
        f"m{max_k}_seed{args.seed}.json",
    )
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\n[summary: mean exact penalty reduction]")
    for name in sampler_names:
        result = samplers[name]
        mean = result["absolute_penalty_reduction"]["mean"]
        gain = result["paired_gain_over_original"]["mean"]
        win = result["paired_win_fraction_over_original"]
        print(
            f"  {name:24s} reduction={mean:.6f} "
            f"gain_vs_original={gain:+.6f} paired_win={win:.3f}"
        )
    print(f"  report saved to {output}")

    del teacher_scores, student_scores
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

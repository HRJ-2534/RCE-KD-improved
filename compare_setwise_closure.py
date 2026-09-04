"""Compare static and residual-aware anchor/blocker selection without training.

All methods receive the same checkpoint, Q1 anchors, Q2 targets, fixed final
budget and student Top-M candidate support.  Unlike the earlier broad sampler
comparison, the primary endpoint here isolates blocker selection: its baseline
already contains the fixed anchors and Q2 targets.  Post-selection quality is
evaluated with exact full-catalog closure masses.
"""

import argparse
import gc
import json
import os
import time

import torch

from compare_soft_closure_samplers import resolve_student_checkpoint
from dataset import implicit_CF_dataset, load_cf_data
from diagnose import build_score_mat
from modeling.KD.anchor_rce import (
    original_count_blocker_probabilities,
    prepare_teacher_anchors,
    rowwise_isin,
    topm_closure_statistics,
    topm_marginal_blocker_probabilities,
)
from setwise_closure import (
    build_fixed_target_closure_state,
    compose_anchor_and_blockers,
    exact_candidate_gains,
    fixed_target_closure_penalty,
    select_by_residual_closure,
)
from soft_closure import (
    exact_soft_closure_rho,
    prepare_soft_closure_catalog,
    soft_closure_bound_terms,
    summarize_tensor,
)
from utils import load_yaml
from utils.parse_utils import parse_cfg


METHODS = (
    "anchor_count_static",
    "anchor_marginal_static",
    "anchor_marginal_topl",
    "anchor_residual_sequential",
    "anchor_residual_greedy",
)
RANDOM_METHODS = {
    "anchor_count_static",
    "anchor_marginal_static",
    "anchor_residual_sequential",
}


def parse_mxk_list(value):
    try:
        result = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "mxK_list must be a comma-separated integer list"
        ) from error
    if not result or any(number <= 0 for number in result):
        raise argparse.ArgumentTypeError("every mxK value must be positive")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("mxK_list must not contain duplicates")
    return sorted(result)


def active_l2_set(interesting_items, teacher_topk, q2_active):
    active_teacher = q2_active & ~rowwise_isin(teacher_topk, interesting_items)
    return (
        torch.cat([interesting_items, teacher_topk], dim=1),
        torch.cat([
            torch.ones_like(interesting_items, dtype=torch.bool),
            active_teacher,
        ], dim=1),
    )


def proposal_entropy(probabilities):
    safe_log = torch.where(
        probabilities > 0., probabilities.clamp_min(1e-30).log(),
        torch.zeros_like(probabilities),
    )
    return -(probabilities * safe_log).sum(dim=1)


def select_static(probabilities, budgets, length, generator=None, top_l=False):
    if top_l:
        positions = torch.topk(probabilities, length, dim=1).indices
    else:
        positions = torch.multinomial(
            probabilities, length, replacement=False, generator=generator,
        )
    steps = torch.arange(length, device=positions.device).unsqueeze(0)
    return positions, steps < budgets.unsqueeze(1)


def synchronize_and_measure(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def evaluate_selection(
        student_scores, teacher_scores, teacher_topk, q2_active,
        target_teacher_scores, catalog_cache, base_fixed_penalty,
        base_full_penalty, interesting_items):
    final_items, final_active = active_l2_set(
        interesting_items, teacher_topk, q2_active,
    )
    fixed_state = build_fixed_target_closure_state(
        student_scores, final_items, final_active,
        teacher_topk, q2_active, target_teacher_scores, catalog_cache,
    )
    fixed_after = fixed_target_closure_penalty(fixed_state)
    rho = exact_soft_closure_rho(
        student_scores, final_items, final_active,
        catalog_cache=catalog_cache,
    )
    full_after = soft_closure_bound_terms(
        student_scores, teacher_scores, final_items, final_active, rho,
    )["penalty"]
    return {
        "fixed_penalty_after": fixed_after,
        "fixed_reduction": base_fixed_penalty - fixed_after,
        "fixed_reduction_ratio": (
            (base_fixed_penalty - fixed_after)
            / base_fixed_penalty.clamp_min(1e-12)
        ),
        "full_penalty_after": full_after,
        "full_penalty_reduction": base_full_penalty - full_after,
    }


def empty_storage(trial_count):
    metrics = (
        "fixed_penalty_after", "fixed_reduction", "fixed_reduction_ratio",
        "full_penalty_after", "full_penalty_reduction",
        "additive_overestimate", "selected_student_rank_mean",
        "proposal_entropy",
    )
    return {
        method: {
            metric: [[] for _ in range(
                trial_count if method in RANDOM_METHODS else 1
            )]
            for metric in metrics
        }
        for method in METHODS
    }


def record_result(storage, method, trial, evaluation, overestimate,
                  selected_rank_mean, entropy):
    values = {
        **evaluation,
        "additive_overestimate": overestimate,
        "selected_student_rank_mean": selected_rank_mean,
        "proposal_entropy": entropy,
    }
    for name, value in values.items():
        storage[method][name][trial].append(value.detach().cpu())


def finalize_method(metric_parts):
    tensors = {
        metric: torch.stack([
            torch.cat(parts).double() for parts in trial_parts
        ])
        for metric, trial_parts in metric_parts.items()
    }
    per_user_mean = {
        metric: value.mean(dim=0) for metric, value in tensors.items()
    }
    return tensors, per_user_mean, {
        metric: summarize_tensor(value.flatten())
        for metric, value in tensors.items()
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
    parser.add_argument(
        "--mxK_list", type=parse_mxk_list, default=parse_mxk_list("200,400,800"),
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--mix_alpha", type=float, default=.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--user_batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("full-catalog checkpoint diagnostics require CUDA")
    if args.trials < 1:
        raise ValueError("trials must be positive")
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
    kd_tau = float(rce_cfg["mkd_tau"])
    count_temperature = float(rce_cfg["mkd_T"])
    max_m = max(args.mxK_list)
    if not (0 < k <= max_m <= num_items):
        raise ValueError("expected 0 < K <= max(mxK_list) <= num_items")
    if any(m < k + length for m in args.mxK_list):
        raise ValueError("every mxK must satisfy mxK >= K + L")

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
    student_topmax = torch.topk(student_scores, max_m, dim=1).indices
    student_topk = student_topmax[:, :k]
    q2_active = ~rowwise_isin(teacher_topk, student_topk)
    eligible_users = q2_active.any(dim=1)
    print(
        f"[users] {eligible_users.sum().item()}/{num_users} have active Q2 targets"
    )

    storage = {m: empty_storage(args.trials) for m in args.mxK_list}
    runtimes = {
        m: {method: 0. for method in METHODS} for m in args.mxK_list
    }
    selections = {
        m: {method: 0 for method in METHODS} for m in args.mxK_list
    }
    base_fixed_parts = []
    base_full_parts = []
    anchor_count_parts = []
    blocker_budget_parts = []
    greedy_topl_equal = {m: [] for m in args.mxK_list}
    generators = {
        (m, method, trial): torch.Generator(device=device).manual_seed(
            args.seed + 1009 * trial + 7919 * (method_index + 1)
            + 104729 * m
        )
        for m in args.mxK_list
        for method_index, method in enumerate(METHODS)
        if method in RANDOM_METHODS
        for trial in range(args.trials)
    }

    print(
        "[comparison] fixed anchors/Q2, exact full-catalog residual closure, "
        f"Top-M={args.mxK_list}"
    )
    for start in range(0, num_users, args.user_batch_size):
        end = min(start + args.user_batch_size, num_users)
        local_keep = eligible_users[start:end]
        if not local_keep.any():
            continue
        scores_s = student_scores[start:end][local_keep]
        scores_t = teacher_scores[start:end][local_keep]
        items_t = teacher_topk[start:end][local_keep]
        active_q2 = q2_active[start:end][local_keep]
        items_topmax = student_topmax[start:end][local_keep]
        target_teacher_scores = scores_t.gather(1, items_t)
        cache = prepare_soft_closure_catalog(scores_s)

        anchor_positions, anchor_active = prepare_teacher_anchors(
            items_topmax, items_t, target_teacher_scores, k, length,
        )
        anchor_items = items_topmax.gather(1, anchor_positions)
        base_items = torch.cat([anchor_items, items_t], dim=1)
        base_active = torch.cat([anchor_active, active_q2], dim=1)
        base_state = build_fixed_target_closure_state(
            scores_s, base_items, base_active, items_t, active_q2,
            target_teacher_scores, cache,
        )
        base_fixed = fixed_target_closure_penalty(base_state)
        base_rho = exact_soft_closure_rho(
            scores_s, base_items, base_active, catalog_cache=cache,
        )
        base_full = soft_closure_bound_terms(
            scores_s, scores_t, base_items, base_active, base_rho,
        )["penalty"]
        anchor_counts = anchor_active.sum(dim=1)
        budgets = length - anchor_counts
        base_fixed_parts.append(base_fixed.cpu())
        base_full_parts.append(base_full.cpu())
        anchor_count_parts.append(anchor_counts.cpu())
        blocker_budget_parts.append(budgets.cpu())

        for max_k in args.mxK_list:
            items_m = items_topmax[:, :max_k]
            scores_m = scores_s.gather(1, items_m)
            eligible = ~rowwise_isin(items_m, items_t)
            local_anchor_positions, local_anchor_active = prepare_teacher_anchors(
                items_m, items_t, target_teacher_scores, k, length,
            )
            if not torch.equal(local_anchor_active, anchor_active):
                raise RuntimeError("Q1 anchors changed when only Top-M was enlarged")
            closure_statistics = topm_closure_statistics(
                scores_m, target_teacher_scores, items_t, items_m, active_q2,
            )
            count_proposal = original_count_blocker_probabilities(
                items_t, items_m, count_temperature,
            )
            marginal_proposal = topm_marginal_blocker_probabilities(
                scores_m, target_teacher_scores, target_teacher_scores,
                items_t, items_m, active_q2, args.mix_alpha,
                closure_statistics=closure_statistics,
            )
            initial_gains = exact_candidate_gains(
                base_state, scores_m, eligible,
            )

            method_proposals = {
                "anchor_count_static": count_proposal,
                "anchor_marginal_static": marginal_proposal,
            }
            for method, proposal in method_proposals.items():
                entropy = proposal_entropy(proposal)
                for trial in range(args.trials):
                    (blocker_positions, blocker_active), elapsed = (
                        synchronize_and_measure(lambda p=proposal, m=method, t=trial: (
                            select_static(
                                p, budgets, length,
                                generator=generators[(max_k, m, t)],
                            )
                        ))
                    )
                    interesting = compose_anchor_and_blockers(
                        items_m, local_anchor_positions, local_anchor_active,
                        blocker_positions, blocker_active, length,
                    )
                    evaluation = evaluate_selection(
                        scores_s, scores_t, items_t, active_q2,
                        target_teacher_scores, cache, base_fixed, base_full,
                        interesting,
                    )
                    singleton_sum = initial_gains.gather(
                        1, blocker_positions,
                    ).masked_fill(~blocker_active, 0.).sum(dim=1)
                    overestimate = (
                        singleton_sum - evaluation["fixed_reduction"]
                    ).clamp_min(0.)
                    selected_rank = (
                        (blocker_positions + 1) * blocker_active
                    ).sum(dim=1) / budgets.clamp_min(1)
                    record_result(
                        storage[max_k], method, trial, evaluation,
                        overestimate, selected_rank, entropy,
                    )
                    runtimes[max_k][method] += elapsed
                    selections[max_k][method] += scores_s.shape[0]

            (top_positions, top_active), elapsed = synchronize_and_measure(
                lambda: select_static(
                    marginal_proposal, budgets, length, top_l=True,
                )
            )
            top_interesting = compose_anchor_and_blockers(
                items_m, local_anchor_positions, local_anchor_active,
                top_positions, top_active, length,
            )
            top_evaluation = evaluate_selection(
                scores_s, scores_t, items_t, active_q2,
                target_teacher_scores, cache, base_fixed, base_full,
                top_interesting,
            )
            top_singleton_sum = initial_gains.gather(
                1, top_positions,
            ).masked_fill(~top_active, 0.).sum(dim=1)
            top_rank = ((top_positions + 1) * top_active).sum(dim=1) / (
                budgets.clamp_min(1)
            )
            record_result(
                storage[max_k], "anchor_marginal_topl", 0, top_evaluation,
                (top_singleton_sum - top_evaluation["fixed_reduction"]).clamp_min(0.),
                top_rank, proposal_entropy(marginal_proposal),
            )
            runtimes[max_k]["anchor_marginal_topl"] += elapsed
            selections[max_k]["anchor_marginal_topl"] += scores_s.shape[0]

            for trial in range(args.trials):
                residual, elapsed = synchronize_and_measure(
                    lambda t=trial: select_by_residual_closure(
                        base_state, scores_m, eligible, budgets, length,
                        mix_alpha=args.mix_alpha, greedy=False,
                        generator=generators[
                            (max_k, "anchor_residual_sequential", t)
                        ],
                    )
                )
                interesting = compose_anchor_and_blockers(
                    items_m, local_anchor_positions, local_anchor_active,
                    residual["positions"], residual["active"], length,
                )
                evaluation = evaluate_selection(
                    scores_s, scores_t, items_t, active_q2,
                    target_teacher_scores, cache, base_fixed, base_full,
                    interesting,
                )
                selected_rank = (
                    (residual["positions"] + 1) * residual["active"]
                ).sum(dim=1) / budgets.clamp_min(1)
                record_result(
                    storage[max_k], "anchor_residual_sequential", trial,
                    evaluation, residual["additive_overestimate"],
                    selected_rank, residual["mean_proposal_entropy"],
                )
                runtimes[max_k]["anchor_residual_sequential"] += elapsed
                selections[max_k]["anchor_residual_sequential"] += scores_s.shape[0]

            greedy, elapsed = synchronize_and_measure(
                lambda: select_by_residual_closure(
                    base_state, scores_m, eligible, budgets, length,
                    mix_alpha=args.mix_alpha, greedy=True,
                )
            )
            greedy_interesting = compose_anchor_and_blockers(
                items_m, local_anchor_positions, local_anchor_active,
                greedy["positions"], greedy["active"], length,
            )
            greedy_evaluation = evaluate_selection(
                scores_s, scores_t, items_t, active_q2,
                target_teacher_scores, cache, base_fixed, base_full,
                greedy_interesting,
            )
            greedy_rank = (
                (greedy["positions"] + 1) * greedy["active"]
            ).sum(dim=1) / budgets.clamp_min(1)
            record_result(
                storage[max_k], "anchor_residual_greedy", 0,
                greedy_evaluation, greedy["additive_overestimate"],
                greedy_rank, greedy["mean_proposal_entropy"],
            )
            runtimes[max_k]["anchor_residual_greedy"] += elapsed
            selections[max_k]["anchor_residual_greedy"] += scores_s.shape[0]
            greedy_topl_equal[max_k].append(
                (greedy_interesting == top_interesting).all(dim=1).cpu()
            )

        if start == 0 or end == num_users or (
                start // args.user_batch_size) % 25 == 0:
            print(f"  users {start}:{end}/{num_users}")

    base_fixed_all = torch.cat(base_fixed_parts).double()
    base_full_all = torch.cat(base_full_parts).double()
    report_mxk = {}
    for max_k in args.mxK_list:
        tensors = {}
        per_user = {}
        summaries = {}
        for method in METHODS:
            tensors[method], per_user[method], summaries[method] = (
                finalize_method(storage[max_k][method])
            )
        baseline = per_user["anchor_marginal_static"]["fixed_reduction"]
        for method in METHODS:
            gain = per_user[method]["fixed_reduction"] - baseline
            summaries[method]["paired_fixed_gain_over_static_marginal"] = (
                summarize_tensor(gain)
            )
            summaries[method]["paired_win_fraction_over_static_marginal"] = (
                (gain > 0.).double().mean().item()
            )
            total_seconds = runtimes[max_k][method]
            summaries[method]["selection_total_seconds"] = total_seconds
            summaries[method]["selection_ms_per_user_draw"] = (
                1000. * total_seconds
                / max(selections[max_k][method], 1)
            )
        report_mxk[str(max_k)] = {
            "methods": summaries,
            "greedy_equals_marginal_topl_fraction": torch.cat(
                greedy_topl_equal[max_k]
            ).double().mean().item(),
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
            "mxK_list": args.mxK_list,
            "trials": args.trials,
            "mix_alpha": args.mix_alpha,
            "seed": args.seed,
            "primary_endpoint": (
                "exact full-catalog fixed-Q2 closure-penalty reduction from "
                "the shared anchor-plus-Q2 starting set"
            ),
        },
        "eligible_users": eligible_users.sum().item(),
        "anchor_count": summarize_tensor(torch.cat(anchor_count_parts)),
        "blocker_budget": summarize_tensor(torch.cat(blocker_budget_parts)),
        "base_fixed_q2_penalty": summarize_tensor(base_fixed_all),
        "base_full_set_penalty": summarize_tensor(base_full_all),
        "by_mxK": report_mxk,
        "guardrails": [
            "This is a checkpoint-level selection diagnostic, not a recommendation-performance result.",
            "Every method keeps exactly the same Q1 anchors and final L-item budget.",
            "Teacher top-K items are excluded from every blocker proposal.",
            "The primary fixed-Q2 objective uses fixed teacher probabilities on Q2 while its closure masses are exact over the full item catalog.",
            "Full-set closure penalties are also reported and re-normalize teacher probability over the actual final L2 set.",
            "Static marginal is the current ARCE-KD proposal; residual sequential recomputes the exact discrete gain after every selected blocker.",
            "Residual greedy is an objective upper bound, not automatically a suitable training sampler; under a total student ranking it is expected to collapse toward the highest-ranked eligible candidates.",
        ],
    }
    mxk_label = "-".join(str(value) for value in args.mxK_list)
    output = args.output or os.path.join(
        "logs", f"setwise_closure_{args.dataset}_"
        f"{args.T_backbone}-{args.S_backbone}_{args.model}_d{args.student_dim}_"
        f"k{k}_l{length}_m{mxk_label}_seed{args.seed}.json",
    )
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\n[summary: exact fixed-Q2 penalty reduction after anchors]")
    for max_k in args.mxK_list:
        print(f"  Top-M={max_k}")
        methods = report_mxk[str(max_k)]["methods"]
        for method in METHODS:
            reduction = methods[method]["fixed_reduction"]["mean"]
            gain = methods[method][
                "paired_fixed_gain_over_static_marginal"
            ]["mean"]
            win = methods[method][
                "paired_win_fraction_over_static_marginal"
            ]
            speed = methods[method]["selection_ms_per_user_draw"]
            print(
                f"    {method:28s} reduction={reduction:.6f} "
                f"gain={gain:+.6f} win={win:.3f} select={speed:.3f}ms/user"
            )
        equality = report_mxk[str(max_k)][
            "greedy_equals_marginal_topl_fraction"
        ]
        print(f"    greedy == marginal Top-L: {equality:.3f}")
    print(f"  report saved to {output}")

    del teacher_scores, student_scores
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

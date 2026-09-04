"""Anchor-preserving RCE-KD training variants.

The KD loss itself is inherited unchanged from :class:`RCEKD`.  This module
only changes how the fixed-size ``interesting_items`` set is constructed at
the start of each epoch.
"""

import torch
import torch.nn.functional as F

from .playground import RCEKD


def rowwise_isin(left, right):
    return (left.unsqueeze(2) == right.unsqueeze(1)).any(dim=2)


def normalize_with_uniform_mixture(raw_weight, eligible, mix_alpha):
    if not 0. <= mix_alpha <= 1.:
        raise ValueError("arce_mix_alpha must be between zero and one")
    eligible = eligible.bool()
    uniform = eligible.to(raw_weight.dtype)
    eligible_count = uniform.sum(dim=1, keepdim=True)
    if (eligible_count == 0).any():
        raise ValueError("at least one blocker candidate is required per user")
    uniform = uniform / eligible_count
    proposal = raw_weight.clamp_min(0.) * eligible
    proposal_sum = proposal.sum(dim=1, keepdim=True)
    proposal = torch.where(
        proposal_sum > 0.,
        proposal / proposal_sum.clamp_min(torch.finfo(raw_weight.dtype).tiny),
        uniform,
    )
    return mix_alpha * proposal + (1. - mix_alpha) * uniform


def original_count_blocker_probabilities(teacher_topk, student_topm,
                                         count_temperature):
    """Original RCE count weights, renormalized after removing teacher items."""
    if count_temperature <= 0.:
        raise ValueError("mkd_T must be positive")
    count = torch.zeros(
        student_topm.shape, device=student_topm.device, dtype=torch.float32,
    )
    matches = (teacher_topk.unsqueeze(2) == student_topm.unsqueeze(1)).nonzero()
    if matches.numel() > 0:
        count[matches[:, 0], matches[:, 2]] += 1.
    count = torch.minimum(
        count.flip(1).cumsum(dim=1).flip(1), count.new_tensor(50.),
    )
    raw_weight = torch.exp((count + 1.) / count_temperature)
    eligible = ~rowwise_isin(student_topm, teacher_topk)
    return normalize_with_uniform_mixture(raw_weight, eligible, 1.)


def topm_closure_statistics(student_scores_m, student_scores_t, teacher_topk,
                            student_topm, q2_active):
    """Shared Top-M quantities for closure difficulty and marginal sampling."""
    if student_scores_m.shape != student_topm.shape:
        raise ValueError("student_scores_m and student_topm must have equal shape")
    if student_scores_t.shape != teacher_topk.shape:
        raise ValueError("student_scores_t and teacher_topk must have equal shape")
    if q2_active.shape != teacher_topk.shape:
        raise ValueError("q2_active and teacher_topk must have equal shape")

    q2_active = q2_active.bool()
    shift = student_scores_m[:, :1]
    exp_m = torch.exp(student_scores_m - shift)
    exp_i = torch.exp(student_scores_t - shift) * q2_active
    mass_j = exp_i.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(student_scores_m.dtype).tiny
    )

    topm_matches_t = student_topm.unsqueeze(1) == teacher_topk.unsqueeze(2)
    topm_is_in_q2 = (topm_matches_t & q2_active.unsqueeze(2)).any(dim=1)
    blocker_relation = (
        student_scores_m.unsqueeze(2) >= student_scores_t.unsqueeze(1)
    ) & q2_active.unsqueeze(1)
    missing_visible = (
        blocker_relation
        * (~topm_is_in_q2).unsqueeze(2)
        * exp_m.unsqueeze(2)
    ).sum(dim=1)
    return {
        "exp_m": exp_m,
        "mass_j": mass_j,
        "blocker_relation": blocker_relation,
        "missing_visible": missing_visible,
        "eligible": ~topm_matches_t.any(dim=1),
    }


def topm_closure_difficulty(teacher_topk_prob, q2_active,
                            closure_statistics):
    """Teacher-mass-weighted Top-M soft-closure penalty per user."""
    missing_visible = closure_statistics["missing_visible"]
    mass_j = closure_statistics["mass_j"]
    if teacher_topk_prob.shape != missing_visible.shape:
        raise ValueError("teacher_topk_prob has an incompatible shape")
    if q2_active.shape != missing_visible.shape:
        raise ValueError("q2_active has an incompatible shape")
    rho = missing_visible / mass_j
    return (
        teacher_topk_prob * q2_active * torch.log1p(rho)
    ).sum(dim=1)


def topm_marginal_blocker_probabilities(student_scores_m, student_scores_t,
                                        teacher_scores_t, teacher_topk,
                                        student_topm, q2_active, mix_alpha,
                                        closure_statistics=None):
    """Top-M approximation of the first-order soft-closure removal value."""
    if teacher_scores_t.shape != teacher_topk.shape:
        raise ValueError("teacher_scores_t and teacher_topk must have equal shape")
    if closure_statistics is None:
        closure_statistics = topm_closure_statistics(
            student_scores_m, student_scores_t, teacher_topk,
            student_topm, q2_active,
        )
    exp_m = closure_statistics["exp_m"]
    mass_j = closure_statistics["mass_j"]
    blocker_relation = closure_statistics["blocker_relation"]
    missing_visible = closure_statistics["missing_visible"]

    masked_teacher = torch.where(
        q2_active, teacher_scores_t,
        torch.full_like(teacher_scores_t, -float("inf")),
    )
    has_q2 = q2_active.any(dim=1, keepdim=True)
    teacher_shift = torch.where(
        has_q2, masked_teacher.max(dim=1, keepdim=True).values,
        torch.zeros_like(masked_teacher[:, :1]),
    )
    teacher_mass = torch.exp(teacher_scores_t - teacher_shift) * q2_active
    p_teacher_q2 = teacher_mass / teacher_mass.sum(
        dim=1, keepdim=True,
    ).clamp_min(torch.finfo(teacher_scores_t.dtype).tiny)

    denominator_i = mass_j + missing_visible
    coefficient = torch.where(
        q2_active,
        p_teacher_q2 / denominator_i.clamp_min(
            torch.finfo(student_scores_m.dtype).tiny
        ),
        torch.zeros_like(p_teacher_q2),
    )
    marginal_raw = exp_m * (
        blocker_relation * coefficient.unsqueeze(1)
    ).sum(dim=2)
    return normalize_with_uniform_mixture(
        marginal_raw, closure_statistics["eligible"], mix_alpha,
    )


def prepare_teacher_anchors(student_topm, teacher_topk, teacher_scores_t,
                            k, length):
    """Select up to L items in Q1, preferring higher teacher confidence."""
    topk_position = (
        torch.arange(student_topm.shape[1], device=student_topm.device)
        .unsqueeze(0) < k
    )
    anchor_mask = topk_position & rowwise_isin(student_topm, teacher_topk)
    topm_matches_t = student_topm.unsqueeze(2) == teacher_topk.unsqueeze(1)
    teacher_scores_m = (
        topm_matches_t * teacher_scores_t.unsqueeze(1)
    ).sum(dim=2)
    anchor_priority = torch.where(
        anchor_mask, teacher_scores_m,
        torch.full_like(teacher_scores_m, -float("inf")),
    )
    anchor_positions = torch.topk(anchor_priority, length, dim=1).indices
    anchor_active = anchor_mask.gather(1, anchor_positions)
    return anchor_positions, anchor_active


def sample_anchor_preserving(student_topm, anchor_positions, anchor_active,
                             blocker_probabilities, length):
    """Keep anchors and fill each user's remaining fixed-L budget."""
    blocker_positions = torch.multinomial(
        blocker_probabilities, length, replacement=False,
    )
    combined_positions = torch.cat([anchor_positions, blocker_positions], dim=1)
    combined_active = torch.cat([
        anchor_active,
        torch.ones_like(blocker_positions, dtype=torch.bool),
    ], dim=1)
    order = torch.arange(
        combined_positions.shape[1], device=combined_positions.device,
    ).unsqueeze(0)
    priority = torch.where(
        combined_active,
        combined_positions.new_tensor(combined_positions.shape[1]) - order,
        combined_positions.new_tensor(-1),
    )
    selected_columns = torch.topk(priority, length, dim=1).indices
    selected_positions = combined_positions.gather(1, selected_columns)
    return student_topm.gather(1, selected_positions)


def select_anchor_preserving_topl(student_topm, anchor_positions,
                                  anchor_active, blocker_scores,
                                  blocker_eligible, length):
    """Keep Q1 anchors and deterministically fill with top-scored blockers."""
    if blocker_scores.shape != student_topm.shape:
        raise ValueError("blocker_scores and student_topm must have equal shape")
    if blocker_eligible.shape != student_topm.shape:
        raise ValueError("blocker_eligible and student_topm must have equal shape")
    blocker_eligible = blocker_eligible.bool()
    if (blocker_eligible.sum(dim=1) < length).any():
        raise ValueError("each user must have at least length eligible blockers")
    masked_scores = torch.where(
        blocker_eligible, blocker_scores,
        torch.full_like(blocker_scores, -float("inf")),
    )
    blocker_positions = torch.topk(masked_scores, length, dim=1).indices
    combined_positions = torch.cat([anchor_positions, blocker_positions], dim=1)
    combined_active = torch.cat([
        anchor_active,
        torch.ones_like(blocker_positions, dtype=torch.bool),
    ], dim=1)
    order = torch.arange(
        combined_positions.shape[1], device=combined_positions.device,
    ).unsqueeze(0)
    priority = torch.where(
        combined_active,
        combined_positions.new_tensor(combined_positions.shape[1]) - order,
        combined_positions.new_tensor(-1),
    )
    selected_columns = torch.topk(priority, length, dim=1).indices
    selected_positions = combined_positions.gather(1, selected_columns)
    return student_topm.gather(1, selected_positions)


def probability_entropy(probabilities):
    safe_log = torch.where(
        probabilities > 0., probabilities.clamp_min(1e-30).log(),
        torch.zeros_like(probabilities),
    )
    return -(probabilities * safe_log).sum(dim=1)


def calibrate_exponential_gamma(trusted_mass, target_mean, iterations=48):
    """Match mean(exp(-beta * trusted_mass)) to ``target_mean``.

    The bisection is deterministic and preserves the original epoch-level
    average L2 weight.  Only its allocation across users changes.  If users
    with exactly zero trusted mass make the requested mean unattainable, the
    closest attainable boundary is returned explicitly.
    """
    if trusted_mass.ndim != 1:
        raise ValueError("trusted_mass must be a rank-1 tensor")
    if trusted_mass.numel() == 0:
        raise ValueError("trusted_mass must be non-empty")
    # Solve the single scalar in CPU float64 after one device transfer. Doing
    # scalar-condition bisection on CUDA would force a host synchronization at
    # every iteration and recreate the utilization stalls this project avoids.
    work_mass = trusted_mass.detach().double().cpu()
    target = torch.as_tensor(target_mean).detach().double().cpu().clamp(0., 1.)
    zero_fraction = (work_mass == 0.).double().mean()
    attainable_target = target.clamp_min(zero_fraction)

    low = work_mass.new_tensor(0.)
    high = work_mass.new_tensor(1.)
    for _ in range(32):
        if torch.exp(-high * work_mass).mean() <= attainable_target:
            break
        high = high * 2.
    for _ in range(iterations):
        middle = (low + high) / 2.
        current = torch.exp(-middle * work_mass).mean()
        if current > attainable_target:
            low = middle
        else:
            high = middle
    beta = ((low + high) / 2.).to(
        device=trusted_mass.device, dtype=trusted_mass.dtype,
    )
    gamma = torch.exp(-beta * trusted_mass)
    return gamma, beta


def redistribute_gamma_by_difficulty(difficulty, reference_gamma):
    """Preserve every gamma value while assigning larger ones to harder users."""
    if difficulty.ndim != 1 or reference_gamma.ndim != 1:
        raise ValueError("difficulty and reference_gamma must be rank-1 tensors")
    if difficulty.shape != reference_gamma.shape:
        raise ValueError("difficulty and reference_gamma must have equal shape")
    difficulty_order = torch.argsort(difficulty, stable=True)
    sorted_gamma = torch.sort(reference_gamma).values
    redistributed = torch.empty_like(reference_gamma)
    redistributed[difficulty_order] = sorted_gamma
    return redistributed


class ARCEKD(RCEKD):
    """RCE-KD with Q1 anchors and alternative blocker-selection rules."""

    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "arcekd"
        self.arce_sampler = getattr(args, "arce_sampler", "marginal")
        if self.arce_sampler not in ("count", "marginal", "marginal_topl"):
            raise ValueError(
                "arce_sampler must be 'count', 'marginal', or 'marginal_topl'"
            )
        self.arce_mix_alpha = float(getattr(args, "arce_mix_alpha", .9))
        if not 0. <= self.arce_mix_alpha <= 1.:
            raise ValueError("arce_mix_alpha must be between zero and one")
        if self.L > self.mxK - self.K:
            raise ValueError(
                "ARCE-KD requires mkd_L <= mkd_mxK - mkd_K so every user "
                "has enough non-teacher blocker candidates"
            )
        self.arce_gamma_mode = getattr(args, "arce_gamma_mode", "sample_overlap")
        if self.arce_gamma_mode not in (
            "sample_overlap", "teacher_mass", "teacher_mass_calibrated",
            "closure_quantile",
        ):
            raise ValueError(
                "arce_gamma_mode must be sample_overlap, teacher_mass, or "
                "teacher_mass_calibrated, or closure_quantile"
            )
        with torch.no_grad():
            users = torch.arange(self.num_users, device=self.T_topk_dict.device)
            self.T_topk_scores = self.teacher.forward_multi_items(
                users, self.T_topk_dict,
            ) / self.tau
            self.T_topk_prob = torch.softmax(self.T_topk_scores, dim=1)

    def do_something_in_each_epoch(self, epoch):
        del epoch
        with torch.no_grad():
            student_score_mat = self.student.get_all_ratings() / self.tau
            student_scores_m, student_topm = torch.topk(
                student_score_mat, self.mxK, dim=-1,
            )
            self.itemS = student_topm[:, :self.K]
            q1_active = rowwise_isin(self.T_topk_dict, self.itemS)
            q2_active = ~q1_active
            student_scores_t = student_score_mat.gather(1, self.T_topk_dict)
            closure_statistics = None
            if (
                self.arce_sampler in ("marginal", "marginal_topl")
                or self.arce_gamma_mode == "closure_quantile"
            ):
                closure_statistics = topm_closure_statistics(
                    student_scores_m, student_scores_t, self.T_topk_dict,
                    student_topm, q2_active,
                )

            if self.arce_sampler == "count":
                blocker_probabilities = original_count_blocker_probabilities(
                    self.T_topk_dict, student_topm, self.T,
                )
            else:
                blocker_probabilities = topm_marginal_blocker_probabilities(
                    student_scores_m, student_scores_t, self.T_topk_scores,
                    self.T_topk_dict, student_topm, q2_active,
                    self.arce_mix_alpha, closure_statistics,
                )

            anchor_positions, anchor_active = prepare_teacher_anchors(
                student_topm, self.T_topk_dict, self.T_topk_scores,
                self.K, self.L,
            )
            if self.arce_sampler == "marginal_topl":
                blocker_eligible = ~rowwise_isin(
                    student_topm, self.T_topk_dict,
                )
                self.interesting_items = select_anchor_preserving_topl(
                    student_topm, anchor_positions, anchor_active,
                    blocker_probabilities, blocker_eligible, self.L,
                )
            else:
                self.interesting_items = sample_anchor_preserving(
                    student_topm, anchor_positions, anchor_active,
                    blocker_probabilities, self.L,
                )

            sampled_teacher = rowwise_isin(
                self.T_topk_dict, self.interesting_items,
            ).float().mean(dim=1)
            sample_gamma = torch.exp(-self.beta * sampled_teacher)
            teacher_missing_mass = (
                self.T_topk_prob * q2_active
            ).sum(dim=1)
            trusted_teacher_mass = 1. - teacher_missing_mass
            closure_difficulty = topm_closure_difficulty(
                self.T_topk_prob, q2_active,
                closure_statistics if closure_statistics is not None
                else topm_closure_statistics(
                    student_scores_m, student_scores_t, self.T_topk_dict,
                    student_topm, q2_active,
                ),
            )
            effective_beta = student_score_mat.new_tensor(self.beta)
            if self.arce_gamma_mode == "sample_overlap":
                gamma = sample_gamma
            elif self.arce_gamma_mode == "teacher_mass":
                gamma = torch.exp(-self.beta * trusted_teacher_mass)
            elif self.arce_gamma_mode == "teacher_mass_calibrated":
                gamma, effective_beta = calibrate_exponential_gamma(
                    trusted_teacher_mass, sample_gamma.mean(),
                )
            else:
                gamma = redistribute_gamma_by_difficulty(
                    closure_difficulty, sample_gamma,
                )
            self.arce_gamma = gamma
            anchor_count = anchor_active.sum(dim=1).float()
            entropy = probability_entropy(blocker_probabilities)
            return {
                "arce_sampler": self.arce_sampler,
                "arce_gamma_mode": self.arce_gamma_mode,
                "arce_anchor_count_mean": anchor_count.mean().item(),
                "arce_blocker_budget_mean": (self.L - anchor_count).mean().item(),
                "arce_sampled_teacher_overlap_mean": sampled_teacher.mean().item(),
                "arce_gamma_mean": gamma.mean().item(),
                "arce_gamma_std": gamma.std(unbiased=False).item(),
                "arce_gamma_min": gamma.min().item(),
                "arce_gamma_max": gamma.max().item(),
                "arce_sample_gamma_mean": sample_gamma.mean().item(),
                "arce_gamma_effective_beta": effective_beta.item(),
                "arce_teacher_missing_mass_mean": teacher_missing_mass.mean().item(),
                "arce_teacher_missing_mass_std": teacher_missing_mass.std(
                    unbiased=False,
                ).item(),
                "arce_closure_difficulty_mean": closure_difficulty.mean().item(),
                "arce_closure_difficulty_std": closure_difficulty.std(
                    unbiased=False,
                ).item(),
                "arce_gamma_reassignment_l1_mean": (
                    gamma - sample_gamma
                ).abs().mean().item(),
                "arce_blocker_entropy_mean": entropy.mean().item(),
            }

    def get_loss(self, *params):
        if self.arce_gamma_mode == "sample_overlap":
            return super().get_loss(*params)

        # This is RCEKD.get_loss verbatim except that the L2 mixture weight is
        # the precomputed teacher-mass gamma. Keeping the two CE terms exactly
        # unchanged isolates the gamma intervention.
        batch_users = params[0]
        itemS = self.itemS[batch_users]
        itemT = self.T_topk_dict[batch_users]
        item_interesting = self.interesting_items[batch_users]
        logit_S_itemS = self.student.forward_multi_items(batch_users, itemS) / self.tau
        logit_S_itemT = self.student.forward_multi_items(batch_users, itemT) / self.tau
        logit_S_interesting = self.student.forward_multi_items(
            batch_users, item_interesting,
        ) / self.tau
        logit_T_itemS = self.teacher.forward_multi_items(batch_users, itemS) / self.tau
        logit_T_itemT = self.teacher.forward_multi_items(batch_users, itemT) / self.tau

        exp_logit_T_itemS = torch.exp(logit_T_itemS)
        Z_T = exp_logit_T_itemS.sum(-1, keepdim=True)
        prob_T_itemS = exp_logit_T_itemS / Z_T
        loss_itemS = F.cross_entropy(
            logit_S_itemS, prob_T_itemS, reduction="none",
        )

        logit_T_interesting = self.teacher.forward_multi_items(
            batch_users, item_interesting,
        ) / self.tau
        exp_logit_T_interesting = torch.exp(logit_T_interesting)
        exp_logit_T_itemT = torch.exp(logit_T_itemT)
        mask = self.rowwise_isin(itemT, item_interesting)
        exp_logit_T_itemT[mask] = 0
        mask2 = self.rowwise_isin(itemT, itemS)
        exp_logit_T_itemT[mask2] = 0
        Z_T = (
            exp_logit_T_interesting.sum(-1, keepdim=True)
            + exp_logit_T_itemT.sum(-1, keepdim=True)
        )
        prob_T_all = torch.cat([
            exp_logit_T_interesting, exp_logit_T_itemT,
        ], dim=-1) / Z_T
        exp_logit_S_itemT = torch.exp(logit_S_itemT)
        exp_logit_S_itemT = (
            exp_logit_S_itemT * (1. - mask.float()) * (1. - mask2.float())
        )
        exp_logit_S_interesting = torch.exp(logit_S_interesting)
        Z_S = (
            exp_logit_S_interesting.sum(-1, keepdim=True)
            + exp_logit_S_itemT.sum(-1, keepdim=True)
        )
        logit_S_all = torch.cat([
            logit_S_interesting, logit_S_itemT,
        ], dim=-1)
        loss_itemT = -(
            prob_T_all * (logit_S_all - torch.log(Z_S))
        ).sum(-1)

        weight = self.arce_gamma[batch_users]
        return ((1. - weight) * loss_itemS + weight * loss_itemT).sum()

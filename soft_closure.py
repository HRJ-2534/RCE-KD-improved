"""Exact and truncated soft-closure quantities for RCE-KD diagnostics.

For a user-specific item set J and student scores s, define

    rho_i = sum_{j not in J, s_j >= s_i} exp(s_j)
            / sum_{j in J} exp(s_j).

The functions below retain gradients with respect to score magnitudes.  The
rank comparisons are necessarily piecewise constant, just like a top-k
operator.  Inactive padded entries are excluded from every quantity.
"""

import math

import torch


def _validate_inputs(student_scores, items, active_mask):
    if student_scores.ndim != 2 or items.ndim != 2:
        raise ValueError("student_scores and items must both be rank-2 tensors")
    if student_scores.shape[0] != items.shape[0]:
        raise ValueError("student_scores and items must have the same batch size")
    if active_mask is None:
        active_mask = torch.ones_like(items, dtype=torch.bool)
    if active_mask.shape != items.shape:
        raise ValueError("active_mask must have the same shape as items")
    return active_mask.bool()


def prepare_soft_closure_catalog(student_scores):
    """Precompute full-catalog quantities reusable across candidate sets."""
    if student_scores.ndim != 2:
        raise ValueError("student_scores must be a rank-2 tensor")
    shift = student_scores.max(dim=1, keepdim=True).values
    sorted_scores = student_scores.sort(dim=1, descending=True).values
    sorted_mass = torch.exp(sorted_scores - shift)
    prefix_mass = torch.cat([
        torch.zeros((student_scores.shape[0], 1), device=student_scores.device,
                    dtype=student_scores.dtype),
        sorted_mass.cumsum(dim=1),
    ], dim=1)
    return {
        "shift": shift,
        "sorted_scores": sorted_scores,
        "prefix_mass": prefix_mass,
    }


def exact_soft_closure_rho(student_scores, items, active_mask=None,
                           catalog_cache=None):
    """Compute rho exactly over every item using sorted prefix masses.

    This avoids materializing a ``batch x |J| x num_items`` tensor.  Scores
    tied with the target are included in H_i, matching the score-based
    definition ``s_j >= s_i``.
    """
    active_mask = _validate_inputs(student_scores, items, active_mask)
    device_items = items.to(student_scores.device)
    active = active_mask.to(student_scores.device)
    scores_j = student_scores.gather(1, device_items)

    if catalog_cache is None:
        catalog_cache = prepare_soft_closure_catalog(student_scores)
    shift = catalog_cache["shift"]
    sorted_scores = catalog_cache["sorted_scores"]
    prefix_mass = catalog_cache["prefix_mass"]
    expected_prefix_shape = (student_scores.shape[0], student_scores.shape[1] + 1)
    if shift.shape != (student_scores.shape[0], 1):
        raise ValueError("catalog_cache shift has an incompatible shape")
    if sorted_scores.shape != student_scores.shape or prefix_mass.shape != expected_prefix_shape:
        raise ValueError("catalog_cache has incompatible sorted-score or prefix shapes")
    exp_j = torch.exp(scores_j - shift) * active
    mass_j = exp_j.sum(dim=1, keepdim=True).clamp_min(torch.finfo(student_scores.dtype).tiny)
    # Negated descending scores are ascending. right=True counts every score
    # greater than or equal to the target, including exact ties.
    prefix_lengths = torch.searchsorted(
        -sorted_scores.contiguous(), -scores_j.contiguous(), right=True,
    )
    prefix_lengths = torch.where(active, prefix_lengths, torch.zeros_like(prefix_lengths))
    mass_h = prefix_mass.gather(1, prefix_lengths)

    # For every target i, subtract active members of J whose scores are at
    # least s_i; what remains is exactly H_i \ J.
    member_ahead = scores_j.unsqueeze(1) >= scores_j.unsqueeze(2)
    mass_h_intersect_j = (
        member_ahead * active.unsqueeze(1) * exp_j.unsqueeze(1)
    ).sum(dim=2)
    missing_mass = (mass_h - mass_h_intersect_j).clamp_min(0.)
    rho = missing_mass / mass_j
    return torch.where(active, rho, torch.zeros_like(rho))


def topm_soft_closure_rho(student_scores, items, topm_items, active_mask=None):
    """Compute the lower, top-M-only approximation of rho.

    The approximation contains exactly the missing blockers visible in the
    supplied top-M list.  It never invents tail mass, so comparison with the
    exact value directly measures how much closure error top-M misses.
    """
    active_mask = _validate_inputs(student_scores, items, active_mask)
    if topm_items.ndim != 2 or topm_items.shape[0] != items.shape[0]:
        raise ValueError("topm_items must be rank 2 with the same batch size")
    device = student_scores.device
    device_items = items.to(device)
    device_topm = topm_items.to(device)
    active = active_mask.to(device)
    scores_j = student_scores.gather(1, device_items)
    scores_m = student_scores.gather(1, device_topm)
    shift = torch.maximum(scores_j.max(dim=1, keepdim=True).values,
                          scores_m.max(dim=1, keepdim=True).values)
    exp_j = torch.exp(scores_j - shift) * active
    exp_m = torch.exp(scores_m - shift)
    mass_j = exp_j.sum(dim=1, keepdim=True).clamp_min(torch.finfo(student_scores.dtype).tiny)

    topm_is_in_j = (
        device_topm.unsqueeze(1) == device_items.unsqueeze(2)
    ) & active.unsqueeze(2)
    topm_is_in_j = topm_is_in_j.any(dim=1)
    topm_is_ahead = scores_m.unsqueeze(1) >= scores_j.unsqueeze(2)
    missing_visible = (
        topm_is_ahead
        * (~topm_is_in_j).unsqueeze(1)
        * exp_m.unsqueeze(1)
    ).sum(dim=2)
    rho = missing_visible / mass_j
    return torch.where(active, rho, torch.zeros_like(rho))


def soft_closure_bound_terms(student_scores, teacher_scores, items,
                             active_mask=None, rho=None):
    """Return CE, soft-closure penalty, and the corresponding NDCG bound.

    The bound is per user and uses natural logarithms:

        log NDCG_J >= -CE_J - E_{p_T^J}[log(1 + rho_i)] + log C_J.

    ``rho`` may be supplied by an approximation for diagnostics.  Omitting it
    computes the exact full-item value.
    """
    active_mask = _validate_inputs(student_scores, items, active_mask)
    if teacher_scores.shape != student_scores.shape:
        raise ValueError("teacher_scores must have the same shape as student_scores")
    device = student_scores.device
    device_items = items.to(device)
    active = active_mask.to(device)
    scores_s_j = student_scores.gather(1, device_items)
    scores_t_j = teacher_scores.gather(1, device_items)
    neg_inf = torch.full_like(scores_s_j, -float("inf"))
    masked_s_j = torch.where(active, scores_s_j, neg_inf)
    masked_t_j = torch.where(active, scores_t_j, neg_inf)

    log_p_s_j = torch.log_softmax(masked_s_j, dim=1)
    p_t_j = torch.softmax(masked_t_j, dim=1)
    ce = -(p_t_j * torch.where(active, log_p_s_j, torch.zeros_like(log_p_s_j))).sum(dim=1)

    if rho is None:
        rho = exact_soft_closure_rho(student_scores, items, active_mask)
    else:
        rho = rho.to(device)
        if rho.shape != items.shape:
            raise ValueError("rho must have the same shape as items")
    penalty = (p_t_j * torch.log1p(rho) * active).sum(dim=1)
    log_c_j = torch.logsumexp(masked_t_j, dim=1) - torch.logsumexp(teacher_scores, dim=1)
    lower_bound = -ce - penalty + log_c_j
    return {
        "ce": ce,
        "penalty": penalty,
        "log_c_j": log_c_j,
        "lower_bound": lower_bound,
        "teacher_prob_j": p_t_j,
    }


def exact_partial_log_ndcg(student_scores, teacher_scores, items, active_mask=None):
    """Compute the paper's partial NDCG using full student ranks."""
    active_mask = _validate_inputs(student_scores, items, active_mask)
    device = student_scores.device
    device_items = items.to(device)
    active = active_mask.to(device)
    scores_s_j = student_scores.gather(1, device_items)
    teacher_prob_full = torch.softmax(teacher_scores, dim=1)
    gains = teacher_prob_full.gather(1, device_items) * active

    # Continuous recommenders almost never tie.  A row-wise binary search on
    # sorted scores gives count(score > score_i) + 1 without constructing the
    # prohibitive batch x |J| x num_items comparison tensor.
    sorted_s = student_scores.sort(dim=1, descending=True).values
    ranks = torch.searchsorted(
        -sorted_s.contiguous(), -scores_s_j.contiguous(), right=False,
    ) + 1
    discounts = torch.log2(1. + ranks.to(student_scores.dtype))
    dcg = (gains / discounts * active).sum(dim=1)

    ideal_gains = torch.where(active, gains, torch.full_like(gains, -1.)).sort(
        dim=1, descending=True,
    ).values
    positions = torch.arange(1, items.shape[1] + 1, device=device,
                             dtype=student_scores.dtype)
    ideal_discounts = torch.log2(1. + positions).unsqueeze(0)
    sorted_active = torch.arange(items.shape[1], device=device).unsqueeze(0) < active.sum(
        dim=1, keepdim=True,
    )
    idcg = (ideal_gains.clamp_min(0.) / ideal_discounts * sorted_active).sum(dim=1)
    ndcg = dcg / idcg.clamp_min(torch.finfo(student_scores.dtype).tiny)
    return ndcg.clamp_min(torch.finfo(student_scores.dtype).tiny).log()


def summarize_tensor(values):
    """JSON-safe descriptive statistics for a one-dimensional tensor."""
    values = values.detach().double().cpu().flatten()
    if values.numel() == 0:
        return None
    q = torch.quantile(values, torch.tensor([0., .25, .5, .75, 1.], dtype=torch.float64))
    return {
        "count": values.numel(),
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "min": q[0].item(),
        "q25": q[1].item(),
        "median": q[2].item(),
        "q75": q[3].item(),
        "max": q[4].item(),
    }


def pearson_correlation(left, right):
    left = left.detach().double().cpu().flatten()
    right = right.detach().double().cpu().flatten()
    if left.numel() != right.numel() or left.numel() < 2:
        return None
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if not math.isfinite(denominator.item()) or denominator.item() == 0:
        return None
    return (left * right).sum().div(denominator).item()

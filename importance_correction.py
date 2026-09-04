"""Utilities for sampled-distillation gradient diagnostics."""

import torch


def masked_softmax(logits, active):
    if logits.shape != active.shape:
        raise ValueError("logits and active must have equal shape")
    active = active.bool()
    if (~active.any(dim=1)).any():
        raise ValueError("every row must contain at least one active entry")
    masked = torch.where(
        active, logits, torch.full_like(logits, -float("inf")),
    )
    probabilities = torch.softmax(masked, dim=1)
    return torch.where(active, probabilities, torch.zeros_like(probabilities))


def sampled_ce_logit_gradient(student_logits, teacher_logits, active,
                              inclusion=None, correct_teacher=True):
    """Gradient of a self-normalized sampled CE with optional log-pi terms.

    The returned tensor is the analytical gradient with respect to the
    *corrected* student logits.  Inclusion probabilities are treated as fixed
    proposal quantities, exactly as in sampled-softmax log-Q correction.
    """
    if not (student_logits.shape == teacher_logits.shape == active.shape):
        raise ValueError("student, teacher, and active tensors must match")
    active = active.bool()
    if inclusion is None:
        inclusion = torch.ones_like(student_logits)
    if inclusion.shape != student_logits.shape:
        raise ValueError("inclusion must match the logit tensors")
    if ((inclusion <= 0.) & active).any():
        raise ValueError("active entries require positive inclusion probability")
    log_inclusion = torch.where(
        active, inclusion.clamp_min(
            torch.finfo(student_logits.dtype).tiny
        ).log(),
        torch.zeros_like(inclusion),
    )
    student_probability = masked_softmax(
        student_logits - log_inclusion, active,
    )
    teacher_probability = masked_softmax(
        teacher_logits - log_inclusion if correct_teacher else teacher_logits,
        active,
    )
    return (student_probability - teacher_probability) * active


def sample_weighted_prefix(probabilities, budgets, draw_length, generator=None):
    """Match ARCE-KD's weighted-without-replacement prefix selection."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be rank 2")
    budgets = budgets.to(probabilities.device).long()
    if budgets.ndim != 1 or budgets.shape[0] != probabilities.shape[0]:
        raise ValueError("budgets must be a batch-length vector")
    if draw_length < 1 or draw_length > probabilities.shape[1]:
        raise ValueError("draw_length is outside the proposal width")
    if (budgets < 0).any() or (budgets > draw_length).any():
        raise ValueError("budgets must lie between zero and draw_length")
    if ((probabilities > 0.).sum(dim=1) < draw_length).any():
        raise ValueError("each proposal needs draw_length positive entries")
    positions = torch.multinomial(
        probabilities, draw_length, replacement=False, generator=generator,
    )
    steps = torch.arange(draw_length, device=probabilities.device).unsqueeze(0)
    return positions, steps < budgets.unsqueeze(1)


def estimate_inclusion_probabilities(
        probabilities, budgets, draw_length, trials, trial_chunk=64,
        generator=None):
    """Estimate prefix inclusion probabilities for weighted sampling.

    Budget-one rows are exact (pi=q).  Other rows use independent Monte Carlo
    draws with Jeffreys smoothing, which prevents unobserved low-probability
    candidates from receiving a spurious zero inclusion probability.
    """
    if trials < 1 or trial_chunk < 1:
        raise ValueError("trials and trial_chunk must be positive")
    device = probabilities.device
    batch_size, width = probabilities.shape
    counts = torch.zeros_like(probabilities)
    completed = 0
    while completed < trials:
        current = min(trial_chunk, trials - completed)
        repeated_probability = probabilities.repeat_interleave(current, dim=0)
        repeated_budget = budgets.repeat_interleave(current)
        positions, active = sample_weighted_prefix(
            repeated_probability, repeated_budget, draw_length, generator,
        )
        repeated_counts = torch.zeros_like(repeated_probability)
        repeated_counts.scatter_add_(
            1, positions, active.to(repeated_counts.dtype),
        )
        counts += repeated_counts.reshape(batch_size, current, width).sum(dim=1)
        completed += current

    support = probabilities > 0.
    estimate = (counts + .5) / (trials + 1.)
    estimate = torch.where(support, estimate, torch.zeros_like(estimate))
    # Fixed-size sampling has sum_j pi_j = budget exactly.  Restore that
    # identity after smoothing so the correction does not gain artificial
    # inclusion mass merely because the candidate catalog is wide.
    estimate = estimate * (
        budgets.to(estimate.dtype).unsqueeze(1)
        / estimate.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(estimate.dtype).tiny
        )
    )
    exact_budget_one = budgets == 1
    estimate = torch.where(
        exact_budget_one.unsqueeze(1), probabilities, estimate,
    )
    estimate = torch.where(
        (budgets > 0).unsqueeze(1), estimate, torch.zeros_like(estimate),
    )
    standard_error = torch.sqrt(
        estimate * (1. - estimate) / max(trials, 1)
    )
    return estimate, standard_error


def embed_sample_gradient(sample_gradient, sample_items, sample_active,
                          reference_items, reference_active):
    """Embed sampled coordinates into a unique active reference item set."""
    if not (
            sample_gradient.shape == sample_items.shape == sample_active.shape):
        raise ValueError("all sampled tensors must have equal shape")
    if reference_items.shape != reference_active.shape:
        raise ValueError("reference items and mask must have equal shape")
    if sample_items.shape[0] != reference_items.shape[0]:
        raise ValueError("sample and reference batch sizes must match")
    matches = (
        (sample_items.unsqueeze(2) == reference_items.unsqueeze(1))
        & reference_active.bool().unsqueeze(1)
    )
    found = matches.any(dim=2)
    if (sample_active.bool() & ~found).any():
        raise ValueError("an active sampled item is absent from the reference set")
    reference_positions = matches.to(torch.int64).argmax(dim=2)
    embedded = torch.zeros(
        reference_items.shape, device=sample_gradient.device,
        dtype=sample_gradient.dtype,
    )
    embedded.scatter_add_(
        1, reference_positions,
        sample_gradient * sample_active.to(sample_gradient.dtype),
    )
    return embedded

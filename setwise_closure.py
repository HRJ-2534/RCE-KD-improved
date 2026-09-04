"""Set-wise soft-closure selection utilities.

The target distribution is fixed on the teacher items that are absent from
the student's top-K (Q2).  For a current item set J, the diagnostic objective
is

    Phi(J) = sum_i p_i^T log(1 + B_i(J) / A(J)),

where A(J) is the student exponential-score mass inside J and B_i(J) is the
mass outside J that outranks target i.  Adding a candidate changes both terms,
so the functions below compute its exact discrete gain rather than the
first-order proxy used by ARCE-KD's current static proposal.
"""

import torch


def _validate_rank_two(name, value):
    if value.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 tensor")


def _fixed_target_probabilities(target_teacher_scores, target_active):
    target_active = target_active.bool()
    if (~target_active.any(dim=1)).any():
        raise ValueError("every row must contain at least one active target")
    masked = torch.where(
        target_active,
        target_teacher_scores,
        torch.full_like(target_teacher_scores, -float("inf")),
    )
    probabilities = torch.softmax(masked, dim=1)
    return torch.where(
        target_active, probabilities, torch.zeros_like(probabilities),
    )


def build_fixed_target_closure_state(
        student_scores, set_items, set_active, target_items, target_active,
        target_teacher_scores, catalog_cache):
    """Build the exact full-catalog state for a fixed set of Q2 targets.

    Active targets must be members of the active current set.  The returned
    missing masses and set mass use the catalog cache's shared exponential
    scale, so they can be updated exactly when candidate items are added.
    """
    for name, value in (
            ("student_scores", student_scores), ("set_items", set_items),
            ("set_active", set_active), ("target_items", target_items),
            ("target_active", target_active),
            ("target_teacher_scores", target_teacher_scores)):
        _validate_rank_two(name, value)
    batch_size = student_scores.shape[0]
    if any(value.shape[0] != batch_size for value in (
            set_items, set_active, target_items, target_active,
            target_teacher_scores)):
        raise ValueError("all inputs must have the same batch size")
    if set_items.shape != set_active.shape:
        raise ValueError("set_items and set_active must have equal shape")
    if not (
            target_items.shape == target_active.shape
            == target_teacher_scores.shape):
        raise ValueError("all target tensors must have equal shape")

    device = student_scores.device
    set_items = set_items.to(device)
    set_active = set_active.to(device).bool()
    target_items = target_items.to(device)
    target_active = target_active.to(device).bool()
    target_teacher_scores = target_teacher_scores.to(device)

    contained = (
        (target_items.unsqueeze(2) == set_items.unsqueeze(1))
        & set_active.unsqueeze(1)
    ).any(dim=2)
    if (target_active & ~contained).any():
        raise ValueError("every active target must be in the active set")

    shift = catalog_cache["shift"]
    sorted_scores = catalog_cache["sorted_scores"]
    prefix_mass = catalog_cache["prefix_mass"]
    if shift.shape != (batch_size, 1):
        raise ValueError("catalog_cache shift has an incompatible shape")
    if sorted_scores.shape != student_scores.shape:
        raise ValueError("catalog_cache sorted_scores has an incompatible shape")
    if prefix_mass.shape != (batch_size, student_scores.shape[1] + 1):
        raise ValueError("catalog_cache prefix_mass has an incompatible shape")

    set_scores = student_scores.gather(1, set_items)
    set_exp = torch.exp(set_scores - shift) * set_active
    set_mass = set_exp.sum(dim=1).clamp_min(
        torch.finfo(student_scores.dtype).tiny
    )
    target_scores = student_scores.gather(1, target_items)
    prefix_lengths = torch.searchsorted(
        -sorted_scores.contiguous(), -target_scores.contiguous(), right=True,
    )
    prefix_lengths = torch.where(
        target_active, prefix_lengths, torch.zeros_like(prefix_lengths),
    )
    ahead_catalog_mass = prefix_mass.gather(1, prefix_lengths)
    set_ahead = set_scores.unsqueeze(1) >= target_scores.unsqueeze(2)
    ahead_set_mass = (
        set_ahead * set_active.unsqueeze(1) * set_exp.unsqueeze(1)
    ).sum(dim=2)
    missing_mass = (ahead_catalog_mass - ahead_set_mass).clamp_min(0.)
    missing_mass = torch.where(
        target_active, missing_mass, torch.zeros_like(missing_mass),
    )
    target_probabilities = _fixed_target_probabilities(
        target_teacher_scores, target_active,
    )
    return {
        "shift": shift,
        "set_mass": set_mass,
        "missing_mass": missing_mass,
        "target_scores": target_scores,
        "target_active": target_active,
        "target_probabilities": target_probabilities,
    }


def fixed_target_closure_penalty(state):
    rho = state["missing_mass"] / state["set_mass"].unsqueeze(1)
    return (
        state["target_probabilities"] * torch.log1p(rho)
    ).sum(dim=1)


def exact_candidate_gains(state, candidate_scores, candidate_eligible):
    """Return the exact one-item reduction in the fixed-target objective."""
    _validate_rank_two("candidate_scores", candidate_scores)
    _validate_rank_two("candidate_eligible", candidate_eligible)
    if candidate_scores.shape != candidate_eligible.shape:
        raise ValueError("candidate scores and eligibility must have equal shape")
    if candidate_scores.shape[0] != state["set_mass"].shape[0]:
        raise ValueError("candidate scores have an incompatible batch size")

    candidate_eligible = candidate_eligible.bool()
    candidate_exp = torch.exp(candidate_scores - state["shift"])
    blocks_target = (
        candidate_scores.unsqueeze(2) >= state["target_scores"].unsqueeze(1)
    ) & state["target_active"].unsqueeze(1)
    after_missing = (
        state["missing_mass"].unsqueeze(1)
        - candidate_exp.unsqueeze(2) * blocks_target
    ).clamp_min(0.)
    after_mass = (
        state["set_mass"].unsqueeze(1) + candidate_exp
    ).unsqueeze(2)
    after_penalty = (
        state["target_probabilities"].unsqueeze(1)
        * torch.log1p(after_missing / after_mass)
    ).sum(dim=2)
    gains = fixed_target_closure_penalty(state).unsqueeze(1) - after_penalty
    return torch.where(
        candidate_eligible, gains,
        torch.full_like(gains, -float("inf")),
    )


def _proposal_from_gains(gains, eligible, mix_alpha):
    if not 0. <= mix_alpha <= 1.:
        raise ValueError("mix_alpha must be between zero and one")
    eligible = eligible.bool()
    eligible_float = eligible.to(gains.dtype)
    counts = eligible_float.sum(dim=1, keepdim=True)
    if (counts == 0).any():
        raise ValueError("every row must have at least one eligible candidate")
    uniform = eligible_float / counts
    raw = torch.where(eligible, gains.clamp_min(0.), torch.zeros_like(gains))
    totals = raw.sum(dim=1, keepdim=True)
    normalized = torch.where(
        totals > 0.,
        raw / totals.clamp_min(torch.finfo(gains.dtype).tiny),
        uniform,
    )
    return mix_alpha * normalized + (1. - mix_alpha) * uniform


def select_by_residual_closure(
        state, candidate_scores, candidate_eligible, budgets, max_length,
        mix_alpha=.9, greedy=False, generator=None):
    """Select candidates while recomputing their exact residual gains.

    ``budgets`` may differ by row.  Returned positions are padded to
    ``max_length`` and accompanied by an active mask.  Sampling uses the exact
    gain proposal at each step; greedy mode is a deterministic theoretical
    upper bound for this fixed-target closure objective.
    """
    if max_length < 1:
        raise ValueError("max_length must be positive")
    budgets = budgets.to(candidate_scores.device).long()
    if budgets.ndim != 1 or budgets.shape[0] != candidate_scores.shape[0]:
        raise ValueError("budgets must be a batch-length vector")
    eligible = candidate_eligible.to(candidate_scores.device).bool().clone()
    if (budgets < 0).any() or (budgets > max_length).any():
        raise ValueError("budgets must lie between zero and max_length")
    if (eligible.sum(dim=1) < budgets).any():
        raise ValueError("a row has fewer eligible candidates than its budget")

    working = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in state.items()
    }
    selected = torch.zeros(
        (candidate_scores.shape[0], max_length),
        device=candidate_scores.device, dtype=torch.long,
    )
    selected_active = torch.zeros_like(selected, dtype=torch.bool)
    candidate_exp = torch.exp(candidate_scores - working["shift"])
    initial_penalty = fixed_target_closure_penalty(working)
    initial_gains = exact_candidate_gains(
        working, candidate_scores, eligible,
    )
    entropy_sum = torch.zeros_like(initial_penalty)

    for step in range(max_length):
        active_rows = step < budgets
        if not active_rows.any():
            break
        gains = exact_candidate_gains(working, candidate_scores, eligible)
        proposal = _proposal_from_gains(gains, eligible, mix_alpha)
        safe_log = torch.where(
            proposal > 0., proposal.clamp_min(1e-30).log(),
            torch.zeros_like(proposal),
        )
        entropy_sum = entropy_sum + (
            -(proposal * safe_log).sum(dim=1) * active_rows
        )
        if greedy:
            positions = gains.argmax(dim=1)
        else:
            positions = torch.multinomial(
                proposal, 1, replacement=False, generator=generator,
            ).squeeze(1)
        selected[:, step] = positions
        selected_active[:, step] = active_rows

        chosen_exp = candidate_exp.gather(1, positions.unsqueeze(1)).squeeze(1)
        chosen_scores = candidate_scores.gather(
            1, positions.unsqueeze(1),
        ).squeeze(1)
        chosen_blocks = (
            chosen_scores.unsqueeze(1) >= working["target_scores"]
        ) & working["target_active"]
        active_float = active_rows.to(candidate_scores.dtype)
        working["set_mass"] = (
            working["set_mass"] + chosen_exp * active_float
        )
        working["missing_mass"] = (
            working["missing_mass"]
            - chosen_exp.unsqueeze(1) * chosen_blocks * active_float.unsqueeze(1)
        ).clamp_min(0.)
        row_indices = torch.arange(
            candidate_scores.shape[0], device=candidate_scores.device,
        )[active_rows]
        eligible[row_indices, positions[active_rows]] = False

    final_penalty = fixed_target_closure_penalty(working)
    selected_singleton_gains = initial_gains.gather(1, selected).masked_fill(
        ~selected_active, 0.,
    ).sum(dim=1)
    realized_gain = initial_penalty - final_penalty
    average_entropy = entropy_sum / budgets.clamp_min(1).to(entropy_sum.dtype)
    return {
        "positions": selected,
        "active": selected_active,
        "final_state": working,
        "initial_penalty": initial_penalty,
        "final_penalty": final_penalty,
        "realized_gain": realized_gain,
        "singleton_gain_sum": selected_singleton_gains,
        "additive_overestimate": (
            selected_singleton_gains - realized_gain
        ).clamp_min(0.),
        "mean_proposal_entropy": average_entropy,
    }


def compose_anchor_and_blockers(
        student_topm, anchor_positions, anchor_active, blocker_positions,
        blocker_active, length):
    """Construct the fixed-L interesting-item set without padding leakage."""
    positions = torch.cat([anchor_positions, blocker_positions], dim=1)
    active = torch.cat([anchor_active, blocker_active], dim=1)
    order = torch.arange(
        positions.shape[1], device=positions.device,
    ).unsqueeze(0)
    priority = torch.where(
        active, positions.new_tensor(positions.shape[1]) - order,
        positions.new_tensor(-1),
    )
    columns = torch.topk(priority, length, dim=1).indices
    chosen_active = active.gather(1, columns)
    if not chosen_active.all():
        raise ValueError("anchors and blockers do not fill the requested length")
    return student_topm.gather(1, positions.gather(1, columns))

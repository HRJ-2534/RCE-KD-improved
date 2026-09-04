import unittest

import torch

from setwise_closure import (
    build_fixed_target_closure_state,
    compose_anchor_and_blockers,
    exact_candidate_gains,
    fixed_target_closure_penalty,
    select_by_residual_closure,
)
from soft_closure import prepare_soft_closure_catalog


class SetwiseClosureTests(unittest.TestCase):
    def setUp(self):
        self.student = torch.tensor([
            [3.0, 2.6, 2.0, 1.8, 1.0, 0.2],
            [2.8, 2.7, 2.3, 1.5, 1.2, 0.1],
        ])
        self.current_items = torch.tensor([[3, 5], [2, 5]])
        self.current_active = torch.ones_like(
            self.current_items, dtype=torch.bool,
        )
        self.targets = torch.tensor([[3, 5], [2, 5]])
        self.target_active = torch.tensor([
            [True, True], [True, False],
        ])
        self.teacher_target_scores = torch.tensor([
            [2.0, 1.0], [1.5, -4.0],
        ])
        self.cache = prepare_soft_closure_catalog(self.student)

    def state(self):
        return build_fixed_target_closure_state(
            self.student, self.current_items, self.current_active,
            self.targets, self.target_active, self.teacher_target_scores,
            self.cache,
        )

    def test_exact_candidate_gain_matches_rebuilding_the_augmented_set(self):
        state = self.state()
        candidates = torch.tensor([[0, 1, 2], [0, 1, 3]])
        candidate_scores = self.student.gather(1, candidates)
        eligible = torch.ones_like(candidates, dtype=torch.bool)
        gains = exact_candidate_gains(state, candidate_scores, eligible)
        initial = fixed_target_closure_penalty(state)

        for column in range(candidates.shape[1]):
            augmented_items = torch.cat([
                self.current_items, candidates[:, column:column + 1],
            ], dim=1)
            augmented_active = torch.cat([
                self.current_active,
                torch.ones((2, 1), dtype=torch.bool),
            ], dim=1)
            rebuilt = build_fixed_target_closure_state(
                self.student, augmented_items, augmented_active,
                self.targets, self.target_active, self.teacher_target_scores,
                self.cache,
            )
            expected = initial - fixed_target_closure_penalty(rebuilt)
            torch.testing.assert_close(gains[:, column], expected)

    def test_sequential_selection_preserves_budget_uniqueness_and_state(self):
        state = self.state()
        candidates = torch.tensor([[0, 1, 2], [0, 1, 3]])
        result = select_by_residual_closure(
            state, self.student.gather(1, candidates),
            torch.ones_like(candidates, dtype=torch.bool),
            torch.tensor([2, 1]), 2, mix_alpha=.9,
            generator=torch.Generator().manual_seed(7),
        )
        self.assertEqual(result["active"].sum(dim=1).tolist(), [2, 1])
        for row, budget in enumerate((2, 1)):
            positions = result["positions"][row][result["active"][row]]
            self.assertEqual(positions.unique().numel(), budget)

        selected_items = candidates.gather(1, result["positions"])
        augmented_items = torch.cat([self.current_items, selected_items], dim=1)
        augmented_active = torch.cat([
            self.current_active, result["active"],
        ], dim=1)
        rebuilt = build_fixed_target_closure_state(
            self.student, augmented_items, augmented_active,
            self.targets, self.target_active, self.teacher_target_scores,
            self.cache,
        )
        torch.testing.assert_close(
            result["final_penalty"], fixed_target_closure_penalty(rebuilt),
        )

    def test_greedy_is_the_fixed_target_upper_bound_for_ranked_candidates(self):
        state = self.state()
        candidates = torch.tensor([[0, 1, 2], [0, 1, 3]])
        result = select_by_residual_closure(
            state, self.student.gather(1, candidates),
            torch.ones_like(candidates, dtype=torch.bool),
            torch.tensor([2, 2]), 2, greedy=True,
        )
        # Higher student-ranked candidates dominate lower ones: they carry
        # more mass and block a superset of the fixed Q2 targets.
        self.assertEqual(result["positions"].tolist(), [[0, 1], [0, 1]])
        self.assertTrue((result["realized_gain"] >= 0.).all())
        self.assertTrue((result["additive_overestimate"] >= 0.).all())

    def test_composition_keeps_active_anchors_and_exact_length(self):
        topm = torch.tensor([[8, 4, 7, 3, 2], [9, 5, 1, 6, 0]])
        anchors = torch.tensor([[1, 4], [0, 2]])
        anchor_active = torch.tensor([[True, False], [True, True]])
        blockers = torch.tensor([[0, 2], [3, 4]])
        blocker_active = torch.tensor([[True, True], [True, False]])
        selected = compose_anchor_and_blockers(
            topm, anchors, anchor_active, blockers, blocker_active, 3,
        )
        self.assertEqual(selected[0].tolist(), [4, 8, 7])
        self.assertEqual(set(selected[1].tolist()), {9, 1, 6})


if __name__ == "__main__":
    unittest.main()

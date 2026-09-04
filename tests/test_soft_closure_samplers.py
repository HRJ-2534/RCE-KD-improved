import unittest

import torch

from compare_soft_closure_samplers import (
    build_sampler_probabilities,
    normalize_with_uniform_mixture,
    prepare_teacher_anchors,
    sample_anchor_preserving,
)
from soft_closure import exact_soft_closure_rho, prepare_soft_closure_catalog


class SoftClosureSamplerTests(unittest.TestCase):
    def test_uniform_mixture_is_normalized_and_respects_support(self):
        raw = torch.tensor([[4., 0., 2., 9.]])
        eligible = torch.tensor([[True, False, True, False]])
        proposal = normalize_with_uniform_mixture(raw, eligible, .8)
        torch.testing.assert_close(proposal.sum(1), torch.ones(1))
        self.assertEqual(proposal[0, 1].item(), 0.)
        self.assertEqual(proposal[0, 3].item(), 0.)
        self.assertGreater(proposal[0, 0], proposal[0, 2])

    def test_mass_sampler_prioritizes_strong_blockers_with_full_support(self):
        student = torch.tensor([[6., 5., 4., 3., 2., 1.]], dtype=torch.float64)
        teacher = torch.tensor([[1., 2., 3., 4., 7., 6.]], dtype=torch.float64)
        teacher_topk = torch.tensor([[4, 5]])
        student_topm = torch.tensor([[0, 1, 2, 3, 4]])
        student_topk = student_topm[:, :2]
        q2 = torch.ones_like(teacher_topk, dtype=torch.bool)
        cache = prepare_soft_closure_catalog(student)
        base_rho = exact_soft_closure_rho(student, teacher_topk, q2, cache)
        proposals = build_sampler_probabilities(
            student, teacher, teacher_topk, student_topm, q2, base_rho, cache,
            mix_alpha=1., score_temperature=1.,
        )
        mass = proposals["teacher_student_mass"]
        # Items 0 and 1 block both teacher targets, but item 0 has more
        # student mass. Full support intentionally retains the teacher item.
        self.assertGreater(mass[0, 0], mass[0, 1])
        self.assertGreater(mass[0, 4].item(), 0.)
        torch.testing.assert_close(mass.sum(1), torch.ones(1, dtype=torch.float64))

        mass_no_teacher = proposals["teacher_student_mass_no_teacher"]
        self.assertEqual(mass_no_teacher[0, 4].item(), 0.)
        torch.testing.assert_close(
            mass_no_teacher.sum(1), torch.ones(1, dtype=torch.float64),
        )

    def test_each_support_condition_has_a_matching_original_baseline(self):
        student = torch.tensor([[6., 5., 4., 3., 2., 1.]], dtype=torch.float64)
        teacher = torch.tensor([[1., 2., 3., 4., 7., 6.]], dtype=torch.float64)
        teacher_topk = torch.tensor([[4, 5]])
        student_topm = torch.tensor([[0, 1, 2, 3, 4]])
        q2 = torch.ones_like(teacher_topk, dtype=torch.bool)
        cache = prepare_soft_closure_catalog(student)
        base_rho = exact_soft_closure_rho(student, teacher_topk, q2, cache)
        proposals = build_sampler_probabilities(
            student, teacher, teacher_topk, student_topm, q2, base_rho, cache,
        )
        self.assertIn("original_code_count", proposals)
        self.assertIn("original_code_count_no_teacher", proposals)
        self.assertEqual(proposals["original_code_count_no_teacher"][0, 4], 0.)

    def test_zero_closure_signal_falls_back_to_uniform_eligible_support(self):
        raw = torch.zeros((1, 4))
        eligible = torch.tensor([[True, True, False, False]])
        proposal = normalize_with_uniform_mixture(raw, eligible, 1.)
        torch.testing.assert_close(proposal, torch.tensor([[.5, .5, 0., 0.]]))

    def test_anchor_sampler_keeps_q1_and_fills_with_unique_nonteacher_items(self):
        student_topm = torch.tensor([
            [0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0],
        ])
        teacher_topk = torch.tensor([[1, 4], [5, 4]])
        teacher_scores = torch.tensor([
            [0., 9., 1., 2., 8., 3.],
            [0., 1., 2., 3., 8., 9.],
        ])
        positions, active = prepare_teacher_anchors(
            student_topm, teacher_topk, teacher_scores, k=3, length=2,
        )
        # Row 0 has one Q1 anchor (item 1); row 1 has two (items 5 and 4).
        self.assertEqual(active.sum(1).tolist(), [1, 2])
        probabilities = torch.tensor([
            [.25, 0., .25, .25, 0., .25],
            [0., 0., .25, .25, .25, .25],
        ])
        sampled = sample_anchor_preserving(
            student_topm, positions, active, probabilities, length=2,
            generator=torch.Generator().manual_seed(9),
        )
        self.assertIn(1, sampled[0].tolist())
        self.assertEqual(set(sampled[1].tolist()), {4, 5})
        self.assertEqual(len(set(sampled[0].tolist())), 2)


if __name__ == "__main__":
    unittest.main()

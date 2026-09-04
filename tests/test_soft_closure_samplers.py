import unittest

import torch

from compare_soft_closure_samplers import (
    build_sampler_probabilities,
    normalize_with_uniform_mixture,
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

    def test_mass_sampler_prioritizes_strong_blockers_and_excludes_teacher(self):
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
        # student mass. The teacher item at position 4 is excluded.
        self.assertGreater(mass[0, 0], mass[0, 1])
        self.assertEqual(mass[0, 4].item(), 0.)
        torch.testing.assert_close(mass.sum(1), torch.ones(1, dtype=torch.float64))

    def test_zero_closure_signal_falls_back_to_uniform_eligible_support(self):
        raw = torch.zeros((1, 4))
        eligible = torch.tensor([[True, True, False, False]])
        proposal = normalize_with_uniform_mixture(raw, eligible, 1.)
        torch.testing.assert_close(proposal, torch.tensor([[.5, .5, 0., 0.]]))


if __name__ == "__main__":
    unittest.main()

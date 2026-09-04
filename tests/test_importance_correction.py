import unittest

import torch

from importance_correction import (
    embed_sample_gradient,
    estimate_inclusion_probabilities,
    masked_softmax,
    sampled_ce_logit_gradient,
)


class ImportanceCorrectionTests(unittest.TestCase):
    def test_uncorrected_gradient_is_student_minus_teacher_probability(self):
        student = torch.tensor([[2., 1., -3.]])
        teacher = torch.tensor([[1., 2., 7.]])
        active = torch.tensor([[True, True, False]])
        actual = sampled_ce_logit_gradient(student, teacher, active)
        expected = masked_softmax(student, active) - masked_softmax(teacher, active)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(dim=1), torch.zeros(1))

    def test_unit_inclusion_is_exact_identity_for_both_correction_modes(self):
        student = torch.tensor([[2., 1., 0.]])
        teacher = torch.tensor([[1., 3., 2.]])
        active = torch.ones_like(student, dtype=torch.bool)
        baseline = sampled_ce_logit_gradient(student, teacher, active)
        for correct_teacher in (False, True):
            actual = sampled_ce_logit_gradient(
                student, teacher, active, torch.ones_like(student),
                correct_teacher=correct_teacher,
            )
            torch.testing.assert_close(actual, baseline, rtol=0., atol=0.)

    def test_budget_one_inclusion_is_exact_proposal_not_monte_carlo(self):
        probabilities = torch.tensor([[.5, .3, .2]])
        estimate, standard_error = estimate_inclusion_probabilities(
            probabilities, torch.tensor([1]), draw_length=2,
            trials=7, trial_chunk=3,
            generator=torch.Generator().manual_seed(5),
        )
        torch.testing.assert_close(estimate, probabilities, rtol=0., atol=0.)
        self.assertTrue((standard_error > 0.).all())

    def test_uniform_inclusion_matches_budget_over_width(self):
        probabilities = torch.full((1, 5), .2)
        estimate, _ = estimate_inclusion_probabilities(
            probabilities, torch.tensor([2]), draw_length=3,
            trials=20000, trial_chunk=200,
            generator=torch.Generator().manual_seed(11),
        )
        torch.testing.assert_close(
            estimate, torch.full_like(estimate, .4), rtol=.03, atol=.015,
        )

    def test_embedding_uses_only_active_unique_reference_coordinates(self):
        sample_items = torch.tensor([[4, 1, 7]])
        sample_active = torch.tensor([[True, True, False]])
        sample_gradient = torch.tensor([[.3, -.3, 9.]])
        reference_items = torch.tensor([[1, 4, 7, 1]])
        reference_active = torch.tensor([[True, True, True, False]])
        embedded = embed_sample_gradient(
            sample_gradient, sample_items, sample_active,
            reference_items, reference_active,
        )
        torch.testing.assert_close(
            embedded, torch.tensor([[-.3, .3, 0., 0.]]),
        )


if __name__ == "__main__":
    unittest.main()

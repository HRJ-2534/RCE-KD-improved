import contextlib
import io
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from modeling.KD.anchor_rce import (
    ARCEKD,
    calibrate_exponential_gamma,
    prepare_teacher_anchors,
    power_sharpen_probabilities,
    redistribute_gamma_by_difficulty,
    rowwise_isin,
    sample_anchor_preserving,
    select_anchor_preserving_topl,
    topm_closure_difficulty,
    topm_closure_statistics,
    topm_marginal_blocker_probabilities,
)


class TinyBackbone(nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.score_parameters = nn.Parameter(scores.clone())

    def get_all_ratings(self):
        return self.score_parameters

    def forward_multi_items(self, users, items):
        return self.score_parameters[users].gather(1, items)


def make_model(sampler, gamma_mode="sample_overlap"):
    student_scores = torch.tensor([
        [9., 8., 7., 6., 5., 4., 3., 2.],
        [1., 3., 5., 7., 9., 8., 6., 4.],
    ])
    teacher_scores = torch.tensor([
        [1., 8., 2., 7., 3., 6., 4., 5.],
        [9., 8., 7., 6., 5., 4., 3., 2.],
    ])
    model = ARCEKD.__new__(ARCEKD)
    nn.Module.__init__(model)
    model.student = TinyBackbone(student_scores)
    model.teacher = TinyBackbone(teacher_scores)
    model.teacher.requires_grad_(False)
    model.num_users, model.num_items = student_scores.shape
    model.K, model.L, model.mxK = 2, 2, 5
    model.T, model.tau, model.beta = 10., 1., 3.
    model.arce_sampler, model.arce_mix_alpha = sampler, .9
    model.arce_sampling_power = 1.
    model.arce_gamma_mode = gamma_mode
    with contextlib.redirect_stdout(io.StringIO()):
        model.T_topk_dict = model.get_topk_dict(model.teacher, model.K)
    users = torch.arange(model.num_users)
    model.T_topk_scores = model.teacher.forward_multi_items(
        users, model.T_topk_dict,
    )
    model.T_topk_prob = torch.softmax(model.T_topk_scores, dim=1)
    return model


class AnchorRCEKDTests(unittest.TestCase):
    def test_sampling_power_one_is_exact_identity_and_larger_power_sharpens(self):
        probabilities = torch.tensor([
            [.5, .3, .2, 0.], [.1, .2, .7, 0.],
        ])
        identity = power_sharpen_probabilities(probabilities, 1.)
        self.assertIs(identity, probabilities)
        sharpened = power_sharpen_probabilities(probabilities, 2.)
        torch.testing.assert_close(sharpened.sum(dim=1), torch.ones(2))
        self.assertTrue(torch.equal(sharpened[:, 3], torch.zeros(2)))
        self.assertGreater(sharpened[0, 0], probabilities[0, 0])
        self.assertGreater(sharpened[1, 2], probabilities[1, 2])

    def test_calibrated_gamma_matches_target_mean_without_changing_order(self):
        trusted = torch.tensor([.1, .3, .6, .9], dtype=torch.float64)
        gamma, beta = calibrate_exponential_gamma(trusted, .4)
        torch.testing.assert_close(gamma.mean(), torch.tensor(.4, dtype=torch.float64))
        self.assertGreater(beta.item(), 0.)
        self.assertTrue(torch.all(gamma[:-1] > gamma[1:]))

    def test_closure_gamma_preserves_full_distribution_and_follows_difficulty(self):
        difficulty = torch.tensor([.8, .1, .5, .3])
        reference = torch.tensor([.05, .7, .2, .1])
        redistributed = redistribute_gamma_by_difficulty(difficulty, reference)
        torch.testing.assert_close(
            torch.sort(redistributed).values, torch.sort(reference).values,
            rtol=0., atol=0.,
        )
        order = torch.argsort(difficulty)
        self.assertTrue(torch.all(
            redistributed[order][:-1] <= redistributed[order][1:]
        ))

    def test_topm_marginal_probabilities_are_finite_supported_and_normalized(self):
        scores_m = torch.tensor([[6., 5., 4., 3., 2.]])
        student_topm = torch.tensor([[0, 1, 2, 3, 4]])
        teacher_topk = torch.tensor([[3, 4]])
        scores_t = torch.tensor([[3., 2.]])
        teacher_t = torch.tensor([[5., 4.]])
        q2 = torch.tensor([[True, True]])
        probabilities = topm_marginal_blocker_probabilities(
            scores_m, scores_t, teacher_t, teacher_topk, student_topm, q2, .9,
        )
        self.assertTrue(torch.isfinite(probabilities).all())
        torch.testing.assert_close(probabilities.sum(1), torch.ones(1))
        self.assertEqual(probabilities[0, 3].item(), 0.)
        self.assertEqual(probabilities[0, 4].item(), 0.)
        self.assertGreater(probabilities[0, 0], probabilities[0, 2])

    def test_topm_marginal_matches_direct_definition(self):
        scores_m = torch.tensor([[6., 5., 4., 3., 2.]], dtype=torch.float64)
        student_topm = torch.tensor([[0, 1, 2, 3, 4]])
        teacher_topk = torch.tensor([[1, 4]])
        scores_t = torch.tensor([[5., 2.]], dtype=torch.float64)
        teacher_t = torch.tensor([[3., 1.]], dtype=torch.float64)
        q2 = torch.tensor([[True, True]])
        actual = topm_marginal_blocker_probabilities(
            scores_m, scores_t, teacher_t, teacher_topk, student_topm, q2, 1.,
        )

        exp_scores = scores_m[0].exp()
        mass_j = scores_t[0].exp().sum()
        p_teacher = torch.softmax(teacher_t[0], dim=0)
        raw = torch.zeros(5, dtype=torch.float64)
        teacher_items = set(teacher_topk[0].tolist())
        for candidate_pos, candidate in enumerate(student_topm[0].tolist()):
            if candidate in teacher_items:
                continue
            for target_pos, target_score in enumerate(scores_t[0]):
                missing_mass = sum(
                    exp_scores[pos]
                    for pos, item in enumerate(student_topm[0].tolist())
                    if item not in teacher_items and scores_m[0, pos] >= target_score
                )
                if scores_m[0, candidate_pos] >= target_score:
                    raw[candidate_pos] += (
                        exp_scores[candidate_pos]
                        * p_teacher[target_pos]
                        / (mass_j + missing_mass)
                    )
        expected = raw / raw.sum()
        torch.testing.assert_close(actual[0], expected)

    def test_topm_closure_difficulty_matches_direct_definition(self):
        scores_m = torch.tensor([[6., 5., 4., 3., 2.]], dtype=torch.float64)
        student_topm = torch.tensor([[0, 1, 2, 3, 4]])
        teacher_topk = torch.tensor([[1, 4]])
        scores_t = torch.tensor([[5., 2.]], dtype=torch.float64)
        q2 = torch.tensor([[True, True]])
        teacher_prob = torch.tensor([[.8, .2]], dtype=torch.float64)
        statistics = topm_closure_statistics(
            scores_m, scores_t, teacher_topk, student_topm, q2,
        )
        actual = topm_closure_difficulty(teacher_prob, q2, statistics)
        mass_j = scores_t[0].exp().sum()
        expected = scores_m.new_tensor(0.)
        teacher_items = set(teacher_topk[0].tolist())
        for target_pos, target_score in enumerate(scores_t[0]):
            missing = sum(
                scores_m[0, pos].exp()
                for pos, item in enumerate(student_topm[0].tolist())
                if item not in teacher_items and scores_m[0, pos] >= target_score
            )
            expected += teacher_prob[0, target_pos] * torch.log1p(missing / mass_j)
        torch.testing.assert_close(actual[0], expected)

    def test_anchor_sampling_preserves_q1_and_fixed_unique_budget(self):
        model = make_model("marginal")
        torch.manual_seed(31)
        diagnostics = model.do_something_in_each_epoch(0)
        q1 = rowwise_isin(model.T_topk_dict, model.itemS)
        q1_items = torch.where(q1, model.T_topk_dict, -1)
        for row in range(model.num_users):
            selected = model.interesting_items[row].tolist()
            expected_anchors = [item for item in q1_items[row].tolist() if item >= 0]
            self.assertTrue(set(expected_anchors).issubset(selected))
            self.assertEqual(len(selected), model.L)
            self.assertEqual(len(set(selected)), model.L)
        self.assertIn("arce_gamma_mean", diagnostics)
        self.assertGreaterEqual(diagnostics["arce_gamma_mean"], 0.)
        self.assertLessEqual(diagnostics["arce_gamma_mean"], 1.)

    def test_all_blocker_selectors_keep_the_rcekd_loss_and_finite_gradients(self):
        for sampler in ("count", "marginal", "marginal_topl"):
            with self.subTest(sampler=sampler):
                model = make_model(sampler)
                torch.manual_seed(47)
                model.do_something_in_each_epoch(0)
                users = torch.tensor([0, 1])
                loss = model.get_loss(users)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(torch.isfinite(model.student.score_parameters.grad).all())

    def test_marginal_topl_is_deterministic_and_keeps_exact_budget(self):
        model = make_model("marginal_topl")
        torch.manual_seed(11)
        diagnostics = model.do_something_in_each_epoch(0)
        first = model.interesting_items.clone()
        torch.manual_seed(997)
        model.do_something_in_each_epoch(1)
        torch.testing.assert_close(
            model.interesting_items, first, rtol=0., atol=0.,
        )
        self.assertEqual(diagnostics["arce_sampler"], "marginal_topl")
        for row in first:
            self.assertEqual(row.unique().numel(), model.L)

    def test_mass_gamma_preserves_mean_when_calibrated_and_changes_allocation(self):
        sample_model = make_model("marginal", "sample_overlap")
        calibrated_model = make_model("marginal", "teacher_mass_calibrated")
        torch.manual_seed(53)
        sample_model.do_something_in_each_epoch(0)
        torch.manual_seed(53)
        diagnostics = calibrated_model.do_something_in_each_epoch(0)
        torch.testing.assert_close(
            calibrated_model.arce_gamma.mean(),
            sample_model.arce_gamma.mean(),
        )
        self.assertEqual(
            diagnostics["arce_gamma_mode"], "teacher_mass_calibrated",
        )
        users = torch.tensor([0, 1])
        loss = calibrated_model.get_loss(users)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            torch.isfinite(calibrated_model.student.score_parameters.grad).all()
        )

    def test_closure_quantile_mode_preserves_every_sample_gamma_value(self):
        model = make_model("marginal", "closure_quantile")
        torch.manual_seed(61)
        model.do_something_in_each_epoch(0)
        sampled_teacher = rowwise_isin(
            model.T_topk_dict, model.interesting_items,
        ).float().mean(dim=1)
        sample_gamma = torch.exp(-model.beta * sampled_teacher)
        torch.testing.assert_close(
            torch.sort(model.arce_gamma).values,
            torch.sort(sample_gamma).values,
            rtol=0., atol=0.,
        )
        users = torch.tensor([0, 1])
        loss = model.get_loss(users)
        self.assertTrue(torch.isfinite(loss))

    def test_gamma_override_changes_only_the_rcekd_mixture_weight(self):
        reference = make_model("marginal", "sample_overlap")
        overridden = make_model("marginal", "teacher_mass")
        torch.manual_seed(71)
        reference.do_something_in_each_epoch(0)
        torch.manual_seed(71)
        overridden.do_something_in_each_epoch(0)
        self.assertTrue(torch.equal(
            reference.interesting_items, overridden.interesting_items,
        ))
        # Force the copied gamma branch to use exactly the original weights.
        # Identical outputs then prove that neither CE term was altered.
        overridden.arce_gamma = reference.arce_gamma.clone()
        users = torch.tensor([0, 1])
        expected = reference.get_loss(users)
        actual = overridden.get_loss(users)
        torch.testing.assert_close(actual, expected, rtol=0., atol=0.)

    def test_anchor_fill_uses_teacher_excluded_blockers(self):
        student_topm = torch.tensor([[0, 1, 2, 3, 4]])
        teacher_topk = torch.tensor([[1, 4]])
        teacher_scores = torch.tensor([[5., 4.]])
        positions, active = prepare_teacher_anchors(
            student_topm, teacher_topk, teacher_scores, k=2, length=2,
        )
        probabilities = torch.tensor([[.4, 0., .3, .3, 0.]])
        sampled = sample_anchor_preserving(
            student_topm, positions, active, probabilities, length=2,
        )
        self.assertIn(1, sampled[0].tolist())
        self.assertNotIn(4, sampled[0].tolist())

    def test_topl_masks_ineligible_items_even_when_their_scores_are_largest(self):
        student_topm = torch.tensor([[0, 1, 2, 3, 4]])
        anchor_positions = torch.tensor([[1, 0]])
        anchor_active = torch.tensor([[True, False]])
        blocker_scores = torch.tensor([[.8, 100., .2, .7, 90.]])
        blocker_eligible = torch.tensor([[True, False, True, True, False]])
        selected = select_anchor_preserving_topl(
            student_topm, anchor_positions, anchor_active,
            blocker_scores, blocker_eligible, length=2,
        )
        self.assertEqual(set(selected[0].tolist()), {0, 1})


if __name__ == "__main__":
    unittest.main()

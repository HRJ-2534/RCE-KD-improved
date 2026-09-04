import contextlib
import io
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from modeling.KD.anchor_rce import (
    ARCEKD,
    prepare_teacher_anchors,
    rowwise_isin,
    sample_anchor_preserving,
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


def make_model(sampler):
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
    with contextlib.redirect_stdout(io.StringIO()):
        model.T_topk_dict = model.get_topk_dict(model.teacher, model.K)
    users = torch.arange(model.num_users)
    model.T_topk_scores = model.teacher.forward_multi_items(
        users, model.T_topk_dict,
    )
    return model


class AnchorRCEKDTests(unittest.TestCase):
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

    def test_count_and_marginal_keep_the_rcekd_loss_and_finite_gradients(self):
        for sampler in ("count", "marginal"):
            with self.subTest(sampler=sampler):
                model = make_model(sampler)
                torch.manual_seed(47)
                model.do_something_in_each_epoch(0)
                users = torch.tensor([0, 1])
                loss = model.get_loss(users)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(torch.isfinite(model.student.score_parameters.grad).all())

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


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from analyze_per_user_k import (
    balanced_overlap_groups,
    paired_bootstrap_delta,
    per_user_recall_ndcg,
    row_overlap_ratio,
    select_and_route,
)


class PerUserKAnalysisTests(unittest.TestCase):
    def test_overlap_preserves_row_set_semantics(self):
        left = torch.tensor([[1, 2, 3], [4, 5, 6]])
        right = torch.tensor([[3, 9, 1], [8, 7, 6]])
        actual = row_overlap_ratio(left, right, chunk_size=1)
        torch.testing.assert_close(actual, torch.tensor([2 / 3, 1 / 3], dtype=torch.float64))

    def test_metrics_include_zero_target_users_like_repository_evaluator(self):
        topk = torch.tensor([[1, 2, 3], [3, 2, 1], [0, 4, 2]])
        truth = {0: torch.tensor([1, 3]), 1: torch.tensor([], dtype=torch.long)}
        metrics = per_user_recall_ndcg(topk, truth, num_users=3, k=3)
        torch.testing.assert_close(metrics["Recall"], torch.tensor([1., 0., 0.], dtype=torch.float64))
        expected = (1 + 1 / torch.log2(torch.tensor(4., dtype=torch.float64))) / (
            1 + 1 / torch.log2(torch.tensor(3., dtype=torch.float64))
        )
        self.assertAlmostEqual(metrics["NDCG"][0].item(), expected.item())
        self.assertEqual(metrics["NDCG"][1:].sum().item(), 0.)

    def test_groups_are_balanced_complete_and_sorted(self):
        overlap = torch.tensor([.8, .1, .9, .2, .7, .3, .6])
        groups = balanced_overlap_groups(overlap, 3)
        users = torch.cat([indices for _, indices in groups])
        self.assertEqual(sorted(users.tolist()), list(range(7)))
        self.assertEqual([len(indices) for _, indices in groups], [3, 2, 2])
        self.assertLessEqual(overlap[groups[0][1]].max(), overlap[groups[1][1]].min())
        self.assertLessEqual(overlap[groups[1][1]].max(), overlap[groups[2][1]].min())

    def test_validation_selects_groups_and_test_does_not_change_selection(self):
        def split(recall, ndcg):
            return {"Recall": torch.tensor(recall, dtype=torch.float64),
                    "NDCG": torch.tensor(ndcg, dtype=torch.float64)}

        metrics = {
            20: {"valid": split([1, 1, 0, 0], [.8, .7, .1, .2]),
                 "test": split([0, 0, 1, 1], [.0, .0, .9, .9])},
            50: {"valid": split([0, 0, 1, 1], [.1, .2, .7, .8]),
                 "test": split([1, 1, 0, 0], [.8, .8, .0, .0])},
        }
        groups = [("low", torch.tensor([0, 1])), ("high", torch.tensor([2, 3]))]
        reports, selected, routing = select_and_route(metrics, groups, [20, 50], 100)
        self.assertEqual([report["selected_K_from_validation"] for report in reports], [20, 50])
        self.assertEqual(selected.tolist(), [20, 20, 50, 50])
        # Test would prefer the exact opposite mapping, proving it did not select K.
        self.assertEqual(routing["group_routed"]["test"]["NDCG@20"], 0.)

    def test_paired_bootstrap_reports_exact_constant_delta(self):
        first = torch.tensor([.4, .6, .8], dtype=torch.float64)
        second = first - .2
        result = paired_bootstrap_delta(first, second, torch.arange(3), samples=200, seed=9)
        self.assertAlmostEqual(result["mean"], .2)
        self.assertAlmostEqual(result["ci95"][0], .2)
        self.assertAlmostEqual(result["ci95"][1], .2)


if __name__ == "__main__":
    unittest.main()

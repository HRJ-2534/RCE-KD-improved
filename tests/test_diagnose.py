import io
import json
import math
import unittest

import torch

from diagnose import recall_ndcg_at_k


class RecallNdcgSerializationTests(unittest.TestCase):
    def assert_json_serializable(self, metrics):
        for value in metrics.values():
            self.assertIsInstance(value, float)
        report = {"bias_and_accuracy": {"overall": {"student": metrics}}}
        self.assertEqual(json.loads(json.dumps(report)), report)
        output = io.StringIO()
        json.dump(report, output)
        self.assertEqual(json.loads(output.getvalue()), report)

    def test_known_ranking_preserves_scores_and_serializes(self):
        metrics = recall_ndcg_at_k(
            torch.tensor([[0, 1, 2], [2, 1, 0]]),
            {0: torch.tensor([0, 2]), 1: torch.tensor([], dtype=torch.long)},
            num_users=2,
            ks=(1, 3),
        )
        self.assertEqual(metrics["Recall@1"], 0.5)
        self.assertEqual(metrics["Recall@3"], 1.0)
        self.assertEqual(metrics["NDCG@1"], 1.0)
        expected_ndcg = 1.5 / (1.0 + 1.0 / math.log2(3))
        self.assertAlmostEqual(metrics["NDCG@3"], expected_ndcg, places=6)
        self.assert_json_serializable(metrics)

    def test_empty_evaluation_serializes_zero_scores(self):
        cases = (
            (torch.tensor([[0, 1, 2]]), {}, 1),
            (torch.tensor([[0, 1, 2]]), {0: torch.tensor([], dtype=torch.long)}, 1),
            (torch.empty((0, 3), dtype=torch.long), {}, 0),
        )
        for recommendations, truth, num_users in cases:
            with self.subTest(num_users=num_users, truth=truth):
                metrics = recall_ndcg_at_k(recommendations, truth, num_users, ks=(1, 3))
                self.assertEqual(
                    metrics,
                    {"Recall@1": 0.0, "NDCG@1": 0.0, "Recall@3": 0.0, "NDCG@3": 0.0},
                )
                self.assert_json_serializable(metrics)


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from analyze_teacher_scope import cutoff_histogram, teacher_scope_features


class TeacherScopeFeatureTests(unittest.TestCase):
    def test_front_cliff_has_early_max_gap_and_larger_head_drop(self):
        flat = torch.linspace(10., 0., 101)
        cliff = torch.cat([
            torch.tensor([10., 9.9, 9.8, 9.7]),
            torch.linspace(3., 0., 97),
        ])
        features = teacher_scope_features(torch.stack([flat, cliff]))
        self.assertEqual(features["max_gap_rank"][1].item(), 4)
        # Floating-point ties in the linear fixture may choose any gap rank;
        # its head-drop ratio, unlike the cliff fixture, is about 19/100.
        self.assertAlmostEqual(features["head_drop_20"][0].item(), .19, places=6)
        self.assertGreater(features["head_drop_20"][1], features["head_drop_20"][0])
        self.assertGreater(features["gap_prominence"][1], features["gap_prominence"][0])

    def test_features_are_invariant_to_positive_affine_score_transform(self):
        generator = torch.Generator().manual_seed(37)
        values = torch.rand((7, 101), generator=generator, dtype=torch.float64).sort(dim=1, descending=True).values
        original = teacher_scope_features(values)
        transformed = teacher_scope_features(values * 7.3 - 19.)
        for name in (
            "head_drop_20", "max_normalized_gap", "gap_prominence",
            "normalized_gap_at_20", "normalized_gap_at_50",
            "standardized_effective_support",
        ):
            torch.testing.assert_close(original[name], transformed[name], rtol=1e-10, atol=1e-12)
        self.assertTrue(torch.equal(original["max_gap_rank"], transformed["max_gap_rank"]))

    def test_cutoff_histogram_is_complete_and_uses_declared_boundaries(self):
        histogram = cutoff_histogram(torch.tensor([1, 20, 21, 50, 51, 100]))
        self.assertEqual([histogram[name]["count"] for name in ("1-20", "21-50", "51-100")], [2, 2, 2])
        self.assertAlmostEqual(sum(histogram[name]["fraction"] for name in histogram), 1.)


if __name__ == "__main__":
    unittest.main()

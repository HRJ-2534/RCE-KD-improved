import unittest

import torch

from soft_closure import (
    exact_partial_log_ndcg,
    exact_soft_closure_rho,
    prepare_soft_closure_catalog,
    soft_closure_bound_terms,
    topm_soft_closure_rho,
)


class SoftClosureTheoryTests(unittest.TestCase):
    def test_exact_rho_is_zero_for_a_closed_top_prefix(self):
        scores = torch.tensor([[4., 3., 2., 1.]], dtype=torch.float64)
        items = torch.tensor([[0, 1]])
        rho = exact_soft_closure_rho(scores, items)
        torch.testing.assert_close(rho, torch.zeros_like(rho))

    def test_nonclosed_set_has_exactly_the_expected_missing_mass(self):
        scores = torch.tensor([[4., 3., 2., 1.]], dtype=torch.float64)
        items = torch.tensor([[1, 2]])
        rho = exact_soft_closure_rho(scores, items)
        denominator = torch.exp(torch.tensor(3., dtype=torch.float64)) + torch.exp(
            torch.tensor(2., dtype=torch.float64)
        )
        expected = torch.exp(torch.tensor(4., dtype=torch.float64)) / denominator
        torch.testing.assert_close(rho, torch.tensor([[expected, expected]]))

    def test_topm_approximation_is_lower_than_exact_and_converges(self):
        scores = torch.tensor([[5., 4., 3., 2., 1.]], dtype=torch.float64)
        items = torch.tensor([[3, 4]])
        exact = exact_soft_closure_rho(scores, items)
        top2 = topm_soft_closure_rho(scores, items, torch.tensor([[0, 1]]))
        top5 = topm_soft_closure_rho(scores, items, torch.tensor([[0, 1, 2, 3, 4]]))
        self.assertTrue(torch.all(top2 <= exact))
        torch.testing.assert_close(top5, exact)

    def test_soft_closure_bound_holds_for_nonclosed_sets(self):
        generator = torch.Generator().manual_seed(91)
        student = torch.randn((8, 17), generator=generator, dtype=torch.float64)
        teacher = torch.randn((8, 17), generator=generator, dtype=torch.float64)
        items = torch.stack([
            torch.randperm(17, generator=generator)[:6] for _ in range(8)
        ])
        terms = soft_closure_bound_terms(student, teacher, items)
        log_ndcg = exact_partial_log_ndcg(student, teacher, items)
        self.assertTrue(torch.all(log_ndcg + 1e-10 >= terms["lower_bound"]))

    def test_padding_does_not_change_active_quantities(self):
        student = torch.tensor([[4., 1., 3., 2.]], dtype=torch.float64)
        teacher = torch.tensor([[1., 3., 4., 2.]], dtype=torch.float64)
        short_items = torch.tensor([[1, 2]])
        padded_items = torch.tensor([[1, 2, 0]])
        active = torch.tensor([[True, True, False]])
        short = soft_closure_bound_terms(student, teacher, short_items)
        padded = soft_closure_bound_terms(student, teacher, padded_items, active)
        for key in ("ce", "penalty", "log_c_j", "lower_bound"):
            torch.testing.assert_close(short[key], padded[key])

    def test_reused_catalog_cache_is_exact(self):
        generator = torch.Generator().manual_seed(17)
        scores = torch.randn((4, 13), generator=generator, dtype=torch.float64)
        items = torch.stack([
            torch.randperm(13, generator=generator)[:5] for _ in range(4)
        ])
        direct = exact_soft_closure_rho(scores, items)
        cached = exact_soft_closure_rho(
            scores, items, catalog_cache=prepare_soft_closure_catalog(scores),
        )
        torch.testing.assert_close(cached, direct)


if __name__ == "__main__":
    unittest.main()

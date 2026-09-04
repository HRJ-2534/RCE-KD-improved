"""Equivalence checks for transfer/indexing optimizations, not model changes.

Run on the training server as well: CUDA-specific checks skip on CPU hosts.
"""
import contextlib
import copy
import io
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation import Evaluator
from modeling.KD.playground import RCEKD, SRCEKD


class TinyBackbone(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Deterministic, non-random fixtures do not consume either RNG stream.
        self.users = nn.Parameter(torch.sin(torch.arange(20).reshape(4, 5).float()).to(device))
        self.items = nn.Parameter(torch.cos(torch.arange(60).reshape(12, 5).float()).to(device))

    def get_all_ratings(self):
        return self.users @ self.items.T

    def forward_multi_items(self, users, items):
        return (self.users[users, None, :] * self.items[items]).sum(-1)


def make_kd(cls, device, sample_rank=False, mode="union", alpha=1., lu=3):
    # Isolate epoch preparation from the constructors' dataset/CUDA setup.
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.student = TinyBackbone(device)
    model.teacher = TinyBackbone(device)
    with torch.no_grad():
        model.teacher.items.mul_(-0.7)
    model.teacher.requires_grad_(False)
    model.num_users, model.num_items = 4, 12
    model.K, model.mxK, model.L = 3, 6, 3
    model.T, model.tau, model.beta = 10., 1., 5.
    model.sample_rank, model.mode = sample_rank, mode
    model.alpha, model.s, model.eta, model.Lu = alpha, 20., 1., lu
    model.kappa = 0.
    model.observed_mat = torch.zeros((4, 12), dtype=torch.bool, device=device)
    model.observed_mat[:, 0] = True
    model.ranking_mat = torch.exp(-(torch.arange(model.mxK) + 1) / model.T).repeat(4, 1)
    with contextlib.redirect_stdout(io.StringIO()):
        model.T_topk_dict = model.get_topk_dict(model.teacher, model.K)
    return model


def legacy_epoch(model):
    """Original sampling, CPU per-user collection and device round trips."""
    device = model.T_topk_dict.device
    with torch.no_grad():
        topk = model.student.get_all_ratings().topk(model.mxK, dim=-1).indices.cpu().to(device)
        is_srce = isinstance(model, SRCEKD)
        if is_srce and model.mode == "union" and model.alpha != 0:
            matches = model.T_topk_dict.unsqueeze(2) == topk.unsqueeze(1)
            positions = matches.float().argmax(2)
            model.rankS_of_T = torch.where(matches.any(2), positions,
                                           torch.full_like(positions, model.mxK))
        if not is_srce and model.sample_rank:
            samples = torch.multinomial(model.ranking_mat, model.L, replacement=False)
        else:
            weights = torch.zeros((model.num_users, model.mxK)).to(device)
            ranks = model.rowwise_index(model.T_topk_dict, topk)
            weights[ranks[:, 0], ranks[:, 1]] += 1
            weights = torch.minimum(weights.flip(-1).cumsum(-1).flip(-1), torch.tensor(50.))
            weights = torch.exp((weights + 1) / model.T)
            samples = torch.multinomial(weights, model.L, replacement=False)
        selected = torch.zeros((model.num_users, model.L)).long()
        for user in range(model.num_users):
            selected[user] = topk[user][samples[user]]
        model.interesting_items = selected.to(device)
        if is_srce:
            if model.Lu > 0:
                model.uniform_items = torch.randint(0, model.num_items,
                                                    (model.num_users, model.Lu)).to(device)
            else:
                model.uniform_items = torch.zeros((model.num_users, 0)).long().to(device)
        model.itemS = topk[:, :model.K]


class LegacyMaskEvaluator(Evaluator):
    @staticmethod
    def _mask_training_items(scores, users, train_dict):
        for row, user in enumerate(users):
            scores[row, train_dict[user.item()]] = -1e10


class PerformanceEquivalenceTests(unittest.TestCase):
    def check_sampling(self, device):
        cases = [(RCEKD, {"sample_rank": flag}) for flag in (False, True)]
        cases += [(SRCEKD, {"mode": mode, "alpha": alpha, "lu": lu})
                  for mode in ("union", "split") for alpha in (0., 1.) for lu in (0, 3)]
        for cls, options in cases:
            with self.subTest(device=device, model=cls.__name__, **options):
                reference = make_kd(cls, device, **options)
                optimized = copy.deepcopy(reference)
                torch.manual_seed(719)
                legacy_epoch(reference)
                cpu_state = torch.get_rng_state().clone()
                cuda_state = torch.cuda.get_rng_state().clone() if device == "cuda" else None
                torch.manual_seed(719)
                with contextlib.redirect_stdout(io.StringIO()):
                    optimized.do_something_in_each_epoch(0)
                self.assertTrue(torch.equal(cpu_state, torch.get_rng_state()))
                if cuda_state is not None:
                    self.assertTrue(torch.equal(cuda_state, torch.cuda.get_rng_state()))
                for attr in ("itemS", "interesting_items", "uniform_items", "rankS_of_T"):
                    if hasattr(reference, attr):
                        self.assertTrue(torch.equal(getattr(reference, attr), getattr(optimized, attr)), attr)
                        self.assertEqual(getattr(optimized, attr).device.type, device)
                users = torch.tensor([3, 0, 3, 1], device=device)
                loss_ref, loss_new = reference.get_loss(users), optimized.get_loss(users)
                torch.testing.assert_close(loss_new, loss_ref, rtol=0, atol=0)
                loss_ref.backward()
                loss_new.backward()
                for old, new in zip(reference.student.parameters(), optimized.student.parameters()):
                    torch.testing.assert_close(new.grad, old.grad, rtol=0, atol=0)
                # Also preserve the following optimizer update exactly.
                torch.optim.Adam(reference.student.parameters(), lr=.001).step()
                torch.optim.Adam(optimized.student.parameters(), lr=.001).step()
                for old, new in zip(reference.student.parameters(), optimized.student.parameters()):
                    torch.testing.assert_close(new, old, rtol=0, atol=0)

    def test_sampling_loss_gradients_and_rng_cpu(self):
        self.check_sampling("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_sampling_loss_gradients_and_rng_cuda(self):
        self.check_sampling("cuda")

    def check_mask(self, device):
        train = {0: torch.tensor([1, 1, 7]), 3: torch.tensor([], dtype=torch.long),
                 8: torch.tensor([0, 2, 5])}
        for user_ids in ([8, 0, 3], [3], [], [0, 0]):
            with self.subTest(device=device, users=user_ids):
                users = torch.tensor(user_ids, dtype=torch.long)
                scores = torch.arange(len(users) * 9, dtype=torch.float).reshape(len(users), 9).to(device)
                expected = scores.clone()
                LegacyMaskEvaluator._mask_training_items(expected, users, train)
                Evaluator._mask_training_items(scores, users, train)
                torch.testing.assert_close(scores, expected, rtol=0, atol=0)
                torch.testing.assert_close(scores.topk(5).indices, expected.topk(5).indices, rtol=0, atol=0)

    def test_mask_empty_duplicate_and_nonordered_users_cpu(self):
        self.check_mask("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_mask_empty_duplicate_and_nonordered_users_cuda(self):
        self.check_mask("cuda")

    def check_evaluation(self, device):
        args = SimpleNamespace(early_stop_patience=2, early_stop_metric="NDCG", task="rec",
                               K_list=[2, 5], early_stop_K=5)
        # More than 1024 users exercises multiple evaluation batches, with
        # non-contiguous/reversed IDs, empty positives, duplicates and ties.
        n = 1100
        train = {u: torch.tensor([u % 17, u % 17]) if u % 3 else torch.empty(0, dtype=torch.long)
                 for u in range(n)}
        valid = SimpleNamespace(inter_dict={u: torch.tensor([(u + 2) % 17]) for u in range(n)})
        test = SimpleNamespace(inter_dict={u: torch.tensor([(u + 3) % 17]) for u in reversed(range(n)) if u % 31})
        loader = SimpleNamespace(dataset=SimpleNamespace(train_dict=train, num_users=n, num_items=17), batch_size=13)
        scores = ((torch.arange(n)[:, None] + torch.arange(17)[None, :]) % 7).float().to(device)
        model = SimpleNamespace(eval=lambda: None, get_ratings=lambda users: scores[users.to(device)].clone())
        reference, optimized = LegacyMaskEvaluator(args), Evaluator(args)
        for epoch in range(3):
            old = reference.evaluate_while_training(model, epoch, loader, valid, test)
            new = optimized.evaluate_while_training(model, epoch, loader, valid, test)
            self.assertEqual(old[:3], new[:3])  # exclude wall-clock time
            self.assertEqual(reference.eval_dict, optimized.eval_dict)

    def test_metrics_and_early_stopping_cpu(self):
        self.check_evaluation("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_metrics_and_early_stopping_cuda(self):
        self.check_evaluation("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/pinned memory")
    def test_pinned_async_batches_preserve_order_values_and_rng(self):
        dataset = TensorDataset(torch.arange(43), torch.arange(86).reshape(43, 2))
        outputs, states = [], []
        for pinned in (False, True):
            torch.manual_seed(119)
            loader = DataLoader(dataset, batch_size=7, shuffle=True, pin_memory=pinned)
            batches = []
            for batch in loader:
                if pinned:
                    self.assertTrue(all(value.is_pinned() for value in batch))
                batches.append([value.cuda(non_blocking=pinned) for value in batch])
            torch.cuda.synchronize()
            outputs.append(batches)
            states.append(torch.get_rng_state())
        self.assertTrue(torch.equal(*states))
        for old, new in zip(*outputs):
            for a, b in zip(old, new):
                self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()

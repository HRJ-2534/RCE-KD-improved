"""Diagnostics for RCE-KD-style distillation (pre-research stage 1).

Quantifies, without any training, the three limitations that motivate SRCE-KD:

  S1  Teacher reliability: how many of the teacher's top-K items are
      confirmed by the user's held-out (valid/test) interactions,
      stratified by item popularity.
  S3  Blindness of the closure approximation: where the teacher's top-K
      items sit in the student's ranking, and how many of them fall beyond
      the student's top-mxK (the only region RCE-KD samples from).
  S4  Popularity-bias inheritance: average popularity (ARP@N) and
      long-tail Recall@N of the teacher vs. the distilled student.

It also reports the teacher-student top-K overlap ratio (the quantity that
drives RCE-KD's adaptive gamma).

Usage example (run on the server, after distillation checkpoints exist):

    python -u diagnose.py --dataset=citeulike --T_backbone=bpr --S_backbone=bpr \
        --model=rcekd --gpu_id=0

Intermediate student checkpoints (EPOCH_*.pt / EPOCH_*_SCORE_MAT.pt, saved
when --ckpt_interval is used) are picked up automatically to show the
dynamics of overlap / blind fraction over training.
"""

import os
import re
import glob
import json
import math
import argparse

import torch

from dataset import load_cf_data, implicit_CF_dataset, implicit_SR_dataset
import modeling.backbone as backbone
from utils import load_yaml
from utils.parse_utils import parse_cfg


def build_score_mat(ckpt_path, backbone_name, cfg, dataset, device):
    """Load a backbone checkpoint and return its full score matrix."""
    if ckpt_path.endswith("SCORE_MAT.pt"):
        return torch.load(ckpt_path, weights_only=True).to(device)
    all_backbones = [e.lower() for e in dir(backbone)]
    cls = getattr(backbone, dir(backbone)[all_backbones.index(backbone_name.lower())])
    model = cls(dataset, cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        return model.get_all_ratings().detach()


def topk_excluding_train(score_mat, train_matrix, k):
    """top-k per user with training items masked out (standard evaluation)."""
    masked = score_mat.clone()
    masked[train_matrix.to_dense().bool().to(score_mat.device)] = -float("inf")
    return torch.topk(masked, k, dim=-1).indices


def recall_ndcg_at_k(topk_items, test_dict, num_users, ks=(10, 20)):
    results = {}
    for k in ks:
        recs = topk_items[:, :k].cpu()
        recalls, ndcgs = [], []
        discounts = 1. / torch.log2(torch.arange(2, k + 2).float())
        for u in range(num_users):
            truth = test_dict.get(u, None)
            if truth is None or len(truth) == 0:
                continue
            truth_set = set(truth.tolist())
            hits = [1.0 if i in truth_set else 0.0 for i in recs[u].tolist()]
            recalls.append(sum(hits) / len(truth_set))
            dcg = sum(h * d for h, d in zip(hits, discounts))
            ideal = min(len(truth_set), k)
            idcg = discounts[:ideal].sum().item()
            ndcgs.append(dcg / idcg)
        results[f"Recall@{k}"] = sum(recalls) / max(len(recalls), 1)
        results[f"NDCG@{k}"] = sum(ndcgs) / max(len(ndcgs), 1)
    return results


def exact_ranks_of_items(score_mat, items, chunk=128):
    """Exact 0-based rank of items[u, k] within row u of score_mat."""
    num_users, K = items.shape
    ranks = torch.empty((num_users, K), dtype=torch.long)
    for s in range(0, num_users, chunk):
        e = min(num_users, s + chunk)
        sc = score_mat[s:e]                                   # b x I
        t = sc.gather(1, items[s:e].to(sc.device))            # b x K
        r = (sc.unsqueeze(1) > t.unsqueeze(2)).sum(dim=-1)    # b x K
        ranks[s:e] = r.cpu()
    return ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--T_backbone", type=str, default="bpr")
    parser.add_argument("--S_backbone", type=str, default="bpr")
    parser.add_argument("--model", type=str, default="rcekd",
                        help="KD model whose student checkpoints are diagnosed")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--eval_K", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # ---------------- data & configs ----------------
    (num_users, num_items, train_pairs, valid_pairs, test_pairs,
     train_dict, valid_dict, test_dict, train_matrix, user_pop, item_pop) = load_cf_data(args.dataset)

    kd_cfg = load_yaml(os.path.join("configs", args.dataset, args.S_backbone, f"{args.model.lower()}.yaml"))
    kd_cfg = kd_cfg[args.T_backbone.lower()]
    K, mxK = int(kd_cfg["mkd_K"]), int(kd_cfg["mkd_mxK"])

    backbone_cfg = load_yaml(os.path.join("configs", args.dataset, args.S_backbone, "base_config.yaml"))
    defaults = {"dataset": args.dataset, "DATA_DIR": "data/"}
    t_cfg = parse_cfg(argparse.Namespace(**defaults), backbone_cfg["teacher"][args.T_backbone.lower()])
    s_cfg = parse_cfg(argparse.Namespace(**defaults), backbone_cfg["student"])

    trainset = implicit_CF_dataset(args.dataset, num_users, num_items, train_pairs,
                                   train_matrix, train_dict, user_pop, item_pop,
                                   num_ns=1, no_neg_sampling=True)

    def wrap_dataset(backbone_name, cfg):
        ds = trainset
        if backbone_name.lower() == "hstu":
            ds = implicit_SR_dataset(trainset, cfg.max_sequence_len)
        return ds

    # ---------------- teacher ----------------
    t_dir = os.path.join("checkpoints", args.dataset, args.T_backbone, f"scratch-{t_cfg.embedding_dim}")
    t_score_path = os.path.join(t_dir, "BEST_SCORE_MAT.pt")
    t_ckpt_path = t_score_path if os.path.exists(t_score_path) else os.path.join(t_dir, "BEST_EPOCH.pt")
    print(f"[teacher] loading {t_ckpt_path}")
    T_scores = build_score_mat(t_ckpt_path, args.T_backbone, t_cfg, wrap_dataset(args.T_backbone, t_cfg), device)
    T_topK = torch.topk(T_scores, K, dim=-1).indices.cpu()   # raw top-K, as used by KD

    # ---------------- student checkpoints (dynamics) ----------------
    s_dir = os.path.join("checkpoints", args.dataset, args.S_backbone,
                         f"{args.model.lower()}-{s_cfg.embedding_dim}" + ("_" + args.suffix if args.suffix else ""))
    s_paths = sorted(glob.glob(os.path.join(s_dir, "EPOCH_*_SCORE_MAT.pt")) +
                     glob.glob(os.path.join(s_dir, "BEST_*SCORE_MAT.pt")))
    if not s_paths:
        s_paths = sorted(glob.glob(os.path.join(s_dir, "EPOCH_*.pt")) +
                         [os.path.join(s_dir, "BEST_EPOCH.pt")])
    s_paths = [p for p in s_paths if os.path.exists(p)]
    assert len(s_paths) > 0, f"no student checkpoints found under {s_dir}"

    def epoch_key(p):
        m = re.search(r"EPOCH_(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 10 ** 9
    s_paths = sorted(set(s_paths), key=epoch_key)

    report = {"dataset": args.dataset, "teacher": args.T_backbone, "student": args.S_backbone,
              "model": args.model, "K": K, "mxK": mxK, "dynamics": []}

    print(f"[student] diagnosing {len(s_paths)} checkpoint(s) from {s_dir}")
    for p in s_paths:
        S_scores = build_score_mat(p, args.S_backbone, s_cfg, wrap_dataset(args.S_backbone, s_cfg), device)
        S_topK = torch.topk(S_scores, K, dim=-1).indices.cpu()
        # overlap ratio (drives RCE-KD's gamma)
        overlap = (S_topK.unsqueeze(2) == T_topK.unsqueeze(1)).any(dim=2).float().mean().item()
        # exact student ranks of teacher top-K items
        ranks = exact_ranks_of_items(S_scores.cpu(), T_topK)
        entry = {
            "checkpoint": os.path.basename(p),
            "epoch": epoch_key(p),
            "overlap_ratio": overlap,
            "teacher_item_rank_in_student": {
                "mean": ranks.float().mean().item(),
                "median": ranks.float().median().item(),
                "frac_beyond_mxK": (ranks >= mxK).float().mean().item(),   # blind region of RCE-KD sampling
                "frac_beyond_1000": (ranks >= 1000).float().mean().item(),
            },
        }
        report["dynamics"].append(entry)
        print(f"  {os.path.basename(p):>32s}  overlap={overlap:.4f}  "
              f"mean_rank={entry['teacher_item_rank_in_student']['mean']:.1f}  "
              f"beyond_mxK={entry['teacher_item_rank_in_student']['frac_beyond_mxK']:.4f}")
        if p == s_paths[-1]:
            S_scores_final, S_topK_final = S_scores, S_topK
        del S_scores
        torch.cuda.empty_cache()

    # ---------------- S1: teacher reliability ----------------
    heldout_dict = {u: torch.cat([valid_dict.get(u, torch.LongTensor([])),
                                  test_dict.get(u, torch.LongTensor([]))])
                    for u in range(num_users)}
    pop = item_pop.float()
    quartiles = torch.quantile(pop, torch.tensor([0.25, 0.5, 0.75]))
    pop_bucket = torch.bucketize(pop, quartiles)              # 0..3, 3 = most popular

    hit_mask = torch.zeros((num_users, K), dtype=torch.bool)
    for u in range(num_users):
        ho = heldout_dict[u]
        if len(ho) > 0:
            hit_mask[u] = torch.isin(T_topK[u], ho)
    rel = {"overall_hit_rate": hit_mask.float().mean().item()}
    for b in range(4):
        in_bucket = pop_bucket[T_topK] == b
        rel[f"hit_rate_popularity_quartile_{b + 1}"] = hit_mask[in_bucket].float().mean().item()
    report["teacher_reliability"] = rel
    print("[S1 teacher reliability]", json.dumps(rel, indent=2))

    # ---------------- S4: popularity-bias inheritance ----------------
    T_topN_eval = topk_excluding_train(T_scores, train_matrix, args.eval_K).cpu()
    S_topN_eval = topk_excluding_train(S_scores_final, train_matrix, args.eval_K).cpu()

    def arp(topk_items):
        return pop[topk_items].mean().item()

    tail_items = (pop_bucket <= 1)                            # bottom-50% popularity = long tail
    tail_test_dict = {u: t[torch.isin(t, torch.nonzero(tail_items).squeeze(-1))]
                      for u, t in test_dict.items()}
    tail_test_dict = {u: t for u, t in tail_test_dict.items() if len(t) > 0}

    bias = {
        f"ARP@{args.eval_K}": {"teacher": arp(T_topN_eval), "student": arp(S_topN_eval)},
        f"longtail_Recall@{args.eval_K}": {
            "teacher": recall_ndcg_at_k(T_topN_eval.to(device), tail_test_dict, num_users, (args.eval_K,))[f"Recall@{args.eval_K}"],
            "student": recall_ndcg_at_k(S_topN_eval.to(device), tail_test_dict, num_users, (args.eval_K,))[f"Recall@{args.eval_K}"],
        },
        "overall": {
            "teacher": recall_ndcg_at_k(T_topN_eval.to(device), test_dict, num_users),
            "student": recall_ndcg_at_k(S_topN_eval.to(device), test_dict, num_users),
        },
    }
    report["bias_and_accuracy"] = bias
    print("[S4 bias & accuracy]", json.dumps(bias, indent=2))

    out_path = os.path.join("logs", f"diagnose_{args.dataset}_{args.T_backbone}-{args.S_backbone}_{args.model.lower()}.json")
    os.makedirs("logs", exist_ok=True)
    json.dump(report, open(out_path, "w"), indent=2)
    print(f"report saved to {out_path}")


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseKD4Rec


class RCEKD(BaseKD4Rec):
    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "rcekd"
        self.tau = args.mkd_tau
        self.K = args.mkd_K
        self.beta = args.mkd_beta
        self.T = args.mkd_T
        self.L = args.mkd_L
        self.mxK = args.mkd_mxK
        self.sample_rank = args.sample_rank
        self.T_topk_dict = self.get_topk_dict(self.teacher, self.K)
        if self.sample_rank:
            ranking_list = torch.exp(-(torch.arange(self.mxK) + 1) / self.T)
            self.ranking_mat = ranking_list.repeat(self.num_users, 1)

    def get_topk_dict(self, model, mxK):
        print('Generating Top-K dict...')
        with torch.no_grad():
            inter_mat = model.get_all_ratings()
            _, topk_dict = torch.topk(inter_mat, mxK, dim=-1)
        return topk_dict.type(torch.LongTensor).cuda()

    # https://discuss.pytorch.org/t/find-indexes-of-elements-from-one-tensor-that-matches-in-another-tensor/147482/3
    def rowwise_index(self, source, target):
        idx = (target.unsqueeze(1) == source.unsqueeze(2)).nonzero()
        idx = idx[:, [0, 2]]
        return idx

    def do_something_in_each_epoch(self, epoch):
        with torch.no_grad():
            S_topk_dict = self.get_topk_dict(self.student, self.mxK)
            self.interesting_items = torch.zeros((self.num_users, self.L)).long()
            if self.sample_rank:
                samples = torch.multinomial(self.ranking_mat, self.L, replacement=False)
            else:
                weight_matrix = torch.zeros((self.num_users, self.mxK)).cuda()
                itemT_rankS = self.rowwise_index(self.T_topk_dict, S_topk_dict)
                weight_matrix[itemT_rankS[:, 0], itemT_rankS[:, 1]] += 1
                weight_matrix = torch.minimum(torch.cumsum(weight_matrix.flip(-1), dim=-1).flip(-1), torch.tensor(50.))
                weight_matrix = torch.exp((weight_matrix + 1) / self.T)
                samples = torch.multinomial(weight_matrix, self.L, replacement=False)
            for user in range(self.num_users):
                self.interesting_items[user] = S_topk_dict[user][samples[user]]
            self.interesting_items = self.interesting_items.cuda()
            self.itemS = S_topk_dict[:, :self.K]

    # https://stackoverflow.com/questions/74946537/can-i-apply-torch-isin-to-each-row-in-2d-tensor-without-loop
    def rowwise_isin(self, tensor_1, target_tensor):
        matches = (tensor_1.unsqueeze(2) == target_tensor.unsqueeze(1))
        result = torch.sum(matches, dim=2, dtype=torch.bool)
        return result

    def get_loss(self, *params):
        batch_users = params[0]
        itemS = self.itemS[batch_users]
        itemT = self.T_topk_dict[batch_users]
        item_interesting = self.interesting_items[batch_users]
        logit_S_itemS = self.student.forward_multi_items(batch_users, itemS) / self.tau
        logit_S_itemT = self.student.forward_multi_items(batch_users, itemT) / self.tau
        logit_S_interesting = self.student.forward_multi_items(batch_users, item_interesting) / self.tau
        logit_T_itemS = self.teacher.forward_multi_items(batch_users, itemS) / self.tau
        logit_T_itemT = self.teacher.forward_multi_items(batch_users, itemT) / self.tau

        exp_logit_T_itemS = torch.exp(logit_T_itemS)
        Z_T = exp_logit_T_itemS.sum(-1, keepdim=True)
        prob_T_itemS = exp_logit_T_itemS / Z_T
        loss_itemS = F.cross_entropy(logit_S_itemS, prob_T_itemS, reduction='none')

        logit_T_interesting = self.teacher.forward_multi_items(batch_users, item_interesting) / self.tau
        exp_logit_T_interesting = torch.exp(logit_T_interesting)
        exp_logit_T_itemT = torch.exp(logit_T_itemT)
        mask = self.rowwise_isin(itemT, item_interesting)
        exp_logit_T_itemT[mask] = 0
        mask2 = self.rowwise_isin(itemT, itemS)
        exp_logit_T_itemT[mask2] = 0
        Z_T = exp_logit_T_interesting.sum(-1, keepdim=True) + exp_logit_T_itemT.sum(-1, keepdim=True)
        prob_T_all = torch.cat([exp_logit_T_interesting, exp_logit_T_itemT], dim=-1) / Z_T
        exp_logit_S_itemT = torch.exp(logit_S_itemT)
        exp_logit_S_itemT = exp_logit_S_itemT * (1. - mask.float()) * (1. - mask2.float())
        exp_logit_S_interesting = torch.exp(logit_S_interesting)
        Z_S = exp_logit_S_interesting.sum(-1, keepdim=True) + exp_logit_S_itemT.sum(-1, keepdim=True)
        logit_S_all = torch.cat([logit_S_interesting, logit_S_itemT], dim=-1)
        loss_itemT = -(prob_T_all * (logit_S_all - torch.log(Z_S))).sum(-1)

        overlap = mask.float().mean(-1)
        weight = torch.exp(-self.beta * overlap)
        loss = ((1 - weight) * loss_itemS + weight * loss_itemT).sum()
        return loss


class SRCEKD(BaseKD4Rec):
    """Soft-closure & Reliability-aware RCE-KD (SRCE-KD).

    Improvements over RCE-KD (see playground.RCEKD):

    1) Soft closure weighting. RCE-KD splits the teacher's top-K into two
       subsets by a hard threshold (whether the item is in the student's
       top-K) and fuses two CE losses with an adaptive scalar gamma. We
       replace this with a single CE computed on the union set
       (student top-K ∪ teacher top-K ∪ closure samples ∪ uniform samples),
       where every teacher item carries a continuous weight
           w_i = 1 + alpha * sigmoid((rank_S(i) - K) / s)
       that grows smoothly with the student's rank of item i. The hard
       split-and-fusion of RCE-KD is recovered in the limit s -> 0.
       rank_S(i) is approximated by the position inside the student's
       top-mxK, and set to mxK when the item falls outside.

    2) Tail coverage. RCE-KD only samples inside the student's top-mxK, so
       "blocking" items beyond mxK are never seen (worst in early training).
       We add a small number of uniformly sampled items per user.

    3) Teacher reliability correction. RCE-KD treats the teacher's
       predictions as ground truth. We reweight each teacher item by
           rel_i = 1 + eta * 1[i in observed(u)]
       so that teacher knowledge confirmed by the user's real (training)
       interactions is emphasized and unconfirmed knowledge is relatively
       downweighted. Only training interactions are used (no leakage).

    New hyperparameters (with defaults in parse.py):
        srce_alpha : max extra weight of badly-ranked teacher items (>=0)
        srce_s     : softness of the rank margin, in rank units (>0);
                     s -> 0 recovers the hard split of RCE-KD
        srce_eta   : reliability boost for ground-truth-confirmed items (>=0)
        srce_Lu    : number of uniform tail samples per user (>=0)
    Reused from RCE-KD: mkd_tau, mkd_K, mkd_L, mkd_T, mkd_mxK.
    """

    def __init__(self, args, teacher, student):
        super().__init__(args, teacher, student)
        self.model_name = "srcekd"
        self.tau = args.mkd_tau
        self.K = args.mkd_K
        self.T = args.mkd_T
        self.L = args.mkd_L
        self.mxK = args.mkd_mxK
        self.alpha = args.srce_alpha
        self.s = args.srce_s
        self.eta = args.srce_eta
        self.Lu = args.srce_Lu
        self.T_topk_dict = self.get_topk_dict(self.teacher, self.K)

        # observed (training) interactions, for the reliability correction
        observed_mat = torch.zeros((self.num_users, self.num_items), dtype=torch.bool)
        for u, items in self.dataset.train_dict.items():
            observed_mat[u, items] = True
        self.observed_mat = observed_mat.cuda()

    def get_topk_dict(self, model, mxK):
        print('Generating Top-K dict...')
        with torch.no_grad():
            inter_mat = model.get_all_ratings()
            _, topk_dict = torch.topk(inter_mat, mxK, dim=-1)
        return topk_dict.type(torch.LongTensor).cuda()

    def do_something_in_each_epoch(self, epoch):
        with torch.no_grad():
            S_topk_dict = self.get_topk_dict(self.student, self.mxK)
            # (approximate) student rank of every teacher top-K item:
            # position inside the student's top-mxK, or mxK if outside
            matches = (self.T_topk_dict.unsqueeze(2) == S_topk_dict.unsqueeze(1))
            present = matches.any(dim=2)
            pos_in_mxK = matches.float().argmax(dim=2)
            self.rankS_of_T = torch.where(present, pos_in_mxK,
                                          torch.full_like(pos_in_mxK, self.mxK))
            # rank-weighted closure samples from the student's top-mxK
            # (same sampling strategy as RCE-KD)
            weight_matrix = torch.zeros((self.num_users, self.mxK)).cuda()
            itemT_rankS = self.rowwise_index(self.T_topk_dict, S_topk_dict)
            weight_matrix[itemT_rankS[:, 0], itemT_rankS[:, 1]] += 1
            weight_matrix = torch.minimum(torch.cumsum(weight_matrix.flip(-1), dim=-1).flip(-1), torch.tensor(50.))
            weight_matrix = torch.exp((weight_matrix + 1) / self.T)
            samples = torch.multinomial(weight_matrix, self.L, replacement=False)
            self.interesting_items = torch.zeros((self.num_users, self.L)).long()
            for user in range(self.num_users):
                self.interesting_items[user] = S_topk_dict[user][samples[user]]
            self.interesting_items = self.interesting_items.cuda()
            # uniform tail samples, covering items beyond the student's top-mxK
            if self.Lu > 0:
                self.uniform_items = torch.randint(0, self.num_items,
                                                   (self.num_users, self.Lu)).cuda()
            else:
                self.uniform_items = torch.zeros((self.num_users, 0)).long().cuda()
            self.itemS = S_topk_dict[:, :self.K]

    # https://discuss.pytorch.org/t/find-indexes-of-elements-from-one-tensor-that-matches-in-another-tensor/147482/3
    def rowwise_index(self, source, target):
        idx = (target.unsqueeze(1) == source.unsqueeze(2)).nonzero()
        idx = idx[:, [0, 2]]
        return idx

    # https://stackoverflow.com/questions/74946537/can-i-apply-torch-isin-to-each-row-in-2d-tensor-without-loop
    def rowwise_isin(self, tensor_1, target_tensor):
        matches = (tensor_1.unsqueeze(2) == target_tensor.unsqueeze(1))
        result = torch.sum(matches, dim=2, dtype=torch.bool)
        return result

    def get_loss(self, *params):
        batch_users = params[0]
        itemS = self.itemS[batch_users]                         # batch_size x K, student top-K
        itemT = self.T_topk_dict[batch_users]                   # batch_size x K, teacher top-K
        itemI = self.interesting_items[batch_users]             # batch_size x L, closure samples
        itemU = self.uniform_items[batch_users]                 # batch_size x Lu, uniform samples

        # drop duplicate entries: later parts lose to earlier ones
        mask_T = self.rowwise_isin(itemT, itemS)                # itemT covered by itemS
        mask_I = self.rowwise_isin(itemI, itemS) | self.rowwise_isin(itemI, itemT)
        mask_U = self.rowwise_isin(itemU, itemS) | self.rowwise_isin(itemU, itemT) | self.rowwise_isin(itemU, itemI)
        keep_T = (~mask_T).float()
        keep_I = (~mask_I).float()
        keep_U = (~mask_U).float()
        # note: itemU may contain self-duplicates; the probability is tiny
        # (Lu << num_items) and they are left as-is

        items_all = torch.cat([itemS, itemT, itemI, itemU], dim=-1)
        keep_all = torch.cat([torch.ones_like(keep_T), keep_T, keep_I, keep_U], dim=-1)

        # ---- target distribution: weighted teacher softmax ----
        logit_T_all = self.teacher.forward_multi_items(batch_users, items_all) / self.tau
        target_weight = torch.ones_like(logit_T_all)
        if self.alpha > 0:
            # soft closure weight for teacher items: entries in itemS that are
            # also teacher top-K use their (exact) student rank < K; deduplicated
            # itemT entries use rankS_of_T
            K = self.K
            rank_itemS = torch.arange(K, device=itemS.device).expand_as(itemS).float()
            in_QT_itemS = self.rowwise_isin(itemS, itemT).float()
            w_itemS = 1 + self.alpha * torch.sigmoid((rank_itemS - K) / self.s) * in_QT_itemS
            rank_T = self.rankS_of_T[batch_users].float()
            w_itemT = 1 + self.alpha * torch.sigmoid((rank_T - K) / self.s)
            target_weight = torch.cat([w_itemS, w_itemT,
                                       torch.ones_like(itemI, dtype=torch.float),
                                       torch.ones_like(itemU, dtype=torch.float)], dim=-1)
        if self.eta > 0:
            # reliability correction: emphasize teacher knowledge confirmed by
            # the user's real training interactions (applied to teacher items)
            rel_T = 1 + self.eta * self.observed_mat[batch_users.unsqueeze(1), itemT].float()
            rel_all = torch.cat([torch.ones_like(keep_T), rel_T,
                                 torch.ones_like(itemI, dtype=torch.float),
                                 torch.ones_like(itemU, dtype=torch.float)], dim=-1)
            target_weight = target_weight * rel_all
        prob_T_all = target_weight * torch.exp(logit_T_all) * keep_all
        prob_T_all = prob_T_all / prob_T_all.sum(-1, keepdim=True)

        # ---- student distribution over the same union set ----
        logit_S_all = self.student.forward_multi_items(batch_users, items_all) / self.tau
        Z_S = (torch.exp(logit_S_all) * keep_all).sum(-1, keepdim=True)
        loss = -(prob_T_all * (logit_S_all - torch.log(Z_S))).sum(-1)
        return loss.sum()

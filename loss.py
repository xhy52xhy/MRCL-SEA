import torch
from torch import nn


from models import *
from utils import *


class CustomMultiLossLayer(nn.Module):
    """
    Inspired by
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
    """
    def __init__(self, loss_num, device=None):
        super(CustomMultiLossLayer, self).__init__()
        self.loss_num = loss_num
        self.log_vars = nn.Parameter(torch.zeros(self.loss_num, ), requires_grad=True)

    def forward(self, loss_list):
        assert len(loss_list) == self.loss_num
        precision = torch.exp(-self.log_vars)
        loss = 0
        for i in range(self.loss_num):
            loss += precision[i] * loss_list[i] + self.log_vars[i]
        return loss


class Hinge_Loss(nn.Module):
    def __init__(self, k=25, gamma=3.0, device=None):
        super(Hinge_Loss, self).__init__()
        self.gamma = gamma
        self.k = k
        self.device = device

    def forward(self, out_emb, ILL):
        left = ILL[:, 0]
        right = ILL[:, 1]
        t = ILL.shape[0]
        left_x = torch.index_select(out_emb, 0, left).to(self.device)
        right_x = torch.index_select(out_emb, 0, right).to(self.device)
        A = torch.sum(torch.abs(left_x - right_x), dim=1)
        neg_left = torch.randint(0, out_emb.shape[0], (t * self.k,), dtype=torch.int32).to(self.device)
        neg_right = torch.randint(0, out_emb.shape[0], (t * self.k,), dtype=torch.int32).to(self.device)
        neg_l_x = torch.index_select(out_emb, 0, neg_left)
        neg_r_x = torch.index_select(out_emb, 0, neg_right)
        B = torch.sum(torch.abs(neg_l_x - neg_r_x), dim=1)
        C = - torch.reshape(B, (t, self.k))
        D = A + self.gamma
        L1 = F.relu(torch.add(C, torch.reshape(D, (t, 1))))

        neg_left = torch.randint(0, out_emb.shape[0], (t * self.k,), dtype=torch.int32).to(self.device)
        neg_right = torch.randint(0, out_emb.shape[0], (t * self.k,), dtype=torch.int32).to(self.device)
        neg_l_x = torch.index_select(out_emb, 0, neg_left)
        neg_r_x = torch.index_select(out_emb, 0, neg_right)
        B = torch.sum(torch.abs(neg_l_x - neg_r_x), dim=1)
        C = - torch.reshape(B, (t, self.k))
        L2 = F.relu(torch.add(C, torch.reshape(D, (t, 1))))

        return (torch.sum(L1) + torch.sum(L2)) / (2.0 * self.k * t)


class BiCl(nn.Module):
    def __init__(self, device, tau=0.1, ab_weight=0.5, n_view=2, intra_weight=1.0, inversion=False):
        super(BiCl, self).__init__()
        self.tau = tau
        self.device = device
        self.sim = cosine_sim
        self.weight = ab_weight
        self.n_view = n_view
        self.intra_weight = intra_weight
        self.inversion = inversion

    def softXEnt(self, target, logits):
        logprobs = F.log_softmax(logits, dim=1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def forward(self, emb, train_links, emb2=None, norm=True):
        if norm:
            emb = F.normalize(emb, dim=1)
            if emb2 is not None:
                emb2 = F.normalize(emb2, dim=1)
        num_ent = emb.shape[0]

        zis = emb[train_links[:, 0]]
        if emb2 is not None:
            zjs = emb2[train_links[:, 1]]
        else:
            zjs = emb[train_links[:, 1]]

        temperature = self.tau
        alpha = self.weight
        n_view = self.n_view

        LARGE_NUM = 1e9

        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        num_classes = batch_size * n_view
        labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=num_classes).float()
        labels = labels.to(self.device)

        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
        masks = masks.to(self.device).float()
        logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large, 0, 1)) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, torch.transpose(hidden2_large, 0, 1)) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature

        # logits_a = torch.cat([logits_ab, self.intra_weight*logits_aa], dim=1)
        # logits_b = torch.cat([logits_ba, self.intra_weight*logits_bb], dim=1)
        if self.inversion:
            logits_a = torch.cat([logits_ab, logits_bb], dim=1)
            logits_b = torch.cat([logits_ba, logits_aa], dim=1)
        else:
            logits_a = torch.cat([logits_ab, logits_aa], dim=1)
            logits_b = torch.cat([logits_ba, logits_bb], dim=1)

        loss_a = self.softXEnt(labels, logits_a)
        loss_b = self.softXEnt(labels, logits_b)

        return alpha * loss_a + (1 - alpha) * loss_b


class CL_loss(nn.Module):
    def __init__(self, device, tau=0.1, ab_weight=0.5, n_view=1, intra_weight=1.0, inversion=False):
        super(CL_loss, self).__init__()
        self.tau = tau
        self.device = device
        self.sim = cosine_sim
        self.weight = ab_weight  # the factor of a->b and b<-a
        self.n_view = n_view
        self.intra_weight = intra_weight  # the factor of aa and bb

    def softCE(self, target, logits):
        logprobs = F.log_softmax(logits, dim=1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def forward(self, emb, train_links, norm=True):
        if norm:
            emb = F.normalize(emb, dim=1)

        zis = emb[train_links[:, 0]]
        zjs = emb[train_links[:, 1]]

        temperature = self.tau
        alpha = self.weight
        n_view = self.n_view
        u = self.sim(zis, zjs) / temperature
        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        num_classes = batch_size * n_view
        labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=num_classes).float()
        labels = labels.to(self.device)

        # logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
        # logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature
        loss_a = self.softCE(labels, u)
        loss_b = self.softCE(labels, torch.transpose(u, 0, 1))

        return alpha * loss_a + (1 - alpha) * loss_b


# Contrastive Alignment Loss + Relation Semantics Modeling Loss
class AliLoss:
    def __init__(self, rel_ht_dict, batch_size, min_rel_win=15, rel_param=0.001, neg_param=0.1, neg_multi=10, neg_margin=1.5, device=None):
        super(AliLoss, self).__init__()
        self.alpha = rel_param
        self.device = device
        self.neg_param = neg_param
        self.neg_margin = neg_margin
        self.rel_ht_dict = rel_ht_dict
        self.rel_win_size = batch_size // len(rel_ht_dict)
        if self.rel_win_size <= 1:
            self.rel_win_size = min_rel_win
        self.neg_multi = neg_multi
        self.batch_size = batch_size

    def compute_loss(self, pos_links, neg_links, emb, only_pos=False):
        index1 = pos_links[:, 0]
        index2 = pos_links[:, 1]
        neg_index1 = neg_links[:, 0]
        neg_index2 = neg_links[:, 1]
        output_embeds = F.normalize(emb, dim=1)

        embeds1 = torch.index_select(output_embeds, 0, index1.long().to(self.device))
        embeds2 = torch.index_select(output_embeds, 0, index2.long().to(self.device))
        pos_loss = torch.sum(torch.sum((embeds1 - embeds2) ** 2, dim=1))

        embeds1 = torch.index_select(output_embeds, 0, neg_index1.long().to(self.device))
        embeds2 = torch.index_select(output_embeds, 0, neg_index2.long().to(self.device))
        neg_distance = torch.sum((embeds1 - embeds2) ** 2, dim=1)
        neg_loss = torch.sum(F.relu(self.neg_margin - neg_distance))

        return pos_loss + self.neg_param * neg_loss

    def compute_rel_loss(self, hs, rs, ts, ent_emb, rel_emb):
        output_embeds = F.normalize(ent_emb, dim=1)
        h_embeds = torch.index_select(output_embeds, 0, torch.tensor(hs, dtype=torch.int32).to(self.device))
        t_embeds = torch.index_select(output_embeds, 0, torch.tensor(ts, dtype=torch.int32).to(self.device))
        r_temp_embeds = (h_embeds - t_embeds).view(-1, self.rel_win_size, output_embeds.shape[-1])
        r_temp_embeds = torch.mean(r_temp_embeds, dim=1, keepdim=True)
        r_embeds = r_temp_embeds.repeat(1, self.rel_win_size, 1)
        r_embeds = torch.sum(r_embeds, dim=1)
        h_embeds = h_embeds.view(-1, self.rel_win_size, h_embeds.shape[-1])
        h_embeds = torch.sum(h_embeds, dim=1)
        t_embeds = t_embeds.view(-1, self.rel_win_size, t_embeds.shape[-1])
        t_embeds = torch.sum(t_embeds, dim=1)
        r_embeds = F.normalize(r_embeds, dim=1)
        rel_emb = torch.index_select(rel_emb, 0, torch.tensor(rs, dtype=torch.int32).to(self.device)) 
        r_embeds = r_embeds + F.normalize(rel_emb, dim=1)
        return torch.sum(torch.sum((h_embeds - t_embeds - r_embeds) ** 2, dim=1)) * self.alpha

    def generate_input_batch(self, train_ill, test_ill, sup_links_set):
        train_left, train_right = list(train_ill[:, 0]), list(train_ill[:, 1])
        test_left, test_right = list(test_ill[:, 0]), list(test_ill[:, 1])
        if self.batch_size > len(train_left):
            batch_size = len(train_left)
        else:
            batch_size = self.batch_size
        index = np.random.choice(len(train_left), batch_size)
        pos_links = train_ill[index, ]
        neg_links = list()

        neg_ent1 = list()
        neg_ent2 = list()
        for i in range(self.neg_multi):
            neg_ent1.extend(random.sample(train_left + test_left, batch_size))
            
            neg_ent2.extend(random.sample(train_right + test_right, batch_size))
        neg_links.extend([(neg_ent1[i], neg_ent2[i]) for i in range(len(neg_ent1))])

        neg_links = set(neg_links) - sup_links_set
        neg_links = np.array(list(neg_links))

        pos_links = torch.tensor(pos_links, dtype=torch.int64)
        neg_links = torch.tensor(neg_links, dtype=torch.int64)
        return pos_links, neg_links

    def generate_rel_batch(self):
        hs, rs, ts = list(), list(), list()
        for r, hts in self.rel_ht_dict.items():
            hts_batch = [random.choice(hts) for _ in range(self.rel_win_size)]
            for h, t in hts_batch:
                hs.append(h)
                ts.append(t)
            rs.append(r)
        return hs, rs, ts


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


class NCA_loss(nn.Module):

    def __init__(self, alpha, beta, ep, device):
        super(NCA_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ep = ep
        self.device = device
        self.sim = cosine_sim

    def forward(self, emb, train_links, test_links, device='cuda:3'):

        emb = F.normalize(emb)
        num_ent = emb.shape[0]

        im = emb[train_links[:, 0]]
        s = emb[train_links[:, 1]]

        im_neg_scores = None
        s_neg_scores = None
        loss_global_neg = None

        # labels = torch.arange(im.size(0))
        # embeddings = torch.cat([im, s], dim=0)
        # labels = torch.cat([labels, labels], dim=0)
        # loss = self.loss_func(embeddings, labels)
        # return loss

        # """

        if len(test_links) != 0:
            test_links = test_links[random.sample([x for x in np.arange(0, len(test_links))], 4500)]

            im_neg_scores = self.sim(im, emb[test_links[:, 1]])
            s_neg_scores = self.sim(s, emb[test_links[:, 0]])

        # im = l2norm(im)
        # s = l2norm(s)

        bsize = im.size()[0]
        # compute prediction-target score matrix
        # print (im)
        # print(s)
        scores = self.sim(im, s)

        # tmp = torch.eye(bsize).to(device)
        tmp = torch.eye(bsize).cuda(device)
        s_diag = tmp * scores

        alpha = self.alpha
        alpha_2 = alpha  # / 3.0
        beta = self.beta
        ep = self.ep
        S_ = torch.exp(alpha * (scores - ep))
        S_ = S_ - S_ * tmp  # clear diagnal

        S_1 = None
        S_2 = None

        if len(test_links) != 0:
            S_1 = torch.exp(alpha * (im_neg_scores - ep))
            S_2 = torch.exp(alpha * (s_neg_scores - ep))

        loss_diag = - torch.log(1 + beta * F.relu(s_diag.sum(0)))

        loss = torch.sum(
            torch.log(1 + S_.sum(0)) / alpha
            + torch.log(1 + S_.sum(1)) / alpha
            + loss_diag) / bsize

        if len(test_links) != 0:
            loss_global_neg = (torch.sum(torch.log(1 + S_1.sum(0)) / alpha_2
                                         + torch.log(1 + S_2.sum(0)) / alpha_2)
                               + torch.sum(torch.log(1 + S_1.sum(1)) / alpha_2
                                           + torch.log(1 + S_2.sum(1)) / alpha_2)) / 4500
        if len(test_links) != 0:
            return loss + loss_global_neg

        return loss


class weight_NCA_loss(nn.Module):

    def __init__(self, a, b, alpha, beta, ep, device):
        super(weight_NCA_loss, self).__init__()
        self.a = a
        self.b = b
        self.alpha = alpha
        self.beta = beta
        self.ep = ep
        self.sim = cosine_sim
        self.device = device
        # from pytorch_metric_learning import losses
        # self.loss_func = losses.MultiSimilarityLoss()

    def forward(self, emb, train_links, corrupt_ill, test_links, device=0):
        alpha = self.alpha
        beta = self.beta
        ep = self.ep
        emb = F.normalize(emb)

        if len(train_links) != 0:
            im_t = emb[train_links[:, 0]]
            s_t = emb[train_links[:, 1]]
            bsize_t = im_t.size()[0]
            scores_t = self.sim(im_t, s_t)
            tmp_t = torch.eye(bsize_t).cuda(device)
            s_diag_t = tmp_t * scores_t
            S_t = torch.exp(alpha * (scores_t - ep))
            S_t = S_t - S_t * tmp_t  # clear diagnal
            loss_diag_t = - torch.log(1 + beta * F.relu(s_diag_t.sum(0)))
            loss_t = torch.sum(
                torch.log(1 + S_t.sum(0)) / alpha
                + torch.log(1 + S_t.sum(1)) / alpha
                + loss_diag_t) / bsize_t
        else:
            loss_t = 0

        if len(corrupt_ill) != 0:
            im_c = emb[corrupt_ill[:, 0]]
            s_c = emb[corrupt_ill[:, 1]]
            bsize_c = im_c.size()[0]
            scores_c = self.sim(im_c, s_c)
  
            tmp_c = torch.eye(bsize_c).cuda(device)
            s_diag_c = tmp_c * scores_c
            S_c = torch.exp(alpha * (scores_c - ep))
            S_c = S_c - S_c * tmp_c  # clear diagnal
            loss_diag_c = - torch.log(1 + beta * F.relu(s_diag_c.sum(0)))
            loss_c = torch.sum(
                torch.log(1 + S_c.sum(0)) / alpha
                + torch.log(1 + S_c.sum(1)) / alpha
                + loss_diag_c) / bsize_c
        else:
            loss_c = 0

        return self.a*loss_t + self.b*loss_c

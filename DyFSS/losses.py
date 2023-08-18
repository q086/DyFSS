import torch
import torch.nn.modules.loss
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np


def re_loss_func(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=labels * pos_weight)

    # Check if the model is simple Graph Auto-encoder
    if logvar is None:
        return cost

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD


def dis_loss(real, d_fake):
    dc_loss_real = torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(real), real))
    dc_loss_fake = torch.mean(F.binary_cross_entropy_with_logits(torch.zeros_like(d_fake), d_fake))
    return dc_loss_real + dc_loss_fake

def pq_loss_func(feat, cluster_centers):
    alpha = 1.0
    q = 1.0 / (1.0 + torch.sum((feat.unsqueeze(1) - cluster_centers) ** 2, dim=2) / alpha)
    q = q ** (alpha + 1.0) / 2.0
    q = (q.t() / torch.sum(q, dim=1)).t()

    weight = q ** 2 / torch.sum(q, dim=0)
    p = (weight.t() / torch.sum(weight, dim=1)).t()
    p = p.detach()

    log_q = torch.log(q)
    loss = F.kl_div(log_q, p)
    return loss, p, q


def dist_2_label(q_t):
    maxlabel, label = torch.max(q_t, dim=1)
    return maxlabel, label


def sim_loss_func(adj_preds, adj_labels, weight_tensor=None):
    cost = 0.
    if weight_tensor is None:
        cost += F.binary_cross_entropy_with_logits(adj_preds, adj_labels)
    else:
        cost += F.binary_cross_entropy_with_logits(adj_preds.view(-1), adj_labels.to_dense().view(-1),
                                                   weight=weight_tensor)

    return cost
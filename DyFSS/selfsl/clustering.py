import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
import numpy as np
import torch
# from .gnn_encoder import GraphConvolution
from numba import njit
import networkx as nx
from sklearn.cluster import KMeans
import os

from layers import GraphConvolution


class Clu(nn.Module):

    def __init__(self, data, processed_data, encoder, nhid1, nhid2, dropout, device, **kwargs):
        super(Clu, self).__init__()
        self.args = kwargs['args']
        self.data = data
        self.processed_data = processed_data
        self.device = device
        self.ncluster = 10
        self.pseudo_labels = self.get_label(self.ncluster)
        self.pseudo_labels = self.pseudo_labels.to(device)
        self.disc1 = nn.Linear(nhid1, self.ncluster)
        self.sampled_indices = (self.pseudo_labels >= 0)

        self.gcn = encoder
        self.gcn2 = GraphConvolution(nhid2, nhid2, dropout, act=lambda x: x)
        self.disc2 = nn.Linear(nhid2, self.ncluster)

    def make_loss_stage_two(self, encoder_features, adj_norm):
        embeddings = self.gcn2_forward(encoder_features, adj_norm)
        embeddings = self.disc2(embeddings)
        output = F.log_softmax(embeddings, dim=1)
        loss = F.nll_loss(output[self.sampled_indices], self.pseudo_labels[self.sampled_indices])
        return loss

    def make_loss_stage_one(self, embeddings):
        embeddings = self.disc1(embeddings)
        output = F.log_softmax(embeddings, dim=1)
        loss = F.nll_loss(output[self.sampled_indices], self.pseudo_labels[self.sampled_indices])
        return loss

    def gcn2_forward(self, input, adj):
        self.train()
        embeddings = self.gcn2(input, adj)
        return embeddings

    def get_label(self, ncluster):
        cluster_file = './saved/' + self.args.dataset + '_cluster_%s.npy' % ncluster
        if not os.path.exists(cluster_file):
            print('perform clustering with KMeans...')
            kmeans = KMeans(n_clusters=ncluster, random_state=0).fit(self.data.features)
            cluster_labels = kmeans.labels_
            return torch.LongTensor(cluster_labels)
        else:
            cluster_labels = np.load(cluster_file)
            return torch.LongTensor(cluster_labels)

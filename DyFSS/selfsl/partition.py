import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
import numpy as np
import torch
import metis
import os
import dgl
from layers import GraphConvolution


class Par(nn.Module):

    def __init__(self, data, processed_data, encoder, nhid1, nhid2, dropout, device, **kwargs):
        super(Par, self).__init__()
        self.args = kwargs['args']
        self.data = data
        self.device = device
        if kwargs['args'].dataset in ['citeseer']:
            self.nparts = 1000
        elif kwargs['args'].dataset in ['photo', 'computers']:
            self.nparts = 100
        elif kwargs['args'].dataset in ['wiki']:
            self.nparts = 20
        else:
            self.nparts = 400
        pseudo_labels = self.get_label(self.nparts)
        self.pseudo_labels = pseudo_labels.to(device)
        self.disc1 = nn.Linear(nhid1, self.nparts)
        self.sampled_indices = (self.pseudo_labels >= 0)

        self.gcn = encoder
        self.gcn2 = GraphConvolution(nhid2, nhid2, dropout, act=lambda x: x)
        self.disc2 = nn.Linear(nhid2, self.nparts)

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
        embeddings = self.gcn2(input, adj)
        return embeddings

    def get_label(self, nparts):
        partition_file = './saved/' + self.args.dataset + '_partition_%s.npy' % nparts
        if not os.path.exists(partition_file):
            print('Perform graph partitioning with Metis...')

            if self.args.dataset == 'chameleon':
                g, _ = dgl.load_graphs('data/' + self.args.dataset + '/' + self.args.dataset + '.bin')
                g = g[0]
                partition_labels = dgl.metis_partition_assignment(g, nparts)
                print("chameleon dgl metis, type(partition_labels) ", type(partition_labels))
                return partition_labels

            adj_coo = self.data.adj.tocoo()
            node_num = adj_coo.shape[0]
            adj_list = [[] for _ in range(node_num)]
            for i, j in zip(adj_coo.row, adj_coo.col):
                if i == j:
                    continue
                adj_list[i].append(j)

            _, partition_labels =  metis.part_graph(adj_list, nparts=nparts, seed=0)
            return torch.LongTensor(partition_labels)
        else:
            partition_labels = np.load(partition_file)
            return torch.LongTensor(partition_labels)


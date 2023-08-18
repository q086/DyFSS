import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
import numpy as np
import torch
import os
from torch_geometric.utils import negative_sampling
from sklearn.neighbors import kneighbors_graph
from layers import GraphConvolution


class PairwiseAttrSim(nn.Module):

    def __init__(self, data, processed_data, encoder, nhid1, nhid2, device, dropout, **kwargs):
        super(PairwiseAttrSim, self).__init__()
        self.args = kwargs['args']
        self.data = data
        self.processed_data = processed_data
        self.device = device
        self.num_nodes = data.adj.shape[0]
        self.nclass = 2
        self.disc1 = nn.Linear(nhid1, self.nclass)
        self.build_knn(self.processed_data.features.cpu(), k=10)

        self.gcn = encoder
        self.gcn2 = GraphConvolution(nhid2, nhid2, dropout, act=lambda x: x)
        self.disc2 = nn.Linear(nhid2, self.nclass)

    def build_knn(self, X, k=10):
        args = self.args
        if not os.path.exists(f'saved/{args.dataset}_knn_{k}.npz'):
            print("performance buliding knn...")
            A_knn = kneighbors_graph(X, k, mode='connectivity',
                            metric='cosine', include_self=True, n_jobs=4)
        else:
            A_knn = sp.load_npz(f'saved/{args.dataset}_knn_{k}.npz')
        self.edge_index_knn = torch.LongTensor(A_knn.nonzero())

    def sample(self, n_samples=4000):
        labels = []
        sampled_edges = []

        num_edges = self.edge_index_knn.shape[1]

        idx_selected = np.random.default_rng(self.args.seed).choice(num_edges,
                        n_samples, replace=False).astype(np.int32)

        labels.append(torch.ones(len(idx_selected), dtype=torch.long))
        sampled_edges.append(self.edge_index_knn[:, idx_selected])

        neg_edges = negative_sampling(
                    edge_index=self.edge_index_knn, num_nodes=self.num_nodes,
                    num_neg_samples=n_samples)
        sampled_edges.append(neg_edges)
        labels.append(torch.zeros(neg_edges.shape[1], dtype=torch.long))

        labels = torch.cat(labels).to(self.device)
        sampled_edges = torch.cat(sampled_edges, axis=1)
        return sampled_edges, labels

    def gcn2_forward(self, input, adj):
        embeddings = self.gcn2(input, adj)
        return embeddings

    def make_loss_stage_two(self, encoder_features, adj_norm):
        node_pairs, labels = self.sample()
        embeddings = self.gcn2_forward(encoder_features, adj_norm)
        embeddings0 = embeddings[node_pairs[0]]
        embeddings1 = embeddings[node_pairs[1]]
        embeddings = self.disc2(torch.abs(embeddings0 - embeddings1))
        output = F.log_softmax(embeddings, dim=1)
        loss = F.nll_loss(output, labels)
        return loss

    def make_loss_stage_one(self, embeddings):
        node_pairs, labels = self.sample()
        embeddings0 = embeddings[node_pairs[0]]
        embeddings1 = embeddings[node_pairs[1]]
        embeddings = self.disc1(torch.abs(embeddings0 - embeddings1))
        output = F.log_softmax(embeddings, dim=1)
        loss = F.nll_loss(output, labels)
        return loss


import os.path as osp
from torch_geometric.datasets import Planetoid, WikiCS, Coauthor, Amazon, CoraFull, DBLP
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import scipy.sparse as sp
import numpy as np
import torch
from deeprobust.graph.data import Dataset


def get_dataset(name, normalize_features=False, transform=None, if_dpr=True):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name)
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, name)
    elif name in ['corafull']:
        dataset = CoraFull(path)
    elif name in ['arxiv']:
        dataset = PygNodePropPredDataset(name='ogbn-'+name)
    elif name in ['cs', 'physics']:
        dataset = Coauthor(path, name)
    elif name in ['computers', 'photo']:
        dataset = Amazon(path, name)
    elif name in ['wiki']:
        dataset = WikiCS(root='data/wiki')
        dataset.name = 'wiki'
    else:
        raise NotImplementedError

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    if not if_dpr:
        return dataset

    return Pyg2Dpr(dataset)


class Pyg2Dpr(Dataset):
    def __init__(self, pyg_data, multi_splits=False, **kwargs):
        try:
            splits = pyg_data.get_idx_split()
        except:
            pass

        dataset_name = pyg_data.name
        pyg_data = pyg_data[0]
        n = pyg_data.num_nodes

        self.adj = sp.csr_matrix((np.ones(pyg_data.edge_index.shape[1]),
            (pyg_data.edge_index[0], pyg_data.edge_index[1])), shape=(n, n))
        self.features = pyg_data.x.numpy()
        self.labels = pyg_data.y.numpy()

        if len(self.labels.shape) == 2 and self.labels.shape[1] == 1:
            self.labels = self.labels.reshape(-1)


def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index]


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)


def dist_2_label(q_t):
    maxlabel, label = torch.max(q_t, dim=1)
    return maxlabel, label


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
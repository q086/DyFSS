import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class VGAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout, cluster_num):
        super(VGAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)   # mu
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)   # logvar
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return hidden1, self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        hidden, mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return hidden, self.dc(z), mu, logvar, z


class Discriminator(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2, hidden_dim3):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim3, hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim1, 1),
        )

    def forward(self, x):
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        z = self.dis(x)
        return z


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

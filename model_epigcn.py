import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import MessagePassing, GATConv, GCNConv




class SIRGCN(MessagePassing):
    def __init__(self, embedding_size):
        super().__init__(aggr='add')
        self.embedding_size = embedding_size
        #self.toS = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.toI = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.toR = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, s, i, r, edge_index, edge_weight):
        # x has shape [N, 3, embedding_size], where the second dim represents S, I, R
        return self.propagate(edge_index, edge_weight=edge_weight, s=s, x=i, r=r)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.reshape(-1,1)

    def update(self, neighbor_i, s, x, r):
        s1 = s - self.toI(torch.cat([s, neighbor_i], dim=1))   # i.shape = 边数x节点数xembedding
        i1 = x + self.toI(torch.cat([s, neighbor_i], dim=1)) - self.toR(x)
        r1 = self.toR(x) + r
        return s1, i1, r1

class EpiGCN(torch.nn.Module):
    def __init__(self, g, w, st_feature, sa_feature, feature_size):
        super(EpiGCN, self).__init__()

        #############################################
        # Parameters
        self.g = g
        self.w = w
        self.feature_size = feature_size
        self.activation = nn.ReLU()
        #############################################
        # generate features
        if (st_feature is not None) and (sa_feature is not None):
            st_feature = torch.cat([f.mean(dim=0, keepdims=True) for f in st_feature], dim=0)
            sa_feature = torch.cat([f.mean(dim=0, keepdims=True) for f in sa_feature], dim=0)
            self.feature = torch.cat([st_feature, sa_feature],dim=-1)
            self.feature_size = feature_size * 2
        elif sa_feature is not None:
            self.feature = torch.cat([f.mean(dim=0, keepdims=True) for f in sa_feature], dim=0)
        elif st_feature is not None:
            self.feature = torch.cat([f.mean(dim=0, keepdims=True) for f in st_feature], dim=0)
        #############################################
        # Define GCN Layers
        self.base_gcn = SIRGCN(self.feature_size)
        #############################################
        # Define Input and Output MLP Layers
        self.input_s = nn.Sequential(nn.Linear(self.feature_size, self.feature_size, bias=True),)
        self.input_i = nn.Sequential(nn.Linear(self.feature_size, self.feature_size, bias=True),)
        self.input_r = nn.Sequential(nn.Linear(self.feature_size, self.feature_size, bias=True),)

        self.output = nn.Sequential(
            nn.Linear(self.feature_size * 3, 3, bias=True),
        )
        #############################################
        # Define Layer Norm
        self.batch_norm = nn.BatchNorm1d(self.feature_size)
        #############################################
        # Initialize
        for l in self.input_s:
            if type(l) == nn.Linear:
                nn.init.kaiming_normal_(l.weight)
                l.bias.data.fill_(0)
        for l in self.input_i:
            if type(l) == nn.Linear:
                nn.init.kaiming_normal_(l.weight)
                l.bias.data.fill_(0)
        for l in self.input_r:
            if type(l) == nn.Linear:
                nn.init.kaiming_normal_(l.weight)
                l.bias.data.fill_(0)
        for l in self.output:
            if type(l) == nn.Linear:
                nn.init.kaiming_normal_(l.weight)
                l.bias.data.fill_(0)

    def forward(self):
        s = self.activation(self.batch_norm(self.input_s(self.feature) + self.feature))
        i = self.activation(self.batch_norm(self.input_i(self.feature) + self.feature))
        r = self.activation(self.batch_norm(self.input_r(self.feature) + self.feature))
        s, i, r = self.base_gcn(s, i, r, self.g, self.w)
        x = self.output(torch.cat([s,i,r],dim=-1))
        x = x.softmax(dim=-1)
        return x


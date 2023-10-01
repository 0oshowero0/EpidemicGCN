import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights, vit_b_32, ViT_B_32_Weights, vgg11, VGG11_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision import transforms
from torch_geometric.nn import MessagePassing, GCNConv, GATConv
import math
from datetime import datetime

class ResNet18(torch.nn.Module):
    def __init__(self, embedding_size, cv_model_batch):
        super(ResNet18, self).__init__()

        #############################################
        # Parameters
        self.embedding_size = embedding_size
        self.cv_model_batch = cv_model_batch
        self.activation = nn.ReLU()
        #############################################
        # Define ResNet Layers
        weights = ResNet18_Weights.DEFAULT
        self.base_model = resnet18(weights=weights)

        num_ftrs = self.base_model.fc.in_features
        fc = nn.Sequential(
        nn.Linear(num_ftrs, self.embedding_size, bias=True),
        )
        self.base_model.fc = fc
        #############################################
        # Initialize
        for l in self.base_model.fc:
            if type(l) == nn.Linear:
                nn.init.kaiming_normal_(l.weight)
                l.bias.data.fill_(0)
    def forward(self, imgs):
        imgs_num = imgs.shape[0]
        if imgs_num >self.cv_model_batch:
            split_num = imgs_num // self.cv_model_batch
            if imgs_num % self.cv_model_batch > 0:
                split_num += 1
            x = [self.base_model(imgs[i*(self.cv_model_batch):(i+1)*(self.cv_model_batch)]) for i in range(split_num)]
            x = torch.cat(x,dim=0)
        else:
            x = self.base_model(imgs)
        return x

class ViTb32(torch.nn.Module):
    def __init__(self, embedding_size, cv_model_batch):
        super(ViTb32, self).__init__()

        #############################################
        # Parameters
        self.embedding_size = embedding_size
        self.cv_model_batch = cv_model_batch
        self.activation = nn.ReLU()
        #############################################
        # Define ViTb32 Layers
        weights = ViT_B_32_Weights.DEFAULT
        self.base_model = vit_b_32(weights=weights)

        num_ftrs = self.base_model.heads[0].in_features
        fc = nn.Sequential(
            nn.Linear(num_ftrs, self.embedding_size, bias=True),
        )
        self.base_model.heads = fc
        #############################################
        # Initialize
        for l in self.base_model.heads:
            if type(l) == nn.Linear:
                nn.init.kaiming_normal_(l.weight)
                l.bias.data.fill_(0)
    def forward(self, imgs):
        imgs_num = imgs.shape[0]
        if imgs_num >self.cv_model_batch:
            split_num = imgs_num // self.cv_model_batch
            if imgs_num % self.cv_model_batch > 0:
                split_num += 1
            x = [self.base_model(imgs[i*(self.cv_model_batch):(i+1)*(self.cv_model_batch)]) for i in range(split_num)]
            x = torch.cat(x,dim=0)
        else:
            x = self.base_model(imgs)
        return x



class FeatureExtractorModel(torch.nn.Module):
    def __init__(self, embedding_size, cv_base_arch, k, cv_model_batch):
        super(FeatureExtractorModel, self).__init__()

        #############################################
        # Parameters
        self.embedding_size = embedding_size
        self.k = k
        self.activation = nn.ReLU()
        #############################################
        # Load CV base model
        if cv_base_arch == 'ResNet18':
            self.cv = ResNet18(self.embedding_size, cv_model_batch)
        elif cv_base_arch == 'ViTb32':
            self.cv = ViTb32(self.embedding_size, cv_model_batch)
        #############################################
        # Define GCN Layers
        self.geo_gcn = GCNConv(self.embedding_size, self.embedding_size)
        #############################################
        # Define Input and Output MLP Layers
        self.output = nn.Sequential(
            nn.Linear(self.embedding_size, 3, bias=True),
        )
        ############################################
        # Initialize

        for l in self.output:
            if type(l) == nn.Linear:
                nn.init.kaiming_normal_(l.weight)
                l.bias.data.fill_(0)

    def forward(self, g, w, images, sample_idxs=None):
        x = self.cv(images)
        if sample_idxs is not None:
            # for satellite images, we have to sample some of them
            X = torch.zeros(g.max() + 1, self.embedding_size).to(x.device)
            X[sample_idxs] = x
            x = X
        # geo gcn
        x = self.activation(self.geo_gcn(x, g, w))
        output = self.output(x)
        output = output.mean(dim=0, keepdim=True).softmax(dim=-1)
        return output

    def get_embedding(self, g, w, images, sample_idxs=None):
        x = self.cv(images)
        if sample_idxs is not None:
            # for satellite images, we have to sample some of them
            X = torch.zeros(g.max() + 1, self.embedding_size).to(x.device)
            X[sample_idxs] = x
            x = X
        # geo gcn
        x = self.activation(self.geo_gcn(x, g, w))
        return x


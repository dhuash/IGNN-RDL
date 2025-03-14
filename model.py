import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter, Linear, Sequential

class IGNNRDL(Module):

    def __init__(self, device, alpha, beta, lamda, gama, fai, dropout, num_class, dim, num_layer, activation=F.relu):
        super(IGNNRDL, self).__init__()
        self.device = device
        self.activation = activation
        self.alpha = alpha
        self.beta = beta
        self.lamda = lamda
        self.gama = gama
        self.fai = fai
        self.num_class = num_class
        self.dim = dim
        self.num_layer = num_layer
        self.dropout = dropout
        self.w1 = Parameter(torch.FloatTensor(dim, dim), requires_grad=True)
        self.w2 = Sequential(Linear(2 * dim, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w1)
        for layer in self.w2:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, feat, org_adj, knn_adj):
        y: Tensor = torch.rand_like(feat)
        z: Tensor = feat
        for i in range(self.num_layer):

            feat = F.dropout(feat, self.dropout, training=self.training)
            temp = torch.mm(z, z.t())
            temp = torch.mm(temp, y)
            temp = torch.mm(temp, self.w1)
            temp = (self.fai / self.lamda) * temp
            temp = torch.relu(temp)
            y_n = feat - temp

            temp = -self.fai * torch.mm(y, y.t())
            temp = torch.relu(temp)
            temp = temp + self.gama * (self.alpha * org_adj + self.beta * knn_adj)
            temp = 1/(self.gama * (1 + self.alpha + self.beta)) * temp
            temp = torch.mm(temp, z)
            z_n = temp
            y = y_n
            z = z_n
        y = F.normalize(y, p=2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        p = torch.cat((y, z), dim=1)
        p = F.dropout(p, self.dropout, training=self.training)
        return self.w2(p), p

# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphGCN(torch.nn.Module):
    def __init__(self, in_channels, data):
        super(GraphGCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(data.num_features, 16)  # 第一层GCN，将输入特征转换到16维
        self.conv2 = GCNConv(16, 1)  # 修改第二层GCN，输出一个单元，用于二分类

    def forward(self, x, edge_index, edge_weight=None):
        # 第一层GCN处理加ReLU激活
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        # 应用dropout
        x = F.dropout(x, p=0.2, training=self.training)
        # 第二层GCN处理，输出一个logit
        x = self.conv2(x, edge_index, edge_weight)
        # 应用sigmoid激活函数来得到属于类别1的概率
        x = torch.sigmoid(x)
        return x.squeeze()  # 用squeeze()去掉多余的维度，以便与目标标签对齐



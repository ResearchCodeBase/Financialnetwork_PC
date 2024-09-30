import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from GCN.model.norm import GraphNorm
# 构建GraphSage模型
class GraphGCN(torch.nn.Module):
    def __init__(self, in_channels,data):
        super(GraphGCN, self).__init__()
        self.conv1 = GCNConv(data.num_features, 16)
        self.conv2 = GCNConv(16, 2)


    def forward(self, x, edge_index,edge_weight):
        # 第一层输入权重矩阵
        x = self.conv1(x, edge_index,edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index,edge_weight)
        return x






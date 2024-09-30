import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

# 构建GraphSage模型
class GraphGCN(torch.nn.Module):
    def __init__(self, in_channels,data):
        super(GraphGCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(data.num_features, 32)
        self.conv2 = GCNConv(32, 7)


    def forward(self, x, edge_index):
        # 第一层输入权重矩阵
        x = self.conv1(x, edge_index)
        # print('第一层网络结果',x.shape)
        x = x.relu()
        # print('relu激活函数后', x.shape)
        x = F.dropout(x, p=0.5, training=self.training)
        # print('droput后', x.shape)
        x = self.conv2(x, edge_index)
        # print('第二层网络结果', x.shape)
        # 注意这里输出的是节点的特征，维度为[节点数,类别数]
        # return x
        # 每一行元素和为1
        # print('Log_softmanx后',F.log_softmax(x, dim=1).shape)

        return F.log_softmax(x, dim=1)






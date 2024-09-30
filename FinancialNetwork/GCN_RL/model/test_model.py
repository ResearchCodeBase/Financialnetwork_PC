import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv,GCNACONV

# 构建GraphSage模型

from torch_geometric.nn.dense.linear import Linear
class Linear(torch.nn.Module):
    def __init__(self, in_channels,data):
        super(self).__init__()
        torch.manual_seed(12345)

        self.lin = Linear(in_channels, 8, bias=False,
                          weight_initializer='glorot')


    def forward(self, x):
        # 第一层输入权重矩阵
        x = self.lin(x)

        # print('第二层网络结果', x.shape)
        # 注意这里输出的是节点的特征，维度为[节点数,类别数]
        # return x
        # 每一行元素和为1
        # print('Log_softmanx后',F.log_softmax(x, dim=1).shape)

        return x






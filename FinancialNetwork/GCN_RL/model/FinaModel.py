import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from GCN.model.norm import GraphNorm
# 构建GraphSage模型
class GraphGCN(torch.nn.Module):
    def __init__(self, in_channels,data):
        super(GraphGCN, self).__init__()
        torch.manual_seed(12345)
        # conv1.lin_src.weight: torch.Size([out_channels, data.num_features]) 是W的T  权重矩阵是8行14列
        # conv1.lin.weight: torch.Size([8, 14])
        self.conv1 = GCNConv(data.num_features, 16)
        # conv2.lin_src.weight: torch.Size([2, 8]) 权重矩阵是2行8列
        self.conv2 = GCNConv(16, 2)

    #     def forward(self, g: dgl.data.DGLDataset, feats, edge_weight=None):
    #         h = self.norm_layers[0](g, feats)
    #         h = self.layers[0](g, h, edge_weight=edge_weight)
    #         for n, l in zip(self.norm_layers[1:], self.layers[1:]):
    #             h = n(g, h)
    #             h = F.dropout(h, p=self.dropout, training=self.training)
    #             h = l(g, h, edge_weight=edge_weight)
    #         return h
    # 输入x,标准话，图卷积，激活，再标准化，丢弃/图卷积，再标准化，丢弃，图卷积，
    # 先标准化，droupout，再图卷积
    # 正确的做法，图卷积，激活，丢弃

    def forward(self, x, edge_index,edge_weight):
        # 第一层输入权重矩阵
        x = self.conv1(x, edge_index,edge_weight)
        # print('第一层网络结果',x.shape)
        x = x.relu()
        # print('relu激活函数后', x.shape)
        x = F.dropout(x, p=0.5, training=self.training)
        # print('droput后', x.shape)
        x = self.conv2(x, edge_index,edge_weight)
        # print('第二层网络结果', x.shape)
        # 注意这里输出的是节点的特征，维度为[节点数,类别数]
        # return x
        # 每一行元素和为1
        # print('Log_softmanx后',F.log_softmax(x, dim=1).shape)
        # 分类结果归一化
        # 选log 不要选 softmax
        return x
        # return F.softmax(x,dim=1)


#     接收图 g，节点特征 feats，以及边权重 edge_weight。
# 首先对输入特征进行图归一化处理：h = self.norm_layers[0](g, feats)。
# 然后通过第一个图卷积层处理：h = self.layers[0](g, h, edge_weight=edge_weight)。
# 对于每一层（除了第一层）：
# 先进行图归一化处理：h = n(g, h)。
# 然后应用 dropout 层：h = F.dropout(h, p=self.dropout, training=self.training)。
# 接着通过图卷积层处理：h = l(g, h, edge_weight=edge_weight)。
# 最后返回经过所有层处理后的特征 h。





import time
import torch_geometric.transforms as T
import torch.nn
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
from GCN.model import *
from GCN.model.FinaModel import GraphGCN




dataset = torch.load('dataset/2021/train0.6_val0.15_test0.25/processed/BankingNetwork.dataset')  #96%
data = dataset[0] #该数据集只有一个图len(dataset)：1
# print(data)
# print(data.y)



# Print the sizes of the resulting sets
print(f'Training set size: {data.train_mask.sum()} nodes')
print(f'Validation set size: {data.val_mask.sum()} nodes')
print(f'Test set size: {data.test_mask.sum()} nodes')
print(f'Dataset: {dataset}:')

print('======================')

print(f'Number of graphs: {len(dataset)}')

print(f'Number of features: {data.num_features}')


print()


print('======================')
'查看图的相关特征'

'查看图的相关特征'

print('节点特征',data.num_features)
print(f'总结点数: {data.num_nodes}')
# print(f'训练节点数： {data.train_mask.sum()}')
# print(f'训练节点比例: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
# print(f'测试节点数： {data.test_mask.sum()}')
# print(f'测试节点比例: {int(data.test_mask.sum()) / data.num_nodes:.2f}')
print(f'是否有孤立节点: {data.contains_isolated_nodes()}')
print(f'是否添加自环: {data.contains_self_loops()}')
print(f'是否无向图: {data.is_undirected()}')
print('======================')

print(data.train_mask)

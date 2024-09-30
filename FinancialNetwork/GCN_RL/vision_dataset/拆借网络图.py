#coding=utf-8
import matplotlib.pyplot as plt
import networkx as nx
import torch
import matplotlib.patches as mpatches
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout, write_dot
def visualize_graph_with_masks(data, train_mask, val_mask, test_mask, seed=5):
    G = nx.DiGraph()

    # 添加边
    edge_index = data.edge_index.cpu().numpy()
    edges = zip(edge_index[0], edge_index[1])
    G.add_edges_from(edges)

    # 使用 circular_layout 算法来定位节点
    pos = nx.circular_layout(G)

    # 将所有节点设置为蓝色
    node_color = ['blue' for _ in range(G.number_of_nodes())]  # 统一使用蓝色表示所有节点

    # 节点大小根据度数来确定
    node_size = [10 * G.degree(node) for node in range(data.num_nodes)]

    # 绘制图形
    plt.figure(figsize=(15, 15))
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.1, alpha=0.5, arrows=True)
    plt.axis('off')

    # 由于所有节点颜色相同，所以不再需要图例
    plt.show()
# 加载数据集和执行可视化的代码保持不变

year = "2022"
country = "Japan"
# 有阈值 和阈值拆借 阈值对应 max_min  cdf
# type 0 原始数据 1是max_min cdf是2
type = 2
# "train0.08_val0.02_test0.9"
# "train0.6_val0.15_test0.25"
# train0.32_val0.08_test0.6
ratio="train0.32_val0.08_test0.6"
# dataset = torch.load(f'../dataset/{year}/{ratio}/processed/BankingNetwork.dataset')  # 96%
dataset = torch.load(f'../foreign_dataset/{country}/{year}/{ratio}/processed/BankingNetwork.dataset')  # 96%
# dataset = torch.load('dataset/2021/train0.7_val0.15/processed/raw_data.dataset')
data = dataset[type]  # 该数据集只有一个图len(dataset)：1
save_path = f'img/{country}/{year}/{ratio}/'
# 假设 `data` 是从您的数据集加载的图数据
# type是图的类型，可以是'raw', 'max_min', 或者 'cdf'
visualize_graph_with_masks(data, data.train_mask, data.val_mask, data.test_mask)

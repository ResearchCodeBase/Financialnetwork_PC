#coding=utf-8
import os

import matplotlib.pyplot as plt
import networkx as nx
import torch
import matplotlib.patches as mpatches
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout, write_dot
def visualize_graph_with_masks(data, train_mask, val_mask, test_mask,save_path, filename='graph_visualization.png', seed=5):
        G = nx.DiGraph()

        # 添加边
        edge_index = data.edge_index.cpu().numpy()
        edges = zip(edge_index[0], edge_index[1])
        G.add_edges_from(edges)

        # 初始化位置字典
        pos = {}

        # 中心位置
        center = np.array([0.5, 0.5])

        # 设置随机种子以确保可重复性
        np.random.seed(seed)

        # 训练和验证节点位置，随机分布在中心区域
        train_val_nodes = [idx for idx, mask in enumerate(train_mask | val_mask) if mask]
        central_radius = 0.1  # 控制中心区域的半径
        for node in train_val_nodes:
            random_angle = np.random.uniform(0, 2 * np.pi)
            random_radius = np.random.uniform(0, central_radius)
            pos[node] = center + np.array([np.cos(random_angle) * random_radius, np.sin(random_angle) * random_radius])

        # 测试节点位置，确保均匀分布在外围
        test_nodes = [idx for idx, mask in enumerate(test_mask) if mask]
        test_radius = 0.25  # 测试节点圆的半径
        test_angles = np.linspace(0, 2 * np.pi, len(test_nodes), endpoint=False)
        for i, node in enumerate(test_nodes):
            pos[node] = center + np.array([np.cos(test_angles[i]) * test_radius, np.sin(test_angles[i]) * test_radius])

        # 确定节点颜色
        node_color = ['#B9B9B9' if test_mask[i] else 'red' for i in range(data.num_nodes)]

        # 节点大小根据度数来确定
        node_size = [10 * G.degree(node) for node in range(data.num_nodes)]

        # 绘制图形
        plt.figure(figsize=(15, 15))
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=0.1, alpha=0.5, arrows=True)
        plt.axis('off')

        # 检查路径是否存在，不存在则创建
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, filename))
        plt.close()
        plt.show()

# 加载数据集和执行可视化的代码保持不变

year = "2022"
country = "America"
# 有阈值 和阈值拆借 阈值对应 max_min  cdf
# type 0 原始数据 1是max_min cdf是2
type = 2
# "train0.08_val0.02_test0.9"
# "train0.6_val0.15_test0.25"
# train0.32_val0.08_test0.6
ratio="train0.7_val0.05_test0.25"
# dataset = torch.load(f'../dataset/{year}/{ratio}/processed/BankingNetwork.dataset')  # 96%
dataset = torch.load(f'../foreign_dataset/{country}/{year}/{ratio}/processed/BankingNetwork.dataset')  # 96%
# dataset = torch.load('dataset/2021/train0.7_val0.15/processed/raw_data.dataset')
data = dataset[type]  # 该数据集只有一个图len(dataset)：1

# 定义保存路径
save_path = f'img/{country}/{year}/{ratio}/'
# 假设 `data` 是从您的数据集加载的图数据
# type是图的类型，可以是'raw', 'max_min', 或者 'cdf'
visualize_graph_with_masks(data, data.train_mask, data.val_mask, data.test_mask, save_path=save_path, filename='training_testing_visualization.png')

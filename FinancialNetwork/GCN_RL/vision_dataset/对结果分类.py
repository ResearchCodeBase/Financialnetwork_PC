# coding=utf-8
import os
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch
import matplotlib.patches as mpatches
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout, write_dot


def visualize_graph_with_masks(data, train_mask, val_mask, test_mask, prob_csv_path, save_path, seed=5):
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

    # 加载概率CSV文件
    probabilities_df = pd.read_csv(prob_csv_path)
    probabilities = probabilities_df['Probability_Class_1'].values

    # 创建节点索引到概率值的映射
    test_node_probabilities = dict(zip(test_nodes, probabilities))

    # 确定节点颜色
    node_color = []
    for i in range(data.num_nodes):
        if (train_mask[i] or val_mask[i]) and not test_mask[i]:
            node_color.append('#B9B9B9')  # 训练和验证节点为灰色
        elif test_mask[i]:
            p = test_node_probabilities.get(i, 0)  # 安全地获取概率值，如果未找到则默认为0
            if p < 0.25:
                node_color.append('#008000')  # 浅绿
            elif p < 0.5:
                node_color.append('#7FFF00')  # 深绿
            elif p < 0.75:
                node_color.append('#FFA500')  # 橙色
            else:
                node_color.append('#FF0000')  # 红色
        else:
            node_color.append('#B9B9B9')  # 未激活的节点为灰色

    # 节点大小根据度数来确定
    node_size = [1 * G.degree(node) for node in range(data.num_nodes)]

    # 绘制图形
    plt.figure(figsize=(10, 10))  # 调整画布大小

    # 获取边的颜色
    edge_colors = ['#4d4d4d' for _ in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.1, edge_color=edge_colors, alpha=0.5, arrows=True, arrowstyle='-|>',
                           arrowsize=10)  # 统一箭头和边的颜色
    # 添加图例
    patches = [mpatches.Patch(color='#008000', label='Safe'),
               mpatches.Patch(color='#7FFF00', label='Caution'),
               mpatches.Patch(color='#FFA500', label='Warning'),
               mpatches.Patch(color='#FF0000', label='Dangerous')]
    # plt.legend(handles=patches, loc='upper right', fontsize=18)  # 调整图例字体大小
    plt.axis('off')

    # 调整子图的布局参数
    # plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.savefig(os.path.join(save_path, "分类结果.png"))
    plt.show()
    # 检查路径是否存在，不存在则创建
    os.makedirs(save_path, exist_ok=True)


    plt.close()



def process_countries(year, countries, ratios, data_type):
    for country in countries:
        for ratio in ratios:
            print(country)
            print(ratio)
            dataset_path = f'../foreign_dataset/{country}/{year}/{ratio}/processed/BankingNetwork.dataset'
            dataset = torch.load(dataset_path)
            data = dataset[data_type]
            prob_csv_path = f'../foreign_results/{country}/{year}/cdf/{ratio}/训练完测试test_prob.csv'
            save_path = f'img/{country}/{year}/{ratio}/'
            visualize_graph_with_masks(data, data.train_mask, data.val_mask, data.test_mask, prob_csv_path, save_path)


year = "2022"
countries = ["America"]
ratios = ["train0.08_val0.02_test0.9", "train0.6_val0.15_test0.25", "train0.32_val0.08_test0.6"]
data_type = 2  # 0: 原始数据, 1: max_min, 2: cdf

process_countries(year, countries, ratios, data_type)

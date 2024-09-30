import os
import matplotlib.pyplot as plt
import networkx as nx
import torch
import matplotlib.patches as mpatches
import numpy as np

def visualize_graph_with_masks(data, train_mask, val_mask, test_mask, save_path, filename='graph_visualization.png', seed=5):
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
        pos[node] = center + np.array(
            [np.cos(random_angle) * random_radius, np.sin(random_angle) * random_radius])

    # 测试节点位置，确保均匀分布在外围
    test_nodes = [idx for idx, mask in enumerate(test_mask) if mask]
    test_radius = 0.25  # 测试节点圆的半径
    test_angles = np.linspace(0, 2 * np.pi, len(test_nodes), endpoint=False)
    for i, node in enumerate(test_nodes):
        pos[node] = center + np.array(
            [np.cos(test_angles[i]) * test_radius, np.sin(test_angles[i]) * test_radius])

    # 确定节点颜色
    node_color = ['#1f78b4' if test_mask[i] else 'red' for i in range(data.num_nodes)]  # 浅红色节点

    # 节点大小根据度数来确定
    node_size = [2 * G.degree(node) for node in range(data.num_nodes)]

    # 绘制图形
    plt.figure(figsize=(15, 15))
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.2, edge_color='#4d4d4d', alpha=0.5, arrows=True)  # 更深的箭头颜色

    # 添加图例
    red_patch = mpatches.Patch(color='red', label='Train & Validation')  # 图例颜色与节点颜色匹配
    blue_patch = mpatches.Patch(color='#1f78b4', label='Test')
    plt.legend(handles=[red_patch, blue_patch], fontsize=30)  # 更大的图例字体

    plt.axis('off')

    # 调整子图的布局参数
    plt.subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.08)

    # 检查路径是否存在，不存在则创建
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, filename), bbox_inches='tight')  # 保存图像时使用 bbox_inches='tight'
    plt.show()
    plt.close()


def generate_visualizations(country, year):
    ratios = ["train0.08_val0.02_test0.9", "train0.6_val0.15_test0.25", "train0.32_val0.08_test0.6"]
    type = 2

    for ratio in ratios:
        dataset = torch.load(f'../foreign_dataset/{country}/{year}/{ratio}/processed/BankingNetwork.dataset')
        data = dataset[type]

        # 定义保存路径
        save_path = f'img/{country}/{year}/{ratio}/'
        # 检查掩码是否有重叠
        overlap_train_val = (data.train_mask & data.val_mask).any()
        overlap_train_test = (data.train_mask & data.test_mask).any()
        overlap_val_test = (data.val_mask & data.test_mask).any()
        print(f"Ratio: {ratio}")
        print("Overlap between train and val masks:", overlap_train_val)
        print("Overlap between train and test masks:", overlap_train_test)
        print("Overlap between val and test masks:", overlap_val_test)

        # 可视化图
        visualize_graph_with_masks(data, data.train_mask, data.val_mask, data.test_mask, save_path=save_path, filename='US_training_testing_visualization.png')

# 使用示例
country = "America"
year = "2022"
generate_visualizations(country, year)

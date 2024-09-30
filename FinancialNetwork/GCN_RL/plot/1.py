import matplotlib.pyplot as plt
import networkx as nx
import torch
import matplotlib.patches as mpatches
import numpy as np

def visualize_graph_with_masks(data, train_mask, val_mask, test_mask, seed=5):
    G = nx.Graph()
    legend_colors = ['#B9B9B9', 'red']
    legend_labels = ['Test', 'Train&Validation']
    # ��ӱ�
    edge_index = data.edge_index.cpu().numpy()
    edges = zip(edge_index[0], edge_index[1])
    G.add_edges_from(edges)

    # ʹ��spring_layout�������нڵ�ĳ�������
    initial_pos = nx.spring_layout(G, seed=seed)

    # �������֣���train��validation�Ľڵ�������ģ�test�ڵ������Χ
    train_val_nodes = [idx for idx, mask in enumerate(train_mask | val_mask) if mask]
    test_nodes = [idx for idx, mask in enumerate(test_mask) if mask]
    center = np.array([0, 0])
    for node in train_val_nodes:
        initial_pos[node] *= 0.5  # ��train��validation�ڵ���������
    for node in test_nodes:
        initial_pos[node] *= 2  # ��test�ڵ�������Χ

    # �ڵ���ɫ�ʹ�С
    node_color = ['#B9B9B9' if not mask else 'red' for mask in train_mask | val_mask]
    node_size = [10 * G.degree(node) for node in range(data.num_nodes)]

    # ����ͼ��
    plt.figure(figsize=(15, 15))
    nx.draw_networkx_nodes(G, initial_pos, node_size=node_size, node_color=node_color, alpha=0.8)
    nx.draw_networkx_edges(G, initial_pos, width=0.1, alpha=0.5)
    plt.axis('off')

    # ����ͼ��
    legend_handles = [mpatches.Patch(color=color, label=label) for color, label in zip(['red', '#B9B9B9'], ['Train&Validation', 'Test'])]
    plt.legend(handles=legend_handles, loc='best')

    # ��ʾͼ��
    plt.show()

# �������ݼ���ִ�п��ӻ��Ĵ��뱣�ֲ���

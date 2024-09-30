# coding=gbk
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
# coding=gbk

def calculate_network_features(L, e, default):
    """
    计算网络特征。
    :param L: 负债矩阵
    :param e: 外部资产
    :param default: 基础违约
    :return: 各种计算出的特征
    """
    '''计算借贷特征'''
    # 计算拆入资金和拆出资金
    row_sums = np.sum(L, axis=1)  # 拆入资金
    column_sums = np.sum(L, axis=0)  # 拆出资金

    # 计算流动性错配指数 (LMI)
    LMI = (column_sums - row_sums) / (e + column_sums)

    # 计算一阶邻居违约率
    neighbor_ratios = []
    for node in range(L.shape[0]):
        in_neighbors = np.where(L[:, node] > 0)[0]
        out_neighbors = np.where(L[node, :] > 0)[0]
        neighbors = np.union1d(in_neighbors, out_neighbors)
        neighbors = np.setdiff1d(neighbors, node)
        neighbor_count = len(neighbors)
        neighbor_default_count = np.sum(default[neighbors] == 1) if neighbor_count > 0 else 0
        neighbor_ratio = neighbor_default_count / neighbor_count if neighbor_count > 0 else 0
        neighbor_ratios.append(neighbor_ratio)

    '''计算网络特征'''

    L[L > 0] = 1
    print(L)
    G = nx.DiGraph(L)
    # 计算每个节点的入度个数 对
    in_degrees = np.sum(L, axis=0)
    print('**********',in_degrees)

    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G)
    closeness_centrality = nx.closeness_centrality(G.reverse())
    betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
    pagerank_centrality = nx.pagerank(G)
    # 计算每个节点的所有入度邻居节点的入度和 对
    average_indegree = []
    for node in range(L.shape[0]):
        in_neighbors = np.where(L[:, node] == 1)[0]
        sum_indegree = np.sum(L[:, in_neighbors], axis=0)
        if len(sum_indegree) > 0:
            average_indegree.append(np.mean(sum_indegree))
        else:
            average_indegree.append(0)

    # 确保字典中的每个项都是Pandas Series
    features = {
        "External_Assets": pd.Series(e),
        "Lent_Funds": pd.Series(column_sums),
        "Borrowed_Funds": pd.Series(row_sums),
        "LMI": pd.Series(LMI),
        "First_Order_Neighbor_Default_Rate": pd.Series(neighbor_ratios),
        "Basic_Default": pd.Series(default),
        'Loan_Count_Number_of_Creditors': pd.Series(in_degrees),
        'In_Degree_Centrality': pd.Series(in_degree_centrality),
        'Out_Degree_Centrality': pd.Series(out_degree_centrality),
        'Closeness_Centrality': pd.Series(closeness_centrality),
        'Betweenness_Centrality': pd.Series(betweenness_centrality),
        'PageRank': pd.Series(pagerank_centrality),
        'Average_Indegree_of_In_Neighbors': pd.Series(average_indegree)
    }
    return features

# def save_features_to_csv(features, filename_prefix="原始特征"):
#     """
#     将特征保存到CSV文件，文件名包含时间戳和指定前缀。
#     :param features: 特征字典
#     :param filename_prefix: 文件名前缀
#     """
#     # 创建DataFrame
#     df = pd.DataFrame(features)
#
#     # 创建文件名，包括时间戳和前缀
#     current_time = datetime.now().strftime("%Y%m%d%H%M%S")
#     filename = f"{filename_prefix}_{current_time}.csv"
#
#     # 保存到CSV
#     df.to_csv(filename, index=False)
#     print(f"结果已保存为CSV文件: {filename}")


def save_features_to_csv(features, directory_name):
    """
    将特征保存到CSV文件，文件名包含时间戳。
    :param features: 特征字典
    :param directory_name: 保存文件的目录
    """
    # 创建DataFrame
    df = pd.DataFrame(features)

    # 创建带时间戳的文件名
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{directory_name}/原始特征.csv"

    # 保存到CSV
    df.to_csv(filename, index=False)
    print(f"原始特征结果已保存为CSV文件，路径是: {filename}")


    return filename

def save_features_to_csv1(features, directory_name):
    """
    将特征保存到CSV文件，文件名包含时间戳。
    :param features: 特征字典
    :param directory_name: 保存文件的目录
    """
    # 创建DataFrame
    df = pd.DataFrame(features)

    # 创建带时间戳的文件名
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{directory_name}/max_min.csv"

    # 保存到CSV
    df.to_csv(filename, index=False)
    print(f"原始特征结果已保存为CSV文件，路径是: {filename}")


    return filename
# 样例
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from torch_geometric.data import InMemoryDataset



# 比predata1增加了训练集测试集 按比例划分 并且验证了结果正确

# 实现了一个自定义的 PyTorch Geometric 数据集，名为 YooChooseBinaryDataset。它继承自 InMemoryDataset 类

# 计算违约影响指数：它接受四个参数：total_liabilities（总负债）、total_assets（总债权）、solvency_status（偿付能力状态）和 impact_factor（影响因子，默认为0.5）。

def load_dataset(l_path, e_path, default_path):
    l = np.loadtxt(l_path)
    e = np.loadtxt(e_path)
    default_data = pd.read_csv(default_path)
    return l, e, default_data

# 计算违约影响指数：它接受四个参数：total_liabilities（总负债）、total_assets（总债权）、solvency_status（偿付能力状态）和 impact_factor（影响因子，默认为0.5）。
def calculate_default_impact(total_liabilities,total_assets, solvency_status, impact_factor=0.5):
    solvency_status = solvency_status.numpy()
    print(solvency_status.shape)
    print(solvency_status)
    default_impact = total_liabilities * (1 + impact_factor * solvency_status) / (1 + total_assets)
    return default_impact



# 将带权重的邻接矩阵转换为边索引和边属性。它接受三个参数：debt_matrix（负债矩阵）、external_assets（外部资产）和 solvency_status（偿付能力状态）。
# 1.先计算总负债和总债权。
# 2.根据这些计算结果和 solvency_status 调用 calculate_default_impact 函数计算违约影响指数。
# 3.计算负债比例和每条边的违约影响指数，并构建边索引和边属性。最后，返回边索引和边属性的张量表示。
def weighted_adjacency_matrix_to_edge_index(debt_matrix, external_assets, solvency_status):
    # 计算总负债和总债权
    total_liabilities = np.sum(debt_matrix, axis=1)
    total_assets = np.sum(debt_matrix, axis=0) + external_assets

    # 计算每家银行的违约影响指数
    default_impact = calculate_default_impact(total_liabilities, total_assets, solvency_status)

    # 计算负债比例和每条边的违约影响指数
    edge_default_impact = np.zeros_like(debt_matrix, dtype=float)
    for i in range(debt_matrix.shape[0]):
        for j in range(debt_matrix.shape[1]):
            debt_ratio = debt_matrix[i, j] / total_liabilities[i] if total_liabilities[i] > 0 else 0
            edge_default_impact[i, j] = (default_impact[i] if solvency_status[i] == 1 else 1) * debt_ratio

    # 构建边索引和边属性
    num_nodes = debt_matrix.shape[0]
    edge_list = []
    edge_weight = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if debt_matrix[i][j] != 0:
                edge_list.append((i, j))
                edge_weight.append(edge_default_impact[i][j])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float).view(-1, 1)

    print('Edge index shape:', edge_index.shape)
    print('Edge weight shape:', edge_weight.shape)
    print('Edge index:', edge_index)
    print('Edge weight:', edge_weight)

    return edge_index, edge_weight

# 用于处理数据，提取特征数据x与标签数据y
def data_process(path):
    df = pd.read_csv(path)
    print(df)

    columns1 = [
        "External_Assets",  # "外部资产"
        "Lent_Funds",  # "拆出资金"
        "Borrowed_Funds",  # "拆入资金"
        "LMI",  # "LMI"
        "Basic_Default",
        "First_Order_Neighbor_Default_Rate",  # "一阶邻居违约率"
        "Loan_Count_Number_of_Creditors",  # "借款次数(债权人个数)"
        "In_Degree_Centrality",  # "入度中心性"
        "Out_Degree_Centrality",  # "出度中心性"
        "Closeness_Centrality",  # "接近中心性"
        "Betweenness_Centrality",  # "中介中心性"
        "PageRank",  # "PageRank"
        "Average_Indegree_of_In_Neighbors"  # "入度邻居平均入度"
    ]

    print(df[columns1])

    x = torch.tensor(df[columns1].values)
    print('x shape', x.shape)
    colunms2 = ['y']

    y = torch.tensor(df[colunms2].values)
    print('y shape', y.shape)

    return x,y
def train_test_split_with_masks(train_size,test_size,val_size,y):
    # 首先进行普通的数据划分
    index = np.arange(len(y))

    indices_train_val, indices_test, y_train_val, y_test = train_test_split(index,y,  stratify=y, test_size=test_size,
                                                                     random_state=48, shuffle=True)

    indices_train, indices_val, y_train, y_val = train_test_split(indices_train_val, y_train_val, stratify=y_train_val,
                                                                  train_size=train_size / (train_size + val_size),
                                                                   random_state=48, shuffle=True)

    # 计算训练集、验证集和测试集的掩码
    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)

    train_mask[indices_train] = True
    val_mask[indices_val] = True
    test_mask[indices_test] = True




    return train_mask,val_mask,test_mask


def feature_process(l,e,feature_path,train_size,test_size,val_size):
    # 提取x,y特征
    x, y = data_process(feature_path)
    # 将带权重的邻接矩阵转换为边索引张量和边属性张量
    # y squeeze后是(182,)符合下面函数的格式 也符合数据集划分的格式
    y = y.squeeze()
    print('***y', y.shape)
    edge_index, edge_weight = weighted_adjacency_matrix_to_edge_index(l, e, y)
    train_mask, val_mask, test_mask = train_test_split_with_masks(train_size,test_size,val_size, y)
    # print('========================================')
    # print('data.train_mask', train_mask)
    # print('data.test_mask', test_mask)
    # print("training samples", torch.sum(train_mask).item())
    # print("validation samples", torch.sum(val_mask).item())
    # print("test samples", torch.sum(test_mask).item())
    # 在PyTorch Geometric（一个流行的用于图神经网络的 PyTorch 扩展库），那么 edge_weight 通常是一个一维张量，其长度等于边索引张量 edge_index 中的列数（即图中边的总数）。在这种情况下，如果有 3845 条边，edge_weight 的形状应该是 (3845,)。
    edge_weight = edge_weight.squeeze()
    print('*******edgeweitg', edge_weight.shape)
    return x, y, edge_index, edge_weight, train_mask, val_mask, test_mask


#
# class FeatureBankingNetworkDataset(InMemoryDataset):
#
#     def __init__(self, root, L_path, e_path, rawfeature_path,processfeature_path, train_size, val_size, test_size,
#                  transform=None, pre_transform=None):
#         self.L_path = L_path
#         self.e_path = e_path
#         self.rawfeature_path = rawfeature_path
#         self.processed_path= processfeature_path
#         self.train_size = train_size
#         self.val_size = val_size
#         self.test_size = test_size
#         super(FeatureBankingNetworkDataset, self).__init__(root, transform, pre_transform)
#             # self.data, self.slices = torch.load(self.processed_paths[0])
#
#     @property
#     def raw_file_names(self):  # 检查self.raw_dir目录下是否存在raw_file_names()属性方法返回的每个文件
#         # 如有文件不存在，则调用download()方法执行原始文件下载
#         return []
#
#     @property
#     def processed_file_names(self):  # 检查self.processed_dir目录下是否存在self.processed_file_names属性方法返回的所有文件，没有就会走process
#         return ['raw_data.dataset', 'processed_data.dataset']
#
#     def download(self):
#         pass
#
#     def process(self):
#         # 处理第一个特征文件
#         data_list1 = self.process_features(self.rawfeature_path)
#         torch.save(data_list1, self.processed_paths[0])
#
#         # 处理第二个特征文件
#         data_list2 = self.process_features(self.processed_path)
#         torch.save(data_list2, self.processed_paths[1])
#
#     def process_features(self, feature_path):
#
#         data_list = []
#         l = np.loadtxt(self.L_path)
#         e = np.loadtxt(self.e_path)
#
#         # 提取x,y特征
#         x, y = data_process( feature_path )
#         # 将带权重的邻接矩阵转换为边索引张量和边属性张量
#         # y squeeze后是(182,)符合下面函数的格式 也符合数据集划分的格式
#         y = y.squeeze()
#         print('***y',y.shape)
#         edge_index, edge_weight = weighted_adjacency_matrix_to_edge_index(l, e, y)
#
#         train_mask, val_mask, test_mask = train_test_split_with_masks(self.train_size,self.test_size,self.val_size,y)
#         # print('========================================')
#         # print('data.train_mask', train_mask)
#         # print('data.test_mask', test_mask)
#         # print("training samples", torch.sum(train_mask).item())
#         # print("validation samples", torch.sum(val_mask).item())
#         # print("test samples", torch.sum(test_mask).item())
#         # 在PyTorch Geometric（一个流行的用于图神经网络的 PyTorch 扩展库），那么 edge_weight 通常是一个一维张量，其长度等于边索引张量 edge_index 中的列数（即图中边的总数）。在这种情况下，如果有 3845 条边，edge_weight 的形状应该是 (3845,)。
#         edge_weight=edge_weight.squeeze()
#         print('*******edgeweitg',edge_weight.shape)
#         data = Data(x = x, edge_index=edge_index,edge_weight=edge_weight,y =y,train_mask=train_mask,val_mask=val_mask,test_mask=test_mask)
#         data_list.append(data)
#         print('*****')
#         torch.save(data_list, self.processed_paths[0])

class FeatureBankingNetworkDataset(InMemoryDataset):

    def __init__(self, root, L_path, e_path, rawfeature_path,normalized_features_path, cdf_scaled_features_path, train_size, val_size, test_size,
                 transform=None, pre_transform=None):
        self.L_path = L_path
        self.e_path = e_path
        self.rawfeature_path = rawfeature_path
        self.normalized_features_path= normalized_features_path
        self.cdf_scaled_features_path =  cdf_scaled_features_path
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        super(FeatureBankingNetworkDataset, self).__init__(root, transform, pre_transform)
            # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):  # 检查self.raw_dir目录下是否存在raw_file_names()属性方法返回的每个文件
        # 如有文件不存在，则调用download()方法执行原始文件下载
        return []

    @property
    def processed_file_names(self):  # 检查self.processed_dir目录下是否存在self.processed_file_names属性方法返回的所有文件，没有就会走process
        return ['BankingNetwork.dataset']

    def download(self):
        pass


    def process(self):
        data_list = []
        l = np.loadtxt(self.L_path)
        e = np.loadtxt(self.e_path)
        # data[0]保存原始数据
        x, y, edge_index, edge_weight, train_mask, val_mask, test_mask = feature_process(l,e,self.rawfeature_path,self.train_size,self.test_size,self.val_size)
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y, train_mask=train_mask, val_mask=val_mask,
                    test_mask=test_mask)
        data_list.append(data)
        print('data[0]保存')

        # data[1]保存最大最小归一化数据
        x1, y1, edge_index1, edge_weight1, train_mask1, val_mask1, test_mask1 = feature_process(l, e, self.normalized_features_path,
                                                                                         self.train_size,
                                                                                         self.test_size, self.val_size)
        data1 = Data(x=x1, edge_index=edge_index1, edge_weight=edge_weight1, y=y1, train_mask=train_mask1, val_mask=val_mask1,
                    test_mask=test_mask1)
        data_list.append(data1)
        print('data[1]保存')

        # data[3]保存cdf缩放数据

        x2, y2, edge_index2, edge_weight2, train_mask2, val_mask2, test_mask2 = feature_process(l, e, self.cdf_scaled_features_path,
                                                                                         self.train_size,
                                                                                         self.test_size, self.val_size)
        data2 = Data(x=x2, edge_index=edge_index2, edge_weight=edge_weight2, y=y, train_mask=train_mask2, val_mask=val_mask2,
                    test_mask=test_mask2)
        data_list.append(data2)
        print('data[2]保存')

        print('*****')
        print('x.shape',x.shape)
        torch.save(data_list, self.processed_paths[0])









if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Banking Network Features')
    parser.add_argument('--L_path', type=str, default='../data/2022-182/2022全-负债矩阵(无其他银行).txt',
                        help='Path to the liability matrix file')
    parser.add_argument('--e_path', type=str, default='../data/2022-182/2022全-外部资产(无其他银行).txt',
                        help='Path to the external assets file')
    parser.add_argument('--rawfeature_path', type=str, default='data_feaature/2022全-finalresult .csv',
                        help='Path to the default data file')
    parser.add_argument('--processfeature_path', type=str, default='2022_1211_200331/cdf_scaled_features.csv',
                        help='Path to the default data file')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test set ratio')
    parser.add_argument('--root', type=str, default='../GCN/1211182/', help='Root directory for saving the dataset')

    args = parser.parse_args()

    dataset = FeatureBankingNetworkDataset(
        root=args.root,
        L_path=args.L_path,
        e_path=args.e_path,
        rawfeature_path=args.rawfeature_path,
        processfeature_path=args.processfeature_path,
        train_size=args.train_ratio,
        val_size=args.val_ratio,
        test_size=args.test_ratio
    )

    # 这里可以添加一个输出语句来确认数据集的保存位置
    print(f'数据集将被保存在 {args.root} 路径下。')

    # 保存或处理dataset...
    print('数据集已成功保存。')
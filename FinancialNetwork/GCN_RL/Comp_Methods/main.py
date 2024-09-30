# -*- coding: UTF-8 -*-
import os

import pandas as pd
import torch
from FCNN1 import FCNN,train_and_test_fcnn
from KNN import train_and_evaluate_knn
from LR import train_and_evaluate_lr
from RF import train_and_evaluate_rf
from svm import train_and_evaluate_svm
# train0.32_val0.08_test0.6 train0.08_val0.02_test0.9train0.6_val0.15_test0.25
ratio = "train0.6_val0.15_test0.25"
country = "United Kingdom"

year = "2022"
dataset = torch.load(f'../dataset/{year}/{ratio}/processed/BankingNetwork.dataset')  #96%
data = dataset[0]
# 确保 data.x 有足够的列
# if data.x.shape[1] > 4:  # 至少需要5列才能删除索引为4的列
#     data.x = torch.cat([data.x[:, :4], data.x[:, 5:]], dim=1)
# else:
#     print("Error: Not enough columns to remove the column at index 4.")
print("y")
print(data.y[data.test_mask])
model = FCNN(in_channels= data.num_features )
#
test_acc_FCNN,test_F1_FCNN, mcc_FCNNN, g_mean_FCNN = train_and_test_fcnn(model,data)
test_acc_KNN, test_F1_KNN, mcc_KNN, g_mean_KNN= train_and_evaluate_knn(data,5)
test_acc_LR,  test_F1_LR, mcc_LR, g_mean_LR= train_and_evaluate_lr(data)
test_acc_RF,  test_F1_RF, mcc_RF, g_mean_RF= train_and_evaluate_rf(data)


print('LR')
print(f'test_acc_LR:{test_acc_LR},  test_F1_LR:{test_F1_LR}, mcc_LR:{mcc_LR}, g_mean_LR:{g_mean_LR}')
print('RF')
print(test_acc_RF,  test_F1_RF, mcc_RF, g_mean_RF)
print('knn')
print(test_acc_KNN, test_F1_KNN, mcc_KNN, g_mean_KNN)
print('FCNN')
print(test_acc_FCNN,  test_F1_FCNN, mcc_FCNNN, g_mean_FCNN)


# 构建保存路径
save_dir = f"../Comp_Methods/{country}/{year}"
# 检查目录是否存在，如果不存在，创建它
os.makedirs(save_dir, exist_ok=True)
# 文件名使用ratio作为一部分来保持唯一性
file_name = f"{ratio}.csv"
save_path = os.path.join(save_dir, file_name)

data = {
    'Model': ['LR', 'RF', 'KNN', 'FCNN'],
    'Accuracy': [test_acc_LR, test_acc_RF, test_acc_KNN, test_acc_FCNN],
    'F1 Score': [test_F1_LR, test_F1_RF, test_F1_KNN, test_F1_FCNN],
    'MCC': [mcc_LR, mcc_RF, mcc_KNN, mcc_FCNNN],
    'G-Mean': [g_mean_LR, g_mean_RF, g_mean_KNN, g_mean_FCNN]
}

df = pd.DataFrame(data)


# 将DataFrame追加写入CSV文件
df.to_csv(save_path, mode='a', index=False, header=not os.path.exists(save_path))
print(f"Results saved to {save_path}")
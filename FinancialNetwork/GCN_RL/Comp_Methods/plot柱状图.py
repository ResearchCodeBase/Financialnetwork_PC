# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 读取CSV文件

import matplotlib.pyplot as plt
# 定义比例和模型MCC分数的均值
ratios = ['France', 'Germany', 'United Kingdom','China']
x = np.arange(len(ratios))  # the label locations
width = 0.15  # 柱子的宽度
# 模型名称
ratio= ['France', 'Germany', 'United Kingdom','China']

RF_MCC_10 = [0.41161481,0.37456022,0.333961672,0.446017515,0.37456022]
RF_MCC_40 = [0.505072239,0.708005442,0.61470184,0.708005442,0.703143536]
RF_MCC_75 = [0.658005442,0.420476767,0.778409091,0.652272727,0.441470184]

FCNN_MCC_10 = [0.557430391,0.601852295,0.558314065,0.607430391,0.657430391]
FCNN_MCC_40 = [0.73000696,0.73000696,0.73000696,0.75505618,0.79405473]
FCNN_MCC_75 = [0.8040971508067066,0.7761,0.841324418,0.7761,0.801819168]

LR_MCC_10 = [0,0,0,0,0]
LR_MCC_40 = [0.670443845,0.670443845,0.526440384,0.670443845,0.670443845]
LR_MCC_75 = [0.781324418,0.526440384,0.526440384,0.526440384,0.526440384]

KNN_MCC_10 = [0,0,0,0,0]
KNN_MCC_40 = [0.247904562,0.247904562,0.567466377,0.177780615,0.093437635]
KNN_MCC_75 = [0.567466377,0.567466377,0.177780615,0.177780615,0.177780615
]

GCN_MCC_10 = [0.7184,0.7184,0.7184,0.7184,0.7184]
GCN_MCC_40 = [0.80,0.80,0.80,0.80,0.80]
GCN_MCC_75 = [0.84,0.8867,0.84,0.8721,0.8867]

# 计算均值和标准差
RF_MCC_10_mean = np.mean(RF_MCC_10)
RF_MCC_10_std = np.std(RF_MCC_10)

RF_MCC_40_mean = np.mean(RF_MCC_40)
RF_MCC_40_std = np.std(RF_MCC_40)

RF_MCC_75_mean = np.mean(RF_MCC_75)
RF_MCC_75_std = np.std(RF_MCC_75)

FCNN_MCC_10_mean = np.mean(FCNN_MCC_10)
FCNN_MCC_10_std = np.std(FCNN_MCC_10)

FCNN_MCC_40_mean = np.mean(FCNN_MCC_40)
FCNN_MCC_40_std = np.std(FCNN_MCC_40)

FCNN_MCC_75_mean = np.mean(FCNN_MCC_75)
FCNN_MCC_75_std = np.std(FCNN_MCC_75)

LR_MCC_10_mean = np.mean(LR_MCC_10)
LR_MCC_10_std = np.std(LR_MCC_10)

LR_MCC_40_mean = np.mean(LR_MCC_40)
LR_MCC_40_std = np.std(LR_MCC_40)

LR_MCC_75_mean = np.mean(LR_MCC_75)
LR_MCC_75_std = np.std(LR_MCC_75)

KNN_MCC_10_mean = np.mean(KNN_MCC_10)
KNN_MCC_10_std = np.std(KNN_MCC_10)

KNN_MCC_40_mean = np.mean(KNN_MCC_40)
KNN_MCC_40_std = np.std(KNN_MCC_40)

KNN_MCC_75_mean = np.mean(KNN_MCC_75)
KNN_MCC_75_std = np.std(KNN_MCC_75)

GCN_MCC_10_mean = np.mean(GCN_MCC_10)
GCN_MCC_10_std = np.std(GCN_MCC_10)

GCN_MCC_40_mean = np.mean(GCN_MCC_40)
GCN_MCC_40_std = np.std(GCN_MCC_40)

GCN_MCC_75_mean = np.mean(GCN_MCC_75)
GCN_MCC_75_std = np.std(GCN_MCC_75)
GCN_MCC_10_mean = np.mean(GCN_MCC_10)
GCN_MCC_10_std = np.std(GCN_MCC_10)

GCN_MCC_40_mean = np.mean(GCN_MCC_40)
GCN_MCC_40_std = np.std(GCN_MCC_40)

GCN_MCC_75_mean = np.mean(GCN_MCC_75)
GCN_MCC_75_std = np.std(GCN_MCC_75)
# 开始绘制
plt.figure(figsize=(12, 8))

# 绘制柱状图
plt.bar(x - 2*width, [RF_MCC_10_mean, RF_MCC_40_mean, RF_MCC_75_mean], width, yerr=[RF_MCC_10_std, RF_MCC_40_std, RF_MCC_75_std], label='RF', capsize=5)
plt.bar(x - width, [FCNN_MCC_10_mean, FCNN_MCC_40_mean, FCNN_MCC_75_mean], width, yerr=[FCNN_MCC_10_std, FCNN_MCC_40_std, FCNN_MCC_75_std], label='FCNN', capsize=5)
plt.bar(x, [LR_MCC_10_mean, LR_MCC_40_mean, LR_MCC_75_mean], width, yerr=[LR_MCC_10_std, LR_MCC_40_std, LR_MCC_75_std], label='LR', capsize=5)
plt.bar(x + width, [KNN_MCC_10_mean, KNN_MCC_40_mean, KNN_MCC_75_mean], width, yerr=[KNN_MCC_10_std, KNN_MCC_40_std, KNN_MCC_75_std], label='KNN', capsize=5)
plt.bar(x + 2*width, [GCN_MCC_10_mean, GCN_MCC_40_mean, GCN_MCC_75_mean], width, yerr=[GCN_MCC_10_std, GCN_MCC_40_std, GCN_MCC_75_std], label='GCN', capsize=5)

# 添加一些文本标签
plt.xlabel('Percentage of pre-labeled nodes')
plt.ylabel('MCC')
plt.title('MCC by model and percentage of pre-labeled nodes')
plt.xticks(x, ratios)
plt.legend()

plt.grid(False)  # 移除网格
plt.tight_layout()
plt.show()
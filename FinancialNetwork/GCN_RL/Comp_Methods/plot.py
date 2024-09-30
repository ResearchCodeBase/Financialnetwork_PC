# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 读取CSV文件

import matplotlib.pyplot as plt

# 模型名称
ratio= ['10', '40', '75']

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
# 绘制误差条形图
plt.figure(figsize=(10, 6))

plt.errorbar(ratio, [RF_MCC_10_mean, RF_MCC_40_mean, RF_MCC_75_mean],
             yerr=[RF_MCC_10_std, RF_MCC_40_std, RF_MCC_75_std],
             fmt='o', color='#ffbe7a', linestyle='-', label='RF')

plt.errorbar(ratio, [FCNN_MCC_10_mean, FCNN_MCC_40_mean, FCNN_MCC_75_mean],
             yerr=[FCNN_MCC_10_std, FCNN_MCC_40_std, FCNN_MCC_75_std],
             fmt='o', color='#96C37D', linestyle='-', label='FCNN')

plt.errorbar(ratio, [LR_MCC_10_mean, LR_MCC_40_mean, LR_MCC_75_mean],
             yerr=[LR_MCC_10_std, LR_MCC_40_std, LR_MCC_75_std],
             fmt='o', color='#999999', linestyle='-', label='LR')

plt.errorbar(ratio, [KNN_MCC_10_mean, KNN_MCC_40_mean, KNN_MCC_75_mean],
             yerr=[KNN_MCC_10_std, KNN_MCC_40_std, KNN_MCC_75_std],
             fmt='o', color='#14517C', linestyle='-', label='KNN')

# 绘制GCN曲线和误差棒
plt.errorbar(ratio, [GCN_MCC_10_mean, GCN_MCC_40_mean, GCN_MCC_75_mean],
             yerr=[GCN_MCC_10_std, GCN_MCC_40_std, GCN_MCC_75_std],
             fmt='o', color='#ffbe7a', linestyle='-', label='GCN')

plt.xlabel('Percentage of pre-labeled nodes')
plt.ylabel('MCC')
plt.legend()
plt.grid(False)  # 移除网格
plt.show()


# 绘制GCN MCC图
plt.figure(figsize=(10, 6))

GCN_MCC_10_mean = np.mean(GCN_MCC_10)
GCN_MCC_10_std = np.std(GCN_MCC_10)

GCN_MCC_40_mean = np.mean(GCN_MCC_40)
GCN_MCC_40_std = np.std(GCN_MCC_40)

GCN_MCC_75_mean = np.mean(GCN_MCC_75)
GCN_MCC_75_std = np.std(GCN_MCC_75)

# 绘制FCNN与GCN MCC图
plt.figure(figsize=(10, 6))

# 绘制FCNN曲线和误差棒
plt.errorbar(ratio, [FCNN_MCC_10_mean, FCNN_MCC_40_mean, FCNN_MCC_75_mean],
             yerr=[FCNN_MCC_10_std, FCNN_MCC_40_std, FCNN_MCC_75_std],
             fmt='o', color='#96C37D', linestyle='-', label='FCNN')

# 绘制GCN曲线和误差棒
plt.errorbar(ratio, [GCN_MCC_10_mean, GCN_MCC_40_mean, GCN_MCC_75_mean],
             yerr=[GCN_MCC_10_std, GCN_MCC_40_std, GCN_MCC_75_std],
             fmt='o', color='#ffbe7a', linestyle='-', label='GCN')

plt.xlabel('Percentage of pre-labeled nodes')
plt.ylabel('MCC')

plt.legend()
plt.grid(False)  # 移除网格
plt.show()
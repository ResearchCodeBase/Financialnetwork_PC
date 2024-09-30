# -*- coding: UTF-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F


class FCNN(nn.Module):
    def __init__(self,in_channels ):

        super(FCNN, self).__init__()

        self.layer1 = nn.Linear(in_channels, 4)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

def evaluate_on_test(model,data,mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x[data.test_mask].float())
    #计算accuracy
    # pred 输出是【0，1，1，1，0】
    pred = out.argmax(dim=1)  # 使用最大概率的类别作为预测结果
    print("Fcnn")
    print(pred)
    # Combine test and validation masks

    test_correct = pred == data.y[mask]  # 获取正确标记的节点
    test_acc = int(test_correct.sum()) / len(data.y[mask])# 计算正确率

    # 计算F1
    # 将log_softmax输出转换为正常概率
    probabilities = torch.exp(out)

    # print('pred',pred)
    # print('probabilities',probabilities)

    # 准备数据用于计算F1得分
    pred_np =  pred.cpu().numpy()
    labels_np = data.y[mask].cpu().numpy()

    # 计算F1得分
    test_f1 = f1_score(labels_np, pred_np, average='macro')

    # 计算Matthews相关系数
    mcc = matthews_corrcoef(labels_np, pred_np)

    # 计算G-Mean
    TP = ((pred_np == 1) & (labels_np == 1)).sum()
    TN = ((pred_np == 0) & (labels_np == 0)).sum()
    FP = ((pred_np == 1) & (labels_np == 0)).sum()
    FN = ((pred_np == 0) & (labels_np == 1)).sum()

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    g_mean = np.sqrt(sensitivity * specificity)

    return test_acc, test_f1, mcc, g_mean

def masked_loss(out,labels,mask,criterion):
    loss = criterion(out[mask], labels[mask])  # 计算loss
    return (loss)


def train_and_test_fcnn(model,data, epochs=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=5e-5)
    criterion = torch.nn.CrossEntropyLoss()


    # Train the model
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        # 图神经网络会用到edge_index和weight，普通的神经网络不需要

        out= model(data.x[data.train_mask].float())

        train_loss = criterion(out, data.y[data.train_mask])
        pred = out.argmax(dim=1)  # 使用最大概率的类别作为预测结果
        test_correct = pred == data.y[data.train_mask]  # 获取正确标记的节点
        test_acc = int(test_correct.sum()) / len(data.y[data.train_mask])  # 计算正确率


        train_loss.backward()  # 反向传播
        optimizer.step()  # 优化器梯度下降

    # 测试集评估
    test_acc,test_f1,mcc,g_mean= evaluate_on_test(model, data, data.test_mask)
    # 保存测试集性能指标
    # metrics_names = ["Accuracy", "F1", "AUC", "MCC", "G-Mean"]
    # for name, metric in zip(metrics_names, test_metrics):
    #     print(f"Test {name}: {metric}")
    print('保存成功')
    return test_acc,test_f1,mcc,g_mean


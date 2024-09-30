# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
import numpy as np
import torch_geometric.transforms as T
import torch.nn
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
from GCN.model import *
from GCN.model.Probability_model import GraphGCN
'比train1加了随机分割, 比train2加了测试集验证集的作用'
'4,5,9,11,13   28 29 31 33'
"31,28能明显识别"
'1.加载数据集'

'比train182是 训练完一个epoch再测试 比(1)增加了 F1-SCOR aOC'



def load_data(data):

    # 打印该数据集 图 的相关信息

    print('===================Data Statistics=====================')

    print("training samples", torch.sum(data.train_mask).item())
    print(f'训练节点比例: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print("validation samples", torch.sum(data.val_mask).item())
    print(f'验证节点比例: {int(data.val_mask.sum()) / data.num_nodes:.2f}')
    print("test samples", torch.sum(data.test_mask).item())
    print(f'测试节点比例: {int(data.test_mask.sum()) / data.num_nodes:.2f}')
    print(f'Number of features: {data.num_features}')
    print(f'总结点数: {data.num_nodes}')
    print('y 形状',data.y.shape)



def define_model(data):
    model = GraphGCN(in_channels=data.num_features, data=data)
    print('===================网络模型结构=====================')
    print('网络结构', model)
    print('模型参数')
    for name, parameter in model.named_parameters():
        print('**************************')
        print(f'{name}: {parameter.shape}')
        print(f'{name},{parameter}')
    return model
def masked_loss(out, labels, mask, criterion):
    return criterion(out[mask], labels[mask])
def masked_accuracy(out, labels, mask):
    # # 使用最大概率的类别作为预测结果
    # 计算准确率
    # out是输出为1的概率
    pred = out>0.5
    pred = pred.int()
    correct = pred[mask] == labels[mask]  # 获取正确标记的节点
    accuracy = int(correct.sum()) / int(mask.sum())  # 计算正确率


    # 计算F1 AUC
    # 计算F1 AUC
    # 将log_softmax输出转换为正常概率
    probabilities = torch.exp(out)
    # 准备数据用于计算F1-Score和AUC
    pred_np = pred[mask].cpu().numpy()
    labels_np = labels[mask].cpu().numpy()
    # 提取正类（例如，第1类）的概率 也就是违约为1的概率

    # 计算F1-Score和AUC
    f1 = f1_score(labels_np, pred_np,average='macro')


    # 计算mcc
    # 计算MCC
    mcc = matthews_corrcoef(labels_np, pred_np)

    # 计算G-Mean
    TP = ((pred_np == 1) & (labels_np == 1)).sum()
    TN = ((pred_np == 0) & (labels_np == 0)).sum()
    FP = ((pred_np == 1) & (labels_np == 0)).sum()
    FN = ((pred_np == 0) & (labels_np == 1)).sum()

    sensitivity = TP / (TP + FN)  # 等同于召回率
    specificity = TN / (TN + FP)
    g_mean = np.sqrt(sensitivity * specificity)
    return accuracy, f1,  mcc, g_mean

def train_step(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()  # 梯度置零
    out = model(data.x.float(), data.edge_index,data.edge_weight)  # 模型前向传播
    train_loss = criterion(out[data.train_mask], data.y.float()[data.train_mask])

    train_loss.backward()  # 反向传播
    optimizer.step()  # 优化器梯度下降

    train_accuracy, train_F1, train_mcc,train_gmean = masked_accuracy(out, data.y, data.train_mask)
    return train_loss, train_accuracy, train_F1, train_mcc,train_gmean

def test(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x.float(), data.edge_index,data.edge_weight)
    test_accuracy, test_F1, test_mcc,test_gmean = masked_accuracy(out, data.y, mask)

    return test_accuracy, test_F1, test_mcc,test_gmean,out



def train_model(model, data, optimizer, criterion,path, epochs):
    train_losses = []
    train_accuracies = []
    train_F1_scores = []
    train_AUC_scores = []
    train_mccs = []
    train_gmeans = []



    val_accuracies = []
    val_F1_scores = []
    val_AUC_scores = []
    val_mccs = []
    val_gmeans = []


    model.train()
    for ep in range(epochs + 1):
        train_loss, train_accuracy, train_F1, train_mcc,train_gmean = train_step(model, data, optimizer, criterion)
        val_accuracy, val_f1, val_mcc,val_gmean ,out= test(model, data, data.val_mask)

        train_losses.append(train_loss.item())
        train_accuracies.append(train_accuracy)
        train_F1_scores.append(train_F1)
        train_mccs.append(train_mcc)
        train_gmeans.append(train_gmean)



        val_accuracies.append(val_accuracy)
        val_F1_scores.append(val_f1)

        val_mccs.append(val_mcc)
        val_gmeans.append(val_gmean)
        print(
            "Epoch {}/{}, Train_Loss: {:.4f}, Train_Accuracy: {:.4f},train_F1: {:.4f}, train_mcc: {:.4f},train_geman: {:.4f}, Val_Accuracy: {:.4f},Val_F1: {:.4f},Val_mcc: {:.4f},Val_gmean: {:.4f}"
            .format(ep + 1, epochs, train_loss.item(), train_accuracy, train_F1, train_mcc,train_gmean,val_accuracy, val_f1,
                   val_mcc,val_gmean))
        output = "Epoch {}/{}, Train_Loss: {:.4f}, Train_Accuracy: {:.4f},train_F1: {:.4f}, train_mcc: {:.4f},train_geman: {:.4f}, Val_Accuracy: {:.4f},Val_F1: {:.4f},Val_mcc: {:.4f},Val_gmean: {:.4f}\n".format(
            ep + 1, epochs, train_loss.item(), train_accuracy, train_F1,  train_mcc, train_gmean,val_accuracy, val_f1,
            val_mcc, val_gmean)
        # 创建完整的文件路径
        file_name = os.path.join(path, 'output.txt')
        with open(file_name, "a") as file:
            file.write(output )

    return train_losses, train_accuracies, val_accuracies,train_F1_scores,val_F1_scores,train_mccs,train_gmeans,val_mccs,val_gmeans

def evaluate_model(model, data):
    test_accuracy, test_F1,test_mcc,test_gmean,out = test(model, data, data.test_mask)


    print(f'该模型的测试集准确率Test Accuracy: {test_accuracy:.4f}')
    print(f'该模型的测试集F1test_F1: {test_F1:.4f}')
    print(f'该模型的测试集auctest_mcc: {test_mcc:.4f}')
    print(f'该模型的测试集auctest_gmean: {test_gmean:.4f}')

    pred = out>0.5
    pred = pred.int()
    print('评估为1')
    print(pred)
    print('y',data.y)



    return test_accuracy, test_F1, test_mcc,test_gmean

def save_and_visualize_results(train_losses, train_accuracies, val_accuracies, train_mccs,train_gmeans,val_mccs,val_gmeans,path):
    # 保存到文件

    if not os.path.exists(path):
        os.makedirs(path)

        # 保存训练损失图像
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    loss_path = os.path.join(path, 'training_loss.png')
    plt.savefig(loss_path)
    print(f"Training Loss Graph saved to: {loss_path}")
    plt.show()
    plt.close()

    # 保存训练准确率图像
    plt.figure()
    plt.plot(train_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    acc_path = os.path.join(path, 'training_accuracy.png')
    plt.savefig(acc_path)
    print(f"Training Accuracy Graph saved to: {acc_path}")
    plt.show()
    plt.close()

    # 保存验证准确率图像
    plt.figure()
    plt.plot(val_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    val_acc_path = os.path.join(path, 'validation_accuracy.png')
    plt.savefig(val_acc_path)
    print(f"Validation Accuracy Graph saved to: {val_acc_path}")
    plt.show()
    plt.close()

    # 保存训练MCC图像
    plt.figure()
    plt.plot(train_mccs)
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.title('Training MCC')
    train_mcc_path = os.path.join(path, 'training_mcc.png')
    plt.savefig(train_mcc_path)
    print(f"Training MCC Graph saved to: {train_mcc_path}")
    plt.show()
    plt.close()

    # 保存验证MCC图像
    plt.figure()
    plt.plot(val_mccs)
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.title('Validation MCC')
    val_mcc_path = os.path.join(path, 'validation_mcc.png')
    plt.savefig(val_mcc_path)
    print(f"Validation MCC Graph saved to: {val_mcc_path}")
    plt.show()
    plt.close()


    # 保存训练G-Means图像
    plt.figure()
    plt.plot(train_gmeans)
    plt.xlabel('Epoch')
    plt.ylabel('G-Means')
    plt.title('Training G-Means')
    train_gmeans_path = os.path.join(path, 'training_gmeans.png')
    plt.savefig(train_gmeans_path)
    print(f"Training G-Means Graph saved to: {train_gmeans_path}")
    plt.show()
    plt.close()

    # 保存验证G-Means图像
    plt.figure()
    plt.plot(val_gmeans)
    plt.xlabel('Epoch')
    plt.ylabel('G-Means')
    plt.title('Validation G-Means')
    val_gmeans_path = os.path.join(path, 'validation_gmeans.png')
    plt.savefig(val_gmeans_path)
    print(f"Validation G-Means Graph saved to: {val_gmeans_path}")
    plt.show()
    plt.close()



def main():
    # # 加载数据集

    # epochs = 300
    # # 'dataset/20222/train0.75_val0.15/processed/BankingNetwork.dataset'
    # # dataset = torch.load(f'dataset/{year}/{ratio}/processed/raw_data.dataset')
    # dataset = torch.load(f'dataset/20222/train0.75_val0.15/processed/BankingNetwork.dataset')
    # data = dataset[type]  # 假设数据集只有一个图
    # load_data(data)
    #



    # # 创建实验结果保存路径

    # # 检查目录是否存在，如果不存在则创建
    # # raw是原始[0]  max_min[1]  cdf[2]   type = '阈值'或者'阈值拆借'

    year = "2020"
    country = "Austria"
    # 有阈值 和阈值拆借 阈值对应 max_min  cdf
    # type 0 原始数据 1是max_min cdf是2

    type = 2
    # "train0.08_val0.02_test0.9"
    # "train0.6_val0.15_test0.25"
    # train0.32_val0.08_test0.6
    ratio="train0.32_val0.08_test0.6"
    dataset = torch.load(f'foreign_dataset/{country}/{year}/{ratio}/processed/BankingNetwork.dataset')  # 96%F1

    # dataset = torch.load('dataset/2021/train0.7_val0.15/processed/raw_data.dataset')
    data = dataset[type]  # 该数据集只有一个图len(dataset)：1
    model = GraphGCN(in_channels=data.num_features, data=data)
    # 定义优化器 损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=5e-5)

    criterion = torch.nn.BCELoss()
    if type == 0:
        name = 'raw'
    elif type == 1:
        name = 'max_min'  # 如果 type 是 1，设置相应的 name 值
    elif type == 2:
        name = 'cdf'  # 如果 t  t ype 是 2，设置相应的 name 值
    else:
        print("数据有问题")
    epochs = 200

    time_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory = os.path.join("Probability_foreign_results", country,year, name, ratio, time_dir)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # 训练模型
    train_losses, train_accuracies, val_accuracies,train_F1_scores,val_F1_scores,train_mccs,train_gmeans,val_mccs,val_gmeans = train_model(model, data, optimizer, criterion,  output_directory,epochs)
    # 测试模型
    test_accuracy, test_F1,test_mcc,test_gmean= evaluate_model(model, data)

    # 测试结果文件路径
    text_file_path = os.path.join(output_directory, 'test_results.txt')

    # 将测试结果写入文本文件

    # 将测试结果和额外的信息写入文本文件
    with open(text_file_path, 'w') as file:
        file.write("Experiment Details\n")
        file.write("==================\n")
        file.write(f"Year: {year}\n")
        file.write(f"Type: {name}\n")
        file.write(f"Ratio: {ratio}\n")
        file.write(f"Learning Rate: {optimizer.param_groups[0]['lr']}\n")
        file.write(f"Epochs: {epochs}\n")
        file.write("\n")
        file.write('===================Data Statistics=====================\n')
        file.write(f"总结点数: {data.num_nodes}\n")
        file.write(f"总边数: {data.edge_index.shape}")
        file.write(f"training samples, {torch.sum(data.train_mask).item()}\n")
        file.write(f"训练节点比例: {int(data.train_mask.sum()) / data.num_nodes:.2f}\n")
        file.write(f"validation samples, {torch.sum(data.val_mask).item()}\n")
        file.write(f"验证节点比例: {int(data.val_mask.sum()) / data.num_nodes:.2f}\n")
        file.write(f"test samples, {torch.sum(data.test_mask).item()}\n")
        file.write(f"测试节点比例: {int(data.test_mask.sum()) / data.num_nodes:.2f}\n")
        file.write(f"Number of features: {data.num_features}\n")

        file.write(f"y 形状, {data.y.shape}\n")

        file.write("Test Results\n")
        file.write("============\n")
        file.write(f"Test Accuracy: {test_accuracy}\n")
        file.write(f"Test F1 Score: {test_F1}\n")
        file.write(f"Test MCC: {test_mcc}\n")
        file.write(f"Test G-Mean: {test_gmean}\n")

    print(f"Test results saved to: {text_file_path}")
    # 将结果保存到 DataFrame
    results_df = pd.DataFrame({
        'Train Loss': train_losses,
        'Train Accuracy': train_accuracies,
        'Validation Accuracy': val_accuracies,
        'Train F1 Score': train_F1_scores,

        'Train mcc': train_mccs,
        'Train gmean': train_gmeans,
        'Validation F1 Score': val_F1_scores,

        'Validation mcc': val_mccs,
        'Validation gmean':val_gmeans,

        'test Accuracy':  test_accuracy,
        'test F1 Score': test_F1,

        'test mcc': test_mcc,
        'test gmean': test_gmean,

    })

    # 保存到 CSV 文件
    csv_path = os.path.join(output_directory, 'training_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    save_and_visualize_results(train_losses, train_accuracies, val_accuracies, train_mccs,train_gmeans,val_mccs,val_gmeans,output_directory )

main()

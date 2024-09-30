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
from GCN.model.RP_model import *
import torch.nn.functional as F


def load_data(data):



    print('===================Data Statistics=====================')

    print("training samples", torch.sum(data.train_mask).item())
    print("validation samples", torch.sum(data.val_mask).item())
    print("test samples", torch.sum(data.test_mask).item())
    print(f'Number of features: {data.num_features}')
    print(f'nodes nums: {data.num_nodes}')



def define_model(data):
    model = GraphGCN(in_channels=data.num_features, data=data)
    print('===================model=====================')
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
    pred = out.argmax(dim=1)
    correct = pred[mask] == labels[mask]
    accuracy = int(correct.sum()) / int(mask.sum())
    pred_np = pred[mask].cpu().numpy()
    labels_np = labels[mask].cpu().numpy()
    f1 = f1_score(labels_np, pred_np,average='macro')
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

    train_loss = masked_loss(out, data.y, data.train_mask, criterion)   #损失
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



def train_model(model, data, optimizer, criterion,path,    best_model_path,epochs):
    train_losses = []
    train_accuracies = []
    train_F1_scores = []
    train_mccs = []
    train_gmeans = []
    val_accuracies = []
    val_F1_scores = []
    val_mccs = []
    val_gmeans = []

    best_val_mcc = 0.0  # 最佳验证准确率初始化
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

        # 保存验证准确率最高的模型
        if val_mcc >=best_val_mcc:
            best_val_mcc = val_mcc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with accuracy: {best_val_mcc:.4f}")
        output = "Epoch {}/{}, Train_Loss: {:.4f}, Train_Accuracy: {:.4f},train_F1: {:.4f}, train_mcc: {:.4f},train_geman: {:.4f}, Val_Accuracy: {:.4f},Val_F1: {:.4f},Val_mcc: {:.4f},Val_gmean: {:.4f}\n".format(
            ep + 1, epochs, train_loss.item(), train_accuracy, train_F1,  train_mcc, train_gmean,val_accuracy, val_f1,
            val_mcc, val_gmean)
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
    pred = out.argmax(dim=1)

    print('out', pred[data.test_mask])  # 使用最大概率的类别作为预测结果)
    print('y',data.y[data.test_mask])


    return test_accuracy, test_F1, test_mcc,test_gmean,out

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



def train(country,year,type,ratio,dataset,epochs):


    data = dataset[type]  # 该数据集只有一个图len(dataset)：1
    model = GraphGCN(in_channels=data.num_features, data=data)
    # 定义优化器 损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=5e-5)

    criterion = torch.nn.CrossEntropyLoss()
    if type == 0:
        name = 'raw'
    elif type == 1:
        name = 'max_min'  # 如果 type 是 1，设置相应的 name 值
    elif type == 2:
        name = 'cdf'  # 如果 t  t ype 是 2，设置相应的 name 值
    else:
        print("数据有问题")


    time_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory = os.path.join("foreign_results", country,year, name, ratio, time_dir)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    #
    # 基本路径 模型
    path = 'save_models'

    # 根据变量创建路径
    model_directory = os.path.join(path, country, year, ratio)

    # 如果目录不存在，则创建它
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    # 获取当前时间并格式化为字符串
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 设置模型保存路径，包括时间戳
    best_model_path = os.path.join(model_directory, f"best_model.pth")
    # 训练模型
    train_losses, train_accuracies, val_accuracies,train_F1_scores,val_F1_scores,train_mccs,train_gmeans,val_mccs,val_gmeans = train_model(model, data, optimizer, criterion,  output_directory,   best_model_path,epochs)
    # 测试模型
    test_accuracy, test_F1,test_mcc,test_gmean,out= evaluate_model(model, data)
    probabilities = F.softmax(out[data.test_mask], dim=1)[:, 1].numpy()  # 应用Softmax并取分类为1的概率

    output_directory1 = os.path.join("foreign_results", country, year, name, ratio)
    csv_path = os.path.join(output_directory1, '训练完测试test_prob.csv')

    # 保存为CSV
    df = pd.DataFrame(probabilities, columns=['Probability_Class_1'])
    df.to_csv(csv_path, index=False)
    print(f"Probabilities saved to: {csv_path}")

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

def test_model(country,year,type,ratio,dataset):
    data = dataset[type]  # 该数据集只有一个图len(dataset)：1
    model = GraphGCN(in_channels=data.num_features, data=data)
    if type == 0:
        name = 'raw'
    elif type == 1:
        name = 'max_min'  # 如果 type 是 1，设置相应的 name 值
    elif type == 2:
        name = 'cdf'  # 如果 type 是 2，设置相应的 name 值
    else:
        print("数据有问题")
    # 模型保存路径
    model_path = f'save_models/{country}/{year}/{ratio}/best_model.pth'

    # 加载模型状态字典
    model.load_state_dict(torch.load(model_path))
    with torch.no_grad():
        # 测试模型
        test_accuracy, test_F1, test_mcc, test_gmean, out = evaluate_model(model, data)
        probabilities = F.softmax(out[data.test_mask], dim=1)[:, 1].numpy()  # 应用Softmax并取分类为1的概率

        # 准备保存概率的路径
    output_directory = f'foreign_results/{country}/{year}/cdf/{ratio}'
    os.makedirs(output_directory, exist_ok=True)  # 创建目录，如果不存在
    csv_path = os.path.join(output_directory, '最优trainmcctest_prob.csv')

    # 保存为CSV
    df = pd.DataFrame(probabilities, columns=['Probability_Class_1'])
    df.to_csv(csv_path, index=False)
    print(f"Probabilities saved to: {csv_path}")

if __name__ == '__main__':
   mode = "train";

   year = "2022"
   country = "China"
   # 有阈值 和阈值拆借 阈值对应 max_min  cdf
   # type 0 原始数据 1是max_min cdf是2
   type = 2
   # "train0.08_val0.02_test0.9"
   # "train0.6_val0.15_test0.25"
   # train0.32_val0.08_test0.6
   ratio = "train0.6_val0.15_test0.25"
   dataset = torch.load(f'foreign_dataset/{country}/{year}/{ratio}/processed/BankingNetwork.dataset')  # 96%F1

   if(mode=="train"):
       print("开始训练")
       epochs = 200
       train(country,year,type,ratio,dataset,epochs)
   if(mode=="test"):
       print("开始测试")
       test_model(country,year,type,ratio,dataset)


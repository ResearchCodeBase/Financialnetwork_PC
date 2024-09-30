import time
from datetime import datetime
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
import numpy as np
import torch_geometric.transforms as T
import torch.nn
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
from GCN.model import *
from GCN.model.FinaModel import GraphGCN
'比train1加了随机分割, 比train2加了测试集验证集的作用'
'4,5,9,11,13   28 29 31 33'
"31,28能明显识别"
'1.加载数据集'

'比train182是 训练完一个epoch再测试 比(1)增加了 F1-SCOR aOC'
# 加载数据集
dataset = torch.load('dataset/2015/train0.6_val0.15_test0.25/processed/BankingNetwork.dataset')  #96%

# dataset = torch.load('dataset/2021/train0.7_val0.15/processed/raw_data.dataset')
data = dataset[1] #该数据集只有一个图len(dataset)：1

data_x_np = data.x.numpy()

# 将数组保存到 txt 文件
np.savetxt('2022.txt', data_x_np, fmt='%f')
# data_edge_weight_np = data.edge_weight.numpy()
# np.savetxt('data_edgeweight1.txt', data_edge_weight_np, fmt='%f')
# 20%的节点为测试集 80%的节点为训练集"
# split = T.RandomNodeSplit(split="train_rest",num_test=0.3,num_val=0)
# 划分训练集 测试集,默认key为y，根据key执行节点级分割
# split = T.RandomNodeSplit(split="train_rest", num_splits = 1, num_val = 0.3, num_test= 0.6)
# split = T.RandomNodeSplit(split="train_rest",num_test=0.4,num_val=0.3)
#split = T.RandomNodeSplit(split="train_rest",num_test=0.15,num_val=0.05,num_test_positive = )
# Apply the transformation to the graph data

print('===================Data Statistics=====================')
print('data.train_mask',data.train_mask)
print('data.test_mask',data.test_mask)
print("training samples",torch.sum(data.train_mask).item())
print("validation samples",torch.sum(data.val_mask ).item())
print("test samples",torch.sum(data.test_mask ).item())
print("train",data.y[data.train_mask])
print("edge_weight",data.edge_weight)
m=data.edge_weight
# '统计样本分布'
# class_counts = torch.bincount(data.y)
# print('class_counts',class_counts)
# # 计算每个类别的权重
# class_weights = 1.0 / class_counts.float()
# # 标准化权重，使其和为1
# class_weights /= torch.sum(class_weights)
#
# print('class_weights',class_weights)

'2.定义模型'
model = GraphGCN(in_channels= data.num_features ,data = data)
print('网络结构', model)
for name, parameter in model.named_parameters():
    print('**************************')
    print(f'{name}: {parameter.shape}')
    print(f'{name},{parameter}')
print('********')
# 打印每一层的神经元结构
# 网络结构 GraphGCN(
#   (conv1): GCNConv(14, 8)
#   (conv2): GCNConv(8, 2)
# )
# conv1.lin.weight: torch.Size([8, 14])
# class_weight =
'3.定义优化器、损失函数'
# 将该模型中可优化的参数model.parameters()注册到优化器中，lr为学习率，weight_decay为学习率衰减系数。之后在训练过程中利用optimizer.step()更新参数。
# 神经元个数设置为8 学习率为0.009是最合适的
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-5)
# 我们选择交叉熵(CrossEntropy)作为loss function，其他loss function见torch.nn-loss function
# 因为CrossEntropyLoss()中做了softmax相关操作，所以在设计网络时，只需要直接输出节点的特征
# criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
criterion = torch.nn.CrossEntropyLoss()

'''4.训练模型
model.train()开启模型的训练模式，由于数据集较小，没有分batch训练（直接作为1个batch），经：
1.梯度置零
2.模型前向传播

3.计算loss
4.反向传播
5.优化器梯度下降'''
def test(model,data,mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x.float(), data.edge_index,data.edge_weight)
    #计算accuracy
    pred = out.argmax(dim=1)  # 使用最大概率的类别作为预测结果
    test_correct = pred[mask] == data.y[mask]  # 获取正确标记的节点
    test_acc = int(test_correct.sum()) / int(mask.sum()) # 计算正确率
    print('pred',pred)
    print('y',data.y)
    # 计算F1 AUC
    # 将log_softmax输出转换为正常概率
    probabilities = torch.exp(out)

    # 准备数据用于计算F1-Score和AUC
    pred_np = pred[mask].cpu().numpy()

    labels_np = data.y[mask].cpu().numpy()

    # 提取正类（例如，第1类）的概率 也就是违约为1的概率
    positive_probabilities_np = probabilities[mask][:, 1].cpu().numpy()

    # 计算F1-Score和AUC micro macro
    f1 = f1_score(labels_np, pred_np,average='macro')

    print('f1',f1)

    auc = roc_auc_score(labels_np, positive_probabilities_np)

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

    return test_acc,f1,auc,mcc,g_mean
def masked_accuracy(out,labels,mask):
    # # 使用最大概率的类别作为预测结果
    # 计算准确率
    pred =out.argmax(dim=1)
    correct= pred[mask] == data.y[mask]  # 获取正确标记的节点
    accuracy= int(correct.sum()) / int(mask.sum()) # 计算正确率

    # 计算F1 AUC
    # 将log_softmax输出转换为正常概率
    probabilities = torch.exp(out)

    # 准备数据用于计算F1-Score和AUC
    pred_np = pred[mask].cpu().numpy()
    labels_np = data.y[mask].cpu().numpy()
    # 提取正类（例如，第1类）的概率 也就是违约为1的概率
    positive_probabilities_np = probabilities[mask][:, 1].cpu().detach().numpy()
    # 计算F1-Score和AUC
    f1 = f1_score(labels_np, pred_np, average='micro')
    auc = roc_auc_score(labels_np, positive_probabilities_np)

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
    return accuracy,f1,auc,mcc,g_mean

def masked_loss(out,labels,mask):
    loss = criterion(out[mask], labels[mask])  # 计算loss
    return (loss)


def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()  # 梯度置零
    out = model(data.x.float(), data.edge_index,data.edge_weight)  # 模型前向传播
    train_loss= masked_loss(out=out,labels=data.y,mask=data.train_mask)

    train_loss.backward()  # 反向传播
    optimizer.step()  # 优化器梯度下降


    train_accuracy,train_F1,train_AUC,mcc,g_mean  = masked_accuracy(out=out,labels=data.y,mask=data.train_mask)


    return train_loss,train_accuracy,train_F1,train_AUC,mcc,g_mean

# 定义空列表来保存损失和准确率
losses = []
accuracies = []
train_losses=[]
train_accuracys=[]



val_losses=[]
val_accuracys=[]

test_losses=[]
test_accuracys=[]
# 训练
best_accuracy =0
epochs=200
# 获取当前时间
current_time = datetime.now()

# 格式化时间字符串
time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")

# 构建文件名
file_name = "output_{}.txt".format(time_string)
for ep in range(epochs+1):
    train_loss, train_accuracy,train_F1,train_AUC,mcc,gmean= train(model, data, optimizer, criterion)

    val_accuracy,val_f1,val_auc,val_mcc,vall_g_mean = test(model,data,data.val_mask)
    print("Epoch {}/{}, Train_Loss: {:.4f}, Train_Accuracy: {:.4f},train_F1: {:.4f},train_AUC: {:.4f}, train_mcc: {:.4f},train_gmean: {:.4f}, Val_Accuracy: {:.4f},Val_F1: {:.4f},Val_AUC: {:.4f},Val_mcc: {:.4f},Val_g-mean: {:.4f}"
              .format(ep + 1, epochs, train_loss.item(), train_accuracy,train_F1,train_AUC, mcc,gmean,val_accuracy,val_f1,val_auc,val_mcc,vall_g_mean))
    if(val_accuracy>best_accuracy):
        best_accuracy=val_accuracy
    # output = "Epoch {}/{}, Train_Loss: {:.4f}, Train_Accuracy: {:.4f}, Val_Accuracy: {:.4f}, Val_F1: {:.4f},Val_AUC: {:.4f},Test_Accuracy: {:.4f}\n".format(ep + 1, epochs, train_loss.item(), train_accuracy, val_accuracy, val_f1,val_auc,test_accuracy)
    #     # 构建文件名
    output = "Epoch {}/{},Train_Loss: {:.4f},Train_Accuracy: {:.4f},train_F1: {:.4f},train_AUC: {:.4f}, Val_Accuracy: {:.4f},Val_F1: {:.4f},Val_AUC: {:.4f}\n".format(
        ep + 1, epochs,train_loss.item(),train_accuracy,train_F1,train_AUC,val_accuracy,val_f1,val_auc)
    # 构建文件名
    file_name = "output_{}.txt".format(time_string)

        # 打开文件，以追加模式写入内容
    with open(file_name, "a") as file:
        file.write(output)

    # 保存损失和准确率
    train_losses.append(train_loss.item())
    train_accuracys.append(train_accuracy)
    val_accuracys.append(val_accuracy)




#精度评价
test_acc,test_F1,test_AUC,mcc,g_mean = test(model,data,data.test_mask)
print(f'该模型的测试集准确率Test Accuracy: {test_acc:.4f}')
print(f'该模型的测试集F1test_F1: {test_F1:.4f}')
print(f'该模型的测试集auctest_AUC: {test_AUC:.4f}')
print(f'该模型的测试集mcc: {mcc:.4f}')
print(f'该模型的测试集g-mean: {g_mean :.4f}')
#


# 打开文件，以追加模式写入内容
with open(file_name, "a") as file:


    file.write('bestval_accuracy: {}\n'.format(best_accuracy))
    file.write('Test_accuracy: {}\n'.format(test_acc))
    file.write('test_F1: {}\n'.format(test_F1))
    file.write('test_AUC: {}\n'.format(test_AUC))

print("文本保存成功！文件名为：{}".format(file_name))












plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 3, 2)
plt.plot(train_accuracys)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')

plt.subplot(1, 3, 3)
plt.plot(val_accuracys)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('val Accuracy')

plt.tight_layout()
plt.show()
print("绘制成功")

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def train_and_evaluate_knn(data, n_neighbors=5):
    # 初始化KNN模型
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    # 训练模型
    train_x = data.x[data.train_mask].float().numpy()
    train_y = data.y[data.train_mask].numpy()
    model.fit(train_x, train_y)

    # 在测试集上做预测
    test_x = data.x[data.test_mask].float().numpy()
    test_y = data.y[data.test_mask].numpy()
    print("KNN")


    # 类别预测
    test_pred = model.predict(test_x)
    print(test_pred)

    # 预测为类别1的概率
    test_proba = model.predict_proba(test_x)[:, 1]  # 预测为类别1的概率

    # 计算性能指标
    test_acc = accuracy_score(test_y, test_pred)
    test_f1 = f1_score(test_y, test_pred,average='macro')  # 计算F1得分

    # 第一个参数是y_true，样本的真实标签，形状（样本数，）
    # 第二个参数y_score：预测为1的概率值，形状（样本数，）

    mcc = matthews_corrcoef(test_y, test_pred)

    # 计算G-Mean
    TP = np.sum((test_pred == 1) & (test_y == 1))
    TN = np.sum((test_pred == 0) & (test_y == 0))
    FP = np.sum((test_pred == 1) & (test_y == 0))
    FN = np.sum((test_pred == 0) & (test_y == 1))

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    g_mean = np.sqrt(sensitivity * specificity)

    return test_acc,  test_f1, mcc, g_mean


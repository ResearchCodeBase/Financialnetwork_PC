


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def train_and_evaluate_rf(data):
    # 分离出训练和测试数据
    train_X = data.x[data.train_mask].float().numpy()
    train_y = data.y[data.train_mask].numpy()
    test_X = data.x[data.test_mask].float().numpy()
    test_y = data.y[data.test_mask].numpy()

    # 训练随机森林回归模型
    model = RandomForestClassifier(n_estimators=2)
    model.fit(train_X, train_y)

    # 在测试集上做预测
    test_pred = model.predict(test_X)
    print("RF")
    print(test_pred)

    # 计算性能指标
    test_acc = accuracy_score(test_y, test_pred)

    test_f1 = f1_score(test_y, test_pred,average='macro')  # 计算F1得分
    test_mcc = matthews_corrcoef(test_y, test_pred)

    # 计算G-Mean
    TP = sum((test_pred == 1) & (test_y == 1))
    TN = sum((test_pred == 0) & (test_y == 0))
    FP = sum((test_pred == 1) & (test_y == 0))
    FN = sum((test_pred == 0) & (test_y == 1))
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    g_mean = np.sqrt(sensitivity * specificity)

    return test_acc, test_f1, test_mcc, g_mean
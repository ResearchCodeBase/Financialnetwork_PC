from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def train_and_evaluate_knn(data, n_neighbors=5):
    # ��ʼ��KNNģ��
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    # ѵ��ģ��
    train_x = data.x[data.train_mask].float().numpy()
    train_y = data.y[data.train_mask].numpy()
    model.fit(train_x, train_y)

    # �ڲ��Լ�����Ԥ��
    test_x = data.x[data.test_mask].float().numpy()
    test_y = data.y[data.test_mask].numpy()
    print("KNN")


    # ���Ԥ��
    test_pred = model.predict(test_x)
    print(test_pred)

    # Ԥ��Ϊ���1�ĸ���
    test_proba = model.predict_proba(test_x)[:, 1]  # Ԥ��Ϊ���1�ĸ���

    # ��������ָ��
    test_acc = accuracy_score(test_y, test_pred)
    test_f1 = f1_score(test_y, test_pred,average='macro')  # ����F1�÷�

    # ��һ��������y_true����������ʵ��ǩ����״������������
    # �ڶ�������y_score��Ԥ��Ϊ1�ĸ���ֵ����״������������

    mcc = matthews_corrcoef(test_y, test_pred)

    # ����G-Mean
    TP = np.sum((test_pred == 1) & (test_y == 1))
    TN = np.sum((test_pred == 0) & (test_y == 0))
    FP = np.sum((test_pred == 1) & (test_y == 0))
    FN = np.sum((test_pred == 0) & (test_y == 1))

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    g_mean = np.sqrt(sensitivity * specificity)

    return test_acc,  test_f1, mcc, g_mean


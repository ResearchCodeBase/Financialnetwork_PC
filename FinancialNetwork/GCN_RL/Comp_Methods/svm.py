import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize

def train_and_evaluate_svm(data):
    # �����ѵ���Ͳ�������
    train_X = data.x[data.train_mask].float().numpy()
    train_y = data.y[data.train_mask].numpy()
    test_X = data.x[data.test_mask].float().numpy()
    test_y = data.y[data.test_mask].numpy()

    # ѵ��SVMģ��
    model = SVC(kernel='linear')  # ʹ�����Ժ�
    model.fit(train_X, train_y)

    # �ڲ��Լ�����Ԥ��
    test_pred = model.predict(test_X)
    print("SVM")
    print(test_pred)


    # ��������ָ��
    test_acc = accuracy_score(test_y, test_pred)
    test_f1 = f1_score(test_y, test_pred, average='macro')  # ����F1�÷�
    test_mcc = matthews_corrcoef(test_y, test_pred)

    # ����G-Mean
    TP = sum((test_pred == 1) & (test_y == 1))
    TN = sum((test_pred == 0) & (test_y == 0))
    FP = sum((test_pred == 1) & (test_y == 0))
    FN = sum((test_pred == 0) & (test_y == 1))
    sensitivity = TP / (TP + FN) if TP + FN else 0
    specificity = TN / (TN + FP) if TN + FP else 0
    g_mean = np.sqrt(sensitivity * specificity) if (sensitivity * specificity) != 0 else 0

    return test_acc, test_f1, test_mcc, g_mean

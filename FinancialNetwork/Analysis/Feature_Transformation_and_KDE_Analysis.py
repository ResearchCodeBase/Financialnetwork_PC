# coding=gbk
import os

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from scipy.stats import yeojohnson
from matplotlib.font_manager import FontProperties
from scipy.stats import yeojohnson
import numpy as np

def yeojohnson_transform(features):
    """
    ��������������Yeo-Johnson�任��������쳣����ԭʼ����
    :param features: ��ά�������� (ÿһ����һ������)
    :return: �任�����������
    """
    transformed_features = features.copy()  # ����ԭʼ�����ĸ���
    # features.shape[1] ��ʾ����������
    for i in range(features.shape[1]):
        try:
            # ���Զ�����Ӧ��Yeo-Johnson�任
            transformed_features[:, i], _ = yeojohnson(features[:, i])
        except Exception as e:
            # ��������쳣����ӡ������Ϣ������ԭʼ����
            print(f"An error occurred while transforming feature column {i}: {e}. Original data is kept.")
            # �Ѿ���ԭʼ���ݵĸ��������Բ���Ҫ�ٴθ�ֵ
    return transformed_features

def save_yeojohnsonfeatures_to_csv(saved_features_file,features, directory_name,filename_prefix="features",column_names=None):
    """
    YEOת��
     ��ת�����������ԭʼ������ĳЩ�б��浽CSV�ļ�
     :param saved_features_file: ����ԭʼ������CSV�ļ�·��
     :param features: ת�������������
     :param directory_name: �����ļ���Ŀ¼��
     :param filename_prefix: �ļ���ǰ׺
     :param column_names: ת�������������
     """
    # ����ԭʼ�����ļ�
    original_df = pd.read_csv(saved_features_file)

    # ��ȡ "Basic_Default" ��
    basic_default_col = original_df[["Basic_Default"]]
    # ��ȡ "Basic_Default" ��
    y = original_df[["y"]]

    # ����ת����������DataFrame
    if column_names is None:
        transformed_df = pd.DataFrame(features)
    else:
        transformed_df = pd.DataFrame(features, columns=column_names)

    # �� "Basic_Default" �� "y" ����ӵ�ת����������DataFrame��
    combined_df = pd.concat([transformed_df, basic_default_col, y], axis=1)

    # ���浽CSV�ļ�
    filename = f"{directory_name}/{filename_prefix}.csv"
    combined_df.to_csv(filename, index=False)
    print(f"Yeo��������ѱ���ΪCSV�ļ���·����: {filename}")
    return filename

def cdf_scale(feature):
    """
    ����������CDF����
    :param feature: һά��������
    :return: ���ź����������
    """
    sorted_feature = np.sort(feature)
    cdf = np.arange(1, len(sorted_feature) + 1) / len(sorted_feature)
    cdf_mapping = dict(zip(sorted_feature, cdf))
    scaled_feature = np.array([cdf_mapping[val] for val in feature])
    return scaled_feature

def kde_difference(feature, labels,feature_name,save_path):
    """
    ʹ�ú��ܶȹ��ƣ�KDE�����������ڲ�ͬ����Ĳ���
    :param feature: һά��������
    :param labels: ��ǩ����
    :return: KDE�������
    """
    # ����������������ֵ
    # ����� labels == 0 ��һ����������������� labels �����е�ÿ��Ԫ���Ƿ����0���������������һ���������飬���е�ÿ��Ԫ�ر�ʾ labels ����Ӧλ�õ�Ԫ���Ƿ��������������Ƿ����0����
    # feature[labels == 0] ��������������������� feature ��ѡ����Щ���Ӧ��ǩΪ0��Ԫ�ء�
    feature_class_0 = feature[labels == 0]
    feature_class_1 = feature[labels == 1]
    # KDE
    kde_class_0 = gaussian_kde(feature_class_0)
    kde_class_1 = gaussian_kde(feature_class_1)

    values = np.linspace(min(feature), max(feature), 100)
    # kde����
    kde_diff = np.sum(np.abs(kde_class_0(values) - kde_class_1(values)))
    # ��ͼ
    plt.figure(figsize=(12, 6))
    # KDE����

    plt.plot(values, kde_class_0(values), color='blue', label='Class 0')
    plt.plot(values, kde_class_1(values), color='orange', label='Class 1')
    plt.title(f'{feature_name} - KDE Difference: {kde_diff:.2f}')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.legend()

    # ȷ�� 'kde' ��Ŀ¼����
    kde_dir = os.path.join(save_path, 'kde')
    os.makedirs(kde_dir, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'kde', f'{feature_name}.png'))
    plt.tight_layout()
    plt.show()






    return kde_diff

def plot_feature_transformation(original_feature, transformed_feature, feature_name, save_path):
    """
    ���������任ǰ���ֱ��ͼ
    :param feature: ԭʼ��������
    :param transformed_feature: �任�����������
    :param feature_name: ��������
    """
    plt.figure(figsize=(8, 4))

    plt.hist(transformed_feature, bins=30, alpha=0.5, color='green', label='Transformed')
    plt.title(f'{feature_name} - Original vs Transformed')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'{feature_name}.png'))
    plt.close()




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
    对所有特征进行Yeo-Johnson变换，如果有异常则保留原始数据
    :param features: 多维特征数组 (每一列是一个特征)
    :return: 变换后的特征数组
    """
    transformed_features = features.copy()  # 创建原始特征的副本
    # features.shape[1] 表示特征的数量
    for i in range(features.shape[1]):
        try:
            # 尝试对特征应用Yeo-Johnson变换
            transformed_features[:, i], _ = yeojohnson(features[:, i])
        except Exception as e:
            # 如果发生异常，打印错误消息并保留原始特征
            print(f"An error occurred while transforming feature column {i}: {e}. Original data is kept.")
            # 已经是原始数据的副本，所以不需要再次赋值
    return transformed_features

def save_yeojohnsonfeatures_to_csv(saved_features_file,features, directory_name,filename_prefix="features",column_names=None):
    """
    YEO转换
     将转换后的特征和原始特征的某些列保存到CSV文件
     :param saved_features_file: 保存原始特征的CSV文件路径
     :param features: 转换后的特征数组
     :param directory_name: 保存文件的目录名
     :param filename_prefix: 文件名前缀
     :param column_names: 转换后的特征列名
     """
    # 加载原始特征文件
    original_df = pd.read_csv(saved_features_file)

    # 提取 "Basic_Default" 列
    basic_default_col = original_df[["Basic_Default"]]
    # 提取 "Basic_Default" 列
    y = original_df[["y"]]

    # 创建转换后特征的DataFrame
    if column_names is None:
        transformed_df = pd.DataFrame(features)
    else:
        transformed_df = pd.DataFrame(features, columns=column_names)

    # 将 "Basic_Default" 和 "y" 列添加到转换后特征的DataFrame中
    combined_df = pd.concat([transformed_df, basic_default_col, y], axis=1)

    # 保存到CSV文件
    filename = f"{directory_name}/{filename_prefix}.csv"
    combined_df.to_csv(filename, index=False)
    print(f"Yeo特征结果已保存为CSV文件，路径是: {filename}")
    return filename

def cdf_scale(feature):
    """
    对特征进行CDF缩放
    :param feature: 一维特征数组
    :return: 缩放后的特征数组
    """
    sorted_feature = np.sort(feature)
    cdf = np.arange(1, len(sorted_feature) + 1) / len(sorted_feature)
    cdf_mapping = dict(zip(sorted_feature, cdf))
    scaled_feature = np.array([cdf_mapping[val] for val in feature])
    return scaled_feature

def kde_difference(feature, labels,feature_name,save_path):
    """
    使用核密度估计（KDE）计算特征在不同类别间的差异
    :param feature: 一维特征数组
    :param labels: 标签数组
    :return: KDE差异度量
    """
    # 分离两个类别的特征值
    # 这里的 labels == 0 是一个布尔条件，它检查 labels 数组中的每个元素是否等于0。这个条件生成了一个布尔数组，其中的每个元素表示 labels 中相应位置的元素是否满足条件（即是否等于0）。
    # feature[labels == 0] 则利用这个布尔数组来从 feature 中选择那些其对应标签为0的元素。
    feature_class_0 = feature[labels == 0]
    feature_class_1 = feature[labels == 1]
    # KDE
    kde_class_0 = gaussian_kde(feature_class_0)
    kde_class_1 = gaussian_kde(feature_class_1)

    values = np.linspace(min(feature), max(feature), 100)
    # kde差异
    kde_diff = np.sum(np.abs(kde_class_0(values) - kde_class_1(values)))
    # 绘图
    plt.figure(figsize=(12, 6))
    # KDE曲线

    plt.plot(values, kde_class_0(values), color='blue', label='Class 0')
    plt.plot(values, kde_class_1(values), color='orange', label='Class 1')
    plt.title(f'{feature_name} - KDE Difference: {kde_diff:.2f}')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.legend()

    # 确保 'kde' 子目录存在
    kde_dir = os.path.join(save_path, 'kde')
    os.makedirs(kde_dir, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'kde', f'{feature_name}.png'))
    plt.tight_layout()
    plt.show()






    return kde_diff

def plot_feature_transformation(original_feature, transformed_feature, feature_name, save_path):
    """
    绘制特征变换前后的直方图
    :param feature: 原始特征数组
    :param transformed_feature: 变换后的特征数组
    :param feature_name: 特征名称
    """
    plt.figure(figsize=(8, 4))

    plt.hist(transformed_feature, bins=30, alpha=0.5, color='green', label='Transformed')
    plt.title(f'{feature_name} - Original vs Transformed')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'{feature_name}.png'))
    plt.close()




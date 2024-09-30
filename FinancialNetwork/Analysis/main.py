#coding=utf-8


import argparse
import os
from datetime import datetime

from  Bankruptcy_Label_Generation1 import label_main
from data_process2 import process_features

from GraphDataset3 import FeatureBankingNetworkDataset

# 主函数
if __name__ == '__main__':
    # 这里是需要其他银行的
    # 注意这里加载的L 和e 都是没有其他银行的
    # Bankruptcy_Label_Generation1.py 函数需要的参数这里是需要其他银行的
    # 初始化解析器
    parser = argparse.ArgumentParser(description='从银行网络数据分析和处理特征。')
    # 注意这里加载的L 和e 都是有其他银行的
    # 用于Bankruptcy_Label_Generation1.py，包括其他银行的参数

    year = '2020'
    # United Kingdom Switzerland Germany
    country = 'United Kingdom'
    parser = argparse.ArgumentParser(description='Process Banking Network Features')
    # type是阈值拆借偏好 或者阈值 ，也会保存在对应的Dataset下
    type='阈值'
    path=f'data/{country}_{year}/删除孤立节点/{type}'
    parser.add_argument('--L1_path', type=str, default=f'{path}/负债矩阵有其他银行_{year}.txt',help='负债矩阵文件路径（包括其他银行）')
    parser.add_argument('--e1_path', type=str, default=f'{path}/外部资产有其他银行_{year}.txt', help='外部资产文件路径')
    parser.add_argument('--name_path', type=str, default=f'{path}/银行名称_{year}.txt', help='银行名称文件路径')
    parser.add_argument('--alpha', type=float, default=0.5, help='模型权重参数alpha')
    parser.add_argument('--beta', type=float, default=0.8, help='模型权重参数beta')

    # 用于data_process2.py，不包括其他银行的参数 生成节点特征，
    parser.add_argument('--L_path', type=str, default=f'{path}/负债矩阵无其他银行_{year}.txt',help='负债矩阵文件路径（不包括其他银行）')
    parser.add_argument('--e_path', type=str, default=f'{path}/外部资产无其他银行_{year}.txt',help='外部资产文件路径（不包括其他银行）')

    # 用于preData.py，数据集分割比例的参数
    parser.add_argument('--train_ratio', type=float, default=0.32, help='用于训练的数据集比例')
    parser.add_argument('--val_ratio', type=float, default=0.08, help='用于验证的数据集比例')
    parser.add_argument('--test_ratio', type=float, default=0.6, help='用于测试的数据集比例')

    # 用于指定保存数据集的根目录的公共参数
    parser.add_argument('--root', type=str, default=f'../GCN/foreign_dataset/{country}/{year}', help='保存处理后数据集的根目录。')

    # 解析参数
    args = parser.parse_args()
    # 生成违约标签路径 Bankruptcy_Label_Generation1.py
    labelcsv_path = label_main(args.L1_path, args.e1_path, args.name_path, args.alpha, args.beta)
    # 完成特征缩放，生成csv文件 最后一列是标签
    # directory_name = create_timestamped_directory(L_path.split('/')[-1][:4]) 路径会获取负债矩阵的前几个也就是年份
    raw_features_path, normalized_features_path,yeojohnson_transformed_path, cdf_scaled_features_path = process_features(args.L_path,
                                                                                                args.e_path,
                                                                                                labelcsv_path,country)

    # 起止这里，验证正确！！！！！开心，与单独执行Bankruptcy_Label_Generation1.py，再执行  # Bankruptcy_Label_Generation1.py 结果相同
    # 制作图数据集
    # 检查root路径是否已经存在数据集
    # 检查root路径是否存在
    if not os.path.exists(args.root):
        # 如果路径不存在，则创建它及其所有父目录
        os.makedirs(args.root, exist_ok=True)
    # 将训练和验证比例转换为字符串，并用下划线分隔
    train_val_dir = f"train{args.train_ratio}_val{args.val_ratio}_test{args.test_ratio}"
    # 在原始root路径后添加当前时间和训练验证比例目录
    args.root = os.path.join(args.root, train_val_dir)

    # 如果路径不存在，则创建它及其所有父目录
    os.makedirs(args.root, exist_ok=True)
    # 注意这里有两个特征文件呀，因为会生成两个数据集方便后面跑实验对比！！！ 是对滴  raw是原始特征，pro是cdf特征
    dataset = FeatureBankingNetworkDataset(
        root=args.root,
        L_path=args.L_path,
        e_path=args.e_path,
        rawfeature_path= raw_features_path,
        normalized_features_path =  normalized_features_path,
        cdf_scaled_features_path=cdf_scaled_features_path,
        train_size=args.train_ratio,
        val_size=args.val_ratio,
        test_size=args.test_ratio
    )



    # 这里可以添加一个输出语句来确认数据集的保存位置
    print(f'数据集将被保存在 {args.root} 路径下。')

    # 保存或处理dataset...
    print('数据集已成功保存。')

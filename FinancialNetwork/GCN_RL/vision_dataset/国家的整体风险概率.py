# coding=utf-8
import os

import pandas as pd
import torch


def calculate_risk_probability(data, train_mask, val_mask, prob_csv_path):
    # 获取所有的节点及其掩码
    total_nodes = data.num_nodes
    train_val_nodes = [idx for idx, mask in enumerate(train_mask | val_mask) if mask]

    # 计算训练和验证集中的违约银行数量
    train_val_default_count = sum(data.y[train_val_nodes].cpu().numpy())

    # 读取测试节点的违约概率
    probabilities_df = pd.read_csv(prob_csv_path)
    test_probabilities = probabilities_df['Probability_Class_1'].values

    # 计算测试集中的违约银行数量，假设概率超过0.5则为违约
    test_default_count = sum(prob >= 0.5 for prob in test_probabilities)

    # 计算整体风险概率
    overall_default_count = train_val_default_count + test_default_count
    overall_risk_probability = overall_default_count / total_nodes

    return overall_risk_probability



def process_countries(year, countries, ratios, data_type, result_path):
    # 初始化结果列表
    results = []

    for country in countries:
        for ratio in ratios:
            print(country)
            print(ratio)
            dataset_path = f'../foreign_dataset/{country}/{year}/{ratio}/processed/BankingNetwork.dataset'
            dataset = torch.load(dataset_path)
            data = dataset[data_type]
            prob_csv_path = f'../foreign_results/{country}/{year}/cdf/{ratio}/训练完测试test_prob.csv'
            save_path = f'img/{country}/{year}/{ratio}/'

            # 调用风险概率计算函数
            risk_probability = calculate_risk_probability(data, data.train_mask, data.val_mask, prob_csv_path)
            print(f'{country} ({ratio}): Overall Risk Probability: {risk_probability:.4f}')

            # 将结果添加到列表中
            results.append({
                'Country': country,
                'Ratio': ratio,
                'Overall Risk Probability': risk_probability
            })

    # 将结果保存到CSV文件
    results_df = pd.DataFrame(results)
    os.makedirs(result_path, exist_ok=True)
    results_df.to_csv(os.path.join(result_path, 'risk_probabilities.csv'), index=False)
    print(f'Results saved to {os.path.join(result_path, "risk_probabilities.csv")}')


year = "2022"
countries = ["America", "Austria", "China", "France", "Germany", "Italy", "Japan", "United Kingdom"]
ratios = ["train0.6_val0.15_test0.25"]
data_type = 2  # 0: 原始数据, 1: max_min, 2: cdf
result_path = 'results/'

process_countries(year, countries, ratios, data_type, result_path)

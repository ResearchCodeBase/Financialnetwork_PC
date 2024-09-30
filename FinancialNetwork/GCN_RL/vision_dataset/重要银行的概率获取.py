# coding=utf-8
import os
import pandas as pd
import torch

def calculate_bank_probabilities(data, test_mask, prob_csv_path):
    # 获取测试集中节点的掩码索引
    test_mask_indices = [idx for idx, mask in enumerate(test_mask) if mask]
    print(test_mask_indices)

    # 读取测试节点的违约概率及其索引
    probabilities_df = pd.read_csv(prob_csv_path)
    test_probabilities = probabilities_df['Probability_Class_1'].values
    # 格式化测试集中的违约概率，保留四位小数
    formatted_probabilities = [f'{prob:.4f}' for prob in test_probabilities]

    # 打印格式化的违约概率
    print(' '.join(formatted_probabilities))

    # # 计算测试集中每个银行的违约概率
    # bank_probabilities = []
    # for idx, prob in zip(test_indices, test_probabilities):
    #     bank_probabilities.append({
    #         'Index': idx,
    #         'Probability_Class_1': prob
    #     })
    #
    # return bank_probabilities

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


            # 调用银行概率计算函数
            bank_probabilities = calculate_bank_probabilities(data, data.test_mask, prob_csv_path)
            print(f'{country} ({ratio}): Bank Probabilities Calculated')

            # 将结果添加到列表中
            for bank_prob in bank_probabilities:
                results.append({
                    'Country': country,
                    'Ratio': ratio,
                    'Index': bank_prob['Index'],
                    'Probability_Class_1': bank_prob['Probability_Class_1']
                })

    # 将结果保存到CSV文件
    results_df = pd.DataFrame(results)
    os.makedirs(result_path, exist_ok=True)
    results_df.to_csv(os.path.join(result_path, 'bank_probabilities.csv'), index=False)
    print(f'Results saved to {os.path.join(result_path, "bank_probabilities.csv")}')

year = "2022"
countries = ["China","America", "Italy","Germany", "United Kingdom","France","Japan",  "Austria",  ]

ratios = ["train0.6_val0.15_test0.25"]
data_type = 2  # 0: 原始数据, 1: max_min, 2: cdf
result_path = 'results/'

process_countries(year, countries, ratios, data_type, result_path)

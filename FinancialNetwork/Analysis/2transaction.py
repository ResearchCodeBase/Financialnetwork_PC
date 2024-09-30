# coding=utf-8
import os

import numpy as np
import pandas as pd

# 读取Excel文件中名为"Germany_2020"的子表
sheet_name = "Austria_2020"
year = 2020
df = pd.read_excel('foreign_data.xlsx', sheet_name=sheet_name)
# 单位是千欧元
# 计算外部资产 银行间资产 银行间负债 核对后都是对的 /1000000是因为单位从元到百万
external_assets = df['2020Cash and Balances with Central Banks']+df['2020_Total Assets']-df['2020_Total Liabilities']
Interbank_Assets = df['2020Net Loans to Banks']
Interbank_Liabilities =df['2020Total Deposits from Banks']

length = len(external_assets)
# 转换为DataFrame并进行转置
external_assets_df = pd.DataFrame(external_assets).T
Interbank_Assets_df = pd.DataFrame(Interbank_Assets).T
Interbank_Liabilities_df = pd.DataFrame(Interbank_Liabilities).T
length = len(external_assets)
print('外部资产长度',length)
print(external_assets)
print('c拆出场长度',len(Interbank_Assets))
print('c拆入长度',len(Interbank_Liabilities))
# 创建子目录

# 保存为3个不同的txt文件，以行的形式展示数据
dir_path = f'Europe/{sheet_name}/原始数据'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

    # Save as different txt files, displaying data in rows
external_assets_file_path = f'{dir_path}/外部资产无其他银行_{year}.txt'
external_assets_df.to_csv(external_assets_file_path, sep='\t', index=False, header=False)
# 在行向量末尾追加一个0
external_assets_df[len(external_assets_df.columns)] = 0
# Save another file with an appended '0'
external_assets_with_zero_file_path = f'{dir_path}/外部资产有其他银行_{year}.txt'
external_assets_df.to_csv(external_assets_with_zero_file_path, sep='\t', index=False, header=False)

Interbank_Assets_df.to_csv(f'{dir_path}/拆出资金_{year}.txt', sep='\t', index=False, header=False)
Interbank_Liabilities_df.to_csv(f'{dir_path}/拆入资金_{year}.txt', sep='\t', index=False, header=False)
# 将 'Name' 列重命名为 'Country'
bank_names = df['Name'].rename('Country')
# 定义保存文件的路径
bank_names_file_path = f'{dir_path}/银行名称_{year}.csv'
# 保存成CSV文件，列名为 'Country'
bank_names.to_csv(bank_names_file_path, index=False)
# 保存银行数量
length_file_path = f'{dir_path}/银行数量.txt'

# Save the integer data to a text file
with open(length_file_path, 'w') as file:
    file.write(str(length))

e1=np.loadtxt(external_assets_file_path)

print('读取的外部资产',len(e1))
print(e1)
print(f'{year}银行简称保存成功')
print(f'{year}保存成功')
# coding=gbk
import pandas as pd

# 定义变量
country_to_filter = 'Italy'
year_to_filter = '2020'

# 加载Excel文件
df = pd.read_excel('foreign_data.xlsx')  # 替换 'foreign_data.xlsx' 为你的文件名

# 由于列名中可能存在空格，需要先处理列名
df.columns = df.columns.str.strip()

# 筛选Country列的数据
filtered_df = df[df['Country'] == country_to_filter]

# 选取特定的列，注意列名周围不要有空格
selected_columns = [
    'Name',
    'Country',
    f'{year_to_filter}_Total Assets',
    f'{year_to_filter}_Total Liabilities',
    f'{year_to_filter}Cash and Balances with Central Banks',
    f'{year_to_filter}Net Loans to Banks',
    f'{year_to_filter}Total Deposits from Banks'
]

# 在进行下一步操作前，先创建包含这些列的DataFrame副本
intermediate_df = filtered_df[selected_columns].copy()

# 删除包含NA值的行，仅检查特定列
final_df = intermediate_df.dropna(subset=[
    f'{year_to_filter}_Total Assets',
    f'{year_to_filter}_Total Liabilities',
    f'{year_to_filter}Cash and Balances with Central Banks',
    f'{year_to_filter}Net Loans to Banks',
    f'{year_to_filter}Total Deposits from Banks'
])

# 构建输出子表的名称
sheet_name = f'{country_to_filter}_{year_to_filter}'

# 将结果保存到foreign_data.xlsx中的一个新的子表中
with pd.ExcelWriter('foreign_data.xlsx', mode='a', engine='openpyxl') as writer:
    final_df.to_excel(writer, sheet_name=sheet_name, index=False)

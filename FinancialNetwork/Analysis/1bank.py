# coding=gbk
import pandas as pd

# �������
country_to_filter = 'Italy'
year_to_filter = '2020'

# ����Excel�ļ�
df = pd.read_excel('foreign_data.xlsx')  # �滻 'foreign_data.xlsx' Ϊ����ļ���

# ���������п��ܴ��ڿո���Ҫ�ȴ�������
df.columns = df.columns.str.strip()

# ɸѡCountry�е�����
filtered_df = df[df['Country'] == country_to_filter]

# ѡȡ�ض����У�ע��������Χ��Ҫ�пո�
selected_columns = [
    'Name',
    'Country',
    f'{year_to_filter}_Total Assets',
    f'{year_to_filter}_Total Liabilities',
    f'{year_to_filter}Cash and Balances with Central Banks',
    f'{year_to_filter}Net Loans to Banks',
    f'{year_to_filter}Total Deposits from Banks'
]

# �ڽ�����һ������ǰ���ȴ���������Щ�е�DataFrame����
intermediate_df = filtered_df[selected_columns].copy()

# ɾ������NAֵ���У�������ض���
final_df = intermediate_df.dropna(subset=[
    f'{year_to_filter}_Total Assets',
    f'{year_to_filter}_Total Liabilities',
    f'{year_to_filter}Cash and Balances with Central Banks',
    f'{year_to_filter}Net Loans to Banks',
    f'{year_to_filter}Total Deposits from Banks'
])

# ��������ӱ������
sheet_name = f'{country_to_filter}_{year_to_filter}'

# ��������浽foreign_data.xlsx�е�һ���µ��ӱ���
with pd.ExcelWriter('foreign_data.xlsx', mode='a', engine='openpyxl') as writer:
    final_df.to_excel(writer, sheet_name=sheet_name, index=False)

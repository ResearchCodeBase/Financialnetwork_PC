# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

# 高级期刊配色方案
color_map = {
    'United States': '#1f78b4',  # 蓝色
    'China': '#33a02c',          # 绿色
    'Japan': '#e31a1c',          # 红色
    'France': '#ff7f00',         # 橙色
    'United Kingdom': '#6a3d9a', # 紫色
    'Germany': '#b15928',        # 棕色
    'Italy': '#cab2d6'           # 淡紫色
}

# 数据
data = {
    'North America': {
        'United States': {
            'JPM': 0.0720,
            'BAC': 0.3665,
            'CITI': 0.2017,
            'GS': 0.3107,
            'MS': 0.0225,
            'BK': 0.4867,
            'STT': 0.4778,
            'WFC': 0.0126
        }
    },
    'Asia': {
        'China': {
            'ABC': 0.2179,
            'BOC': 0.0019,
            'COMM': 0.2081
        },
        'Japan': {
            'MUFG': 0.4500,
            'SMFG': 0.0243
        }
    },
    'Europe': {
        'France': {
            'BNP': 0.1444,
            'BPCE': 0.1888,
            'CA': 0.1921
        },
        'United Kingdom': {
            'HSBC': 0.0143
        },
        'Germany': {
            'DB': 0.4766
        },
        'Italy': {
            'UCG': 0.4361
        }
    }
}

# 提取数据
banks = []
probabilities = []
colors = []

for region, countries in data.items():
    for country, banks_probs in countries.items():
        for bank, prob in banks_probs.items():
            banks.append(bank)
            probabilities.append(prob)
            colors.append(color_map[country])

# 转换为数组
banks = np.array(banks)
probabilities = np.array(probabilities)
colors = np.array(colors)

# 排序
indices = np.argsort(probabilities)
banks = banks[indices]
probabilities = probabilities[indices]
colors = colors[indices]

# 创建图形
fig, ax = plt.subplots(figsize=(14, 10))

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

# 绘制水平条形图
y_pos = np.arange(len(banks))
bars = ax.barh(y_pos, probabilities, align='center', color=colors)

# 设置内刻度
ax.xaxis.set_ticks_position('bottom')
ax.xaxis.set_label_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_label_position('left')
ax.tick_params(axis='x', which='both', direction='in', labelsize=24, colors='black')
ax.tick_params(axis='y', which='both', direction='in', labelsize=24, colors='black')

# 设置标签和标题
ax.set_yticks(y_pos)
ax.set_yticklabels(banks, fontsize=24, fontname='Times New Roman', color='black')
ax.invert_yaxis()  # 倒置Y轴，使最高概率的银行在顶部
ax.set_xlabel('Default Probability', fontsize=32, fontname='Times New Roman', color='black')
ax.set_ylabel('Banks', fontsize=32, fontname='Times New Roman', color='black')

# 添加每个条形的值标签
for i in range(len(banks)):
    ax.text(probabilities[i] + 0.01, y_pos[i], f'{probabilities[i]:.4f}', va='center', fontsize=24, fontname='Times New Roman', color='black')

# 创建图例
legend_elements = [plt.Line2D([0], [0], color=color_map[country], lw=6, label=country) for country in color_map]
ax.legend(handles=legend_elements, title='Country', fontsize=22, title_fontsize=24, loc='upper right', frameon=True, edgecolor='black')

# 减少图形的上下留白
plt.subplots_adjust(top=0.95, bottom=0.05, left=0.15, right=0.95)

# 设置轴的颜色
for spine in ax.spines.values():
    spine.set_edgecolor('black')

# 增加横坐标范围
ax.set_xlim(0, 0.6)

# 显示图形
plt.tight_layout()

# 保存图形
plt.savefig('default_probabilities_of_banks.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()

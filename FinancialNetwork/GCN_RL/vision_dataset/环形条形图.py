import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 示例数据
data = {
    'Bank': ["Bank A", "Bank B", "Bank C", "Bank D", "Bank E",
             "Bank F", "Bank G", "Bank H", "Bank I", "Bank J",
             "Bank K", "Bank L", "Bank M", "Bank N", "Bank O",
             "Bank P", "Bank Q", "Bank R", "Bank S", "Bank T"],
    'Region': ["North America", "South America", "Europe", "Asia", "Africa",
               "North America", "South America", "Europe", "Asia", "Africa",
               "North America", "South America", "Europe", "Asia", "Africa",
               "North America", "South America", "Europe", "Asia", "Africa"],
    'Default Probability': [0.12, 0.18, 0.25, 0.14, 0.20, 0.28, 0.24, 0.33, 0.30, 0.21,
                            0.16, 0.19, 0.25, 0.31, 0.13, 0.12, 0.16, 0.20, 0.23, 0.28]
}

df = pd.DataFrame(data)

# 将数据按违约概率排序
df = df.sort_values(by='Default Probability', ascending=True)

# 计算角度
num_banks = len(df)
angles = np.linspace(0, 2 * np.pi, num_banks, endpoint=False).tolist()
angles += angles[:1]  # 为了闭合环形图

# 颜色映射
region_colors = {
    "North America": "blue",
    "South America": "green",
    "Europe": "red",
    "Asia": "purple",
    "Africa": "orange"
}

colors = [region_colors[region] for region in df['Region']]

# 设置图表参数
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

# 绘制条形图（银行）
bars = ax.bar(
    angles[:-1], df['Default Probability'], width=0.3, color=colors, edgecolor='black', alpha=0.8
)

# 添加银行标签
for bar, angle, label in zip(bars, angles, df['Bank']):
    rotation = np.rad2deg(angle)
    alignment = 'right' if angle > np.pi else 'left'
    ax.text(
        angle, bar.get_height() + 0.02, label, rotation=rotation, rotation_mode='anchor',
        ha=alignment, va='center', color='black', fontsize=10
    )

# 添加国家标签
outer_labels = df['Region'].unique()
outer_angles = np.linspace(0, 2 * np.pi, len(outer_labels), endpoint=False).tolist()
outer_angles += outer_angles[:1]

for label, angle in zip(outer_labels, outer_angles):
    rotation = np.rad2deg(angle)
    alignment = 'center'
    ax.text(
        angle, max(df['Default Probability']) + 0.1, label, rotation=rotation, rotation_mode='anchor',
        ha=alignment, va='center', color='black', fontsize=12, fontweight='bold'
    )

# 设置极轴
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_ylim(0, max(df['Default Probability']) + 0.2)

# 创建图例
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=color, lw=4, label=region) for region, color in region_colors.items()]
ax.legend(handles=legend_elements, title='Region', bbox_to_anchor=(1.1, 1.05))

# 显示图表
plt.title('Default Probability of Banks by Region', size=20, color='black', y=1.1)
plt.show()

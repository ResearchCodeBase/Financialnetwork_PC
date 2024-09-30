import matplotlib.pyplot as plt
import numpy as np

# 示例数据
bank_names = ["Bank A", "Bank B", "Bank C", "Bank D", "Bank E",
              "Bank F", "Bank G", "Bank H", "Bank I", "Bank J",
              "Bank K", "Bank L", "Bank M", "Bank N", "Bank O",
              "Bank P", "Bank Q", "Bank R", "Bank S", "Bank T"]
regions = ["North America", "South America", "Europe", "Asia", "Africa",
           "North America", "South America", "Europe", "Asia", "Africa",
           "North America", "South America", "Europe", "Asia", "Africa",
           "North America", "South America", "Europe", "Asia", "Africa"]

default_probabilities = [0.15, 0.20, 0.25, 0.10, 0.08,
                         0.18, 0.22, 0.30, 0.12, 0.09,
                         0.14, 0.19, 0.28, 0.11, 0.10,
                         0.17, 0.23, 0.29, 0.15, 0.13]

# 颜色映射
region_colors = {
    "North America": "blue",
    "South America": "green",
    "Europe": "red",
    "Asia": "purple",
    "Africa": "orange"
}

colors = [region_colors[region] for region in regions]

# 创建条形图
fig, ax = plt.subplots(figsize=(12, 8))

bars = ax.barh(bank_names, default_probabilities, color=colors)

# 添加标签和标题
ax.set_xlabel('Default Probability')
ax.set_title('Default Probability of Banks by Region')

# 创建图例
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=color, lw=4, label=region) for region, color in region_colors.items()]
ax.legend(handles=legend_elements, title='Region')

plt.show()

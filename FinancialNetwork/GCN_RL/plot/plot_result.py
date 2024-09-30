import matplotlib.pyplot as plt
import numpy as np

name = ["name1", "name2", "name3", "name4", "name5", "name6", "name7", "name8", "name9", "name10"]
y1 = [6, 5, 8, 5, 6, 6, 8, 9, 8, 10]
y2 = [5, 3, 6, 4, 3, 4, 7, 4, 4, 6]
y3 = [4, 1, 2, 1, 2, 1, 6, 2, 3, 2]

x = np.arange(len(name))
width = 0.25

plt.bar(x, y1, width=width, label='label1', color='darkorange')
plt.bar(x + width, y2, width=width, label='label2', color='deepskyblue', tick_label=name)
plt.bar(x + 2 * width, y3, width=width, label='label3', color='green')

# 显示在图形上的值
for a, b in zip(x, y1):
    plt.text(a, b + 0.1, b, ha='center', va='bottom')
for a, b in zip(x, y2):
    plt.text(a + width, b + 0.1, b, ha='center', va='bottom')
for a, b in zip(x, y3):
    plt.text(a + 2 * width, b + 0.1, b, ha='center', va='bottom')

plt.xticks(rotation=340)
plt.legend(loc="upper left")  # 防止label和图像重合显示不出来
plt.rcParams['figure.figsize'] = (15.0, 8.0)  # 尺寸
plt.show()
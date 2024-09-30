import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw_rl_flowchart():
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 定义节点位置
    positions = {
        "Initialize": (5, 9),
        "Compute Node Embeddings\nand Risk Probabilities": (5, 7.5),
        "Policy Network Selects\nNodes and Rescue Ratios": (5, 6),
        "Update Node Features\nand Compute Total Risk": (5, 4.5),
        "Compute Reward\nand Store Transitions": (5, 3),
        "Optimize Policy and\nValue Networks": (5, 1.5),
        "Repeat Until Convergence": (5, 0)
    }

    # 绘制节点
    for key, pos in positions.items():
        ax.text(pos[0], pos[1], key, horizontalalignment='center', verticalalignment='center', fontsize=10,
                bbox=dict(facecolor='lightyellow', edgecolor='black', boxstyle='round,pad=1'))

    # 定义箭头样式
    arrowprops = dict(arrowstyle='->', color='black', linewidth=1.5)

    # 绘制箭头
    arrow_start_end = [
        ("Initialize", "Compute Node Embeddings\nand Risk Probabilities"),
        ("Compute Node Embeddings\nand Risk Probabilities", "Policy Network Selects\nNodes and Rescue Ratios"),
        ("Policy Network Selects\nNodes and Rescue Ratios", "Update Node Features\nand Compute Total Risk"),
        ("Update Node Features\nand Compute Total Risk", "Compute Reward\nand Store Transitions"),
        ("Compute Reward\nand Store Transitions", "Optimize Policy and\nValue Networks"),
        ("Optimize Policy and\nValue Networks", "Repeat Until Convergence"),
        ("Repeat Until Convergence", "Compute Node Embeddings\nand Risk Probabilities")
    ]

    for start, end in arrow_start_end:
        ax.annotate('', xy=positions[end], xytext=positions[start], arrowprops=arrowprops)

    plt.show()


draw_rl_flowchart()

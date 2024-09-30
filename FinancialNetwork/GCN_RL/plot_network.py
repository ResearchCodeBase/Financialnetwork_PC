import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def draw_flowchart():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define box properties
    box_props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")

    # Define positions for each box
    positions = {
        "Feature": (0.5, 0.9),
        "GCN": (0.5, 0.75),
        "GRU": (0.5, 0.6),
        "MLP1": (0.3, 0.45),
        "MLP2": (0.7, 0.45),
        "SoftMax": (0.3, 0.3)
    }

    # Add boxes
    for label, pos in positions.items():
        ax.text(pos[0], pos[1], label, transform=ax.transAxes, fontsize=14, verticalalignment='center', horizontalalignment='center', bbox=box_props)

    # Draw arrows
    arrowprops = dict(arrowstyle="->", connectionstyle="arc3", color="black")
    ax.annotate("", xy=positions["Feature"], xytext=positions["GCN"], xycoords='axes fraction', textcoords='axes fraction', arrowprops=arrowprops)
    ax.annotate("", xy=positions["GCN"], xytext=positions["GRU"], xycoords='axes fraction', textcoords='axes fraction', arrowprops=arrowprops)
    ax.annotate("", xy=positions["GRU"], xytext=positions["MLP1"], xycoords='axes fraction', textcoords='axes fraction', arrowprops=arrowprops)
    ax.annotate("", xy=positions["GRU"], xytext=positions["MLP2"], xycoords='axes fraction', textcoords='axes fraction', arrowprops=arrowprops)
    ax.annotate("", xy=positions["MLP1"], xytext=positions["SoftMax"], xycoords='axes fraction', textcoords='axes fraction', arrowprops=arrowprops)

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    plt.show()

draw_flowchart()

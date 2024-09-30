import networkx as nx
import torch
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

def plot_train_test(self, name, network='eur', mpatches=None):
    graph = self.nx_graph

    legend_colors = ['#B9B9B9', 'red']
    legend_labels = ['Test', 'Train&Validation']
    mask = torch.logical_or(self.graphs[0].ndata['train_mask'], self.graphs[0].ndata['val_mask']).int()
    node_color = [legend_colors[self.graphs[0].ndata['train_mask'][idx]] for idx, node in enumerate(graph.nodes)]

    # Set node size based on degree
    node_size = [10 * d for n, d in graph.degree()]

    # Use force-directed layout algorithm to position nodes
    if network == 'eur':
        pos = nx.spring_layout(graph, iterations=200, k=120, scale=2, seed=self.seed)
    else:
        pos = nx.spring_layout(graph, iterations=200, k=0.5, scale=1, seed=self.seed)

    # Plot the graph
    fig = plt.figure(figsize=(15, 15))
    nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color=node_color, alpha=0.8)
    nx.draw_networkx_edges(graph, pos, width=0.1, alpha=0.5)
    plt.axis('off')
    legend_handles = []
    for color, label in zip(legend_colors, legend_labels):
        legend_handles.append(mpatches.Patch(color=color, label=label))

    plt.legend(handles=legend_handles, title='Node colors', loc='best')
    fig.savefig(name, format='svg')

    return fig
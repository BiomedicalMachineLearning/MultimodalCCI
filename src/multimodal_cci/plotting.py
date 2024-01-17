from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import numpy as np

from . import plot_helper


def network_plot(
    network,
    p_vals=None,
    normalise=True,
    p_val_cutoff=0.05,
    edge_weight=20,
    text_size=15,
    node_size=700,
    figsize=(10, 8),
    arrowsize=20,
    node_label_dist=1,
    cmap=None,
):
    """Plots a network with optional edge significance highlighting and node
    coloring based on in-degree and out-degree difference.

    Args:
        network (pandas.DataFrame or numpy.ndarray): The adjacency matrix representing
        the network.
        p_vals (pandas.DataFrame or numpy.ndarray, optional): A matrix of p-values
        corresponding to the edges in `network`. If not provided, significance values
        will not be plotted. Defaults to None.
        normalise (bool, optional): Whether to normalize the network matrix before
        plotting. Defaults to True.
        p_val_cutoff (float, optional): The p-value cutoff for determining significant
        edges. Defaults to 0.05.
        edge_weight (float, optional): The base weight for edges. Defaults to 20.
        text_size (int, optional): The font size for node labels. Defaults to 15.
        node_size (int, optional): The size of the nodes. Defaults to 700.
        figsize (tuple, optional): The size of the figure. Defaults to (10, 8).
        arrowsize (int, optional): The size of the arrow heads for edges. Defaults to
        20.
        node_label_dist (float, optional): A factor for adjusting the distance between
        nodes and labels. Defaults to 1.
        cmap (matplotlib.colors.Colormap, optional): A custom colormap for node
        coloring. Defaults to None.
    """

    plt.figure(figsize=figsize)

    if normalise:
        network = network / network.sum().sum()
    network = network.astype(float)
    G_network = nx.from_pandas_adjacency(network, create_using=nx.DiGraph)
    pos = nx.circular_layout(G_network)
    weights = nx.get_edge_attributes(G_network, "weight")

    # Calculate the in-degree and out-degree for each node
    in_degree = dict(G_network.in_degree(weight="weight"))
    out_degree = dict(G_network.out_degree(weight="weight"))

    in_out_diff = {node: in_degree[node] - out_degree[node] for node in G_network.nodes}

    if cmap is None:
        # Create a color scale based on the in-degree and out-degree difference
        max_diff = max(abs(value) for value in in_out_diff.values())
        color_scale = np.linspace(-max_diff, max_diff, 256)
        cmap_colors = [(1, 0, 0), (0.7, 0.7, 0.7), (0, 0, 1)]  # Blue, Grey, Red
        cmap = LinearSegmentedColormap.from_list("custom_cmap", cmap_colors)

    # Map node colors to the in-degree and out-degree difference
    node_colors = [
        cmap(int(np.interp(in_out_diff[node], color_scale, range(256))))
        for node in G_network.nodes
    ]

    if p_vals is None:
        # Create a non-significant matrix
        p_vals = network.replace(network.values, 1, inplace=False)

    # Get edges that are significant
    G_p_vals = nx.from_pandas_adjacency(p_vals, create_using=nx.DiGraph)
    non_sig = [
        (u, v) for (u, v, d) in G_p_vals.edges(data=True) if d["weight"] > p_val_cutoff
    ]
    non_sig = [edge for edge in non_sig if edge in weights.keys()]
    sig = [
        (u, v) for (u, v, d) in G_p_vals.edges(data=True) if d["weight"] <= p_val_cutoff
    ]
    sig = [edge for edge in sig if edge in weights.keys()]

    edge_thickness_non_sig = [weights[edge] * edge_weight for edge in non_sig]
    edge_thickness_sig = [weights[edge] * edge_weight for edge in sig]

    nx.draw_networkx_nodes(
        G_network,
        pos,
        node_size=node_size,
        node_color=node_colors,
    )

    nx.draw_networkx_edges(
        G_network,
        pos,
        node_size=node_size * 2,
        connectionstyle="arc3,rad=0.08",
        width=edge_thickness_non_sig,
        arrows=True,
        arrowstyle="->",
        arrowsize=arrowsize,
        edgelist=non_sig,
    )

    nx.draw_networkx_edges(
        G_network,
        pos,
        node_size=node_size * 2,
        connectionstyle="arc3,rad=0.08",
        width=edge_thickness_sig,
        arrows=True,
        arrowstyle="->",
        edgelist=sig,
        arrowsize=arrowsize,
        edge_color="purple",
    )

    edge_labels = nx.get_edge_attributes(G_p_vals, "weight")
    edge_labels = {
        key: edge_labels[key] for key in G_network.edges().keys() if key in edge_labels
    }

    # Add edge labels for significant edges
    for key, value in edge_labels.items():
        if value > p_val_cutoff:
            edge_labels[key] = ""
        else:
            edge_labels[key] = round(value, 3)

    def offset(d, pos, dist=0.05, loop_shift=0.1):
        for (u, v), obj in d.items():
            if u != v:
                par = dist * (pos[v] - pos[u])
                dx, dy = par[1], -par[0]
                x, y = obj.get_position()
                obj.set_position((x + dx, y + dy))
            else:
                x, y = obj.get_position()
                obj.set_position((x, y + loop_shift))

    d = nx.draw_networkx_edge_labels(G_network, pos, edge_labels)

    offset(d, pos)

    pos.update(
        (x, [y[0] * 1.4 * node_label_dist, y[1] * (1.25 + 0.05) * node_label_dist])
        for x, y in pos.items()
    )
    nx.draw_networkx_labels(
        G_network,
        pos,
        font_weight="bold",
        font_color="black",
        font_size=text_size,
        clip_on=False,
        horizontalalignment="center",
    )

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def chord_plot(network, show=True, title=None, label_size=10):
    """Plots a chord plot of a network

    Args:
        network (pandas.DataFrame or numpy.ndarray): The adjacency matrix representing
        the network.
        show (bool): Whether to show plot or not. Defaults to True.
        title (str): Title of the plot. Defaults to None.
        label_size (int): Font size of the labels. Defaults to None.
    """

    network = network.transpose()
    fig = plt.figure(figsize=(8, 8))

    flux = network.values
    cell_names = network.index.values.astype(str)
    nodes = cell_names

    ax = plt.axes([0, 0, 1, 1])
    nodePos = plot_helper.chordDiagram(flux, ax, lim=1.25)
    ax.axis("off")
    prop = dict(fontsize=label_size, ha="center", va="center")

    for i in range(len(cell_names)):
        x, y = nodePos[i][0:2]
        ax.text(x, y, nodes[i], rotation=nodePos[i][2], **prop)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    if show:
        plt.show()
    else:
        return fig, ax


def dissim_hist(dissimilarity_scores):
    """Plots a histogram of dissimilarity scores.

    Args:
        dissimilarity_scores (dict): A dictionary of dissimilarity scores.
    """

    plt.hist(list(dissimilarity_scores.values()))
    plt.xlim(0, 1)
    plt.xlabel("Dissimilarity Score")
    plt.ylabel("Count")
    plt.show()


def silhouette_scores_plot(silhouette_scores):
    """Plots a line plot of silhouette scores.

    Args:
        silhouette_scores (list): A list of silhouette scores.
    """

    plt.figure(figsize=(10, 7))
    plt.plot(range(2, 11), silhouette_scores, marker="o")
    plt.title("Silhouette Score for Different Numbers of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.show()

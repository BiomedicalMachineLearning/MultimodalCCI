from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import numpy as np

from . import plot_helper
from . import analysis as an


def network_plot(
    network,
    p_vals=None,
    diff_plot=False,
    normalise=True,
    remove_unconnected=True,
    show_labels=True,
    p_val_cutoff=0.05,
    edge_weight=20,
    text_size=15,
    node_size=700,
    figsize=(10, 8),
    arrowsize=20,
    node_label_dist=1,
    p_val_text_size=10,
    node_colors=None,
    node_palette=None,
    outer_node_palette=None,
    show=True,
):
    """Plots a network with optional edge significance highlighting and node
    coloring based on in-degree and out-degree difference.

    Args:
        network (pandas.DataFrame or numpy.ndarray): The adjacency matrix representing
        the network.
        p_vals (pandas.DataFrame or numpy.ndarray, optional): A matrix of p-values
        corresponding to the edges in `network`. If not provided, significance values
        will not be plotted. Defaults to None.
        diff_plot (bool, optional): Whether you are plotting the network difference, to
        show up and down-regulated edges. Defaults to False.
        normalise (bool, optional): Whether to normalize the network matrix before
        plotting. Defaults to True.
        remove_unconnected (bool, optional): Whether to remove cell types that
        do not interact with any cell types. Defaults to True.
        show_labels (bool, optional): Whether to show node labels. Defaults to True.
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
        p_val_text_size (int, optional): The font size for p-value labels. Defaults to
        10.
        node_colors (dict, optional): A dictionary of colors for each node. Overwrites
        node_palette. Defaults to None.
        node_palette (str, optional): The name of the color palette to use for nodes.
        Defaults to None.
        outer_node_palette (str, optional): The name of the color palette to use for
        outer nodes to show sender/reciever nodes. Defaults to None.
        show (bool, optional): Whether to show the plot or not. Defaults to True.

    Returns:
        tuple: A tuple containing the figure and axis objects.
    """

    if isinstance(network, dict):
        raise ValueError(
            "Input should be a single matrix, not a sample. You may need to run \
                calculate_overall_interactions() first or select an LR pair.")

    fig = plt.figure(figsize=figsize)

    if remove_unconnected:
        network = network.loc[(network != 0).any(axis=1), (network != 0).any(axis=0)]
        network = network.loc[network.index.isin(
            network.columns), network.columns.isin(network.index)]

    if normalise:
        if network.min().min() >= 0:
            network = network / network.sum().sum()
    network_abs = abs(network)

    if normalise:
        network_abs = network_abs / network_abs.sum().sum()
    network_abs = network.astype(float)
    G_network = nx.from_pandas_adjacency(network_abs, create_using=nx.DiGraph)
    pos = nx.circular_layout(G_network)
    weights = nx.get_edge_attributes(G_network, "weight")

    # Calculate the in-degree and out-degree for each node
    in_degree = dict(G_network.in_degree(weight="weight"))
    out_degree = dict(G_network.out_degree(weight="weight"))

    in_out_diff = {node: in_degree[node] - out_degree[node] for node in G_network.nodes}

    max_diff = max(abs(value) for value in in_out_diff.values())
    color_scale = np.linspace(-max_diff, max_diff, 256)

    if outer_node_palette is None:
        # Create a color scale based on the in-degree and out-degree difference
        cmap_colors = [(1, 0, 0), (0.7, 0.7, 0.7), (0, 0, 1)]  # Blue, Grey, Red
        outer_node_cmap = LinearSegmentedColormap.from_list("custom_cmap", cmap_colors)
    else:
        outer_node_cmap = plt.get_cmap(outer_node_palette)

    
    edge_colors = []
    # Map node colors to the in-degree and out-degree difference
    if sum(abs(value) for value in in_out_diff.values()) == 0:
        edge_colors = ['grey' for node in G_network.nodes]
    else:
        edge_colors = [
            outer_node_cmap(int(np.interp(in_out_diff[node], color_scale, range(256))))
            for node in G_network.nodes
        ]

    if node_colors is not None:
        node_colors = [node_colors[node] for node in G_network.nodes]
    else:
        if node_palette is not None:
            node_colors = list(plt.get_cmap(node_palette).colors)[:len(G_network.nodes)]
        else:
            node_colors = ["grey" for node in G_network.nodes]

    if p_vals is None or diff_plot == False:
        # Create a non-significant matrix
        p_vals = network_abs.replace(network_abs.values, 1, inplace=False)
    else:
        # Prevent removal of pvals of 0
        p_vals[p_vals == 0] = 1e-300

    # Get edges that are significant
    G_p_vals = nx.from_pandas_adjacency(p_vals, create_using=nx.DiGraph)
    G_network_updown = nx.from_pandas_adjacency(network, create_using=nx.DiGraph)

    non_sig_up = [
        (u, v) for (u, v, d) in G_p_vals.edges(data=True)
        if d["weight"] > p_val_cutoff
        and u in G_network_updown
        and v in G_network_updown[u]
        and G_network_updown[u][v]["weight"] > 0]
    non_sig_up = [edge for edge in non_sig_up if edge in weights.keys()]

    non_sig_down = [
        (u, v) for (u, v, d) in G_p_vals.edges(data=True)
        if d["weight"] > p_val_cutoff
        and u in G_network_updown
        and v in G_network_updown[u]
        and G_network_updown[u][v]["weight"] < 0]
    non_sig_down = [edge for edge in non_sig_down if edge in weights.keys()]

    sig_up = [
        (u, v) for (u, v, d) in G_p_vals.edges(data=True)
        if d["weight"] <= p_val_cutoff
        and u in G_network_updown
        and v in G_network_updown[u]
        and G_network_updown[u][v]["weight"] > 0]
    sig_up = [edge for edge in sig_up if edge in weights.keys()]

    sig_down = [
        (u, v) for (u, v, d) in G_p_vals.edges(data=True) if d["weight"] <= p_val_cutoff
        and u in G_network_updown
        and v in G_network_updown[u]
        and G_network_updown[u][v]["weight"] < 0]
    sig_down = [edge for edge in sig_down if edge in weights.keys()]

    edge_thickness_non_sig_up = []
    edge_thickness_non_sig_down = []
    edge_thickness_sig_up = []
    edge_thickness_sig_down = []

    for edge in weights.keys():
        if edge in non_sig_up:
            edge_thickness_non_sig_up.append(weights[edge] * edge_weight)
        else:
            edge_thickness_non_sig_up.append(0)

        if edge in non_sig_down:
            edge_thickness_non_sig_down.append(weights[edge] * edge_weight)
        else:
            edge_thickness_non_sig_down.append(0)

        if edge in sig_up:
            edge_thickness_sig_up.append(weights[edge] * edge_weight)
        else:
            edge_thickness_sig_up.append(0)

        if edge in sig_down:
            edge_thickness_sig_down.append(weights[edge] * edge_weight)
        else:
            edge_thickness_sig_down.append(0)

    # if node_colors is None:
    nx.draw_networkx_nodes(
        G_network,
        pos,
        node_size=node_size,
        node_color=node_colors,
        edgecolors=edge_colors,
        linewidths=8.0,
    )

    if diff_plot:
        nx.draw_networkx_edges(
            G_network,
            pos,
            node_size=node_size * 2,
            connectionstyle="arc3,rad=0.08",
            width=edge_thickness_non_sig_up,
            arrows=True,
            arrowstyle="->",
            arrowsize=arrowsize,
            edge_color="pink"
            # edgelist=non_sig,
        )

        nx.draw_networkx_edges(
            G_network,
            pos,
            node_size=node_size * 2,
            connectionstyle="arc3,rad=0.08",
            width=edge_thickness_non_sig_down,
            arrows=True,
            arrowstyle="->",
            arrowsize=arrowsize,
            edge_color="lightgreen"
            # edgelist=non_sig,
        )

    else:
        nx.draw_networkx_edges(
            G_network,
            pos,
            node_size=node_size * 2,
            connectionstyle="arc3,rad=0.08",
            width=edge_thickness_non_sig_up,
            arrows=True,
            arrowstyle="->",
            arrowsize=arrowsize,
            # edgelist=non_sig,
        )

        nx.draw_networkx_edges(
            G_network,
            pos,
            node_size=node_size * 2,
            connectionstyle="arc3,rad=0.08",
            width=edge_thickness_non_sig_down,
            arrows=True,
            arrowstyle="->",
            arrowsize=arrowsize,
            # edgelist=non_sig,
        )

    nx.draw_networkx_edges(
        G_network,
        pos,
        node_size=node_size * 2,
        connectionstyle="arc3,rad=0.08",
        width=edge_thickness_sig_up,
        arrows=True,
        arrowstyle="->",
        # edgelist=sig_up,
        arrowsize=arrowsize,
        edge_color="purple",
    )

    nx.draw_networkx_edges(
        G_network,
        pos,
        node_size=node_size * 2,
        connectionstyle="arc3,rad=0.08",
        width=edge_thickness_sig_down,
        arrows=True,
        arrowstyle="->",
        # edgelist=sig_down,
        arrowsize=arrowsize,
        edge_color="green",
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

    def offset(d, pos, dist=0.05, loop_shift=0.22):
        for (u, v), obj in d.items():
            if u != v:
                par = dist * (pos[v] - pos[u])
                dx, dy = par[1], -par[0]
                x, y = obj.get_position()
                obj.set_position((x + dx, y + dy))
            else:
                x, y = obj.get_position()
                obj.set_position((x, y + loop_shift))

    d = nx.draw_networkx_edge_labels(
        G_network, pos, edge_labels, font_size=p_val_text_size)

    offset(d, pos)

    pos.update(
        (x, [y[0] * 1.4 * node_label_dist, y[1] * (1.25 + 0.05) * node_label_dist])
        for x, y in pos.items()
    )

    if show_labels:
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

    if show:
        plt.show()
    else:
        return fig, ax


def chord_plot(
    network,
    min_int=0.01,
    n_top_ccis=10,
    colors=None,
    show=True,
    title=None,
    label_size=10,
    figsize=(8, 8),
):
    """Plots a chord plot of a network

    Args:
        network (pandas.DataFrame or numpy.ndarray): The adjacency matrix representing
        the network.
        min_int (float): Minimum interactions to display cell type. Defaults to 0.01.
        n_top_ccis (int): Number of top cell types to display. Defaults to 10.
        colors (dict): Dict of colors for each cell type to use for the plot. Defaults
        to None.
        show (bool): Whether to show plot or not. Defaults to True.
        title (str): Title of the plot. Defaults to None.
        label_size (int): Font size of the labels. Defaults to None.
        figsize (tuple): Size of the figure. Defaults to None.

    Returns:
        tuple: A tuple containing the figure and axis objects.
    """

    if isinstance(network, dict):
        raise ValueError(
            "Input should be a single matrix, not a sample. You may need to run \
                calculate_overall_interactions() first or select an LR pair.")

    network = network.transpose()
    fig = plt.figure(figsize=figsize)

    flux = network.values

    total_ints = flux.sum(axis=1) + flux.sum(axis=0) - flux.diagonal()
    keep = total_ints > min_int
    # Limit of 10 for good display #
    if sum(keep) > n_top_ccis:
        keep = np.argsort(-total_ints)[0:n_top_ccis]
    flux = flux[:, keep]
    flux = flux[keep, :].astype(float)
    cell_names = network.index.values.astype(str)[keep]
    nodes = cell_names

    color_list = []
    if colors is not None:
        for cell in cell_names:
            color_list.append(colors[cell])
    else:
        color_list = None

    ax = plt.axes([0, 0, 1, 1])
    nodePos = plot_helper.chordDiagram(flux, ax, lim=1.25, colors=color_list)
    ax.axis("off")
    prop = dict(fontsize=label_size, ha="center", va="center")

    for i in range(len(cell_names)):
        x, y = nodePos[i][0:2]
        if label_size != 0:
            ax.text(x, y, nodes[i], rotation=nodePos[i][2], **prop)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    if show:
        plt.show()
    else:
        return fig, ax


def dissim_hist(
    dissimilarity_scores,
    x_label_size=24,
    y_label_size=24,
    x_tick_size=14,
    y_tick_size=12,
    figsize=(6, 5),
    show=True,
):
    """Plots a histogram of dissimilarity scores.

    Args:
        dissimilarity_scores (dict): A dictionary of dissimilarity scores.
        x_label_size (int): Font size for x-axis label. Defaults to 24.
        y_label_size (int): Font size for y-axis label. Defaults to 24.
        x_tick_size (int): Font size for ticks. Defaults to 14.
        y_tick_size (int): Font size for ticks. Defaults to 12.
        figsize (tuple): Size of the figure. Defaults to (10, 8).

    Returns:
        matplotlib.figure.Figure: The figure
    """

    fig = plt.figure(figsize=figsize)
    plt.style.use('default')
    plt.hist(list(dissimilarity_scores.values()))
    plt.xlim(0, 1)
    plt.xlabel("Dissimilarity Score", fontsize=x_label_size)
    plt.ylabel("Count", fontsize=y_label_size)
    plt.tick_params(axis='x', which='major', labelsize=x_tick_size)
    plt.tick_params(axis='y', which='major', labelsize=y_tick_size)

    if show:
        plt.show()
    else:
        plt.close(fig)
        return fig


def lr_top_dissimilarity(
    dissimilarity_scores,
    n=10,
    top=True,
    x_label_size=24,
    y_label_size=24,
    x_tick_size=14,
    y_tick_size=12,
    figsize=(6, 5),
    show=True,
):
    """Plots a bar plot of LR pairs with highest/lowest dissimilarity scores.

    Args:
        dissimilarity_scores (dict): A dictionary of dissimilarity scores.
        n (int): Number of LR pairs to plot.
        top (bool): If True, plot LR pairs with highest dissimilarity scores.
        If False, plot LR pairs with lowest dissimilarity scores.
        x_label_size (int): Font size for x-axis label. Defaults to 24.
        y_label_size (int): Font size for y-axis label. Defaults to 24.
        x_tick_size (int): Font size for ticks. Defaults to 14.
        y_tick_size (int): Font size for ticks. Defaults to 12.
        figsize (tuple): Size of the figure. Defaults to (10, 8).

    Returns:
        matplotlib.figure.Figure: The figure
    """

    reverse = not top
    sorted_items = sorted(
        dissimilarity_scores.items(), key=lambda x: x[1], reverse=reverse
    )
    top_n_items = sorted_items[-n:]
    keys, values = zip(*top_n_items)

    fig = plt.figure(figsize=figsize)
    plt.style.use('default')
    plt.barh(keys, values)
    plt.xlabel("Dissimilarity Score", fontsize=x_label_size)
    plt.ylabel("LR Pair", fontsize=y_label_size)
    plt.tick_params(axis='x', which='major', labelsize=x_tick_size)
    plt.tick_params(axis='y', which='major', labelsize=y_tick_size)

    if show:
        plt.show()
    else:
        plt.close(fig)
        return fig


def silhouette_scores_plot(
    silhouette_scores,
    figsize=(10, 7),
    show=True,
):
    """Plots a line plot of silhouette scores.

    Args:
        silhouette_scores (list): A list of silhouette scores.
        figsize (tuple): Size of the figure. Defaults to (10, 7).

    Returns:
        matplotlib.figure.Figure: The figure
    """

    fig = plt.figure(figsize=figsize)
    plt.style.use('fast')
    plt.plot(range(2, 11), silhouette_scores, marker="o")
    plt.title("Silhouette Score for Different Numbers of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")

    if show:
        plt.show()
    else:
        plt.close(fig)
        return fig


def lr_barplot(
    sample,
    n=15,
    x_label_size=24,
    y_label_size=24,
    x_tick_size=14,
    y_tick_size=12,
    figsize=(6, 5),
    show=True,
):
    """Plots a bar plot of LR pairs and their proportions for a sample.

    Args:
        sample (dict): A dictionary of LR pairs.
        n (int): Number of LR pairs to plot. If None, plot all LR pairs. Defaults to
        15.
        x_label_size (int): Font size for x-axis label. Defaults to 24.
        y_label_size (int): Font size for y-axis label. Defaults to 24.
        x_tick_size (int): Font size for tick labels. Defaults to 14.
        y_tick_size (int): Font size for tick labels. Defaults to 12.
        figsize (tuple): Size of the figure. Defaults to (10, 8).

    Returns:
        matplotlib.figure.Figure: The figure
    """

    if not isinstance(sample, dict):
        raise ValueError("The sample must be a dict of LR matrices.")

    interactions = [(lr, df.sum().sum()) for (lr, df) in sample.items()]
    interactions.sort(key=lambda x: x[1])
    interactions = interactions[-n:]
    keys, values = zip(*interactions)
    values = [value / sum(values) for value in values]

    fig = plt.figure(figsize=figsize)
    plt.style.use('default')
    plt.barh(keys, values)
    plt.xlabel("Relative Interaction Strength", fontsize=x_label_size)
    plt.ylabel("LR Pair", fontsize=y_label_size)
    plt.tick_params(axis='x', which='major', labelsize=x_tick_size)
    plt.tick_params(axis='y', which='major', labelsize=y_tick_size)

    if show:
        plt.show()
    else:
        plt.close(fig)
        return fig


def lrs_per_celltype(
    sample,
    sender,
    receiver,
    p_vals=None,
    n=15,
    x_label_size=24,
    y_label_size=24,
    x_tick_size=14,
    y_tick_size=12,
    figsize=(6, 5),
    show=True,
):
    """Plots a bar plot of LR pairs and their proportions for a sender and receiver cell
    type pair along with p_values (optional).

    Args:
        sample (dict): A dictionary of LR pairs.
        sender (str): The sender cell type.
        receiver (str): The receiver cell type.
        p_vals (dict): A dictionary of p-values. Defaults to None.
        n (int): Number of LR pairs to plot. Defaults to 15.
        x_label_size (int): Font size for x-axis label. Defaults to 24.
        y_label_size (int): Font size for y-axis label. Defaults to 24.
        x_tick_size (int): Font size for tick labels. Defaults to 14.
        y_tick_size (int): Font size for tick labels. Defaults to 12.
        figsize (tuple): Size of the figure. Defaults to (10, 8).

    Returns:
        matplotlib.figure.Figure: The figure
    """

    pairs = an.get_lrs_per_celltype(sample, sender, receiver)
    keys = list(pairs.keys())[:n]
    values = list(pairs.values())[:n]
    keys.reverse()
    values.reverse()
    if p_vals is not None:
        p_val_pairs = an.get_p_vals_per_celltype(p_vals, sender, receiver)
        labels = [p_val_pairs[key] for key in keys]

        # make labels readable (if less than 0.00001, show as <0.00001)
        for i in range(len(labels)):
            if labels[i] < 0.001:
                labels[i] = "<0.001"
            else:
                labels[i] = f"{labels[i]:.3f}"

        # Define colors based on p-values
        colors = [
            '#1f77b4' if val < 0.05 else 'grey' for val in [
                p_val_pairs[key] for key in keys]]

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    plt.style.use('default')

    if p_vals is None:
        ax.barh(keys, values)
    else:
        bars = ax.barh(keys, values, color=colors)
        ax.bar_label(bars, labels)

    ax.set_xlabel("Proportion", fontsize=x_label_size)
    ax.set_ylabel("LR Pair", fontsize=y_label_size)
    ax.tick_params(axis='x', which='major', labelsize=x_tick_size)
    ax.tick_params(axis='y', which='major', labelsize=y_tick_size)

    if show:
        plt.show()
    else:
        plt.close(fig)
        return fig

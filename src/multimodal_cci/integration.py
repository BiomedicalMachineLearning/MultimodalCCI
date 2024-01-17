import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, ClusterWarning
from warnings import simplefilter
from tqdm import tqdm

from . import scoring as sc
from . import plotting as pl

simplefilter("ignore", ClusterWarning)


def normalise_samples_to_target(samples, sample_sizes, target=None):
    """Normalizes LR matrices within a list of samples to a target sample based on
    number of spots to account for differences in sample size.

    Args:
        samples (list): A list of dictionaries of LR matrices
        sample_sizes (list): A list of number of spots for each sample
        target (int) (optional): A target sample size to normalize to. If not provided,
        the first sample's size is used.

    Returns:
        list: The list of samples with normalized matrices.
    """

    if target is None:
        target = sample_sizes[0]

    for i, sample in enumerate(samples):
        for lr, matrix in sample.items():
            samples[i][lr] = matrix * (target / sample_sizes[i])

    return samples


def get_majority_lr_pairs(samples, equal_to=False, return_counts=False):
    """Identifies the LR pairs present in a majority of samples.

    Args:
        samples (list): A list of dictionaries of LR matrices
        equal_to (bool) (optional): If True, includes LR pairs present in exactly half
        of the samples. Defaults to False.
        return_counts (bool) (optional): If True, the function returns the number of
        samples each LR pair is present in.

    Returns:
        list or dict: A list of LR pairs that are present in a majority of samples or a
        dictionary of LR pairs and counts if return_counts=True.
    """

    lr_pairs_counts = {}
    lr_pairs = []

    for sample in samples:
        for lr_pair, matrix in sample.items():
            if sum(sum(matrix.values)) != 0:
                lr_pairs_counts[lr_pair] = lr_pairs_counts.setdefault(lr_pair, 0) + 1

    for lr_pair, count in lr_pairs_counts.items():
        if equal_to:
            if count >= len(samples) / 2:
                lr_pairs.append(lr_pair)
        else:
            if count > len(samples) / 2:
                lr_pairs.append(lr_pair)

    if return_counts:
        return lr_pairs_counts
    return lr_pairs


def get_avg_lr_pairs(samples, lr_pairs):
    """Calculates the average matrix for each LR pair across a list of samples.

    Args:
        samples (list): A list of dictionaries of LR matrices
        lr_pairs (list): A list of LR pairs to calculate averages for.

    Returns:
        dict: A dictionary where keys are LR pairs and values are the corresponding
        average matrices.
    """

    lr_pairs_matrices = {}

    for sample in samples:
        for lr_pair, matrix in sample.items():

            if lr_pair in lr_pairs:
                current_matrix = lr_pairs_matrices.get(lr_pair)

                if current_matrix is not None:
                    common_rows = list(set(matrix.index) & set(current_matrix.index))
                    common_cols = list(
                        set(matrix.columns) & set(current_matrix.columns)
                    )
                    matrix = matrix.loc[common_rows, common_cols]
                    current_matrix = current_matrix.loc[common_rows, common_cols]
                    lr_pairs_matrices[lr_pair] = current_matrix + matrix / len(samples)
                else:
                    lr_pairs_matrices[lr_pair] = matrix / len(samples)

    return lr_pairs_matrices


def normalise_samples_between_tech(samples, method="mean", tech_norm=0):
    """Normalizes matrices between different technologies.

    Args:
        samples (list): A nested list of samples, organized by technology.
        Example: [[t1s1, t1s2], [t2s1, t2s2]].
        method (str) (optional): The normalization method, either "mean" or "sum".
        Defaults to "mean". Normalises based on either the mean or sum of the values in
        the matrix.
        tech_norm (str) (optional): The index of the technology to use as the
        normalization reference. Defaults to 0.

    Returns:
        The normalized nested list of samples.
    """

    tech_counts = []
    for tech_samples in samples:
        group_counts = []
        lr_pair_counts = []
        for samples in tech_samples:
            total_counts = 0
            lr_pair_count = 0
            for pair, matrix in samples.items():
                lr_pair_count += 1
                if method == "mean":
                    total_counts += matrix.mean().mean()
                elif method == "sum":
                    total_counts += matrix.sum().sum()
                else:
                    raise ValueError("Invalid method option.")

            total_counts = total_counts / lr_pair_count
            group_counts.append(total_counts)
        tech_counts.append(statistics.mean(group_counts))

    for tech in range(len(samples)):
        for group in range(len(samples[tech])):
            for lr, df in samples[tech][group].items():
                factor = tech_counts[tech_norm] / tech_counts[tech]
                samples[tech][group][lr] = df * factor

    return samples


def integrate_between_tech(samples):
    """Integrates matrices from different technologies.

    Args:
        samples (list): A list of samples with different technologies.

    Returns:
        dict: A dictionary where keys are LR pairs and values are the integrated
        matrices.
    """

    integrated = {}
    lr_matrices = {}

    if len(samples) == 2:
        lr_pairs = get_majority_lr_pairs(samples, equal_to=False)
        lr_pairs_counts = {lr: 2 for lr in lr_pairs}

    elif len(samples) > 2:
        lr_pairs = sorted(get_majority_lr_pairs(samples, equal_to=True))
        lr_pairs_counts = get_majority_lr_pairs(
            samples, equal_to=True, return_counts=True
        )

    else:
        raise ValueError("Integration needs at least two samples")

    for i in range(len(lr_pairs)):
        lr = lr_pairs[i]
        lr_count = lr_pairs_counts[lr]

        for tech in range(len(samples)):
            if lr in samples[tech]:
                if lr in lr_matrices:
                    lr_matrices[lr].append(samples[tech][lr])
                else:
                    lr_matrices[lr] = [samples[tech][lr]]

    for lr, matrices in lr_matrices.items():
        if len(matrices) == 2:
            integrated[lr] = (matrices[0] * matrices[1]).fillna(0)
            integrated[lr] = np.sqrt(integrated[lr]).fillna(0)
        elif len(matrices) > 2:
            integrated[lr] = sc.multiply_non_zero_values(matrices)

    return integrated


def calculate_overall_interactions(sample):
    """Calculates an overall interaction matrix by combining matrices for different LR
    pairs within a sample.

    Args:
        sample (dict): A sample containing matrices for different LR pairs.

    Returns:
        pd.DataFrame: A matrix representing the overall interactions.
    """

    total = None
    for lr in sample.keys():
        if sample[lr].sum().sum() > 0:
            if total is not None:
                total = total + sample[lr] / sample[lr].sum().sum()
                total = total.fillna(0)
            else:
                total = sample[lr] / sample[lr].sum().sum()
                total = total.fillna(0)

    if total is None:
        return None

    total = total / total.sum().sum()
    total = total.fillna(0)

    return total


def calculate_dissim(sample1, sample2):
    """Calculates a dissimilarity score between two samples for each common LR pair.

    Args:
        sample1, sample2 (dict): Two samples containing matrices for LR pairs.

    Returns:
        dict: A dictionary where keys are common LR pairs and values are the
        dissimilarity scores.
    """

    dissims = {}
    for lr in set(sample1.keys()).intersection(set(sample2.keys())):
        dissims[lr] = sc.dissimilarity_score(sample1[lr], sample2[lr])

    return dissims


def get_lrs_per_celltype(samples, sender, reciever):
    """Compares LR pairs between two samples for specific cell types.

    Args:
        samples (list): A list of dictionaries containing LR matrices for the
        first sample.
        sender (str): The sender cell type.
        reciever (str): The receiver cell type.

    Returns:
        dict: A dictionary with keys 'sample1', 'sample2', etc., each containing
        a set of LR pairs and proportion of its weighting.
    """

    names = []

    for i in range(len(samples)):
        samples[i] = {
            key: df.loc[[sender]]
            for key, df in samples[i].items()
            if sender in df.index
        }

        samples[i] = {
            key: df[[reciever]]
            for key, df in samples[i].items()
            if reciever in df.columns
        }

        samples[i] = {
            key: df
            for key, df in samples[i].items()
            if not df.map(lambda x: x == 0).all().all()
        }

        samples[i] = {
            key: df
            for key, df in samples[i].items()
            if not df.map(lambda x: x == 0).all().all()
        }
        names.append("sample" + str(i + 1))

    data = {}

    for group_name, lr_dict in zip(names, samples):

        lr_props = {}
        total = 0
        for lr_pair in set(lr_dict.keys()):
            score = lr_dict[lr_pair].at[sender, reciever]
            total += score

        for lr_pair in set(lr_dict.keys()):
            lr_props[lr_pair] = lr_dict[lr_pair].at[sender, reciever] / total

        lr_props = dict(
            sorted(lr_props.items(), key=lambda item: item[1], reverse=True)
        )

        data[group_name] = lr_props

    return data


def lr_clustering(sample, n_clusters=0):
    """Clusters LR pairs based on LR matrix similarities.

    Args:
        sample (dict): A dictionary containing LR matrices.
        n_clusters (int) (optional): The desired number of clusters. If 0, the optimal
        number is determined using silhouette analysis. Defaults to 0.

    Returns:
        pd.DataFrame: A DataFrame with the cluster assignments for each sample.
    """

    if "per_lr_cci_cell_type" in sample:
        sample.pop("per_lr_cci_cell_type")

    # Function to check if the entire dataframe is zero
    def has_non_zero_values(df):
        return not (df == 0).all().all()

    # Filtering out key-value pairs with dataframes containing all zeros
    sample = {key: value for key, value in sample.items() if has_non_zero_values(value)}

    # Initialize an empty dataframe to store the results
    result_df = pd.DataFrame(index=sample.keys(), columns=sample.keys())

    # Iterate through the keys and compare the dataframes
    print("Computing Dissimilarity Scores ...")
    with tqdm(total=len(sample), desc="Processing") as pbar:
        for key1, df1 in sample.items():
            for key2, df2 in sample.items():
                result = dissimilarity_score(df1, df2, lmbda=0.5, only_non_zero=True)

                # Store the result in the result_df
                result_df.loc[key1, key2] = result
            pbar.update(1)

    # Compute distance matrix from disimilarity matrix
    result_df = result_df.astype("float64")
    result_df = result_df.fillna(0)
    distances = pdist(result_df.values, metric="euclidean")
    dist_matrix = squareform(distances)
    dist_matrix = pd.DataFrame(
        pp.MinMaxScaler(feature_range=(0, 100)).fit_transform(
            pd.DataFrame(dist_matrix).T.values
        )
    )

    print("Computing Principal Components of weighted graph ...")
    # Perform PCA on weighted edges of interaction network
    flatten_dfs = [df.to_numpy().flatten() for df in sample.values()]
    flatten_dfs = pd.DataFrame(flatten_dfs)
    scaler = pp.StandardScaler()
    data_scaled = scaler.fit_transform(flatten_dfs)
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)

    # Concatenate PC-1 and PC-2 with distance matrix
    pc_com_dist_matrix = pd.concat([dist_matrix, pd.DataFrame(data_pca)], axis=1)

    print("Performing Clustering and Ranking within clusters...")
    if n_clusters > 0:
        # Number of clusters (adjust as needed)
        n_clusters = n_clusters

        # Perform hierarchical clustering
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        clusters = model.fit_predict(pc_com_dist_matrix)

    if n_clusters == 0:

        # Evaluate silhouette score for different numbers of clusters
        silhouette_scores = []
        for n_clusters in range(2, 11):
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
            cluster_labels = clusterer.fit_predict(pc_com_dist_matrix)
            silhouette_avg = silhouette_score(pc_com_dist_matrix, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        silhouette_scores_plot(silhouette_scores)

        # Perform hierarchical clustering
        model = AgglomerativeClustering(
            n_clusters=np.argmax(silhouette_scores) + 2, linkage="ward"
        )  # as indexing starts from 0
        clusters = model.fit_predict(pc_com_dist_matrix)

    clusters = pd.DataFrame(clusters)
    clusters.index = sample.keys()
    clusters.rename(columns={0: "Cluster"}, inplace=True)

    # Rank LRs in each cluster based on increasing dissimilarity
    dist_matrix.columns = list(sample.keys())
    dist_matrix.index = list(sample.keys())
    columns = ["LRs", "Cluster"]
    final_clusters = pd.DataFrame(columns=columns)

    for i in range(0, len(set(clusters["Cluster"]))):
        clusters_index = list(clusters[clusters["Cluster"] == i].index)
        dist_matrix_cluster = dist_matrix[dist_matrix.index.isin(clusters_index)]
        similarity_matrix = 1 / (1 + dist_matrix_cluster)
        linkage_matrix = hierarchy.linkage(similarity_matrix, method="ward")
        sorted_indices = hierarchy.leaves_list(linkage_matrix)
        cluster_df = pd.DataFrame(clusters_index)
        cluster_df = cluster_df.iloc[sorted_indices[::-1]].reset_index(drop=True)
        cluster_df["Cluster"] = i
        final_clusters = pd.concat([final_clusters, cluster_df])
    final_clusters = final_clusters.iloc[:, 1:]
    final_clusters = final_clusters.rename(columns={0: "LRs"})
    final_clusters = final_clusters[["LRs", "Cluster"]]

    return final_clusters


def subset_clusters(sample, clusters):
    """Groups LR pairs using their clusters.

    Args:
        sample (dict): A dictionary containing LR matrices.
        clusters (pd.DataFrame): A DataFrame with the cluster assignments for each
        sample.

    Returns:
        dict: A dictionary with clusters as keys and the subsetted sample per cluster as
        values.
    """

    cluster_dict = {}

    for ind in clusters.index:
        cluster = clusters["Cluster"][ind]

        if cluster not in cluster_dict:
            cluster_dict[cluster] = {}
        cluster_dict[cluster][ind] = sample[ind]

    cluster_dict = dict(sorted(cluster_dict.items()))

    return cluster_dict


def calculate_cluster_interactions(sample):
    """Calculates an overall interaction matrix per cluster by combining matrices for
    different LR pairs within a sample.

    Args:
        sample (dict): A dictionary of samples containing matrices for different LR
        pairs.

    Returns:
        dict: A dictionary with clusters as keys and the overall interaction matrix for
        that cluster as the value
    """

    cluster_dict = {}

    for key in sample.keys():
        cluster_dict[key] = calculate_overall_interactions(sample[key])

    return cluster_dict

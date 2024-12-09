import pandas as pd
import numpy as np
import networkx as nx
import scipy as sp
import scanpy as sc
import gseapy as gp
import seaborn as sns
import statistics
import subprocess
import umap
import matplotlib.pyplot as plt
import importlib.resources
import anndata as ad

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from gseapy import barplot, dotplot
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, ClusterWarning
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn import preprocessing as pp
from sklearn.decomposition import PCA
from warnings import simplefilter

from . import scoring as sco
from . import plotting as pl
from . import integration as it
from . import tools as tl

simplefilter("ignore", Warning)


def calculate_dissim(sample1, sample2):
    """Calculates a dissimilarity score between two samples for each common LR pair.

    Args:
        sample1, sample2 (dict): Two samples containing matrices for LR pairs.

    Returns:
        dict: A dictionary where keys are common LR pairs and values are the
        dissimilarity scores.
    """

    if not (isinstance(sample1, dict) and isinstance(sample2, dict)):
        raise ValueError("The sample must be a dict of LR matrices.")

    dissims = {}
    for lr in set(sample1.keys()).intersection(set(sample2.keys())):
        dissims[lr] = sco.dissimilarity_score(sample1[lr], sample2[lr])

    return dissims


def get_network_diff(m1, m2):
    """Calculates the difference between two networks.

    Args:
        m1, m2 (pd.DataFrame): Two matrices to compare (as DataFrames).

    Returns:
        pd.DataFrame: A DataFrame of the differences between the two matrices.
    """

    dfs = tl.align_dataframes(m1, m2)

    return dfs[0] - dfs[1]


def perm_test(m1, m2, num_perms=100000):
    """Performs permutation testing to assess the significance of a dissimilarity score.

    Args:
        m1, m2 (pd.DataFrame): Two matrices to compare (as DataFrames).
        num_perms (int) (optional): Number of permutations to perform. Defaults to
        100000.

    Returns:
        pd.DataFrame: A DataFrame of p-values for each element in the matrices.
    """

    diff = abs(get_network_diff(m1, m2))

    result_matrix = diff.values.copy()

    def permtr(x):
        return np.apply_along_axis(
            np.random.permutation,
            axis=0,
            arr=np.apply_along_axis(np.random.permutation, axis=1, arr=x),
        )

    # Permute and test for matrix1
    perm = [permtr(diff) for _ in range(num_perms)]
    sums = np.sum([result_matrix < perm_result for perm_result in perm], axis=0)

    # Calculate p-values
    p_vals = sums / num_perms

    p_vals = pd.DataFrame(p_vals, index=diff.index, columns=diff.columns)

    return p_vals


def get_lrs_per_celltype(sample, sender, reciever):
    """Compares LR pairs between two samples for specific cell types.

    Args:
        samples (dict): A dictionary containing LR matrices for the
        first sample.
        sender (str): The sender cell type.
        reciever (str): The receiver cell type.

    Returns:
        dict: A list of LR pairs and proportion of its weighting.
    """

    if not isinstance(sample, dict):
        raise ValueError("The sample must be a dict of LR matrices.")
    
    keys = sample.keys()
    for key in keys:
        if type(sample[key]) != pd.DataFrame:
            del sample[key]

    sample = {key: df.loc[[sender]] for key, df in sample.items() if sender in df.index}

    sample = {
        key: df[[reciever]] for key, df in sample.items() if reciever in df.columns
    }

    sample = {
        key: df
        for key, df in sample.items()
        if not df.map(lambda x: x == 0).all().all()
    }

    sample = {
        key: df
        for key, df in sample.items()
        if not df.map(lambda x: x == 0).all().all()
    }

    lr_props = {}
    total = 0
    for lr_pair in set(sample.keys()):
        score = sample[lr_pair].at[sender, reciever]
        total += score

    for lr_pair in set(sample.keys()):
        lr_props[lr_pair] = sample[lr_pair].at[sender, reciever] / total

    lr_props = dict(sorted(lr_props.items(), key=lambda item: item[1], reverse=True))

    return lr_props


def get_p_vals_per_celltype(sample, sender, reciever):
    """Compares LR pairs between two samples for specific cell types.

    Args:
        samples (dict): A dictionary containing LR matrices for the
        first sample.
        sender (str): The sender cell type.
        reciever (str): The receiver cell type.

    Returns:
        dict: A dictionary containing the p-values for each interaction
    """

    if not isinstance(sample, dict):
        raise ValueError("The sample must be a dict of LR matrices.")

    keys = sample.keys()
    for key in keys:
        if type(sample[key]) != pd.DataFrame:
            del sample[key]
            
    sample = {key: df.loc[[sender]] for key, df in sample.items() if sender in df.index}

    sample = {
        key: df[[reciever]] for key, df in sample.items() if reciever in df.columns
    }

    result = {}
    for lr_pair in set(sample.keys()):
        result[lr_pair] = sample[lr_pair].at[sender, reciever]
        
    return result


def cell_network_clustering(sample, n_clusters=0, method="KMeans"):
    """Groups and ranks LR pairs using their clusters and dissimilarities.

    Args:
        sample (dict): A dictionary containing LR matrices.
        n_clusters (int) (optional): The desired number of clusters. If 0, the optimal
        number is determined using silhouette analysis. Defaults to 0.
        method (str) (optional): The clustering method to use. Defaults to 'KMeans'.

    Returns:
        pd.DataFrame: A DataFrame with the cluster assignments for each sample.
    """

    if not isinstance(sample, dict):
        raise ValueError("The sample must be a dict of LR matrices.")

    one_interaction_sample = {}

    if "per_lr_cci_cell_type" in sample:
        sample.pop("per_lr_cci_cell_type")

    # Function to check if the entire dataframe is zero
    def has_non_zero_values(df):
        return not (df == 0).all().all()

    # Filtering out key-value pairs with dataframes containing all zeros
    sample = {key: value for key, value in sample.items() if has_non_zero_values(value)}
    for key, df in list(sample.items()):
        # Check if the dataframe has more than one unique value (excluding 0)
        if (df.values.diagonal() == 0).sum() == df.shape[0] - 1 and (
            df.values == 0
        ).sum() == (df.shape[0] * df.shape[0]) - 1:
            # If only one unique value (excluding 0), remove the entry from the
            # dictionary
            one_interaction_sample[key] = df
            del sample[key]

    if sample is not None:
        # Initialize an empty dataframe to store the results
        result_df = pd.DataFrame(index=sample.keys(), columns=sample.keys())
        # Iterate through the keys and compare the dataframes
        print("Computing Dissimilarity Scores for multiple interactions...")
        with tqdm(total=len(sample), desc="Processing") as pbar:
            for key1, df1 in sample.items():
                for key2, df2 in sample.items():
                    result = sco.dissimilarity_score(
                        df1, df2, lmbda=0.5, only_non_zero=True
                    )
                    # Store the result in the result_df
                    result_df.loc[key1, key2] = result
                pbar.update(1)
        final_clusters_multiple = _lr_cluster_helper(
            result_df, sample, n_clusters, method
        )

    if one_interaction_sample is not None:
        # Initialize an empty dataframe to store the results
        result_df_one = pd.DataFrame(
            index=one_interaction_sample.keys(), columns=one_interaction_sample.keys()
        )
        # Iterate through the keys and compare the dataframes
        print("Computing Dissimilarity Scores for single interactions...")
        with tqdm(total=len(one_interaction_sample), desc="Processing") as pbar:
            for key1, df1 in one_interaction_sample.items():
                for key2, df2 in one_interaction_sample.items():
                    result = sco.dissimilarity_score(
                        df1, df2, lmbda=0.5, only_non_zero=True
                    )

                    # Store the result in the result_df
                    result_df_one.loc[key1, key2] = result
                pbar.update(1)
        n_clusters = (
            it.calculate_overall_interactions(one_interaction_sample).values.diagonal()
            == 0
        ).sum()
        final_clusters_single = _lr_cluster_helper(
            result_df_one, one_interaction_sample, n_clusters
        )
        final_clusters_single["Cluster"] = (
            final_clusters_single["Cluster"]
            + max(final_clusters_multiple["Cluster"])
            + 1
        )

    final_clusters = pd.concat([final_clusters_multiple, final_clusters_single])
    return final_clusters


def _lr_cluster_helper(result_df, sample, n_clusters=0, method="KMeans"):
    """
    Args:
        result_df (pd.DataFrame): A DataFrame containing dissimilarity scores for LRs
        sample (dict): A dictionary containing LR matrices.
        n_clusters (int) (optional): The desired number of clusters. If 0, the optimal
        number is determined using silhouette analysis. Defaults to 0.
        method (str) (optional): The clustering method to use. Defaults to 'KMeans'.

    Returns:
        pd.DataFrame: A DataFrame with the cluster assignments for each sample.
    """

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
    flatten_dfs = flatten_dfs.fillna(0)
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

        if method == "Hierarchial":
            # Perform hierarchical clustering
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
            clusters = model.fit_predict(pc_com_dist_matrix)
        if method == "KMeans":
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(pc_com_dist_matrix)

    if n_clusters == 0:
        if method == "Hierarchial":
            # Evaluate silhouette score for different numbers of clusters
            silhouette_scores = []
            for n_clusters in range(2, 11):
                clusterer = AgglomerativeClustering(
                    n_clusters=n_clusters, linkage="ward"
                )
                cluster_labels = clusterer.fit_predict(pc_com_dist_matrix)
                silhouette_avg = silhouette_score(pc_com_dist_matrix, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            plt.silhouette_scores_plot(silhouette_scores)
            # Perform hierarchical clustering
            model = AgglomerativeClustering(
                # Add 2 to account for starting with k=2
                n_clusters=np.argmax(silhouette_scores) + 2,
                linkage="ward",
            )  # as indexing starts from 0
            clusters = model.fit_predict(pc_com_dist_matrix)

        if method == "KMeans":
            # Find optimal numer of clusters Davies-Bouldin index
            db_scores = []
            for k in range(2, 11):
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(pc_com_dist_matrix)
                db_scores.append(davies_bouldin_score(pc_com_dist_matrix, labels))
            # Add 2 to account for starting with k=2
            optimal_clusters = np.argmin(db_scores) + 2
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
            clusters = kmeans.fit_predict(pc_com_dist_matrix)

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
    final_clusters.set_index("LRs", inplace=True)
    return final_clusters

    
def lr_interaction_clustering(
    sample, 
    resolution=0.5, 
    palette="Dark2_r", 
    cell_colors=None, 
    spot_size=1.5, 
    spatial_plot=True, 
    proportion_plot=True,
    return_adata=False,
    **kwargs
    ):
    """Clustering of spatial LR interaction scores on AnnData objects processed through
    stLearn.

    Args:
    sample (AnnData): An AnnData object that has been run through stLearn.
    resolution (float) (optional): The resolution to use for the clustering. Defaults to
    0.5.
    palette (str) (optional): The palette to use for the UMAP plot. Defaults to
    'Dark2_r'.
    cell_colors (dict) (optional): A dictionary mapping cell types to colors. Defaults
    to None.
    spot_size (float) (optional): The size of the spots in the spatial plot. Defaults to
    1.5.
    spatial_plot (bool) (optional): Whether to show the spatial plot. Defaults to True.
    proportion_plot (bool) (optional): Whether to show the proportion plot. Defaults to
    True.
    return_adata (bool) (optional): Whether to return the AnnData object with the
    clustering results. Defaults to False.
    **kwargs: Additional keyword arguments to pass to the scanpy spatial plot function.
    
    Returns:
    AnnData: An AnnData object with the clustering results.
    """

    LR = pd.DataFrame(sample.obsm["lr_scores"])
    LR.columns = list(sample.uns["lr_summary"].index)
    LR.index = sample.obs.index

    LR = sc.AnnData(LR)
    sc.pp.normalize_total(LR, inplace=True)
    sc.pp.log1p(LR)
    sc.pp.pca(LR)
    sc.pp.highly_variable_genes(LR, flavor="seurat", n_top_genes=2000)
    sc.pp.neighbors(LR, use_rep="X_pca", n_neighbors=15)
    sc.tl.leiden(LR, resolution=resolution)
    
    sc.tl.rank_genes_groups(LR, groupby='leiden', method='wilcoxon')
    # sc.pl.rank_genes_groups(LR, n_genes=25, sharey=False)
    sc.tl.dendrogram(LR, groupby='leiden')
    sc.pl.rank_genes_groups_dotplot(LR, n_genes=10, groupby='leiden')
    
    LR.obsm = sample.obsm
    LR.uns = sample.uns
    LR.obs["leiden"] = LR.obs["leiden"].astype("int64")

    sc.pp.pca(LR)
    sc.pp.highly_variable_genes(LR, flavor="seurat", n_top_genes=2000)
    sc.pp.neighbors(LR, use_rep="X_pca", n_neighbors=15)
    sc.tl.umap(LR)
    
    sample.obs["LR_Cluster"] = LR.obs["leiden"].astype("str")
    
    if spatial_plot:        
        # sc.pl.umap(LR, color="LR_Cluster", palette=palette, legend_loc=None)
        sc.pl.spatial(sample, color="LR_Cluster", size=spot_size, palette=palette, 
                      **kwargs)
        
    if proportion_plot:
        if "cell_type" in sample.uns:
            # make any value less than 0.1 == 0 in sample.uns['cell_type']
            sample.uns['cell_type'] = sample.uns['cell_type'].applymap(
                lambda x: 0 if x < 0.1 else x)
            
            merged_df = sample.uns['cell_type'].merge(sample.obs["LR_Cluster"], 
                                                    left_index=True, right_index=True)
            proportions = merged_df.groupby('LR_Cluster').mean()
            proportions = proportions.div(proportions.sum(axis=1), axis=0)
        else:
            proportions = sample.obs.groupby('LR_Cluster')['cell_type'] \
                .value_counts(normalize=True).unstack()

        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]})

        bars = proportions.plot(kind="bar", stacked=True, ax=ax1, 
                                color=proportions.columns.to_series().map(cell_colors))  
        # Use colormap based on cell types
        ax1.set_ylabel("Proportion")
        ax1.legend().remove()  # Remove default legend

        # Create a legend without plot elements
        handles, labels = bars.get_legend_handles_labels()  # Get handles and labels 
        # from the bar plot
        ax2.legend(handles, labels, loc='center')
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

    if return_adata:
        return LR


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

    if not isinstance(sample, dict):
        raise ValueError("The sample must be a dict of LR matrices.")

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

    if not isinstance(sample, dict):
        raise ValueError("The sample must be a dict of LR matrices.")

    cluster_dict = {}

    for key in sample.keys():
        cluster_dict[key] = it.calculate_overall_interactions(sample[key])

    return cluster_dict


def run_gsea(
    sample,
    lrs=None,
    organism="human",
    gene_sets=["KEGG_2021_Human", "MSigDB_Hallmark_2020"],
    show_dotplot=False,
    show_barplot=True,
    top_term=5,
    figsize=(3,5),
):
    """Runs GSEA analysis on a sample.

    Args:
        sample (dict): A dictionary containing LR matrices.
        lrs (list) (optional): A list of LR pairs to use for GSEA analysis instead of
        sample. Defaults to None.
        organism (str) (optional): The organism to use. Defaults to 'human'.
        gene_sets (list) (optional): The gene sets to use for gseapy analysis. Defaults
        to ['KEGG_2021_Human',
        'MSigDB_Hallmark_2020'].
        show_dotplot (bool) (optional): Whether to show the dotplot. Defaults to False.
        show_barplot (bool) (optional): Whether to show the barplot. Defaults to True.
        top_term (int) (optional): The number of top terms to show. Defaults to 5.
        figsize (tuple) (optional): The size of the figure. Defaults to (3,5).

    Returns:
        pd.DataFrame: A DataFrame with the GSEA results.
    """

    gene_list = set()

    if lrs is None:
        if not isinstance(sample, dict):
            raise ValueError("The sample must be a dict of LR matrices.")
        lrs = sample.keys()

    for lr in lrs:
        gene1, gene2 = lr.split("_")
        gene_list.add(gene1)
        gene_list.add(gene2)

    gene_list = list(gene_list)

    enr = gp.enrichr(
        gene_list=gene_list,
        gene_sets=gene_sets,
        organism=organism,
        outdir=None,
    )
        
    if show_dotplot:
        
        try:
            ax = dotplot(
                enr.results,
                column="Adjusted P-value",
                x="Gene_set",
                size=10,
                top_term=top_term,
                figsize=figsize,
                xticklabels_rot=45,
                show_ring=True,
                marker="o",
            )
        except:
            print("Could not plot dotplot.")
        
    if show_barplot:
        colours = list(plt.cm.Dark2.colors)
        colour_dict = {gene_set: colours[i] for i, gene_set in enumerate(gene_sets)}
    
        try:
            ax = barplot(
                enr.results,
                column="Adjusted P-value",
                group="Gene_set",
                size=10,
                top_term=top_term,
                figsize=figsize,
                color=colour_dict
            )
        except:
            print("Could not plot barplot.")

    return enr.results


def pathway_subset(sample, gsea_results, terms, strict=False):
    """Subsets a sample to only include interactions between genes in a set of
    pathways.

    Args:
        sample (dict): The sample to subset.
        gsea_results (pd.DataFrame): The GSEA results to use to subset the sample.
        terms (list): The terms to subset the sample with.
        strict (bool): Whether to only include interactions between genes in the
        same pathway.

    Returns:
        dict: The subsetted sample.
    """

    genes = []
    grouped = {}

    for term in terms:
        filtered_df = gsea_results[gsea_results['Term'] == term]
        gene_list = filtered_df['Genes'].tolist()

        for gene in gene_list:
            genes.extend(gene.lower().split(";"))

    for key in sample.keys():
        lig, rec = key.lower().split("_")
        if strict:
            if lig in genes and rec in genes:
                grouped[key] = sample[key]
        else:
            if lig in genes or rec in genes:
                grouped[key] = sample[key]

    return grouped


def add_lr_module_score(sample, lr_list, key_name="score"):
    """Adds a module score to an AnnData object run through stLearn based on the
    interactions in a list of ligand-receptor pairs.

    Args:
        sample (AnnData): The AnnData object to add the score to. Must be processed
        through stLearn.
        lr_list (list): The list of ligand-receptor pairs to use.
        key_name (str): The key to use for the score.

    Returns:
        AnnData: The AnnData object with the module score added.
    """

    lr_counts = pd.DataFrame(sample.obsm['lr_sig_scores'])
    lr_counts.index = sample.obs.index
    lr_counts.columns = sample.uns['lr_summary'].index

    adata = ad.AnnData(lr_counts)
    sc.tl.score_genes(adata, gene_list=lr_list)
    sample.obs[key_name] = adata.obs['score']

    return sample

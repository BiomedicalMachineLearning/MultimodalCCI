import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

from scipy import stats
from tqdm import tqdm

from . import scoring as sc
from . import tools as tl


def normalise_within_tech(samples, sample_sizes, target=None):
    """Normalizes LR matrices within a list of samples to a target sample based on
    number of spots to account for differences in sample size.

    Args:
        samples (list): A list of dictionaries of LR matrices.
        sample_sizes (list): A list of number of spots for each sample.
        target (int) (optional): A target sample size to normalize to. If not provided,
        the first sample's size is used.

    Returns:
        list: The list of samples with normalized matrices.
    """

    if not isinstance(samples, list):
        raise ValueError("Samples must be a list of dicts of LR matrices.")

    if target is None:
        target = sample_sizes[0]

    for i, sample in enumerate(samples):
        for lr, matrix in sample.items():
            samples[i][lr] = matrix * (target / sample_sizes[i])

    return samples


def get_lr_pairs(samples, method=">=50%"):
    """Identifies the LR pairs present in a list of samples according to the given
    method.

    Args:
        samples (list): A list of dictionaries of LR matrices
        method (str) (optional): The method to use for identifying LR pairs. Options are
        "all", ">=50%", ">50%", and "any". Defaults to ">=50%".

    Returns:
        list: A list of LR pairs that are present in a majority of samples
    """

    if not isinstance(samples, list):
        raise ValueError("Samples must be a list of dicts of LR matrices.")

    lr_pairs_counts = {}
    lr_pairs = []

    for sample in samples:
        for lr_pair, matrix in sample.items():
            if sum(sum(matrix.values)) != 0:
                lr_pairs_counts[lr_pair] = lr_pairs_counts.setdefault(lr_pair, 0) + 1

    for lr_pair, count in lr_pairs_counts.items():
        if method == "all":
            if count == len(samples):
                lr_pairs.append(lr_pair)
        elif method == ">=50%":
            if count >= len(samples) / 2:
                lr_pairs.append(lr_pair)
        elif method == ">50%":
            if count > len(samples) / 2:
                lr_pairs.append(lr_pair)
        elif method == "any":
            lr_pairs.append(lr_pair)
        else:
            raise ValueError("Method must be 'all', '>=50%', '>50%', or 'any'.")

    return lr_pairs


def subset_samples(samples, lr_pairs):
    """Subsets each sample in samples to only contain the LR pairs in lr_pairs.

    Args:
        samples (list): A list of dictionaries of LR matrices.
        lr_pairs (list): A list of LR pairs to include in the subsetted samples.

    Returns:
        list: A list of subsetted dictionaries of LR matrices containing only
        the LR pairs in lr_pairs.
    """

    if not isinstance(samples, list):
        raise ValueError("Samples must be a list of dicts of LR matrices.")

    subsetted_samples = []

    for sample in samples:
        subsetted_samples.append({lr: sample[lr] for lr in lr_pairs})

    return subsetted_samples


def normalise_between_tech(samples, method="mean"):
    """Normalizes matrices between different technologies.

    Args:
        samples (list): A nested list of samples, organized by technology.
        Example: [[t1s1, t1s2], [t2s1, t2s2]].
        method (str) (optional): The normalization method, either "mean" or "sum".
        Defaults to "mean". Normalises based on either the mean or sum of the values in
        the matrix.

    Returns:
        The normalized nested list of samples.
    """

    if not isinstance(samples, list):
        raise ValueError("Samples must be a list of dicts of LR matrices.")

    tech_counts = []
    for tech_samples in samples:
        total_counts = 0
        lr_pair_count = 0
        for pair, matrix in tech_samples.items():
            lr_pair_count += 1
            if method == "mean":
                total_counts += matrix.mean().mean()
            elif method == "sum":
                total_counts += matrix.sum().sum()
            else:
                raise ValueError("Invalid method option.")

        total_counts = total_counts / lr_pair_count
        tech_counts.append(total_counts)

    for tech in range(len(samples)):
        for lr, df in samples[tech].items():
            factor = max(tech_counts) / tech_counts[tech]
            samples[tech][lr] = df * factor

    return samples


def integrate_samples(samples, method=">=50%", sum=False, strict=False):
    """Integrates matrices from different technologies.

    Args:
        samples (list): A list of samples with different technologies.
        method (str) (optional): The method to use for identifying LR pairs. Options are
        "all", ">=50%", ">50%", and "any". Defaults to ">=50%".
        sum (bool) (optional): Whether to sum instead of multiply the matrices. Defaults
        to False.
        strict (bool) (optional): If True, only interactions where more than 50% of the 
        values are non-zero will be multiplied. Defaults to False.

    Returns:
        dict: A dictionary where keys are LR pairs and values are the integrated
        matrices.
    """

    if not isinstance(samples, list):
        raise ValueError("Samples must be a list of dicts of LR matrices.")

    integrated = {}
    lr_matrices = {}

    if len(samples) >= 2:
        lr_pairs = sorted(get_lr_pairs(samples, method=method))
    else:
        raise ValueError("Integration needs at least two samples")

    for i in range(len(lr_pairs)):
        lr = lr_pairs[i]

        for tech in range(len(samples)):
            if lr in samples[tech]:
                if lr in lr_matrices:
                    lr_matrices[lr].append(samples[tech][lr])
                else:
                    lr_matrices[lr] = [samples[tech][lr]]

    with tqdm(total=len(lr_pairs), desc="Integrating LR matrices") as pbar:
        for lr, matrices in lr_matrices.items():
            if len(matrices) == 2:
                if sum:
                    matrices[0], matrices[1] = tl.align_dataframes(matrices[0], 
                                                                   matrices[1])
                    integrated[lr] = matrices[0] + matrices[1]
                    integrated[lr] = integrated[lr] / 2
                    integrated[lr] = integrated[lr].fillna(0)
                else:
                    integrated[lr] = (matrices[0] * matrices[1]).fillna(0)
                    integrated[lr] = np.sqrt(integrated[lr]).fillna(0)
            elif len(matrices) > 2:
                if sum:
                    integrated[lr] = matrices[0]
                    for i in range(1, len(matrices)):
                        integrated[lr], matrices[i] = tl.align_dataframes(
                            integrated[lr], matrices[i])
                        integrated[lr] = integrated[lr] + matrices[i]
                    integrated[lr] = integrated[lr] / len(matrices)
                    integrated[lr] = integrated[lr].fillna(0)
                else:
                    integrated[lr] = sc.multiply_non_zero_values(matrices, 
                                                                 strict=strict)
            else:
                integrated[lr] = matrices[0]
            tqdm.update(pbar, 1)

    return integrated


def calculate_overall_interactions(sample, normalisation=True):
    """Calculates an overall interaction matrix by combining matrices for different LR
    pairs within a sample.

    Args:
        sample (dict): A sample containing matrices for different LR pairs.

    Returns:
        pd.DataFrame: A matrix representing the overall interactions.
    """

    if not isinstance(sample, dict):
        raise ValueError("The sample must be a dict of dicts of LR matrices.")

    total = None
    for lr in sample.keys():
        if sample[lr].sum().sum() > 0:
            if total is not None:
                if normalisation:
                    total = total + sample[lr] / sample[lr].sum().sum()
                else:
                    total = total + sample[lr]
                total = total.fillna(0)
            else:
                if normalisation:
                    total = sample[lr] / sample[lr].sum().sum()
                else:
                    total = sample[lr]
                total = total.fillna(0)

    if total is None:
        return None

    total = total / total.sum().sum()
    total = total.fillna(0)

    return total


def correct_pvals_matrix(dataframes, method="stouffer"):
    """Corrects p-values across a list of pandas DataFrames.

    Args:
        dataframes (list): A list of pandas DataFrames with p vals to combine.
        method (str) (optional): The method to use for combining p-values. Options are
        "stouffer" and "fisher". Defaults to "stouffer".

    Returns:
        pd.DataFrame: A DataFrame with corrected p-values.
    """

    result_df = dataframes[0]

    for i in range(len(dataframes)):
        dataframes[i], result_df = tl.align_dataframes(
            dataframes[i], result_df, fill_value=np.NaN)

    for i in range(len(dataframes)):
        dataframes[i], result_df = tl.align_dataframes(
            dataframes[i], result_df, fill_value=np.NaN)

    result_df = result_df.astype(np.float64)
    for i, row in result_df.iterrows():
        for j in row.index:
            values = [df.loc[i, j] for df in dataframes]
            values = [
                0.00000000001 if x == 0 else (
                    0.9999999999 if x == 1 else x) for x in values]
            values = [x for x in values if not np.isnan(x)]
            result_df.loc[i, j] = stats.combine_pvalues(values, method=method, )[1]

    return result_df


def integrate_p_vals(samples, method="stouffer"):
    """Integrates p-values from different samples.

    Args:
        samples (list): A list of dictionaries of LR matrices.
        method (str) (optional): The method to use for combining p-values. Options
        are "stouffer" and "fisher". Defaults to "stouffer".

    Returns:
        dict: A dictionary where keys are LR pairs and values are the integrated
        p-values.
    """

    if not isinstance(samples, list):
        raise ValueError("Samples must be a list of dicts of LR matrices.")

    integrated = {}
    lr_matrices = {}

    lr_pairs = set()
    for sample in samples:
        lr_pairs.update(sample.keys())
    lr_pairs = list(lr_pairs)

    for i in range(len(lr_pairs)):
        lr = lr_pairs[i]

        for tech in range(len(samples)):
            if lr in samples[tech]:
                if lr in lr_matrices:
                    lr_matrices[lr].append(samples[tech][lr])
                else:
                    lr_matrices[lr] = [samples[tech][lr]]

    with tqdm(total=len(lr_matrices), desc="Integrating p values") as pbar:
        for lr, matrices in lr_matrices.items():
            integrated[lr] = correct_pvals_matrix(matrices, method=method)
            pbar.update(1)

    return integrated


def remove_insignificant(sample, p_vals, cutoff=0.05):
    """Removes insignificant interactions from a sample based on p-values.

    Args:
        sample (dict): A sample containing matrices for different LR pairs.
        p_vals (dict): A dictionary of p-values for the LR pairs in the sample.

    Returns:
        dict: A dictionary of matrices with insignificant interactions set to 0.
    """

    corrected_sample = {}
    sample_copy = copy.deepcopy(sample)
    for lr, matrix in sample_copy.items():
        for i, row in matrix.iterrows():
            for j in row.index:
                if p_vals[lr].loc[i, j] > cutoff:
                    matrix.loc[i, j] = 0
        corrected_sample[lr] = matrix

    return corrected_sample

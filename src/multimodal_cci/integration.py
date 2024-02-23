import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import scoring as sc


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


def get_majority_lr_pairs(samples, equal_to=False):
    """Identifies the LR pairs present in a majority of samples.

    Args:
        samples (list): A list of dictionaries of LR matrices
        equal_to (bool) (optional): If True, includes LR pairs present in exactly half
        of the samples. Defaults to False.

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
        if equal_to:
            if count >= len(samples) / 2:
                lr_pairs.append(lr_pair)
        else:
            if count > len(samples) / 2:
                lr_pairs.append(lr_pair)

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


def integrate_samples(samples, all_lr_pairs=False):
    """Integrates matrices from different technologies.

    Args:
        samples (list): A list of samples with different technologies.
        all_lr_pairs (bool) (optional): If True and two samples are used, use all the LR
        pairs from either sample. Defaults to False.

    Returns:
        dict: A dictionary where keys are LR pairs and values are the integrated
        matrices.
    """

    if not isinstance(samples, list):
        raise ValueError("Samples must be a list of dicts of LR matrices.")

    integrated = {}
    lr_matrices = {}

    if len(samples) == 2:
        if all_lr_pairs:
            lr_pairs = get_majority_lr_pairs(samples, equal_to=True)
        else:
            lr_pairs = get_majority_lr_pairs(samples, equal_to=False)

    elif len(samples) > 2:
        lr_pairs = sorted(get_majority_lr_pairs(samples, equal_to=True))

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

    for lr, matrices in lr_matrices.items():
        if len(matrices) == 2:
            integrated[lr] = (matrices[0] * matrices[1]).fillna(0)
            integrated[lr] = np.sqrt(integrated[lr]).fillna(0)
        elif len(matrices) > 2:
            integrated[lr] = sc.multiply_non_zero_values(matrices)

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

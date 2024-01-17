import numpy as np
import pandas as pd


def align_dataframes(m1, m2):
    """Aligns two DataFrames by matching their indices and columns, filling missing
    values with 0.

    Args:
        m1, m2 (pd.DataFrame): The DataFrames to align.

    Returns:
        tuple: A tuple of the aligned DataFrames.
    """

    m1, m2 = m1.align(m2, fill_value=0)

    columns = sorted(set(m1.columns) | set(m2.columns))
    m1 = m1.reindex(columns, fill_value=0)
    m2 = m2.reindex(columns, fill_value=0)

    rows = sorted(set(m1.index) | set(m2.index))
    m1 = m1.reindex(rows, fill_value=0)
    m2 = m2.reindex(rows, fill_value=0)

    return m1, m2


def dissimilarity_score(
    m1, m2, lmbda=0.5, normalise=False, binary=False, trim=False, only_non_zero=False
):
    """Calculates a dissimilarity score between two matrices.

    Args:
        m1, m2 (pd.DataFrame): Two matrices to compare.
        lmbda (float) (optional): Weighting factor for weighted vs binary dissimilarity
        (0-1). 0 is fully binary and 1 is fully weighted. Defaults to 0.5.
        normalise (bool) (optional): Normalizes matrices before comparison. Defaults to
        False.
        binary (bool) (optional): Treats matrices as binary (0 or 1). Defaults to False.
        trim (bool) (optional): Trims matrices to common rows and columns. Otherwise
        pads 0s to uncommon rows and columns. Defaults to False.
        only_non_zero (bool) (optional): Only considers non-zero edges for calculation.
        Defaults to False.

    Returns:
        pd.DataFrame: The dissimilarity scores between the two matrices.
    """

    if trim:
        common_rows = list(set(m1.index) & set(m2.index))
        common_cols = list(set(m1.columns) & set(m2.columns))

        m1 = m1.loc[common_rows, common_cols]
        m2 = m2.loc[common_rows, common_cols]

    else:
        m1, m2 = align_dataframes(m1, m2)

    m1 = m1.values
    m2 = m2.values

    if normalise:
        if m1.sum().sum() == 0 and m2.sum().sum() == 0:
            return 0
        if m1.sum().sum() != 0 and m2.sum().sum() != 0:
            m1 = m1 / m1.sum().sum()
            m2 = m2 / m2.sum().sum()

    if binary:
        m1 = np.where(m1 > 0, 1, 0)
        m2 = np.where(m2 > 0, 1, 0)

    n_of_edges = len(m1) ** 2

    if only_non_zero:
        n_of_edges = np.where((m1 + m2) > 0, 1, 0).sum().sum()

    abs_weight_difference = np.abs(m1 - m2)
    weight_sum = m1 + m2

    # Avoid division by zero
    weight_sum[weight_sum == 0] = -1
    norm_weight_difference = abs_weight_difference / weight_sum
    norm_weight_difference_sum = np.sum(np.sum(norm_weight_difference))
    wt_dissim = lmbda * (norm_weight_difference_sum / n_of_edges)
    wt_dissim = wt_dissim.fillna(0)

    n_diff = np.where(abs_weight_difference > 0, 1, 0).sum().sum()
    bin_dissim = (1 - lmbda) * (n_diff / n_of_edges)
    bin_dissim = bin_dissim.fillna(0)

    return wt_dissim + bin_dissim


def perm_test(m1, m2, num_perms=100000):
    """Performs permutation testing to assess the significance of a dissimilarity score.

    Args:
        m1, m2 (pd.DataFrame): Two matrices to compare (as DataFrames).
        num_perms (int) (optional): Number of permutations to perform. Defaults to
        100000.

    Returns:
        pd.DataFrame: A DataFrame of p-values for each element in the matrices.
    """

    dfs = align_dataframes(m1, m2)

    matrix1 = dfs[0].values
    matrix2 = dfs[1].values

    result_matrix1 = dfs[0].values.copy()
    result_matrix2 = dfs[1].values.copy()

    def permtr(x):
        return np.apply_along_axis(
            np.random.permutation,
            axis=0,
            arr=np.apply_along_axis(np.random.permutation, axis=1, arr=x),
        )

    # Permute and test for matrix1
    perm1 = [permtr(matrix2) for _ in range(num_perms)]
    sums1 = np.sum([result_matrix1 < perm_result for perm_result in perm1], axis=0)

    # Permute and test for matrix2
    perm2 = [permtr(matrix1) for _ in range(num_perms)]
    sums2 = np.sum([result_matrix2 < perm_result for perm_result in perm2], axis=0)

    # Calculate p-values
    p_vals = (sums1 + sums2) / (2 * num_perms)

    p_vals = pd.DataFrame(p_vals, index=m1.index, columns=m1.columns)

    return p_vals


def multiply_non_zero_values(dataframes):
    """Multiply non-zero values across a list of pandas DataFrames.

    Parameters:
    - dataframes (list): A list of pandas DataFrames with the same shape and column/row
    names.

    Returns:
    - pd.DataFrame: A new DataFrame where each cell contains the product of non-zero
    values or zero if more than 50% of the values in the corresponding cells are zero.
    """

    result_df = dataframes[0]

    for i in range(len(dataframes)):
        dataframes[i], result_df = align_dataframes(dataframes[i], result_df)

    for i in range(len(dataframes)):
        dataframes[i], result_df = align_dataframes(dataframes[i], result_df)

    for i, row in result_df.iterrows():
        for j in row.index:
            values = [df.loc[i, j] for df in dataframes]
            non_zero_values = [value for value in values if value != 0]

            if len(non_zero_values) / len(values) <= 0.5:
                result_df.loc[i, j] = 0
            else:
                result_df.loc[i, j] = np.prod(non_zero_values)

    result_df = np.power(result_df, 1 / len(values)).fillna(0)

    return result_df

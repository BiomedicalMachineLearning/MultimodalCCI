import numpy as np

from . import tools as tl


def dissimilarity_score(
    m1,
    m2,
    lmbda=0.5,
    normalise=False,
    binary=False,
    trim=False,
    only_non_zero=False
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
        m1, m2 = tl.align_dataframes(m1, m2)

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

    n_diff = np.where(abs_weight_difference > 0, 1, 0).sum().sum()
    bin_dissim = (1 - lmbda) * (n_diff / n_of_edges)

    return wt_dissim + bin_dissim


def multiply_non_zero_values(dataframes, strict=False):
    """Multiply non-zero values across a list of pandas DataFrames.

    Parameters:
    - dataframes (list): A list of pandas DataFrames with the same shape and column/row
    names.
    - strict (bool) (optional): If True, only interactions where more than 50% of the 
    values are non-zero will be multiplied. Defaults to False.

    Returns:
    - pd.DataFrame: A new DataFrame where each cell contains the product of non-zero
    values or zero if more than 50% of the values in the corresponding cells are zero.
    """

    result_df = dataframes[0]

    for i in range(len(dataframes)):
        dataframes[i], result_df = tl.align_dataframes(dataframes[i], result_df)

    for i in range(len(dataframes)):
        dataframes[i], result_df = tl.align_dataframes(dataframes[i], result_df)

    result_df = result_df.astype(np.float64)
    for i, row in result_df.iterrows():
        for j in row.index:
            values = [df.loc[i, j] for df in dataframes]
            non_zero_values = [value for value in values if value != 0]

            if strict:
                if len(non_zero_values) / len(values) <= 0.5:
                    result_df.loc[i, j] = 0
                else:
                    result_df.loc[i, j] = np.prod(non_zero_values, dtype=np.float64)
            else:
                result_df.loc[i, j] = np.prod(non_zero_values, dtype=np.float64)

    result_df = np.power(result_df, 1 / len(values)).fillna(0)

    return result_df

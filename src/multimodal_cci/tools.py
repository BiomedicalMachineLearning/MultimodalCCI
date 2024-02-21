import pandas as pd
import numpy as np
import scanpy as sc


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


def rename_celltypes(sample, replacements):
    """Renames cell types in a sample.

    Args:
        sample (dict): A dict of LR matrices.
        replacements (dict): A dictionary of replacements, where the keys are the old
            cell type names and the values are the new cell type names.

    Returns:
        dict: A dict of LR matrices with the cell type names replaced.
    """

    if not isinstance(sample, dict):
        raise ValueError("The sample must be a dict of LR matrices.")

    renamed = sample.copy()
    for lr_pair in renamed.keys():
        renamed[lr_pair].rename(
            index=replacements,
            columns=replacements,
            inplace=True)

    return renamed


def read_stLearn(path):
    """Reads a stLearn ligand-receptor analysis output and converts it to a dictionary
    of LR matrices that can be used with the multimodal cci functions.

    Args:
        path (str): The path to the stLearn ligand-receptor analysis output.

    Returns:
        dict: A dictionary of LR matrices.
        int: The number of cells or spots in the dataset.
    """

    adata = sc.read_h5ad(path)

    return adata.uns['per_lr_cci_cell_type'], adata.shape[0]


def read_CellPhoneDB(path):
    """Reads a CellPhoneDB interaction scores txt file and converts it to a dictionary
    of LR matrices that can be used with the multimodal cci functions.

    Args:
        path (str): the path to the interaction scores txt file.

    Returns:
        dict: A dictionary of LR matrices.
    """

    cp_obj = pd.read_csv(path, delimiter="\t")

    sample = {}

    for ind in cp_obj.index:

        key = cp_obj["interacting_pair"][ind]

        val = cp_obj.iloc[ind, cp_obj.columns.get_loc("classification") + 1:]
        val = pd.DataFrame({"Index": val.index, "Value": val.values})
        val[["Row", "Col"]] = val["Index"].str.split("|", expand=True)
        val = val.drop("Index", axis=1)
        val = val.pivot(index="Row", columns="Col", values="Value")
        val = val.rename_axis(None, axis=1).rename_axis(None)

        sample[key] = val

    return sample


def read_Squidpy(result):
    """Reads a Squidpy ligand-receptor analysis output and converts it to a dictionary
    of LR matrices that can be used with the multimodal cci functions.

    Args:
        result (dict): The output from squidpy.gr.ligrec.

    Returns:
        dict: A dictionary of LR matrices.
    """

    lr_dict = {}
    pvals = pd.DataFrame(result["pvalues"])

    cci_names = np.array([col[0] + "--" + col[1] for col in pvals.columns])
    cell_type_set = np.unique([col[0] for col in pvals.columns])

    for i, row in enumerate(pvals.index):

        int_matrix = np.zeros((len(cell_type_set), len(cell_type_set)))
        lr_ = "_".join(list(row))

        # Getting sig CCIs for this lr #
        lr_pvals = np.array(pvals.values[i, :])
        sig_bool = lr_pvals < 0.05
        lr_ccis = cci_names[sig_bool]
        lig = list(row)[0]
        rec = list(row)[1]
        for j, cci in enumerate(lr_ccis):
            c1, c2 = cci.split("--")
            row = np.where(cell_type_set == c1)[0][0]
            col = np.where(cell_type_set == c2)[0][0]
            int_matrix[row, col] = result["means"][c1][c2][lig][rec]

        lr_dict[lr_] = pd.DataFrame(
            int_matrix, index=cell_type_set, columns=cell_type_set
        )

    return lr_dict


def read_CellChat(path):
    """Reads a CellChat ligand-receptor analysis output (cellchat@dr) and converts it to
    a dictionary of LR matrices that can be used with the multimodal cci functions.

    Args:
        path (str): The output from cellchat@dr.

    Returns:
        dict: A dictionary of LR matrices.
    """

    result = pd.read_csv(path)
    lr_dict = {}

    cell_type_set = np.unique(np.concatenate([result["source"], result["target"]]))

    for i in result.index:
        lr_ = result['interaction_name'][i]
        if lr_ not in lr_dict.keys():
            int_matrix = np.zeros((len(cell_type_set), len(cell_type_set)))
        else:
            int_matrix = lr_dict[lr_].values
        row = np.where(cell_type_set == result["source"][i])[0][0]
        col = np.where(cell_type_set == result["target"][i])[0][0]
        int_matrix[row, col] = result["prob"][i]

        lr_dict[lr_] = pd.DataFrame(
            int_matrix, index=cell_type_set, columns=cell_type_set
        )

    return lr_dict


def read_NATMI(path):
    """Reads a NATMI ligand-receptor analysis output (Edges_lrc2p.csv) and converts it
    to a dictionary of LR matrices that can be used with the multimodal cci functions.

    Args:
        path (str): The path to Edges_lrc2p.csv.

    Returns:
        dict: A dictionary of LR matrices.
    """

    result = pd.read_csv(path)
    lr_dict = {}

    cell_type_set = np.unique(np.concatenate(
        [result["Sending cluster"], result["Target cluster"]]))

    for i in result.index:
        lr_ = result['Ligand symbol'][i] + "_" + result['Receptor symbol'][i]
        if lr_ not in lr_dict.keys():
            int_matrix = np.zeros((len(cell_type_set), len(cell_type_set)))
        else:
            int_matrix = lr_dict[lr_].values
        row = np.where(cell_type_set == result["Sending cluster"][i])[0][0]
        col = np.where(cell_type_set == result["Target cluster"][i])[0][0]
        int_matrix[row, col] = result["Edge average expression weight"][i]

        lr_dict[lr_] = pd.DataFrame(
            int_matrix, index=cell_type_set, columns=cell_type_set
        )

    return lr_dict

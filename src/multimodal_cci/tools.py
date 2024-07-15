import pandas as pd
import numpy as np
import scanpy as sc


def align_dataframes(m1, m2, fill_value=0):
    """Aligns two DataFrames by matching their indices and columns, filling missing
    values with 0.

    Args:
        m1, m2 (pd.DataFrame): The DataFrames to align.

    Returns:
        tuple: A tuple of the aligned DataFrames.
    """

    m1, m2 = m1.align(m2, fill_value=fill_value)

    columns = sorted(set(m1.columns) | set(m2.columns))
    m1 = m1.reindex(columns, fill_value=fill_value)
    m2 = m2.reindex(columns, fill_value=fill_value)

    rows = sorted(set(m1.index) | set(m2.index))
    m1 = m1.reindex(rows, fill_value=fill_value)
    m2 = m2.reindex(rows, fill_value=fill_value)

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


def read_stLearn(path, key="cell_type", return_adata=False):
    """Reads a stLearn ligand-receptor analysis output and converts it to a dictionary
    of LR matrices that can be used with the multimodal cci functions.

    Args:
        path (str): The path to the stLearn ligand-receptor analysis output.
        key (str) (optional): The key in adata.obs that was used for CCI. Defaults to
            "cell_type".
        return_adata (bool) (optional): Whether to return the AnnData object as well.
            Defaults to False.

    Returns:
        dict: The n_spots, lr_scores, lr_pvals, and AnnData (optional).
    """

    adata = sc.read_h5ad(path)

    if return_adata:
        out = {
            "n_spots": adata.shape[0],
            "lr_scores": adata.uns[f"per_lr_cci_raw_{key}"],
            "lr_pvals": adata.uns[f"per_lr_cci_pvals_{key}"],
            "adata": adata
        }
    else:
        out = {
            "n_spots": adata.shape[0],
            "lr_scores": adata.uns[f"per_lr_cci_raw_{key}"],
            "lr_pvals": adata.uns[f"per_lr_cci_pvals_{key}"]
        }

    return out


def convert_stLearn(adata, key="cell_type", return_adata=False):
    """Reads a stLearn ligand-receptor analysis output and converts it to a dictionary
    of LR matrices that can be used with the multimodal cci functions.

    Args:
        adata (AnnData): The stLearn ligand-receptor analysis output.
        key (str) (optional): The key in adata.obs that was used for CCI. Defaults to
            "cell_type".
        return_adata (bool) (optional): Whether to return the AnnData object as well.
            Defaults to False.

    Returns:
        dict: The n_spots, lr_scores, lr_pvals, and AnnData (optional).
    """

    if return_adata:
        out = {
            "n_spots": adata.shape[0],
            "lr_scores": adata.uns[f"per_lr_cci_raw_{key}"],
            "lr_pvals": adata.uns[f"per_lr_cci_pvals_{key}"],
            "adata": adata
        }
    else:
        out = {
            "n_spots": adata.shape[0],
            "lr_scores": adata.uns[f"per_lr_cci_raw_{key}"],
            "lr_pvals": adata.uns[f"per_lr_cci_pvals_{key}"]
        }

    return out


def read_CellPhoneDB(means_path, pvals_path):
    """Reads a CellPhoneDB interaction scores txt file and converts it to a dictionary
    of LR matrices that can be used with the multimodal cci functions.

    Args:
        means_path (str): Path to the means txt file.
        pvals_path (str): Path to the pvals txt file.

    Returns:
        dict: The LR scores and p_vals
    """

    cp_obj = pd.read_csv(means_path, delimiter="\t")
    lr_dict = {}
    for ind in cp_obj.index:

        key = cp_obj["interacting_pair"][ind]

        val = cp_obj.iloc[ind, cp_obj.columns.get_loc("classification") + 1:]
        val = pd.DataFrame({"Index": val.index, "Value": val.values})
        val[["Row", "Col"]] = val["Index"].str.split("|", expand=True)
        val = val.drop("Index", axis=1)
        val = val.pivot(index="Row", columns="Col", values="Value")
        val = val.rename_axis(None, axis=1).rename_axis(None)
        lr_dict[key] = val

    cp_obj = pd.read_csv(pvals_path, delimiter="\t")
    p_vals_dict = {}
    for ind in cp_obj.index:

        key = cp_obj["interacting_pair"][ind]

        val = cp_obj.iloc[ind, cp_obj.columns.get_loc("classification") + 1:]
        val = pd.DataFrame({"Index": val.index, "Value": val.values})
        val[["Row", "Col"]] = val["Index"].str.split("|", expand=True)
        val = val.drop("Index", axis=1)
        val = val.pivot(index="Row", columns="Col", values="Value")
        val = val.rename_axis(None, axis=1).rename_axis(None)
        val = val.fillna(1)
        p_vals_dict[key] = val

    out = {
        "lr_scores": lr_dict,
        "lr_pvals": p_vals_dict
    }

    return out


def read_Squidpy(result):
    """Reads a Squidpy ligand-receptor analysis output and converts it to a dictionary
    of LR matrices that can be used with the multimodal cci functions.

    Args:
        result (dict): The output from squidpy.gr.ligrec.

    Returns:
        dict: The LR scores and p_vals.
    """

    lr_dict = {}
    pvals = pd.DataFrame(result["pvalues"])

    cci_names = np.array([col[0] + "--" + col[1] for col in pvals.columns])
    cell_type_set = np.unique([col[0] for col in pvals.columns])

    for i, row in enumerate(pvals.index):

        int_matrix = np.zeros((len(cell_type_set), len(cell_type_set)))
        lr_ = "_".join(list(row))
        lr_ccis = cci_names
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

    pvals_dict = {}

    for i, row in enumerate(pvals.index):

        int_matrix = np.zeros((len(cell_type_set), len(cell_type_set)))
        lr_ = '_'.join(list(row))
        lr_ccis = cci_names
        lig = list(row)[0]
        rec = list(row)[1]
        for j, cci in enumerate(lr_ccis):
            c1, c2 = cci.split('--')
            row = np.where(cell_type_set == c1)[0][0]
            col = np.where(cell_type_set == c2)[0][0]
            int_matrix[row, col] = result['pvalues'][c1][c2][lig][rec]

        df = pd.DataFrame(int_matrix, index=cell_type_set, columns=cell_type_set)
        df = df.fillna(1)
        pvals_dict[lr_] = df

    out = {
        "lr_scores": lr_dict,
        "lr_pvals": pvals_dict
    }

    return out


def read_CellChat(path):
    """Reads a CellChat ligand-receptor analysis output (cellchat@dr) and converts it to
    a dictionary of LR matrices that can be used with the multimodal cci functions.

    Args:
        path (str): The output from cellchat@dr.

    Returns:
        dict: The LR scores and p_vals
    """

    result = pd.read_csv(path)
    lr_dict = {}
    pvals_dict = {}

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

    for i in result.index:
        lr_ = result['interaction_name'][i]
        if lr_ not in lr_dict.keys():
            int_matrix = np.zeros((len(cell_type_set), len(cell_type_set)))
        else:
            int_matrix = lr_dict[lr_].values
        row = np.where(cell_type_set == result["source"][i])[0][0]
        col = np.where(cell_type_set == result["target"][i])[0][0]
        int_matrix[row, col] = result["pval"][i]

        df = pd.DataFrame(
            int_matrix, index=cell_type_set, columns=cell_type_set
        )
        df = df.fillna(1)
        pvals_dict[lr_] = df

    out = {
        "lr_scores": lr_dict,
        "lr_pvals": pvals_dict
    }

    return out


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

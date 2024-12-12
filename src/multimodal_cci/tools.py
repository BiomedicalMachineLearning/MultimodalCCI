import pandas as pd
import numpy as np
import scanpy as sc
from typing import Dict, List, Optional, Union
from CCIData import CCIData


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


def read_stLearn(path, key="cell_type", save_anndata=True) -> CCIData:
    """Reads a stLearn ligand-receptor analysis output and converts it to a CCIData 
    object.

    Args:
        path (str): The path to the stLearn ligand-receptor analysis output.
        key (str) (optional): The key in adata.obs that was used for CCI. Defaults to
            "cell_type".
        save_anndata (bool) (optional): Whether to save the AnnData object in the 
            CCIData object. Defaults to True.

    Returns:
        CCIData: The CCIData object.
    """

    adata = sc.read_h5ad(path)

    if save_anndata:
        cci_data = CCIData(
            n_spots=adata.shape[0], 
            cci_scores=adata.uns[f"per_lr_cci_raw_{key}"], 
            p_values=adata.uns[f"per_lr_cci_pvals_{key}"],
            adata=adata
            )
    else:
        cci_data = CCIData(
            n_spots=adata.shape[0], 
            cci_scores=adata.uns[f"per_lr_cci_raw_{key}"], 
            p_values=adata.uns[f"per_lr_cci_pvals_{key}"]
            )

    return cci_data


def convert_stLearn(adata, key="cell_type", save_anndata=True) -> CCIData:
    """Reads a stLearn ligand-receptor analysis output and converts it to a CCIData 
    object.

    Args:
        adata (AnnData): The stLearn ligand-receptor analysis output.
        key (str) (optional): The key in adata.obs that was used for CCI. Defaults to
            "cell_type".
        save_anndata (bool) (optional): Whether to save the AnnData object in the 
            CCIData object. Defaults to True.

    Returns:
        CCIData: The CCIData object.
    """

    if save_anndata:
        cci_data = CCIData(
            n_spots=adata.shape[0], 
            cci_scores=adata.uns[f"per_lr_cci_raw_{key}"], 
            p_values=adata.uns[f"per_lr_cci_pvals_{key}"],
            adata=adata
            )
    else:
        cci_data = CCIData(
            n_spots=adata.shape[0], 
            cci_scores=adata.uns[f"per_lr_cci_raw_{key}"], 
            p_values=adata.uns[f"per_lr_cci_pvals_{key}"]
            )

    return cci_data


def read_CellPhoneDB(means_path, pvals_path, n_spots=None) -> CCIData:
    """Reads a CellPhoneDB interaction scores txt file and converts it to a CCIData
    object.

    Args:
        means_path (str): Path to the means txt file.
        pvals_path (str): Path to the pvals txt file.
        n_spots (int) (optional): The number of spots. Defaults to None.

    Returns:
        CCIData: The CCIData object.
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

    cci_data = CCIData(n_spots=n_spots, cci_scores=lr_dict, p_values=p_vals_dict)

    return cci_data


def read_Squidpy(result, n_spots=None) -> CCIData:
    """Reads a Squidpy ligand-receptor analysis output and converts it to a CCIData
    object.

    Args:
        result (dict): The output from squidpy.gr.ligrec.
        n_spots (int) (optional): The number of spots. Defaults to None.

    Returns:
        CCIData: The CCIData object.
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

    p_vals_dict = {}

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
        p_vals_dict[lr_] = df

    cci_data = CCIData(n_spots=n_spots, cci_scores=lr_dict, p_values=p_vals_dict)

    return cci_data


def read_CellChat(path, n_spots=None) -> CCIData:
    """Reads a CellChat ligand-receptor analysis output (cellchat@dr) and converts it to
    a CCIData object.

    Args:
        path (str): The output from cellchat@dr.
        n_spots (int) (optional): The number of spots. Defaults to None.

    Returns:
        CCIData: The CCIData object.
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
        if lr_ not in pvals_dict.keys():
            int_matrix = np.ones((len(cell_type_set), len(cell_type_set)))
        else:
            int_matrix = pvals_dict[lr_].values
        row = np.where(cell_type_set == result["source"][i])[0][0]
        col = np.where(cell_type_set == result["target"][i])[0][0]
        int_matrix[row, col] = result["pval"][i]

        df = pd.DataFrame(
            int_matrix, index=cell_type_set, columns=cell_type_set
        )
        df = df.fillna(1)
        pvals_dict[lr_] = df

    cci_data = CCIData(n_spots=n_spots, cci_scores=lr_dict, p_values=pvals_dict)

    return cci_data


def read_NATMI(path, n_spots=None) -> CCIData:
    """Reads a NATMI ligand-receptor analysis output (Edges_lrc2p.csv) and converts it
    to a CCIData object.
    Args:
        path (str): The path to Edges_lrc2p.csv.
        n_spots (int) (optional): The number of spots. Defaults to None.

    Returns:
        CCIData: The CCIData object.
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

    cci_data = CCIData(n_spots=n_spots, cci_scores=lr_dict)
    
    return cci_data
    
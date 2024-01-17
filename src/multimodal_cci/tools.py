import pandas as pd
import numpy as np


def read_cellphone_db(path):
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

        val = cp_obj.iloc[ind, cp_obj.columns.get_loc("classification") + 1 :]
        val = pd.DataFrame({"Index": val.index, "Value": val.values})
        val[["Row", "Col"]] = val["Index"].str.split("|", expand=True)
        val = val.drop("Index", axis=1)
        val = val.pivot(index="Row", columns="Col", values="Value")
        val = val.rename_axis(None, axis=1).rename_axis(None)

        sample[key] = val

    return sample


def read_squidpy(result):
    """Reads a squidpy ligand-receptor analysis output and converts it to a dictionary
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

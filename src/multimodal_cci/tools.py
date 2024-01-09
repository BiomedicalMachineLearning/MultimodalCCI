import pandas as pd

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
        
        key = cp_obj['interacting_pair'][ind]
        
        val = cp_obj.iloc[ind, cp_obj.columns.get_loc('classification') + 1:]
        val = pd.DataFrame({'Index': val.index, 'Value': val.values})
        val[['Row', 'Col']] = val['Index'].str.split('|', expand=True)
        val = val.drop('Index', axis=1)
        val = val.pivot(index='Row', columns='Col', values='Value')
        val = val.rename_axis(None, axis=1).rename_axis(None)
    
        sample[key] = val

    return sample
    
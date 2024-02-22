
################################################################################
import pickle
import squidpy as sq
import scanpy as sc
import pandas as pd
import numpy as np
import os
# Environment setup #
################################################################################
work_dir = '/home/uqlhocke/Aging/Levi/Scripts/LR_CCI/benchmarking/'
data_dir = '/scratch/project/stseq/Levi/mmcci_benchmarking/'
out_dir = data_dir

os.chdir(work_dir)


################################################################################
# Loading the data #
################################################################################
data = sc.read_h5ad(data_dir + 'sample1.h5ad')

################################################################################
# Running squidpy #
################################################################################
result = sq.gr.ligrec(data, n_perms=10000, cluster_key="cell_type",
                      copy=True, use_raw=False,
                      show_progress_bar=True,
                      corr_method='fdr_bh', corr_axis='clusters',
                      threshold=0
                      )
pvals = pd.DataFrame(result['pvalues'])

################################################################################
# Creating a CCI interaction matrix equivalent to other methods #
################################################################################
cci_names = np.array([col[0] + '--' + col[1] for col in pvals.columns])
cell_type_set = np.unique([col[0] for col in pvals.columns])

lr_dict = {}

for i, row in enumerate(pvals.index):

    int_matrix = np.zeros((len(cell_type_set), len(cell_type_set)))
    lr_ = '_'.join(list(row))

    # Getting sig CCIs for this lr #
    lr_pvals = np.array(pvals.values[i, :])
    sig_bool = lr_pvals < .05
    lr_ccis = cci_names[sig_bool]
    lig = list(row)[0]
    rec = list(row)[1]
    for j, cci in enumerate(lr_ccis):
        c1, c2 = cci.split('--')
        row = np.where(cell_type_set == c1)[0][0]
        col = np.where(cell_type_set == c2)[0][0]
        int_matrix[row, col] = result['means'][c1][c2][lig][rec]

    lr_dict[lr_] = pd.DataFrame(int_matrix, index=cell_type_set, columns=cell_type_set)

with open(data_dir + 'squidpy_sample1.pkl', 'wb') as f:
    pickle.dump(lr_dict, f)


################################################################################
    # Loading the data #
################################################################################
data = sc.read_h5ad(data_dir + 'sample2.h5ad')

################################################################################
# Running squidpy #
################################################################################
result = sq.gr.ligrec(data, n_perms=10000, cluster_key="cell_type",
                      copy=True, use_raw=False,
                      show_progress_bar=True,
                      corr_method='fdr_bh', corr_axis='clusters',
                      threshold=0
                      )
pvals = pd.DataFrame(result['pvalues'])

################################################################################
# Creating a CCI interaction matrix equivalent to other methods #
################################################################################
cci_names = np.array([col[0] + '--' + col[1] for col in pvals.columns])
cell_type_set = np.unique([col[0] for col in pvals.columns])

lr_dict = {}

for i, row in enumerate(pvals.index):

    int_matrix = np.zeros((len(cell_type_set), len(cell_type_set)))
    lr_ = '_'.join(list(row))

    # Getting sig CCIs for this lr #
    lr_pvals = np.array(pvals.values[i, :])
    sig_bool = lr_pvals < .05
    lr_ccis = cci_names[sig_bool]
    lig = list(row)[0]
    rec = list(row)[1]
    for j, cci in enumerate(lr_ccis):
        c1, c2 = cci.split('--')
        row = np.where(cell_type_set == c1)[0][0]
        col = np.where(cell_type_set == c2)[0][0]
        int_matrix[row, col] = result['means'][c1][c2][lig][rec]

    lr_dict[lr_] = pd.DataFrame(int_matrix, index=cell_type_set, columns=cell_type_set)

with open(data_dir + 'squidpy_sample2.pkl', 'wb') as f:
    pickle.dump(lr_dict, f)


################################################################################
    # Loading the data #
################################################################################
data = sc.read_h5ad(data_dir + 'sample3.h5ad')

# ################################################################################
#                         # Running squidpy #
# ################################################################################
result = sq.gr.ligrec(data, n_perms=10000, cluster_key="cell_type",
                      copy=True, use_raw=False,
                      show_progress_bar=True,
                      corr_method='fdr_bh', corr_axis='clusters',
                      threshold=0
                      )
pvals = pd.DataFrame(result['pvalues'])

################################################################################
# Creating a CCI interaction matrix equivalent to other methods #
################################################################################
cci_names = np.array([col[0] + '--' + col[1] for col in pvals.columns])
cell_type_set = np.unique([col[0] for col in pvals.columns])

lr_dict = {}

for i, row in enumerate(pvals.index):

    int_matrix = np.zeros((len(cell_type_set), len(cell_type_set)))
    lr_ = '_'.join(list(row))

    # Getting sig CCIs for this lr #
    lr_pvals = np.array(pvals.values[i, :])
    sig_bool = lr_pvals < .05
    lr_ccis = cci_names[sig_bool]
    lig = list(row)[0]
    rec = list(row)[1]
    for j, cci in enumerate(lr_ccis):
        c1, c2 = cci.split('--')
        row = np.where(cell_type_set == c1)[0][0]
        col = np.where(cell_type_set == c2)[0][0]
        int_matrix[row, col] = result['means'][c1][c2][lig][rec]

    lr_dict[lr_] = pd.DataFrame(int_matrix, index=cell_type_set, columns=cell_type_set)

with open(data_dir + 'squidpy_sample3.pkl', 'wb') as f:
    pickle.dump(lr_dict, f)


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
data_dir = '/scratch/project/stseq/Levi/mmcci/mmcci_benchmarking/'
out_dir = data_dir

# os.chdir(work_dir)


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

with open(data_dir + 'squidpy_sample1.pkl', 'wb') as f:
    pickle.dump(result, f)


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

with open(data_dir + 'squidpy_sample2.pkl', 'wb') as f:
    pickle.dump(result, f)


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

with open(data_dir + 'squidpy_sample3.pkl', 'wb') as f:
    pickle.dump(result, f)
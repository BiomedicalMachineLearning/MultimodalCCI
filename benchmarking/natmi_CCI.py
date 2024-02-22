import scanpy as sc
import pandas as pd

out_dir = '/scratch/project/stseq/Levi/mmcci/mmcci_benchmarking/'

data = sc.read_h5ad(out_dir + 'sample1.h5ad')
counts = data.to_df()
counts.index = ["sample_" + name for name in counts.index]
counts.T.to_csv(out_dir + 'sample1_em.csv', index=True, header=True)
metadata = pd.DataFrame(data.obs['cell_type'])
metadata.index = ["sample_" + name for name in metadata.index]
metadata['_'] = metadata.index
metadata = metadata.rename(columns={'cell_type': 'Annotation', '_': 'Cell'})
metadata = metadata[['Cell', 'Annotation']]
metadata.to_csv(out_dir + 'sample1_metadata.csv', index=False, header=True)

data = sc.read_h5ad(out_dir + 'sample2.h5ad')
counts = data.to_df()
counts.index = ["sample_" + name for name in counts.index]
counts.T.to_csv(out_dir + 'sample2_em.csv', index=True, header=True)
metadata = pd.DataFrame(data.obs['cell_type'])
metadata.index = ["sample_" + name for name in metadata.index]
metadata['_'] = metadata.index
metadata = metadata.rename(columns={'cell_type': 'Annotation', '_': 'Cell'})
metadata = metadata[['Cell', 'Annotation']]
metadata.to_csv(out_dir + 'sample2_metadata.csv', index=False, header=True)

data = sc.read_h5ad(out_dir + 'sample3.h5ad')
counts = data.to_df()
counts.index = ["sample_" + name for name in counts.index]
counts.T.to_csv(out_dir + 'sample3_em.csv', index=True, header=True)
metadata = pd.DataFrame(data.obs['cell_type'])
metadata.index = ["sample_" + name for name in metadata.index]
metadata['_'] = metadata.index
metadata = metadata.rename(columns={'cell_type': 'Annotation', '_': 'Cell'})
metadata = metadata[['Cell', 'Annotation']]
metadata.to_csv(out_dir + 'sample3_metadata.csv', index=False, header=True)

# Run the following in the terminal to generate the CCI interaction matrix:

# python ExtractEdges.py --emFile
# /scratch/project/stseq/Levi/mmcci_benchmarking/sample1_em.csv --annFile
# /scratch/project/stseq/Levi/mmcci_benchmarking/sample1_metadata.csv
# --interDB lrc2p --coreNum 4

# python ExtractEdges.py --emFile
# /scratch/project/stseq/Levi/mmcci_benchmarking/sample2_em.csv --annFile
# /scratch/project/stseq/Levi/mmcci_benchmarking/sample2_metadata.csv
# --interDB lrc2p --coreNum 4

# python ExtractEdges.py --emFile
# /scratch/project/stseq/Levi/mmcci_benchmarking/sample3_em.csv --annFile
# /scratch/project/stseq/Levi/mmcci_benchmarking/sample3_metadata.csv
# --interDB lrc2p --coreNum 4

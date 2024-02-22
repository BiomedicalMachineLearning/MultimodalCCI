
<!-- <table align="center">
  <tr>
    <td>
      <b>Package</b>
    </td>
    <td>
      <a href="https://pypi.python.org/pypi//">
      <img src="https://img.shields.io/pypi/v/.svg" alt="PyPI Version">
      </a>
      <a href="https://pepy.tech/project/">
      <img src="https://static.pepy.tech/personalized-badge/?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads"
        alt="PyPI downloads">
      </a>
      <a href="https://anaconda.org/conda-forge/">
      <img src="https://anaconda.org/conda-forge//badges/downloads.svg" alt="Conda downloads">
      </a>
      <a href="https://anaconda.org/conda-forge/">
      <img src="https://anaconda.org/conda-forge//badges/downloads.svg" alt="Install">
      </a>
    </td>
  </tr>
  <tr>
    <td>
      <b>Documentation</b>
    </td>
    <td>
      <a href="https://multimodalcci.readthedocs/en/latest/">
      <img src="https://multimodalcci.readthedocs.org/projects//badge/?version=latest" alt="Documentation Status">
      </a>
    </td>
  </tr>
  <tr>
    <td>
     <b>Paper</b>
    </td>
    <td>
      <a href="https://doi.org/"><img src="https://zenodo.org/badge/DOI/.svg"
        alt="DOI"></a>
    </td>
  </tr>
  <tr>
    <td>
      <b>License</b>
    </td>
    <td>
      <a href="https://github.com/BiomedicalMachineLearning/MultimodalCCI/blob/main/LICENSE.txt"><img src="https://img.shields.io/badge/License-BSD-blue.svg"
        alt="LICENSE"></a>
    </td>
  </tr>
</table> -->


# MMCCI: Multimodal Cell-Cell Interaction Integration, Analysis, and Visualisation

**MMCCI** is a fast and lightweight Python package for integrating and visualizing CCI networks within and between multiple modalities at the level of the individual LR pair. It works on **scRNA-seq** and **spatial transcriptomics** data samples that have been processed through the following CCI algorithms:
1. stLearn
2. CellChat
3. CellPhoneDB
4. NATMI
5. Squidpy

## Getting Started

### Installation

Coming soon

### Documentation

Documentation and Tutorials are available at our **Read the Docs** page (coming soon).

- There is a tutorial notebook [here](examples/brain_aging_example.ipynb)
- To understand how to load CCI results from different tools, look at this notebook [here](example/loading_CCI_results.ipynb)

## CCI Integration

MMCCI allows users to integrate multiple CCI results together, both:
1. Samples from a single modality (eg. Visium)
2. Samples from multiple modalities (eg. Visium, Xenium and CosMX)

![Integration Method](docs/images/integration_method.png)

## CCI Analysis

MMCCI provides multiple useful analyses that can be run on the integrated networks or from a single sample:
1. Network comparison between groups with permutation testing
2. CLustering of LR pairs with similar networks
3. Clustering of spots/cells with similar interaction scores
4. Sender-receiver LR querying
5. GSEA pathway analysis

![Downstream Analyses](docs/images/analyses.png)

### Pipeline Diagram

![MMCCI Pipeline](docs/images/pipeline.png)

## Citing MMCCI

If you have used MMCCI in your research, please consider citing us: (coming soon).


# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: scvi
#     language: python
#     name: python3
# ---

# %% Cell Lists
# This module contains lists of cell types and markers.

# %% Imports and Setup
import importlib
import os
import sys

import scanpy as sc

# Add repository root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set working directory to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plotting_functions

import bar_nick_utils

importlib.reload(plotting_functions)
importlib.reload(bar_nick_utils)


# load /home/barroz/projects/Mapping-scRNA-seq-Variability-Using-Co-Embedding-to-Understand-Cell-Neighborhood-Effects/CODEX_RNA_seq/data/trained_data/protein_vae_trained.h5ad
adata = sc.read(
    "/home/barroz/projects/Mapping-scRNA-seq-Variability-Using-Co-Embedding-to-Understand-Cell-Neighborhood-Effects/CODEX_RNA_seq/data/trained_data/protein_vae_trained.h5ad"
)

# %%
terminal_exhaustion = [
    "CD3G",
    "FASLG",
    "ID2",
    "LAG3",
    "RGS1",
    "CCL3",
    "CCL3L1",
    "KIAA1671",
    "SH2D2A",
    "DUSP2",
    "PDCD1",
    "CD7",
    "NR4A2",
    "CD160",
    "PTPN22",
    "ABI3",
    "PTGER4",
    "GZMK",
    "GZMA",
    "MBNL1",
    "VMP1",
    "PLAC8",
    "RGS3",
    "EFHD2",
    "GLRX",
    "CXCR6",
    "ARL6IP1",
    "CCL4",
    "ISG15",
    "LAX1",
    "CD8A",
    "SERPINA3",
    "GZMB",
    "TOX",
]

precursor_exhaustion = [
    "TCF7",
    "MS4A4A",
    "TNFSF8",
    "CXCL10",
    "EEF1B2",
    "ID3",
    "IL7R",
    "JUN",
    "LTB",
    "XCL1",
    "SOCS3",
    "TRAF1",
    "EMB",
    "CRTAM",
    "EEF1G",
    "CD9",
    "ITGB1",
    "GPR183",
    "ZFP36L1",
    "SLAMF6",
    "LY6E",
]

cd8_t_cell_activation = [
    "CD69",
    "CCR7",
    "CD27",
    "BTLA",
    "CD40LG",
    "IL2RA",
    "CD3E",
    "CD47",
    "EOMES",
    "GNLY",
    "GZMA",
    "GZMB",
    "PRF1",
    "IFNG",
    "CD8A",
    "CD8B",
    "CD95L",
    "LAMP1",
    "LAG3",
    "CTLA4",
    "HLA-DRA",
    "TNFRSF4",
    "ICOS",
    "TNFRSF9",
    "TNFRSF18",
]

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

# %%
# preprocess the real data from Elham lab and peprform the archetype analysis

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"sys.path added: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
print(f"sys.path added: {os.path.dirname(os.path.abspath(__file__))}")
print(f"Current working directory: {os.getcwd()}")

# %%
# Imports
# %%
import importlib
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import plotting_functions as pf
import scanpy as sc
import torch
from scipy.io import mmread

import bar_nick_utils
import covet_utils

importlib.reload(bar_nick_utils)
importlib.reload(covet_utils)
importlib.reload(pf)

from bar_nick_utils import clean_uns_for_h5ad


def setup_environment():
    """Setup environment variables and random seeds"""
    np.random.seed(8)
    warnings.filterwarnings("ignore")
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.max_rows", 10)
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def check_data_exists(data_dir):
    """Check if required data files exist"""
    required_files = [
        "tonsil/tonsil_codex.csv",
        "tonsil/tonsil_rna_counts.txt",
        "tonsil/tonsil_rna_names.csv",
        "tonsil/tonsil_rna_meta.csv",
    ]
    return all((data_dir / file).exists() for file in required_files)


def download_maxfuse_data(data_dir):
    """Download and extract MaxFuse data"""
    if check_data_exists(data_dir):
        print("Data files already exist, skipping download.")
        return

    print("Downloading MaxFuse data...")
    import io
    import zipfile

    import requests

    r = requests.get("http://stat.wharton.upenn.edu/~zongming/maxfuse/data.zip")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(str(data_dir.parent))


# fmt: off
def load_protein_data(data_dir):
    """Load and process protein data"""
    print("Loading protein data...")
    protein = pd.read_csv(data_dir / "tonsil/tonsil_codex.csv")

    protein_features = [
        'CD38', 'CD19', 'CD31', 'Vimentin', 'CD22', 'Ki67', 'CD8',
        'CD90', 'CD123', 'CD15', 'CD3', 'CD152', 'CD21', 'cytokeratin', 'CD2',
        'CD66', 'collagen IV', 'CD81', 'HLA-DR', 'CD57', 'CD4', 'CD7', 'CD278',
        'podoplanin', 'CD45RA', 'CD34', 'CD54', 'CD9', 'IGM', 'CD117', 'CD56',
        'CD279', 'CD45', 'CD49f', 'CD5', 'CD16', 'CD63', 'CD11b', 'CD1c',
        'CD40', 'CD274', 'CD27', 'CD104', 'CD273', 'FAPalpha', 'Ecadherin'
    ]

    protein_locations = ['centroid_x', 'centroid_y']
    protein_adata = ad.AnnData(
        protein[protein_features].to_numpy(), dtype=np.float32
    )
    protein_adata.obsm["spatial"] = protein[protein_locations].to_numpy()
    protein_adata.obs['cell_types'] = protein['cluster.term'].to_numpy()

    return protein_adata
# fmt: on


def load_rna_data(data_dir):
    """Load and process RNA data"""
    print("Loading RNA data...")
    rna = mmread(data_dir / "tonsil/tonsil_rna_counts.txt")
    rna_names = pd.read_csv(data_dir / "tonsil/tonsil_rna_names.csv")["names"].to_numpy()

    rna_adata = ad.AnnData(rna.tocsr(), dtype=np.float32)
    rna_adata.var_names = rna_names

    metadata_rna = pd.read_csv(data_dir / "tonsil/tonsil_rna_meta.csv")
    rna_adata.obs["cell_types"] = metadata_rna["cluster.info"].to_numpy()

    return rna_adata


def preprocess_rna_maxfuse(adata):
    """Preprocess RNA data using MaxFuse method"""
    print("Preprocessing RNA data...")
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=5000)
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata)
    return adata


def filter_and_subsample_data(adata_1, adata_2, num_rna_cells=80000, num_protein_cells=80000):
    """Filter and subsample data"""
    print("Filtering and subsampling data...")
    # Filter out tumor and dead cells
    adata_2 = adata_2[adata_2.obs["cell_types"] != "tumor"]
    adata_2 = adata_2[adata_2.obs["cell_types"] != "dead"]

    # Subsample cells
    subsample_n_obs_rna = min(adata_1.shape[0], num_rna_cells)
    subsample_n_obs_protein = min(adata_2.shape[0], num_protein_cells)
    sc.pp.subsample(adata_1, n_obs=subsample_n_obs_rna)
    sc.pp.subsample(adata_2, n_obs=subsample_n_obs_protein)

    # Remove NK cells
    adata_1 = adata_1[adata_1.obs["cell_types"] != "nk cells"]
    adata_2 = adata_2[adata_2.obs["cell_types"] != "nk cells"]

    # Sort by cell types
    adata_1 = adata_1[adata_1.obs["cell_types"].argsort(), :]
    adata_2 = adata_2[adata_2.obs["cell_types"].argsort(), :]

    return adata_1, adata_2


def process_spatial_data(adata):
    """Process spatial data for protein dataset"""
    print("Processing spatial data...")
    x_coor = adata.obsm["spatial"][:, 0]
    y_coor = adata.obsm["spatial"][:, 1]
    temp = pd.DataFrame([x_coor, y_coor], index=["x", "y"]).T
    temp.index = adata.obs.index
    adata.obsm["spatial_location"] = temp
    adata.obs["X"] = x_coor
    adata.obs["Y"] = y_coor
    return adata


def save_processed_data(adata_1, adata_2, save_dir):
    """Save processed data"""
    print("Saving processed data...")
    clean_uns_for_h5ad(adata_2)
    clean_uns_for_h5ad(adata_1)
    time_stamp = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    adata_1.write(save_dir / f"preprocessed_adata_rna_maxfuse_{time_stamp}.h5ad")
    adata_2.write(save_dir / f"preprocessed_adata_prot_maxfuse_{time_stamp}.h5ad")


# %%
# Setup and run preprocessing
# %%
# Setup environment
device = setup_environment()

# Setup paths
root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = root_dir / "CODEX_RNA_seq" / "data" / "raw_data"
save_dir = root_dir / "CODEX_RNA_seq" / "data" / "processed_data"
plot_flag = True

# Create data directory if it doesn't exist
data_dir.mkdir(parents=True, exist_ok=True)

# %%
# Download and load data
# %%
# Download data if needed
download_maxfuse_data(data_dir)

# Load data
protein_adata = load_protein_data(data_dir)
rna_adata = load_rna_data(data_dir)

# %%
# Plot initial data overview
# %%
if plot_flag:
    # Compute PCA for both datasets
    sc.tl.pca(rna_adata)
    sc.tl.pca(protein_adata)

    pf.plot_data_overview(rna_adata, protein_adata)
    pf.plot_cell_type_distribution(rna_adata, protein_adata)
    pf.plot_spatial_data(protein_adata)

# %%
# Filter and subsample data
# %%
num_rna_cells = 400  # Reduced from 8000 for testing
num_protein_cells = 400  # Reduced from 8000 for testing
rna_adata, protein_adata = filter_and_subsample_data(
    rna_adata, protein_adata, num_rna_cells, num_protein_cells
)

# %%
# Preprocess RNA data
# %%
rna_adata = preprocess_rna_maxfuse(rna_adata)

if plot_flag:
    pf.plot_highly_variable_genes(rna_adata)

# %%
# Process spatial data
# %%
protein_adata = process_spatial_data(protein_adata)

# %%
# Save processed data
# %%
save_processed_data(rna_adata, protein_adata, save_dir)

# %%
# Plot preprocessing results
# %%
if plot_flag:
    # Compute UMAP for both datasets
    sc.pp.neighbors(rna_adata)
    sc.tl.umap(rna_adata)
    sc.pp.neighbors(protein_adata)
    sc.tl.umap(protein_adata)

    pf.plot_preprocessing_results(rna_adata, protein_adata)

print("Preprocessing completed successfully!")

# %%

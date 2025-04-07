# %%

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

import json

# %%
# Setup paths
# %%
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# %%
# Imports
# %%
import importlib
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotting_functions as pf
import scanpy as sc
import scipy
import torch

import bar_nick_utils

importlib.reload(pf)
importlib.reload(bar_nick_utils)

from plotting_functions import (
    plot_archetype_heatmaps,
    plot_b_cells_analysis,
    plot_cell_type_distribution,
    plot_original_data_visualizations,
    plot_pca_and_umap,
    plot_protein_umap,
    plot_umap_visualizations_original_data,
)

from bar_nick_utils import (
    clean_uns_for_h5ad,
    get_latest_file,
    get_umap_filtered_fucntion,
    match_datasets,
)

if not hasattr(sc.tl.umap, "_is_wrapped"):
    sc.tl.umap = get_umap_filtered_fucntion()
    sc.tl.umap._is_wrapped = True

# Load config if exists
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    num_rna_cells = config["subsample"]["num_rna_cells"]
    num_protein_cells = config["subsample"]["num_protein_cells"]
    plot_flag = config["plot_flag"]
else:
    num_rna_cells = num_protein_cells = 2000
    plot_flag = True


def setup_environment():
    """Setup environment variables and random seeds"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.max_rows", 10)
    warnings.filterwarnings("ignore")
    pd.options.display.max_rows = 10
    pd.options.display.max_columns = 10
    np.set_printoptions(threshold=100)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return device


def load_data(folder, file_prefixes):
    """Load data files and return AnnData objects"""
    print("Loading data files...")
    latest_files = {prefix: get_latest_file(folder, prefix) for prefix in file_prefixes}
    print(latest_files)
    adata_rna = sc.read(latest_files["adata_rna_"])
    print("Loaded RNA data:", adata_rna.shape)
    adata_prot = sc.read(latest_files["adata_prot_"])
    print("Loaded protein data:", adata_prot.shape)

    return adata_rna, adata_prot


def subsample_data(adata_rna, adata_prot, sample_size=2000):
    """Subsample data to specified size"""
    print("\nSubsampling data...")
    rna_sample_size = min(len(adata_rna), sample_size)
    prot_sample_size = min(len(adata_prot), sample_size)
    adata_rna_subset = sc.pp.subsample(adata_rna, n_obs=rna_sample_size, copy=True)
    adata_prot_subset = sc.pp.subsample(adata_prot, n_obs=prot_sample_size, copy=True)
    print(f"Subsampled to {rna_sample_size} RNA cells and {prot_sample_size} protein cells")
    return adata_rna_subset, adata_prot_subset


def order_cells_by_type(adata_rna_subset, adata_prot_subset):
    """Order cells by major and minor cell types"""
    print("\nOrdering cells by major and minor cell types...")
    adata_rna_subset.obs_names_make_unique()
    adata_prot_subset.obs_names_make_unique()

    new_order_rna = adata_rna_subset.obs.sort_values(by=["major_cell_types", "cell_types"]).index
    new_order_prot = adata_prot_subset.obs.sort_values(by=["major_cell_types", "cell_types"]).index
    adata_rna_subset = adata_rna_subset[new_order_rna]
    adata_prot_subset = adata_prot_subset[new_order_prot]
    return adata_rna_subset, adata_prot_subset


def compute_archetype_distances(adata_rna_subset, adata_prot_subset):
    """Compute archetype distances between RNA and protein data"""
    print("Computing archetype distances...")
    archetype_distances = scipy.spatial.distance.cdist(
        adata_rna_subset.obsm["archetype_vec"].values,
        adata_prot_subset.obsm["archetype_vec"].values,
        metric="cosine",
    )

    return archetype_distances


def compute_pca_and_umap(adata_rna_subset, adata_prot_subset):
    """Compute PCA and UMAP for visualization"""
    sc.pp.pca(adata_rna_subset)
    sc.pp.pca(adata_prot_subset)
    sc.pp.neighbors(adata_rna_subset, key_added="original_neighbors")
    sc.tl.umap(adata_rna_subset, neighbors_key="original_neighbors")
    adata_rna_subset.obsm["X_original_umap"] = adata_rna_subset.obsm["X_umap"]
    sc.pp.neighbors(adata_prot_subset, key_added="original_neighbors")
    sc.tl.umap(adata_prot_subset, neighbors_key="original_neighbors")
    adata_prot_subset.obsm["X_original_umap"] = adata_prot_subset.obsm["X_umap"]
    return adata_rna_subset, adata_prot_subset


def save_processed_data(adata_rna_subset, adata_prot_subset, save_dir):
    """Save processed data"""
    clean_uns_for_h5ad(adata_prot_subset)
    clean_uns_for_h5ad(adata_rna_subset)
    save_dir = Path(save_dir).absolute()
    time_stamp = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(save_dir, exist_ok=True)

    print(
        f"\nRNA subset dimensions: {adata_rna_subset.shape[0]} samples x {adata_rna_subset.shape[1]} features"
    )
    print(
        f"Protein subset dimensions: {adata_prot_subset.shape[0]} samples x {adata_prot_subset.shape[1]} features\n"
    )

    sc.write(
        Path(f"{save_dir}/adata_rna_subset_prepared_for_training_{time_stamp}.h5ad"),
        adata_rna_subset,
    )
    sc.write(
        Path(f"{save_dir}/adata_prot_subset_prepared_for_training_{time_stamp}.h5ad"),
        adata_prot_subset,
    )


def load_and_subsample_data(folder, file_prefixes, sample_size=2000):
    """Load and subsample data files"""
    device = setup_environment()
    adata_rna, adata_prot = load_data(folder, file_prefixes)
    adata_rna_subset, adata_prot_subset = subsample_data(adata_rna, adata_prot, sample_size)
    del adata_prot, adata_rna
    return adata_rna_subset, adata_prot_subset


# Load and subsample data
# %%
folder = "CODEX_RNA_seq/data/processed_data"
file_prefixes = ["adata_rna_", "adata_prot_", "adata_archetype_rna_", "adata_archetype_prot_"]
sample_size = 8000  # Adjust this value as needed

adata_rna_subset, adata_prot_subset = load_and_subsample_data(folder, file_prefixes, sample_size)
# adata_rna_subset = adata_rna_subset[:2000]  # todo remove
# adata_prot_subset = adata_prot_subset[:1200]  # todo remove
# %%
# Process and visualize data
# %%

sc.pp.pca(adata_rna_subset)
sc.pp.pca(adata_prot_subset)
sc.pp.neighbors(adata_rna_subset)
sc.pp.neighbors(adata_prot_subset)

# Plot UMAP visualizations
if plot_flag:
    plot_umap_visualizations_original_data(adata_rna_subset, adata_prot_subset)

# Order cells by type
adata_rna_subset, adata_prot_subset = order_cells_by_type(adata_rna_subset, adata_prot_subset)

# Compute archetype distances
archetype_distances = compute_archetype_distances(adata_rna_subset, adata_prot_subset)
matching_distance_before = np.diag(archetype_distances).mean()
# %%
# Plot archetype heatmaps
if plot_flag:
    plot_archetype_heatmaps(adata_rna_subset, adata_prot_subset, archetype_distances)
# %%
# Match datasets
adata_rna_subset_matched, adata_prot_subset_matched = match_datasets(
    adata_rna_subset, adata_prot_subset, threshold=0.1, plot_flag=plot_flag
)

# %%

# Rest of the code below will not execute
adata_rna_subset, adata_prot_subset = adata_rna_subset_matched, adata_prot_subset_matched

# Find closest protein cells to each RNA cell using archetype vectors
dist_matrix = scipy.spatial.distance.cdist(
    adata_rna_subset.obsm["archetype_vec"], adata_prot_subset.obsm["archetype_vec"], metric="cosine"
)
closest_prot_indices = np.argmin(dist_matrix, axis=1)
plt.figure()
plt.plot(closest_prot_indices)
plt.show()
# Set CN values based on closest protein cells
adata_rna_subset.obs["CN"] = adata_prot_subset.obs["CN"].values[closest_prot_indices]

# Compute PCA and UMAP
adata_rna_subset, adata_prot_subset = compute_pca_and_umap(adata_rna_subset, adata_prot_subset)

# Additional visualizations
if plot_flag:
    plot_pca_and_umap(adata_rna_subset, adata_prot_subset)
    plot_b_cells_analysis(adata_rna_subset)
    one_cell_type = plot_protein_umap(adata_prot_subset)
    plot_cell_type_distribution(adata_rna_subset, adata_prot_subset)
    plot_original_data_visualizations(adata_rna_subset, adata_prot_subset)

# %%
# Save processed data
# %%
save_dir = "CODEX_RNA_seq/data/processed_data"
save_processed_data(adata_rna_subset, adata_prot_subset, save_dir)

# %%
# Additional analysis cells - uncomment and modify as needed
# %%
# Plot specific visualizations
if plot_flag:
    sc.pl.embedding(
        adata_rna_subset, color=["cell_types"], basis="X_original_umap", title="RNA data cell types"
    )

# %%
# Analyze specific cell types
if plot_flag:
    one_cell_type = adata_prot_subset.obs["major_cell_types"][0]
    plot_b_cells_analysis(
        adata_rna_subset[adata_rna_subset.obs["major_cell_types"] == one_cell_type]
    )

# %%
# Compute additional metrics
archetype_distances = compute_archetype_distances(adata_rna_subset, adata_prot_subset)
print(f"Initial matching distance: {matching_distance_before:.3f}")
print(f"Average matching distance: {np.diag(archetype_distances).mean():.3f}")

# %%

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

# %%

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set working directory to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import scvi
import seaborn as sns
from anndata import AnnData
from matplotlib.patches import Arc
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.optimize import linear_sum_assignment
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Add repository root to Python path without changing working directory


importlib.reload(scvi)
import re

import torch
import torch.nn.functional as F
from scvi.model import SCVI
from scvi.train import TrainingPlan
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

import bar_nick_utils

importlib.reload(bar_nick_utils)

from bar_nick_utils import (
    archetype_vs_latent_distances_plot,
    calculate_cLISI,
    calculate_iLISI,
    clean_uns_for_h5ad,
    compare_distance_distributions,
    compute_pairwise_kl,
    compute_pairwise_kl_two_items,
    get_latest_file,
    get_umap_filtered_fucntion,
    match_datasets,
    mixing_score,
    plot_cosine_distance,
    plot_inference_outputs,
    plot_latent,
    plot_latent_mean_std,
    plot_normalized_losses,
    plot_rna_protein_matching_means_and_scale,
    plot_similarity_loss_history,
    select_gene_likelihood,
    verify_gradients,
)

if not hasattr(sc.tl.umap, "_is_wrapped"):
    sc.tl.umap = get_umap_filtered_fucntion()
    sc.tl.umap._is_wrapped = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pd.set_option("display.max_columns", 10)
pd.set_option("display.max_rows", 10)
warnings.filterwarnings("ignore")
pd.options.display.max_rows = 10
pd.options.display.max_columns = 10
np.set_printoptions(threshold=100)
np.random.seed(0)
save_dir = "CODEX_RNA_seq/data/processed_data"

plot_flag = True

# %%
folder = "CODEX_RNA_seq/data/processed_data"
file_prefixes = ["adata_rna_", "adata_prot_", "adata_archetype_rna_", "adata_archetype_prot_"]


# Load the latest files
latest_files = {prefix: get_latest_file(folder, prefix) for prefix in file_prefixes}
print(latest_files)
adata_rna = sc.read(latest_files["adata_rna_"])
adata_prot = sc.read(latest_files["adata_prot_"])
adata_archetype_rna = sc.read(latest_files["adata_archetype_rna_"])
adata_archetype_prot = sc.read(latest_files["adata_archetype_prot_"])

# %%

# %%
sample_size = min(len(adata_prot), len(adata_rna), 2000)
adata_rna_subset = sc.pp.subsample(adata_rna, n_obs=sample_size, copy=True)
adata_prot_subset = sc.pp.subsample(adata_prot, n_obs=int(sample_size) - 1, copy=True)
del adata_prot, adata_rna
if plot_flag:
    # making sure that the archetypes make sense in original data context
    sc.pp.neighbors(adata_rna_subset)
    sc.pp.neighbors(adata_prot_subset)
    sc.tl.umap(adata_rna_subset)
    sc.tl.umap(adata_prot_subset)
    sc.pl.umap(
        adata_rna_subset,
        color="archetype_label",
        title=["Original RNA cells associated Archetypes"],
    )
    sc.pl.umap(
        adata_prot_subset,
        color="archetype_label",
        title=["Original Protein cells associated Archetypes"],
    )

# %%
# order cells by major and minor cell type for easy visualization
new_order_rna = adata_rna_subset.obs.sort_values(by=["major_cell_types", "cell_types"]).index
new_order_prot = adata_prot_subset.obs.sort_values(by=["major_cell_types", "cell_types"]).index
adata_rna_subset = adata_rna_subset[new_order_rna]
adata_prot_subset = adata_prot_subset[new_order_prot]
archetype_distances = scipy.spatial.distance.cdist(
    adata_rna_subset.obsm["archetype_vec"].values,
    adata_prot_subset.obsm["archetype_vec"].values,
    metric="cosine",
)
matching_distance_before = np.diag(archetype_distances).mean()

if plot_flag:
    plt.figure(figsize=(10, 5))
    plt.suptitle("Heatmap of archetype coor before matching\nordered by cell types only")
    plt.subplot(1, 2, 1)
    sns.heatmap(np.log1p(adata_rna_subset.obsm["archetype_vec"].values), cbar=False)
    plt.title("RNA Archetype Vectors")
    plt.ylabel("RNA cell index")
    plt.xlabel("Archetype Betas")
    plt.subplot(1, 2, 2)
    sns.heatmap(np.log1p(adata_prot_subset.obsm["archetype_vec"].values), cbar=False)
    plt.xlabel("Archetype Betas")
    plt.ylabel("Protein cell index")
    plt.title("Protein Archetype Vectors")
    plt.show()

# %%

# %%
# use_matched = input('Do you want to use previous saved matched data? (y/n)')
# if use_matched == 'y':
#     adata_rna_subset, adata_prot_subset = adata_rna_subset_matched, adata_prot_subset_matched
# else:
#     adata_rna_subset_matched,adata_prot_subset_matched = match_datasets(adata_rna_subset,adata_prot_subset,0.05,plot_flag=plot_flag)
#     adata_rna_subset,adata_prot_subset = adata_rna_subset_matched,adata_prot_subset_matched
#     # save the matched data
#     adata_rna_subset_matched.write(f'{save_dir}/adata_rna_matched.h5ad')
#     adata_prot_subset_matched.write(f'{save_dir}/adata_prot_matched.h5ad')

adata_rna_subset_matched, adata_prot_subset_matched = match_datasets(
    adata_rna_subset, adata_prot_subset, threshold=0.1, plot_flag=plot_flag
)
# ok threhols example  = 0.01

adata_rna_subset, adata_prot_subset = adata_rna_subset_matched, adata_prot_subset_matched
adata_rna_subset.obs["CN"] = adata_prot_subset.obs["CN"].values  # add the CN to the rna data


# %%
if plot_flag:
    plt.figure(figsize=(10, 5))
    plt.suptitle("Heatmap of archetype coor after matching")
    plt.subplot(1, 2, 1)
    sns.heatmap(np.log1p(adata_rna_subset.obsm["archetype_vec"].values), cbar=False)
    plt.ylabel("RNA cell index")
    plt.xlabel("Archetype Betas")
    plt.subplot(1, 2, 2)
    sns.heatmap(np.log1p(adata_prot_subset.obsm["archetype_vec"].values), cbar=False)
    plt.ylabel("Protein cell index")
    plt.xlabel("Archetype Betas")

# %%
if plot_flag:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)

    sns.heatmap(np.log1p(archetype_distances[::5, ::5].T))
    plt.xlabel("RNA cell index")
    plt.ylabel("Protein cell index")
    plt.gca().invert_yaxis()

    plt.subplot(1, 2, 2)
    plt.scatter(
        np.arange(len(archetype_distances.argmin(axis=1))),
        archetype_distances.argmin(axis=1),
        s=1,
        rasterized=True,
    )
    plt.xlabel("RNA cell index")
    plt.ylabel("Protein cell index")
    # plt.title('If this looks like a line, then the matching ARE THE SAME AND NOT ACROSS MODALITIES')
    plt.show()


# %%
adata_rna_subset

# %%
sc.pp.pca(adata_rna_subset)
sc.pp.pca(adata_prot_subset)
sc.pp.neighbors(adata_rna_subset, key_added="original_neighbors")
sc.tl.umap(adata_rna_subset, neighbors_key="original_neighbors")
adata_rna_subset.obsm["X_original_umap"] = adata_rna_subset.obsm["X_umap"]
sc.pp.neighbors(adata_prot_subset, key_added="original_neighbors")
sc.tl.umap(adata_prot_subset, neighbors_key="original_neighbors")
adata_prot_subset.obsm["X_original_umap"] = adata_prot_subset.obsm["X_umap"]

if plot_flag:
    sc.pl.pca(adata_rna_subset, color=["cell_types", "major_cell_types"])
    sc.pl.pca(adata_prot_subset, color=["cell_types", "major_cell_types"])
    sc.pl.embedding(
        adata_rna_subset, basis="X_original_umap", color=["cell_types", "major_cell_types"]
    )
    sc.pl.embedding(
        adata_prot_subset, basis="X_original_umap", color=["cell_types", "major_cell_types"]
    )


# %%
adata_rna_subset.obs["major_cell_types"][0]

# %%
if plot_flag:
    adata_B_cells = adata_rna_subset[
        adata_rna_subset.obs["major_cell_types"] == adata_rna_subset.obs["major_cell_types"][0]
    ]
    sc.pp.pca(adata_B_cells)
    sc.pp.neighbors(adata_B_cells, use_rep="X_pca")
    sc.tl.umap(adata_B_cells)
    if "tissue" in adata_B_cells.obs:
        sc.pl.umap(
            adata_B_cells, color=["tissue"], title="verifying tissue does not give a major effect"
        )
    else:
        sc.pl.umap(
            adata_B_cells, color=["cell_types"], title="verifying cell types are well separated"
        )


# %%
adata_prot_subset.obs.columns

# %%

sc.pp.neighbors(adata_prot_subset, use_rep="X_pca", key_added="X_neighborhood")
sc.tl.umap(adata_prot_subset, neighbors_key="X_neighborhood")
adata_prot_subset.obsm["X_original_umap"] = adata_prot_subset.obsm["X_umap"]
sc.pl.umap(
    adata_prot_subset,
    color="CN",
    title="Protein UMAP of CN vectors colored by CN label",
    neighbors_key="original_neighbors",
)
one_cell_type = adata_prot_subset.obs["major_cell_types"][0]
sc.pl.umap(
    adata_prot_subset[adata_prot_subset.obs["major_cell_types"] == one_cell_type],
    color="cell_types",
    title="Protein UMAP of CN vectors colored by minor cell type label",
)
adata_prot_subset

# %%
if plot_flag:
    sns.histplot(
        adata_prot_subset[adata_prot_subset.obs["major_cell_types"] == one_cell_type].obs,
        x="cell_types",
        hue="CN",
        multiple="fill",
        stat="proportion",
    )
    # sns.histplot(adata_prot_subset.obs, x='cell_types',hue='CN', multiple='fill', stat='proportion')
    plt.title("Showcasing the signature CN progile of each minor B cell type")

# %%
if plot_flag:
    # sc.pl.embedding(adata_rna_subset, color=["major_cell_types","cell_types"], basis='X_original_umap',title='Original data major minor cell types')
    # sc.pl.embedding(adata_prot_subset, color=["major_cell_types","cell_types"], basis='X_original_umap',title='Original data major and minor cell types')

    # sc.pl.umap(adata_rna_subset, color="CN",neighbors_key='original_neighbors',title='Original RNA data CN')
    sc.pl.embedding(
        adata_rna_subset,
        color=["cell_types"],
        basis="X_original_umap",
        title="Original rna data CN",
    )
    sc.pl.embedding(
        adata_rna_subset,
        color=["CN", "cell_types"],
        basis="X_original_umap",
        title="Original rna data CN",
    )
    # sc.pl.umap(adata_prot_subset, color="CN",neighbors_key='original_neighbors',title='Original protein data CN')
    sc.pl.embedding(
        adata_prot_subset, color=["CN", "cell_types", "archetype_label"], basis="X_original_umap"
    )
    sc.pl.embedding(
        adata_prot_subset,
        color=[
            "archetype_label",
            "cell_types",
        ],
        basis="X_original_umap",
    )
    sc.pl.umap(
        adata_prot_subset[adata_prot_subset.obs["major_cell_types"] == one_cell_type],
        color="cell_types",
        neighbors_key="original_neighbors",
        title="Latent space MINOR cell types, B cells only",
    )

# %%
# DO NOT DELETE - save the adata of external processing
clean_uns_for_h5ad(adata_prot_subset)
clean_uns_for_h5ad(adata_rna_subset)
save_dir = Path("CODEX_RNA_seq/data/processed_data").absolute()

sc.write(Path(f"{save_dir}/adata_rna_subset"), adata_rna_subset)
sc.write(Path(f"{save_dir}/adata_prot_subset"), adata_prot_subset)


# %%

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

# %% Plotting Functions
# This file contains various plotting functions used in the analysis.

# %%
# %%
# Setup paths
# %%
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# %%
# Imports
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData, concat
from matplotlib.patches import Arc
from scipy.sparse import issparse


def plot_neighbor_means(adata_2_prot, neighbor_means):
    """Plot neighbor means and raw protein expression."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Mean Protein Expression of Cell Neighborhoods")
    sns.heatmap(neighbor_means)
    plt.subplot(1, 2, 2)
    plt.title("Raw Protein Expression per Cell")
    sns.heatmap(adata_2_prot.X)
    plt.show()


def plot_spatial_clusters(adata_2_prot, neighbor_means):
    """Plot spatial clusters and related visualizations."""
    fig, ax = plt.subplots()
    sc.pl.scatter(
        adata_2_prot,
        x="X",
        y="Y",
        color="CN",
        title="Cluster cells by their CN, can see the different CN in different regions, \nthanks to the different B cell types in each region",
        ax=ax,
        show=False,
    )

    neighbor_adata = AnnData(neighbor_means)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(neighbor_adata.X)
    plt.title("convet sqrt")
    plt.subplot(1, 2, 2)
    sns.heatmap(adata_2_prot.X.todense() if issparse(adata_2_prot.X) else adata_2_prot.X)
    plt.title("Proteins expressions of each cell")
    plt.show()

    neighbor_adata.obs["CN"] = pd.Categorical(adata_2_prot.obs["CN"])
    sc.pp.pca(neighbor_adata)
    sc.pp.neighbors(neighbor_adata)
    sc.tl.umap(neighbor_adata)
    sc.pl.umap(neighbor_adata, color="CN", title="UMAP of CN embedding")

    # Combine protein and CN data
    adata_prot_cn_concat = concat(
        [adata_2_prot, neighbor_adata], join="outer", label="modality", keys=["Protein", "CN"]
    )
    X = (
        adata_prot_cn_concat.X.toarray()
        if issparse(adata_prot_cn_concat.X)
        else adata_prot_cn_concat.X
    )
    X = np.nan_to_num(X)
    adata_prot_cn_concat.X = X
    sc.pp.pca(adata_prot_cn_concat)
    sc.pp.neighbors(adata_prot_cn_concat)
    sc.tl.umap(adata_prot_cn_concat)
    sc.pl.umap(
        adata_prot_cn_concat,
        color=["CN", "modality"],
        title=[
            "UMAP of CN embedding to make sure they are not mixed",
            "UMAP of CN embedding to make sure they are not mixed",
        ],
    )
    sc.pl.pca(
        adata_prot_cn_concat,
        color=["CN", "modality"],
        title=[
            "PCA of CN embedding to make sure they are not mixed",
            "PCA of CN embedding to make sure they are not mixed",
        ],
    )


def plot_modality_embeddings(adata_1_rna, adata_2_prot):
    """Plot PCA and UMAP embeddings for both modalities."""
    sc.tl.umap(adata_2_prot, neighbors_key="original_neighbors")
    sc.pl.pca(
        adata_1_rna,
        color=["cell_types", "major_cell_types"],
        title=["RNA pca minor cell types", "RNA pca major cell types"],
    )
    sc.pl.pca(
        adata_2_prot,
        color=["cell_types", "major_cell_types"],
        title=["Protein pca minor cell types", "Protein pca major cell types"],
    )
    sc.pl.embedding(
        adata_1_rna,
        basis="X_umap",
        color=["major_cell_types", "cell_types"],
        title=["RNA UMAP major cell types", "RNA UMAP major cell types"],
    )
    sc.pl.embedding(
        adata_2_prot,
        basis="X_original_umap",
        color=["major_cell_types", "cell_types"],
        title=["Protein UMAp major cell types", "Protein UMAP major cell types"],
    )


def plot_elbow_method(evs_protein, evs_rna):
    """Plot elbow method results."""
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(evs_protein)), evs_protein, marker="o", label="Protein")
    plt.plot(range(len(evs_rna)), evs_rna, marker="s", label="RNA")
    plt.xlabel("Number of Archetypes (k)")
    plt.ylabel("Explained Variance")
    plt.title("Elbow Plot: Explained Variance vs Number of Archetypes")
    plt.legend()
    plt.grid()


def plot_archetype_proportions(archetype_proportion_list_rna, archetype_proportion_list_protein):
    """Plot archetype proportions."""
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap((archetype_proportion_list_rna[0]), cbar=False)
    plt.xticks()
    plt.title("RNA Archetypes")
    plt.yticks([])
    plt.ylabel("Archetypes")
    plt.subplot(1, 2, 2)
    plt.title("Protein Archetypes")
    sns.heatmap((archetype_proportion_list_protein[0]), cbar=False)
    plt.suptitle("Non-Aligned Archetypes Profiles")
    plt.yticks([])
    plt.ylabel("Archetypes")
    plt.show()


def plot_archetype_weights(best_archetype_rna_prop, best_archetype_prot_prop, row_order):
    """Plot archetype weights."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("RNA Archetype Weights vs Cell Types")
    plt.ylabel("Archetypes")
    sns.heatmap(pd.DataFrame(best_archetype_rna_prop).iloc[row_order], cbar=False)
    plt.yticks([])
    plt.ylabel("Archetypes")
    plt.subplot(1, 2, 2)
    plt.ylabel("Archetypes")
    plt.title("Protein Archetype Weights vs Cell Types")
    sns.heatmap(pd.DataFrame(best_archetype_prot_prop).iloc[row_order], cbar=False)
    plt.ylabel("Archetypes")
    plt.suptitle(
        "Archetype Weight Distribution Across Cell Types (Higher Similarity = Better Alignment)"
    )
    plt.yticks([])
    plt.xticks(rotation=45)
    plt.show()


def plot_archetype_visualizations(
    adata_archetype_rna, adata_archetype_prot, adata_1_rna, adata_2_prot
):
    """Plot archetype visualizations."""
    sc.pp.pca(adata_archetype_rna)
    sc.pp.pca(adata_archetype_prot)
    sc.pl.pca(adata_archetype_rna, color=["major_cell_types", "archetype_label", "cell_types"])
    sc.pl.pca(adata_archetype_prot, color=["major_cell_types", "archetype_label", "cell_types"])
    sc.pp.neighbors(adata_archetype_rna)
    sc.pp.neighbors(adata_archetype_prot)
    sc.tl.umap(adata_archetype_rna)
    sc.tl.umap(adata_archetype_prot)
    sc.pl.umap(adata_archetype_rna, color=["major_cell_types", "archetype_label", "cell_types"])
    sc.pl.umap(adata_archetype_prot, color=["major_cell_types", "archetype_label", "cell_types"])

    sc.pp.neighbors(adata_1_rna)
    sc.pp.neighbors(adata_2_prot)
    sc.tl.umap(adata_1_rna)
    sc.tl.umap(adata_2_prot)
    sc.pl.umap(
        adata_1_rna, color="archetype_label", title="RNA UMAP Embedding Colored by Archetype Labels"
    )
    sc.pl.umap(
        adata_2_prot,
        color="archetype_label",
        title="Protein UMAP Embedding Colored by Archetype Labels",
    )


def plot_umap_visualizations_original_data(adata_rna_subset, adata_prot_subset):
    """Generate UMAP visualizations for original RNA and protein data"""
    print("\nGenerating UMAP visualizations...")
    sc.tl.umap(adata_rna_subset)
    sc.tl.umap(adata_prot_subset)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sc.pl.umap(
        adata_rna_subset,
        color="archetype_label",
        title="Original RNA cells associated Archetypes",
        show=False,
    )

    plt.subplot(1, 2, 2)
    sc.pl.umap(
        adata_prot_subset,
        color="archetype_label",
        title="Original Protein cells associated Archetypes",
        show=False,
    )
    plt.tight_layout()
    plt.show()


def plot_archetype_heatmaps(adata_rna_subset, adata_prot_subset, archetype_distances):
    """Plot heatmaps of archetype coordinates"""
    plt.figure(figsize=(10, 5))
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
    # this is the heatmap of the archetype distances
    plt.figure(figsize=(10, 5))
    plt.title("Archetype Distances")
    plt.subplot(1, 2, 1)
    sns.heatmap(np.log1p(archetype_distances[::5, ::5].T))
    plt.xlabel("RNA cell index")
    plt.ylabel("Protein cell index")
    plt.gca().invert_yaxis()

    plt.subplot(1, 2, 2)
    plt.title("minimum Archetype Distances between RNA and Protein cells")
    plt.scatter(
        np.arange(len(archetype_distances.argmin(axis=1))),
        archetype_distances.argmin(axis=1),
        s=1,
        rasterized=True,
    )
    plt.xlabel("RNA cell index")
    plt.ylabel("Protein cell index")
    plt.show()


def plot_pca_and_umap(adata_rna_subset, adata_prot_subset):
    """Plot PCA and UMAP visualizations"""
    sc.pl.pca(adata_rna_subset, color=["cell_types", "major_cell_types"])
    sc.pl.pca(adata_prot_subset, color=["cell_types", "major_cell_types"])
    sc.pl.embedding(
        adata_rna_subset, basis="X_original_umap", color=["cell_types", "major_cell_types"]
    )
    sc.pl.embedding(
        adata_prot_subset, basis="X_original_umap", color=["cell_types", "major_cell_types"]
    )


def plot_b_cells_analysis(adata_rna_subset):
    """Plot analysis for B cells"""
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


def plot_protein_umap(adata_prot_subset):
    """Plot protein UMAP visualizations"""
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
    return one_cell_type


def plot_cell_type_distribution(adata_prot_subset, one_cell_type):
    """Plot cell type distribution"""
    sns.histplot(
        adata_prot_subset[adata_prot_subset.obs["major_cell_types"] == one_cell_type].obs,
        x="cell_types",
        hue="CN",
        multiple="fill",
        stat="proportion",
    )
    plt.title("Showcasing the signature CN progile of each minor B cell type")


def plot_original_data_visualizations(adata_rna_subset, adata_prot_subset):
    """Plot original data visualizations"""
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

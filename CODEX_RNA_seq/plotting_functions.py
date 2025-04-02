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
from matplotlib.patches import Arc


def plot_umap_visualizations(adata_rna_subset, adata_prot_subset):
    """Generate UMAP visualizations for RNA and protein data"""
    print("\nGenerating UMAP visualizations...")
    sc.pp.neighbors(adata_rna_subset)
    sc.pp.neighbors(adata_prot_subset)
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

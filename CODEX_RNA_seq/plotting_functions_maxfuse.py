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


def plot_data_overview(adata_1, adata_2):
    """Plot overview of RNA and protein data"""
    print("\nPlotting data overview...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # RNA data
    sc.pl.pca(adata_1, color="cell_types", show=False, ax=axes[0])
    axes[0].set_title("RNA PCA")

    # Protein data
    sc.pl.pca(adata_2, color="cell_types", show=False, ax=axes[1])
    axes[1].set_title("Protein PCA")

    plt.tight_layout()
    plt.show()


def plot_cell_type_distribution(adata_1, adata_2):
    """Plot cell type distribution for both datasets"""
    print("\nPlotting cell type distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # RNA data
    sns.countplot(data=adata_1.obs, x="cell_types", ax=axes[0])
    axes[0].set_title("RNA Cell Types")
    axes[0].tick_params(axis="x", rotation=45)

    # Protein data
    sns.countplot(data=adata_2.obs, x="cell_types", ax=axes[1])
    axes[1].set_title("Protein Cell Types")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def plot_spatial_data(adata):
    """Plot spatial data for protein dataset"""
    print("\nPlotting spatial data...")
    plt.figure(figsize=(10, 10))

    # Get unique cell types and create a color map
    unique_cell_types = adata.obs["cell_types"].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cell_types)))
    color_dict = dict(zip(unique_cell_types, colors))

    # Create scatter plot
    for cell_type in unique_cell_types:
        mask = adata.obs["cell_types"] == cell_type
        plt.scatter(
            adata.obsm["spatial"][mask, 0],
            adata.obsm["spatial"][mask, 1],
            c=[color_dict[cell_type]],
            label=cell_type,
            s=1.5,
            alpha=0.6,
        )

    plt.title("Protein Spatial Data")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()
    plt.show()


def plot_highly_variable_genes(adata):
    """Plot highly variable genes"""
    print("\nPlotting highly variable genes...")
    sc.pl.highly_variable_genes(adata, show=False)
    plt.title("Highly Variable Genes")
    plt.show()


def plot_preprocessing_results(adata_1, adata_2):
    """Plot results after preprocessing"""
    print("\nPlotting preprocessing results...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    # RNA data
    sc.pl.pca(adata_1, color="cell_types", show=False, ax=axes[0, 0])
    axes[0, 0].set_title("RNA PCA")

    sc.pl.umap(adata_1, color="cell_types", show=False, ax=axes[0, 1])
    axes[0, 1].set_title("RNA UMAP")

    # Protein data
    sc.pl.pca(adata_2, color="cell_types", show=False, ax=axes[1, 0])
    axes[1, 0].set_title("Protein PCA")

    sc.pl.umap(adata_2, color="cell_types", show=False, ax=axes[1, 1])
    axes[1, 1].set_title("Protein UMAP")

    plt.tight_layout()
    plt.show()

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
# Setup paths
# %%
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# %%
# Imports
# %%
import importlib

import cell_lists
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData, concat
from scipy.sparse import issparse
from scipy.spatial import distance as scipy
from sklearn.decomposition import PCA

import bar_nick_utils

importlib.reload(cell_lists)
importlib.reload(bar_nick_utils)

# %% MaxFuse Plotting Functions
# This module contains functions for plotting MaxFuse-specific visualizations.


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


def plot_spatial_data(adata_prot):
    """Plot spatial data for protein dataset"""
    print("\nPlotting spatial data...")
    plt.figure(figsize=(10, 10))

    # Get unique cell types and create a color map
    unique_cell_types = adata_prot.obs["cell_types"].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cell_types)))
    color_dict = dict(zip(unique_cell_types, colors))

    # Create scatter plot
    for cell_type in unique_cell_types:
        mask = adata_prot.obs["cell_types"] == cell_type
        plt.scatter(
            adata_prot.obsm["spatial"][mask, 0],
            adata_prot.obsm["spatial"][mask, 1],
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


def plot_spatial_data_comparison(adata_rna, adata_prot):
    """Plot spatial data comparison between RNA and protein datasets"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # RNA data
    axes[0].scatter(
        adata_rna.obsm["spatial"][:, 0],
        adata_rna.obsm["spatial"][:, 1],
        c=adata_rna.obs["CN"],
        cmap="tab10",
        alpha=0.6,
    )
    axes[0].set_title("RNA Spatial Data")
    axes[0].set_xlabel("X coordinate")
    axes[0].set_ylabel("Y coordinate")

    # Protein data
    axes[1].scatter(
        adata_prot.obsm["spatial"][:, 0],
        adata_prot.obsm["spatial"][:, 1],
        c=adata_prot.obs["CN"],
        cmap="tab10",
        alpha=0.6,
    )
    axes[1].set_title("Protein Spatial Data")
    axes[1].set_xlabel("X coordinate")
    axes[1].set_ylabel("Y coordinate")

    plt.tight_layout()
    plt.show()


# %% VAE Plotting Functions
# This module contains functions for plotting VAE-specific visualizations.


def plot_latent(latent_rna, latent_prot, adata_rna, adata_prot, index_prot=None, index_rna=None):
    """Plot latent space representations."""
    if index_rna is None:
        index_rna = range(len(adata_rna.obs.index))
    if index_prot is None:
        index_prot = range(len(adata_prot.obs.index))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    cn_rna = adata_rna.obs["CN"].cat.codes.values
    cn_prot = adata_prot.obs["CN"].cat.codes.values

    pca_latent_rna = PCA(n_components=2).fit_transform(latent_rna)
    pca_latent_prot = PCA(n_components=2).fit_transform(latent_prot)
    scatter = axes[0].scatter(
        pca_latent_rna[:, 0],
        pca_latent_rna[:, 1],
        c=cn_rna,
        alpha=0.5,
    )
    axes[0].set_title("RNA Latent Space")
    axes[0].set_xlabel("Latent Dimension 1")
    axes[0].set_ylabel("Latent Dimension 2")

    # Plot protein latent space
    scatter = axes[1].scatter(
        pca_latent_prot[:, 0],
        pca_latent_prot[:, 1],
        c=cn_prot,
        alpha=0.5,
    )
    axes[1].set_title("Protein Latent Space")
    axes[1].set_xlabel("Latent Dimension 1")
    axes[1].set_ylabel("Latent Dimension 2")
    plt.tight_layout()
    plt.show()
    return fig


def plot_latent_mean_std(
    rna_inference_outputs,
    protein_inference_outputs,
    adata_rna,
    adata_prot,
    index_rna=None,
    index_prot=None,
    use_subsample=True,
):
    """Plot latent space visualization combining heatmaps and PCA plots.

    Args:
        rna_inference_outputs: RNA inference outputs containing qz means and scales
        protein_inference_outputs: Protein inference outputs containing qz means and scales
        adata_rna: RNA AnnData object
        adata_prot: Protein AnnData object
        index_rna: Indices for RNA data (optional)
        index_prot: Indices for protein data (optional)
        use_subsample: Whether to subsample to 300 points (default: True)
    """
    if index_rna is None:
        index_rna = range(len(adata_rna.obs.index))
    if index_prot is None:
        index_prot = range(len(adata_prot.obs.index))

    # Convert tensors to numpy if needed
    rna_mean = rna_inference_outputs["qz"].mean.detach().cpu().numpy()
    protein_mean = protein_inference_outputs["qz"].mean.detach().cpu().numpy()
    rna_std = rna_inference_outputs["qz"].scale.detach().cpu().numpy()
    protein_std = protein_inference_outputs["qz"].scale.detach().cpu().numpy()

    # Subsample if requested
    if use_subsample:
        n_subsample = min(300, len(index_rna))
        subsample_idx = np.random.choice(len(index_rna), n_subsample, replace=False)
        index_rna = np.array(index_rna)[subsample_idx]
        index_prot = np.array(index_prot)[subsample_idx]
        rna_mean = rna_mean[subsample_idx]
        protein_mean = protein_mean[subsample_idx]
        rna_std = rna_std[subsample_idx]
        protein_std = protein_std[subsample_idx]

    # Plot heatmaps
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    sns.heatmap(rna_mean)
    plt.title("RNA Mean Latent Space")

    plt.subplot(122)
    sns.heatmap(protein_mean)
    plt.title("Protein Mean Latent Space")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    sns.heatmap(rna_std)
    plt.title("RNA Std Latent Space")

    plt.subplot(122)
    sns.heatmap(protein_std)
    plt.title("Protein Std Latent Space")
    plt.tight_layout()
    plt.show()

    # Create AnnData objects for PCA visualization
    rna_ann = AnnData(X=rna_mean, obs=adata_rna.obs.iloc[index_rna].copy())
    protein_ann = AnnData(X=protein_mean, obs=adata_prot.obs.iloc[index_prot].copy())

    # Plot PCA and distributions
    plt.figure(figsize=(15, 5))

    # RNA PCA
    plt.subplot(131)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(rna_ann.X)

    df = pd.DataFrame(
        {
            "PC1": pca_result[:, 0],
            "PC2": pca_result[:, 1],
            "CN": rna_ann.obs["CN"],  # Add the CN column
        }
    )
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="CN", palette="viridis")
    pca = PCA(n_components=2)
    # pca_result = pca.fit_transform(rna_ann.X)
    # plt.scatter(pca_result[:, 0], pca_result[:, 1], c=rna_ann.obs["CN"])
    # plt.xlabel("PC1")
    # plt.ylabel("PC2")
    # plt.title("RNA Latent Space PCA")

    # Protein PCA
    plt.subplot(132)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(protein_ann.X)
    df = pd.DataFrame(
        {
            "PC1": pca_result[:, 0],
            "PC2": pca_result[:, 1],
            "CN": protein_ann.obs["CN"],  # Add the CN column
        }
    )
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="CN", palette="viridis")
    plt.title("Protein Latent Space PCA")

    # Standard deviation distributions
    plt.subplot(133)
    plt.hist(rna_std.flatten(), bins=50, alpha=0.5, label="RNA", density=True)
    plt.hist(protein_std.flatten(), bins=50, alpha=0.5, label="Protein", density=True)
    plt.title("Latent Space Standard Deviations")
    plt.xlabel("Standard Deviation")
    plt.ylabel("Density")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_rna_protein_matching_means_and_scale(rna_inference_outputs, protein_inference_outputs):
    """Plot RNA and protein matching means and scales"""
    plt.figure(figsize=(15, 5))

    # Plot means
    plt.subplot(131)
    plt.scatter(
        rna_inference_outputs["qz"].mean.detach().cpu().numpy().flatten(),
        protein_inference_outputs["qz"].mean.detach().cpu().numpy().flatten(),
        alpha=0.1,
    )
    plt.xlabel("RNA Latent Mean")
    plt.ylabel("Protein Latent Mean")
    plt.title("Latent Means Comparison")

    # Plot scales
    plt.subplot(132)
    plt.scatter(
        rna_inference_outputs["qz"].scale.detach().cpu().numpy().flatten(),
        protein_inference_outputs["qz"].scale.detach().cpu().numpy().flatten(),
        alpha=0.1,
    )
    plt.xlabel("RNA Latent Scale")
    plt.ylabel("Protein Latent Scale")
    plt.title("Latent Scales Comparison")

    # Plot mean vs scale
    plt.subplot(133)
    plt.scatter(
        rna_inference_outputs["qz"].mean.detach().cpu().numpy().flatten(),
        rna_inference_outputs["qz"].scale.detach().cpu().numpy().flatten(),
        alpha=0.1,
        label="RNA",
    )
    plt.scatter(
        protein_inference_outputs["qz"].mean.detach().cpu().numpy().flatten(),
        protein_inference_outputs["qz"].scale.detach().cpu().numpy().flatten(),
        alpha=0.1,
        label="Protein",
    )
    plt.xlabel("Latent Mean")
    plt.ylabel("Latent Scale")
    plt.title("Mean vs Scale")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_inference_outputs(
    rna_inference_outputs,
    protein_inference_outputs,
    latent_distances,
    rna_distances,
    prot_distances,
):
    """Plot inference outputs"""
    print("\nPlotting inference outputs...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Plot latent distances
    axes[0, 0].hist(latent_distances.detach().cpu().numpy().flatten(), bins=50)
    axes[0, 0].set_title("Latent Distances")

    # Plot RNA distances
    axes[0, 1].hist(rna_distances.detach().cpu().numpy().flatten(), bins=50)
    axes[0, 1].set_title("RNA Distances")

    # Plot protein distances
    axes[0, 2].hist(prot_distances.detach().cpu().numpy().flatten(), bins=50)
    axes[0, 2].set_title("Protein Distances")

    # Plot latent vs RNA distances
    axes[1, 0].scatter(
        rna_distances.detach().cpu().numpy().flatten(),
        latent_distances.detach().cpu().numpy().flatten(),
        alpha=0.1,
    )
    axes[1, 0].set_title("Latent vs RNA Distances")

    # Plot latent vs protein distances
    axes[1, 1].scatter(
        prot_distances.detach().cpu().numpy().flatten(),
        latent_distances.detach().cpu().numpy().flatten(),
        alpha=0.1,
    )
    axes[1, 1].set_title("Latent vs Protein Distances")

    # Plot RNA vs protein distances
    axes[1, 2].scatter(
        rna_distances.detach().cpu().numpy().flatten(),
        prot_distances.detach().cpu().numpy().flatten(),
        alpha=0.1,
    )
    axes[1, 2].set_title("RNA vs Protein Distances")

    plt.tight_layout()
    plt.show()


def plot_similarity_loss_history(similarity_loss_history, active_history):
    """Plot similarity loss history and active state"""
    plt.figure(figsize=(15, 5))

    # Plot similarity loss
    plt.subplot(121)
    plt.plot(similarity_loss_history)
    plt.title("Similarity Loss History")
    plt.xlabel("Step")
    plt.ylabel("Loss")

    # Plot active state
    plt.subplot(122)
    plt.plot(active_history)
    plt.title("Similarity Loss Active State")
    plt.xlabel("Step")
    plt.ylabel("Active (1) / Inactive (0)")

    plt.tight_layout()
    plt.show()


def plot_normalized_losses(history):
    """Plot normalized training losses."""
    plt.figure(figsize=(15, 5))

    # Get all loss keys from history
    loss_keys = [k for k in history.keys() if "loss" in k.lower() and len(history[k]) > 0]

    # Normalize each loss
    normalized_losses = {}
    for key in loss_keys:
        values = history[key]
        if len(values) > 0:  # Only process non-empty lists
            values = np.array(values)
            # Remove inf and nan
            values = values[~np.isinf(values) & ~np.isnan(values)]
            if len(values) > 0:  # Check again after filtering
                min_val = np.min(values)
                max_val = np.max(values)
                if max_val > min_val:  # Avoid division by zero
                    normalized_losses[key] = (values - min_val) / (max_val - min_val)

    # Plot each normalized loss
    for key, values in normalized_losses.items():
        plt.plot(values, label=key, alpha=0.7)

    plt.title("Normalized Training Losses")
    plt.xlabel("Step")
    plt.ylabel("Normalized Loss")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_cosine_distance(rna_batch, protein_batch):
    """Plot cosine distance between archetype vectors"""
    print("\nPlotting cosine distances...")
    archetype_dis = scipy.cdist(
        rna_batch["archetype_vec"].detach().cpu().numpy(),
        protein_batch["archetype_vec"].detach().cpu().numpy(),
        metric="cosine",
    )

    plt.figure(figsize=(10, 6))
    plt.hist(archetype_dis.flatten(), bins=50)
    plt.title("Cosine Distance Distribution")
    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_archetype_vs_latent_distances(archetype_dis_tensor, latent_distances, threshold):
    """Plot archetype vs latent distances"""
    print("\nPlotting archetype vs latent distances...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot archetype distances
    axes[0].hist(archetype_dis_tensor.detach().cpu().numpy().flatten(), bins=50)
    axes[0].axvline(x=threshold, color="r", linestyle="--", label=f"Threshold: {threshold}")
    axes[0].set_title("Archetype Distances")
    axes[0].legend()

    # Plot latent distances
    axes[1].hist(latent_distances.detach().cpu().numpy().flatten(), bins=50)
    axes[1].axvline(x=threshold, color="r", linestyle="--", label=f"Threshold: {threshold}")
    axes[1].set_title("Latent Distances")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_latent_distances(latent_distances, threshold):
    """Plot latent distances and threshold"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot latent distances heatmap
    sns.heatmap(latent_distances.detach().cpu().numpy(), ax=axes[0])
    axes[0].set_title("Latent Distances")
    axes[0].legend()

    # Plot latent distances
    axes[1].hist(latent_distances.detach().cpu().numpy().flatten(), bins=50)
    axes[1].axvline(x=threshold, color="r", linestyle="--", label=f"Threshold: {threshold}")
    axes[1].set_title("Latent Distances")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_combined_latent_space(combined_latent):
    """Plot combined latent space visualizations"""
    # Plot UMAP
    sc.tl.umap(combined_latent, min_dist=0.1)
    sc.pl.umap(
        combined_latent,
        color=["CN", "modality"],
        title=["UMAP Combined Latent space CN", "UMAP Combined Latent space modality"],
        alpha=0.5,
    )
    sc.pl.umap(
        combined_latent,
        color=["CN", "modality", "cell_types"],
        title=[
            "UMAP Combined Latent space CN",
            "UMAP Combined Latent space modality",
            "UMAP Combined Latent space cell types",
        ],
        alpha=0.5,
    )

    # Plot PCA
    sc.pl.pca(
        combined_latent,
        color=["CN", "modality"],
        title=["PCA Combined Latent space CN", "PCA Combined Latent space modality"],
        alpha=0.5,
    )


def plot_cell_type_distributions(combined_latent, top_n=3):
    """Plot UMAP for top N most common cell types"""
    top_cell_types = combined_latent.obs["cell_types"].value_counts().index[:top_n]

    for cell_type in top_cell_types:
        cell_type_data = combined_latent[combined_latent.obs["cell_types"] == cell_type]
        sc.pl.umap(
            cell_type_data,
            color=["CN", "modality", "cell_types"],
            title=[
                f"Combined latent space UMAP {cell_type}, CN",
                f"Combined latent space UMAP {cell_type}, modality",
                f"Combined latent space UMAP {cell_type}, cell types",
            ],
            alpha=0.5,
        )


def plot_rna_protein_cn_cell_type_umap(rna_vae_new, protein_vae):
    """Plot RNA and protein embeddings"""
    sc.pl.embedding(
        rna_vae_new.adata,
        color=["CN", "cell_types"],
        basis="X_scVI",
        title=["RNA Latent space, CN", "RNA Latent space, cell types"],
    )
    sc.pl.embedding(
        protein_vae.adata,
        color=["CN", "cell_types"],
        basis="X_scVI",
        title=["Protein Latent space UMAP, CN ", "Protein Latent space UMAP, cell types"],
    )


def plot_combined_latent_space_umap(combined_latent):
    """Plot UMAP of combined latent space"""
    sc.tl.umap(combined_latent, min_dist=0.1)
    sc.pl.umap(
        combined_latent,
        color=["CN", "modality"],
        title=["UMAP Combined Latent space CN", "UMAP Combined Latent space modality"],
        alpha=0.5,
    )
    sc.pl.umap(
        combined_latent,
        color=["CN", "modality", "cell_types"],
        title=[
            "UMAP Combined Latent space CN",
            "UMAP Combined Latent space modality",
            "UMAP Combined Latent space cell types",
        ],
        alpha=0.5,
    )


def plot_archetype_embedding(rna_vae_new, protein_vae):
    """Plot archetype embedding"""
    rna_archtype = AnnData(rna_vae_new.adata.obsm["archetype_vec"])
    rna_archtype.obs = rna_vae_new.adata.obs
    sc.pp.neighbors(rna_archtype)
    sc.tl.umap(rna_archtype)

    prot_archtype = AnnData(protein_vae.adata.obsm["archetype_vec"])
    prot_archtype.obs = protein_vae.adata.obs
    sc.pp.neighbors(prot_archtype)
    sc.tl.umap(prot_archtype)

    # Plot archetype vectors
    sc.pl.umap(
        rna_archtype,
        color=["CN", "cell_types"],
        title=["RNA Archetype embedding UMAP CN", "RNA Archetype embedding UMAP cell types"],
    )
    sc.pl.umap(
        prot_archtype,
        color=["CN", "cell_types"],
        title=[
            "Protein Archetype embedding UMAP CN",
            "Protein Archetype embedding UMAP cell types",
        ],
    )


# %%


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
    plt.subplot(1, 2, 2)
    plt.title("Protein Archetypes")
    sns.heatmap((archetype_proportion_list_protein[0]), cbar=False)
    plt.suptitle("Non-Aligned Archetypes Profiles")
    plt.yticks([])
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


def plot_cell_type_distribution_single(adata_prot_subset, one_cell_type):
    """Plot cell type distribution for a single dataset"""
    sns.histplot(
        adata_prot_subset[adata_prot_subset.obs["cell_types"] == one_cell_type].obs,
        x="cell_types",
        hue="CN",
        multiple="fill",
        stat="proportion",
    )


def plot_original_data_visualizations(adata_rna_subset, adata_prot_subset):
    """Plot original data visualizations"""

    sc.pl.embedding(
        adata_rna_subset,
        color=["CN", "cell_types", "archetype_label"],
        basis="X_original_umap",
        title=[
            "Original rna data CN",
            "Original rna data cell types",
            "Original rna data archetype label",
        ],
    )

    sc.pl.embedding(
        adata_prot_subset,
        color=[
            "CN",
            "archetype_label",
            "cell_types",
        ],
        basis="X_original_umap",
        title=[
            "Original protein data CN",
            "Original protein data archetype label",
            "Original protein data cell types",
        ],
    )


# %%


def plot_latent_single(means, adata, index, color_label="CN", title=""):
    plt.figure(figsize=(10, 5))
    pca = PCA(n_components=3)
    means_cpu = means.detach().cpu().numpy()
    index_cpu = index.detach().cpu().numpy().flatten()
    pca.fit(means_cpu)
    rna_pca = pca.transform(means_cpu)
    plt.subplot(1, 1, 1)
    plt.scatter(
        rna_pca[:, 0],
        rna_pca[:, 1],
        c=adata[index_cpu].obs[color_label].values.astype(float),
        cmap="jet",
    )
    plt.title(title)
    plt.show()


# %%

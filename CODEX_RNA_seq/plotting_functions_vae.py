# %%
# Setup paths
# %%
import os
import sys

from sklearn.decomposition import PCA

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anndata

# %%
# Imports
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.spatial
import scipy.spatial.distance as scipy
import seaborn as sns
import torch
import torch.nn.functional as F
from anndata import AnnData
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE


def plot_latent(latent_rna, latent_prot, adata_rna, adata_prot, index=None):
    """Plot latent space visualization"""
    if index is None:
        index = range(len(adata_prot.obs.index))

    # Create AnnData objects for visualization
    rna_ann = AnnData(X=latent_rna)
    prot_ann = AnnData(X=latent_prot)

    # Add metadata
    rna_ann.obs = adata_rna.obs.iloc[index].copy()
    prot_ann.obs = adata_prot.obs.iloc[index].copy()

    # Compute PCA
    sc.pp.pca(rna_ann)
    sc.pp.pca(prot_ann)

    # Plot PCA
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # RNA PCA
    axes[0].scatter(
        rna_ann.obsm["X_pca"][:, 0],
        rna_ann.obsm["X_pca"][:, 1],
        c=rna_ann.obs["CN"],
        cmap="tab10",
        alpha=0.6,
    )
    axes[0].set_title("RNA Latent Space PCA")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    # Protein PCA
    axes[1].scatter(
        prot_ann.obsm["X_pca"][:, 0],
        prot_ann.obsm["X_pca"][:, 1],
        c=prot_ann.obs["CN"],
        cmap="tab10",
        alpha=0.6,
    )
    axes[1].set_title("Protein Latent Space PCA")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    plt.tight_layout()
    plt.show()


def plot_latent_mean_std(
    rna_inference_outputs, protein_inference_outputs, adata_rna, adata_prot, index=None
):
    """Plot latent space visualization"""
    if index is None:
        index = range(len(adata_prot.obs.index))

    """Plot mean and standard deviation of latent space."""
    # Extract latent means and convert to numpy arrays
    rna_latent = rna_inference_outputs["qz"].mean.detach().cpu().numpy()
    protein_latent = protein_inference_outputs["qz"].mean.detach().cpu().numpy()

    # Create AnnData objects with proper initialization
    rna_ann = AnnData(X=rna_latent, obs=adata_rna.obs.iloc[index].copy())
    protein_ann = AnnData(X=protein_latent, obs=adata_prot.obs.iloc[index].copy())

    # Compute PCA for both datasets
    sc.pp.pca(rna_ann)
    sc.pp.pca(protein_ann)

    # Plot PCA for RNA latent space
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(rna_ann.X)
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=rna_ann.obs["CN"])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("RNA Latent Space PCA")

    # Plot PCA for protein latent space
    plt.subplot(132)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(protein_ann.X)
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=protein_ann.obs["CN"])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Protein Latent Space PCA")

    # Plot standard deviations
    plt.subplot(133)
    rna_std = rna_inference_outputs["qz"].scale.detach().cpu().numpy()
    protein_std = protein_inference_outputs["qz"].scale.detach().cpu().numpy()

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
    """Plot normalized training and validation losses"""
    plt.figure(figsize=(15, 5))

    # Plot training losses
    plt.subplot(121)
    train_keys = [key for key in history.keys() if key.startswith("train_")]
    for key in train_keys:
        loss_data = np.array(history[key])
        if len(loss_data) > 0:
            min_val = loss_data.min()
            max_val = loss_data.max()
            normalized_loss = (loss_data - min_val) / (max_val - min_val + 1e-8)
            plt.plot(normalized_loss, label=f"{key} (min: {min_val:.2f}, max: {max_val:.2f})")
    plt.title("Normalized Training Losses")
    plt.xlabel("Training Step")
    plt.ylabel("Normalized Loss Value")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Plot validation losses
    plt.subplot(122)
    val_keys = [key for key in history.keys() if key.startswith("validation_")]
    for key in val_keys:
        loss_data = np.array(history[key])
        if len(loss_data) > 0:
            min_val = loss_data.min()
            max_val = loss_data.max()
            normalized_loss = (loss_data - min_val) / (max_val - min_val + 1e-8)
            plt.plot(normalized_loss, label=f"{key} (min: {min_val:.2f}, max: {max_val:.2f})")
    plt.title("Normalized Validation Losses")
    plt.xlabel("Validation Step")
    plt.ylabel("Normalized Loss Value")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

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


def plot_spatial_data(adata_rna, adata_prot):
    """Plot spatial data with cell types and CN"""
    plt.figure(figsize=(12, 6))

    # Plot with cell type as color
    plt.subplot(1, 2, 1)
    sns.scatterplot(
        x=adata_prot.obs["X"],
        y=adata_prot.obs["Y"],
        hue=adata_prot.obs["cell_types"],
        palette="tab10",
        s=10,
    )
    plt.title("Protein cells colored by cell type")
    plt.legend(loc="upper right", fontsize="small", title_fontsize="small")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.title("RNA cells colored by cell types")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend([], [], frameon=False)

    # Plot with CN as color
    plt.subplot(1, 2, 2)
    sns.scatterplot(
        x=adata_prot.obs["X"],
        y=adata_prot.obs["Y"],
        hue=adata_prot.obs["CN"],
        s=10,
    )
    plt.title("Protein cells colored by CN")
    plt.xlabel("X")
    plt.ylabel("Y")

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
                f"UMAP {cell_type} CN",
                f"UMAP {cell_type} modality",
                f"UMAP {cell_type} cell types",
            ],
            alpha=0.5,
        )


def plot_latent_distances(rna_latent, prot_latent, distances):
    """Plot latent space distances between modalities"""
    # Randomize RNA latent space to compare distances
    rand_rna_latent = rna_latent.copy()
    shuffled_indices = np.random.permutation(rand_rna_latent.obs.index)
    rand_rna_latent = rand_rna_latent[shuffled_indices].copy()
    rand_distances = np.linalg.norm(rand_rna_latent.X - prot_latent.X, axis=1)

    # Plot randomized latent space distances
    rand_rna_latent.obs["latent_dis"] = np.log(distances)

    sc.tl.umap(rand_rna_latent)
    sc.pl.umap(
        rand_rna_latent,
        cmap="coolwarm",
        color="latent_dis",
        title="Latent space distances between RNA and Protein cells",
    )

    return rand_distances


def plot_rna_protein_embeddings(rna_vae_new, protein_vae):
    """Plot RNA and protein embeddings"""
    sc.pl.embedding(
        rna_vae_new.adata,
        color=["CN", "cell_types"],
        basis="X_scVI",
        title=["Latent space, CN RNA", "Latent space, minor cell types RNA"],
    )
    sc.pl.embedding(
        protein_vae.adata,
        color=["CN", "cell_types"],
        basis="X_scVI",
        title=["Latent space, CN Protein", "Latent space, minor cell types Protein"],
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


def plot_archetype_vectors(rna_vae_new, protein_vae):
    """Plot archetype vectors"""
    # Process archetype vectors
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
        title=["RNA Archetype UMAP CN", "RNA Archetype UMAP cell types"],
    )
    sc.pl.umap(
        prot_archtype,
        color=["CN", "cell_types"],
        title=["Protein Archetype UMAP CN", "Protein Archetype UMAP cell types"],
    )


# %%

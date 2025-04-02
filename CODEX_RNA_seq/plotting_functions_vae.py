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
import seaborn as sns
from anndata import AnnData


def plot_latent(rna_latent, protein_latent, rna_adata, protein_adata, index=None):
    """Plot latent space visualization"""
    print("\nPlotting latent space...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Create AnnData objects and compute PCA for RNA data
    rna_ann = AnnData(rna_latent)
    if index is not None:
        rna_ann.obs_names = [f"RNA_{i}" for i in range(len(rna_latent))]
        rna_ann.obs["cell_types"] = rna_adata.obs["cell_types"].iloc[index].values
    else:
        rna_ann.obs_names = [f"RNA_{i}" for i in range(len(rna_latent))]
        rna_ann.obs["cell_types"] = rna_adata.obs["cell_types"].values
    sc.pp.pca(rna_ann)
    sc.pl.pca(rna_ann, color="cell_types", show=False, ax=axes[0])
    axes[0].set_title("RNA Latent Space")

    # Create AnnData objects and compute PCA for protein data
    protein_ann = AnnData(protein_latent)
    if index is not None:
        protein_ann.obs_names = [f"Protein_{i}" for i in range(len(protein_latent))]
        protein_ann.obs["cell_types"] = protein_adata.obs["cell_types"].iloc[index].values
    else:
        protein_ann.obs_names = [f"Protein_{i}" for i in range(len(protein_latent))]
        protein_ann.obs["cell_types"] = protein_adata.obs["cell_types"].values
    sc.pp.pca(protein_ann)
    sc.pl.pca(protein_ann, color="cell_types", show=False, ax=axes[1])
    axes[1].set_title("Protein Latent Space")

    plt.tight_layout()
    plt.show()


def plot_latent_mean_std(rna_inference_outputs, protein_inference_outputs):
    """Plot mean and standard deviation of latent space."""
    # Extract latent means and convert to numpy arrays
    rna_latent = rna_inference_outputs["qz"].mean.detach().cpu().numpy()
    protein_latent = protein_inference_outputs["qz"].mean.detach().cpu().numpy()

    # Create AnnData objects with proper initialization
    rna_ann = AnnData(X=rna_latent)
    protein_ann = AnnData(X=protein_latent)

    # Compute PCA for both datasets
    sc.pp.pca(rna_ann)
    sc.pp.pca(protein_ann)

    # Plot PCA for RNA latent space
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(rna_ann.X)
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("RNA Latent Space PCA")

    # Plot PCA for protein latent space
    plt.subplot(132)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(protein_ann.X)
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
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


def plot_rna_protein_matching_means_and_scale(
    rna_inference_outputs, protein_inference_outputs, use_subsample=True
):
    if use_subsample:
        n_samples = min(
            300,
            min(
                rna_inference_outputs["qz"].mean.shape[0],
                protein_inference_outputs["qz"].mean.shape[0],
            ),
        )
        subsample_indexes = np.random.choice(
            rna_inference_outputs["qz"].mean.shape[0], n_samples, replace=False
        )
    else:
        subsample_indexes = np.arange(rna_inference_outputs["qz"].mean.shape[0])
    rna_means = rna_inference_outputs["qz"].mean.detach().cpu().numpy()[subsample_indexes]
    rna_scales = protein_inference_outputs["qz"].scale.detach().cpu().numpy()[subsample_indexes]
    protein_means = protein_inference_outputs["qz"].mean.detach().cpu().numpy()[subsample_indexes]
    protein_scales = protein_inference_outputs["qz"].scale.detach().cpu().numpy()[subsample_indexes]

    # Combine means for PCA
    combined_means = np.concatenate([rna_means, protein_means], axis=0)

    # Fit PCA on means
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined_means)

    # Transform scales using the same PCA transformation
    combined_scales = np.concatenate([rna_scales, protein_scales], axis=0)
    scales_transformed = pca.transform(combined_scales)

    # Plot with halos
    plt.figure(figsize=(8, 6))

    # Plot RNA points and halos
    for i in range(rna_means.shape[0]):
        # Add halo using scale information
        circle = plt.Circle(
            (pca_result[i, 0], pca_result[i, 1]),
            radius=np.linalg.norm(scales_transformed[i]) * 0.05,
            color="blue",
            alpha=0.1,
        )
        plt.gca().add_patch(circle)
    # Plot Protein points and halos
    for i in range(protein_means.shape[0]):
        # Add halo using scale information
        circle = plt.Circle(
            (pca_result[rna_means.shape[0] + i, 0], pca_result[rna_means.shape[0] + i, 1]),
            radius=np.linalg.norm(scales_transformed[rna_means.shape[0] + i]) * 0.05,
            color="orange",
            alpha=0.1,
        )
        plt.gca().add_patch(circle)

    # Add connecting lines
    for i in range(rna_means.shape[0]):
        color = "red" if (i % 2 == 0) else "green"
        plt.plot(
            [pca_result[i, 0], pca_result[rna_means.shape[0] + i, 0]],
            [pca_result[i, 1], pca_result[rna_means.shape[0] + i, 1]],
            "k-",
            alpha=0.2,
            color=color,
        )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA of RNA and Protein with Scale Halos")
    plt.legend()
    plt.gca().set_aspect("equal")
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


def plot_similarity_loss_history(similarity_loss_history, active_similarity_loss_active_history):
    """Plot similarity loss history"""
    print("\nPlotting similarity loss history...")
    plt.figure(figsize=(12, 6))

    # Plot loss values
    plt.plot(similarity_loss_history, label="Similarity Loss")

    # Plot active/inactive periods
    active_periods = np.array(active_similarity_loss_active_history)
    plt.fill_between(
        range(len(active_periods)),
        min(similarity_loss_history),
        max(similarity_loss_history),
        where=active_periods,
        alpha=0.3,
        color="green",
        label="Active Periods",
    )

    plt.title("Similarity Loss History")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_normalized_losses(history):
    """Plot normalized training losses"""
    print("\nPlotting normalized losses...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    # Plot reconstruction losses
    axes[0, 0].plot(history["train_rna_reconstruction_loss"], label="RNA")
    axes[0, 0].plot(history["train_protein_reconstruction_loss"], label="Protein")
    axes[0, 0].set_title("Reconstruction Losses")
    axes[0, 0].legend()

    # Plot contrastive loss
    axes[0, 1].plot(history["train_contrastive_loss"])
    axes[0, 1].set_title("Contrastive Loss")

    # Plot matching loss
    axes[1, 0].plot(history["train_matching_rna_protein_loss"])
    axes[1, 0].set_title("Matching Loss")

    # Plot total loss
    axes[1, 1].plot(history["train_total_loss"])
    axes[1, 1].set_title("Total Loss")

    plt.tight_layout()
    plt.show()


def plot_cosine_distance(rna_batch, protein_batch):
    """Plot cosine distance between archetype vectors"""
    print("\nPlotting cosine distances...")
    archetype_dis = scipy.spatial.distance.cdist(
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


# %%

import copy
import os
import re
import time
from itertools import product, zip_longest
from typing import Any, Dict, List, Union

import anndata as ad
import cvxpy as cp
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import scipy.sparse as sp
import seaborn as sns
import torch
from anndata import AnnData
from scipy.optimize import linear_sum_assignment, nnls
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.manifold import TSNE

# wrap UMAP to filter duplicates to avoid UMAP distortion


def get_umap_filtered_fucntion():
    # Save original UMAP function if not already wrapped
    _original_umap = sc.tl.umap

    def umap_filtered(adata, *args, **kwargs):
        if "duplicate" in adata.obs.columns:
            # Filter duplicates and remove the triggering column
            adata_filtered = adata[~adata.obs["duplicate"]].copy()
            adata_filtered.obs["duplicate_temp"] = adata_filtered.obs["duplicate"]
            del adata_filtered.obs["duplicate"]
            # Run original UMAP on filtered data
            _original_umap(adata_filtered, *args, **kwargs)
            adata_filtered.obs["duplicate"] = adata_filtered.obs["duplicate_temp"]
            # Map results back to original adata
            umap_results = np.full((adata.n_obs, adata_filtered.obsm["X_umap"].shape[1]), np.nan)
            umap_results[~adata.obs["duplicate"].values] = adata_filtered.obsm["X_umap"]
            adata.obsm["X_umap"] = umap_results
        else:
            _original_umap(adata, *args, **kwargs)

    return umap_filtered


if not hasattr(sc.tl.umap, "_is_wrapped"):
    sc.tl.umap = get_umap_filtered_fucntion()
    sc.tl.umap._is_wrapped = True


# Function to get the latest file based on the timestamp
# plot the mean and std of each in subplots
def archetype_vs_latent_distances_plot(
    archetype_dis_tensor, latent_distances, threshold, use_subsample=True
):
    if use_subsample:
        subsample_indexes = torch.tensor(np.arange(min(300, archetype_dis_tensor.shape[0])))
    else:
        subsample_indexes = torch.tensor(np.arange(archetype_dis_tensor.shape[0]))
    archetype_dis_tensor_ = archetype_dis_tensor.detach().cpu()
    archetype_dis_tensor_ = torch.index_select(
        archetype_dis_tensor_, 0, subsample_indexes
    )  # Select rows
    archetype_dis_tensor_ = torch.index_select(
        archetype_dis_tensor_, 1, subsample_indexes
    )  # Select columns
    latent_distances_ = latent_distances.detach().cpu()
    latent_distances_ = torch.index_select(latent_distances_, 0, subsample_indexes)
    latent_distances_ = torch.index_select(latent_distances_, 1, subsample_indexes)
    latent_distances_ = latent_distances_.numpy()
    archetype_dis = archetype_dis_tensor_.numpy()
    all_distances = np.sort(archetype_dis.flatten())
    below_threshold_distances = np.sort(archetype_dis)[latent_distances_ < threshold].flatten()
    fig, ax1 = plt.subplots()
    counts_all, bins_all, _ = ax1.hist(
        all_distances, bins=100, alpha=0.5, label="All Distances", color="blue"
    )
    ax1.set_ylabel("Count (All Distances)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2 = ax1.twinx()
    counts_below, bins_below, _ = ax2.hist(
        below_threshold_distances, bins=bins_all, alpha=0.5, label="Below Threshold", color="green"
    )
    ax2.set_ylabel("Count (Below Threshold)", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    plt.title(
        f"Number Below Threshold: {np.sum(latent_distances.detach().cpu().numpy() < threshold)}"
    )
    plt.show()

    plt.figure()
    plt.subplot(1, 2, 1)
    sns.heatmap(latent_distances_)
    plt.title("latent_distances")
    plt.subplot(1, 2, 2)
    sns.heatmap(archetype_dis)
    plt.title("archetype_distances")
    plt.show()


def compare_distance_distributions(rand_distances, rna_latent, prot_latent, distances):
    # cannot use subsample here, since we get distaces indexs that need to be matched

    # Randomize RNA cells within cell types
    rand_rna_latent = rna_latent.copy()
    for cell_type in rand_rna_latent.obs["cell_types"].unique():
        cell_type_indices = rand_rna_latent.obs["cell_types"] == cell_type
        shuffled_indices = np.random.permutation(rand_rna_latent[cell_type_indices].obs.index)
        rand_rna_latent.X[cell_type_indices] = (
            rand_rna_latent[cell_type_indices][shuffled_indices].copy().X
        )
    rand_distances_cell_type = torch.cdist(
        torch.tensor(rand_rna_latent.X), torch.tensor(prot_latent.X)
    )

    # Filter distances using 95th percentile
    rand_dist_filtered = rand_distances[rand_distances < np.percentile(rand_distances, 95)]
    rand_dist_cell_filtered = rand_distances_cell_type[
        rand_distances_cell_type < np.percentile(rand_distances_cell_type, 95)
    ]
    true_dist_filtered = distances[distances < np.percentile(distances, 95)]

    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Calculate common bins based on all data
    all_data = np.concatenate([rand_dist_filtered, rand_dist_cell_filtered, true_dist_filtered])
    bins = np.linspace(all_data.min(), all_data.max(), 200)

    # Plot random distances on first y-axis
    counts_rand, bins_rand, _ = ax1.hist(
        rand_dist_filtered, bins=bins, alpha=0.5, color="green", label="Randomized distances"
    )
    ax1.set_ylabel("Count (Random)", color="green")
    ax1.tick_params(axis="y", labelcolor="green")

    # Create second y-axis for cell type distances
    ax2 = ax1.twinx()
    counts_cell, bins_cell, _ = ax2.hist(
        rand_dist_cell_filtered, bins=bins, alpha=0.5, color="red", label="Within cell types"
    )
    ax2.set_ylabel("Count (Cell Types)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Create third y-axis for true distances
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.2))
    counts_true, bins_true, _ = ax3.hist(
        true_dist_filtered, bins=bins, alpha=0.5, color="blue", label="True distances"
    )
    ax3.set_ylabel("Count (True)", color="blue")
    ax3.tick_params(axis="y", labelcolor="blue")

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax3.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc="upper right")

    plt.title("Distribution of Distances (95th percentile)")
    plt.show()


def select_gene_likelihood(adata):
    """Select an appropriate gene likelihood based on the data."""
    if issparse(adata.X):
        # For sparse matrices, assume it's count data (common in scRNA-seq)
        return "zinb"
    elif adata.X.min() >= 0:
        # If all values are non-negative
        if np.allclose(np.round(adata.X), adata.X):
            # If data is integer-valued (like counts)
            return "zinb"
        else:
            # If data contains non-integer values
            return "normal"
    else:
        # For data with negative values
        return "normal"


def calculate_cLISI(adata, label_key="cell_type", neighbors_key="neighbors", plot_flag=False):
    """
    Calculate cell-type Local Inverse Simpson's Index (LISI) using precomputed neighbors.

    The cLISI score measures how well cell types are separated in the embedding space.
    Higher scores indicate better cell type separation, with a minimum value of 1
    (all neighbors same cell type) and maximum of k+1 (all neighbors different cell types),
    where k is the number of neighbors used.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with precomputed neighbors
    label_key : str, default='cell_type'
        Column in adata.obs containing cell type labels
    neighbors_key : str, default='neighbors'
        Key where neighbor information is stored in adata.uns

    Returns
    -------
    float
        Median cLISI score across all cells. Higher values indicate better
        cell type separation in the embedding space.
    """

    if neighbors_key not in adata.uns:
        raise ValueError(f"Run sc.pp.neighbors with key='{neighbors_key}' first")

    connectivities = adata.obsp[f"connectivities"]
    n_cells = adata.n_obs

    lisi_scores = []
    for i in range(n_cells):
        neighbors = connectivities[i].indices
        neighbors = np.append(neighbors, i)

        labels = adata.obs[label_key].iloc[neighbors].values
        unique_labels, counts = np.unique(labels, return_counts=True)

        proportions = counts / len(neighbors)
        simpson = np.sum(proportions**2)
        lisi = 1 / simpson if simpson > 0 else 0
        lisi_scores.append(lisi)

    return np.median(lisi_scores)


def mixing_score(
    rna_inference_outputs_mean,
    protein_inference_outputs_mean,
    adata_rna_subset,
    adata_prot_subset,
    index_rna=None,
    index_prot=None,
    plot_flag=False,
):
    if index_rna is None:
        index_rna = np.arange(len(rna_inference_outputs_mean))
    if index_prot is None:
        index_prot = np.arange(len(protein_inference_outputs_mean))
    if isinstance(rna_inference_outputs_mean, torch.Tensor):
        rna_latent = rna_inference_outputs_mean.clone().detach().cpu().numpy()
        prot_latent = protein_inference_outputs_mean.clone().detach().cpu().numpy()
    else:
        rna_latent = rna_inference_outputs_mean
        prot_latent = protein_inference_outputs_mean
    combined_latent = ad.concat(
        [AnnData(rna_latent), AnnData(prot_latent)],
        join="outer",
        label="modality",
        keys=["RNA", "Protein"],
    )
    combined_major_cell_types = pd.concat(
        (
            adata_rna_subset[index_rna].obs["major_cell_types"],
            adata_prot_subset[index_prot].obs["major_cell_types"],
        ),
        join="outer",
    )
    combined_latent.obs["major_cell_types"] = combined_major_cell_types.values
    sc.pp.pca(combined_latent)
    sc.pp.neighbors(combined_latent, use_rep="X")
    iLISI = calculate_iLISI(combined_latent, "modality", plot_flag=plot_flag)
    cLISI = calculate_cLISI(combined_latent, "major_cell_types", plot_flag=plot_flag)
    return {"iLISI": iLISI, "cLISI": cLISI}


def calculate_iLISI(
    adata,
    batch_key="batch",
    neighbors_key="neighbors",
    plot_flag=False,
    use_subsample=True,
    global_step=None,
):
    """
    Calculate integration Local Inverse Simpson's Index (LISI) using precomputed neighbors.

    The iLISI score measures how well different batches are mixed in the embedding space.
    Higher scores indicate better batch mixing, with a minimum value of 1
    (all neighbors same batch) and maximum of k+1 (all neighbors different batches),
    where k is the number of neighbors used.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with precomputed neighbors
    batch_key : str, default='batch'
        Column in adata.obs containing batch labels
    neighbors_key : str, default='neighbors'
        Key where neighbor information is stored in adata.uns
    use_subsample: bool:
        for ploting part of the batch

    Returns
    -------
    float
        Median iLISI score across all cells. Higher values indicate better
        batch mixing in the embedding space.
    """

    if neighbors_key not in adata.uns:
        raise ValueError(f"Run sc.pp.neighbors with key='{neighbors_key}' first")

    connectivities = adata.obsp[f"connectivities"]
    n_cells = adata.n_obs
    if False:  # ignore for now
        if use_subsample:
            sample_size = min(300, n_cells)  # Use smaller of 300 or total cells
            subset_indices = np.random.choice(n_cells, sample_size, replace=False)
            subset_connectivities = connectivities[subset_indices][:, subset_indices]
        else:
            subset_connectivities = connectivities
        plt.figure(figsize=(10, 8))
        plt.title(
            "neighbors, first half are RNA cells \nthe second half, protein cells (subset of 300)"
        )
        sns.heatmap(subset_connectivities.todense())
        mid_point = subset_connectivities.shape[1] // 2
        plt.axvline(x=mid_point, color="red", linestyle="--", linewidth=2)
        plt.axhline(y=mid_point, color="red", linestyle="--", linewidth=2)
        if global_step is not None:
            mlflow.log_figure(plt.gcf(), f"step_{global_step}_neighbor_heatmap.png")
        else:
            mlflow.log_figure(plt.gcf(), "neighbor_heatmap.png")
        plt.show()
    lisi_scores = []
    for i in range(n_cells):
        neighbors = connectivities[i].indices
        neighbors = np.append(neighbors, i)

        batches = adata.obs[batch_key].iloc[neighbors].values
        unique_batches, counts = np.unique(batches, return_counts=True)

        proportions = counts / len(neighbors)
        simpson = np.sum(proportions**2)
        lisi = 1 / simpson if simpson > 0 else 0
        lisi_scores.append(lisi)

    return np.median(lisi_scores)


def plot_merged_pca_tsne(
    adata1, adata2, unmatched_rna_indices, unmatched_prot_indices, pca_components=5
):
    """
    1) Combines Protein + RNA 'archetype_vec' data.
    2) Dynamically adjusts PCA components if needed.
    3) Plots the first two principal components in one figure.
    4) Applies TSNE to the PCA output, and generates TWO separate figures:
       - Figure A: Colors by modality (Protein vs. RNA).
       - Figure B: Colors by matched vs. unmatched status.

    Parameters
    ----------
    adata1 : anndata.AnnData
        Protein subset with 'archetype_vec' in obsm.
    adata2 : anndata.AnnData
        RNA subset with 'archetype_vec' in obsm.
    unmatched_prot_indices : list or np.ndarray
        Indices of unmatched protein cells.
    unmatched_rna_indices : list or np.ndarray
        Indices of unmatched RNA cells.
    pca_components : int
        Requested number of principal components before TSNE.
    """

    # -------------------- MERGE DATA --------------------
    rna_data = adata1.obsm["archetype_vec"]
    prot_data = adata2.obsm["archetype_vec"]
    merged_data = np.vstack([prot_data, rna_data])

    max_valid_components = min(merged_data.shape[0], merged_data.shape[1])
    final_pca_components = min(pca_components, max_valid_components)

    # -------------------- PCA --------------------
    pca_model = PCA(n_components=final_pca_components)
    pca_result = pca_model.fit_transform(merged_data)

    # Split PCA results back
    prot_pca = pca_result[: len(prot_data)]
    rna_pca = pca_result[len(prot_data) :]

    # -------------------- PCA PLOT --------------------
    plt.figure(figsize=(6, 5))

    # Matched vs. unmatched (Protein)
    plt.scatter(prot_pca[:, 0], prot_pca[:, 1], c="blue", s=5, label="Matched Protein")
    plt.scatter(
        prot_pca[unmatched_prot_indices, 0],
        prot_pca[unmatched_prot_indices, 1],
        c="black",
        marker="x",
        s=10,
        label="Unmatched Protein",
        alpha=0.5,
    )

    # Matched vs. unmatched (RNA)
    plt.scatter(rna_pca[:, 0], rna_pca[:, 1], c="red", s=5, label="Matched RNA", alpha=0.5)
    plt.scatter(
        rna_pca[unmatched_rna_indices, 0],
        rna_pca[unmatched_rna_indices, 1],
        c="green",
        marker="D",
        s=10,
        label="Unmatched RNA",
        alpha=0.5,
    )

    plt.title("PCA (First Two PCs)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.show()

    # -------------------- TSNE --------------------
    tsne_model = TSNE(n_components=2)
    tsne_result = tsne_model.fit_transform(pca_result)

    prot_tsne = tsne_result[: len(prot_data)]
    rna_tsne = tsne_result[len(prot_data) :]

    # -------------------- FIGURE A (Modality Colors) --------------------
    plt.figure(figsize=(6, 5))

    plt.scatter(prot_tsne[:, 0], prot_tsne[:, 1], c="blue", s=5, label="Protein")
    plt.scatter(rna_tsne[:, 0], rna_tsne[:, 1], c="red", s=5, label="RNA", alpha=0.5)

    plt.title("TSNE by Modality (Protein vs. RNA)")
    plt.xlabel("TSNE 1")
    plt.ylabel("TSNE 2")
    plt.legend()
    plt.show()

    # -------------------- FIGURE B (Matched vs. Unmatched) --------------------
    plt.figure(figsize=(6, 5))

    # Matched Protein
    plt.scatter(prot_tsne[:, 0], prot_tsne[:, 1], c="red", s=5, label="Matched Protein")
    # Unmatched Protein
    plt.scatter(
        prot_tsne[unmatched_prot_indices, 0],
        prot_tsne[unmatched_prot_indices, 1],
        c="green",
        marker="x",
        s=10,
        label="Unmatched Protein",
        alpha=0.5,
    )

    # Matched RNA
    plt.scatter(rna_tsne[:, 0], rna_tsne[:, 1], c="red", s=5, label="Matched RNA", alpha=0.5)
    # Unmatched RNA
    plt.scatter(
        rna_tsne[unmatched_rna_indices, 0],
        rna_tsne[unmatched_rna_indices, 1],
        c="black",
        marker="x",
        s=10,
        label="Unmatched RNA",
        alpha=0.5,
    )

    plt.title("TSNE by Match Status")
    plt.xlabel("TSNE 1")
    plt.ylabel("TSNE 2")
    plt.legend()
    plt.show()


def match_datasets(
    adata_rna: AnnData,
    adata_prot: AnnData,
    threshold: Union[float, str] = "auto",
    obs_key1: str = "archetype_vec",
    obs_key2="archetype_vec",
    plot_flag=False,
    batch_size=1000,
):
    # Compute pairwise distance matrix in batches to prevent memory issues
    print("Computing distance matrix in batches...")
    n1, n2 = len(adata_rna), len(adata_prot)
    dist_matrix = np.zeros((n1, n2))

    for i in range(0, n1, batch_size):
        end_idx = min(i + batch_size, n1)
        batch_dist = scipy.spatial.distance.cdist(
            adata_rna[i:end_idx].obsm[obs_key1], adata_prot.obsm[obs_key2], metric="cosine"
        )
        dist_matrix[i:end_idx] = batch_dist
        print(f"Processed batch {i//batch_size + 1}/{(n1-1)//batch_size + 1}", end="\r")

    print("\nDistance matrix computation complete.")
    matching_distance_before = np.diag(dist_matrix).mean()

    # Rest of the function continues as before
    if plot_flag:
        # subsample the data for plotting
        adata_rna_sub = adata_rna[np.random.choice(len(adata_rna), 1000, replace=False)]
        adata_prot_sub = adata_prot[np.random.choice(len(adata_prot), 1000, replace=False)]
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        sns.heatmap(adata_rna_sub.obsm[obs_key1], cmap="viridis")
        plt.title("RNA Archetype Vectors")
        plt.subplot(122)
        sns.heatmap(adata_prot_sub.obsm[obs_key2], cmap="viridis")
        plt.title("Protein Archetype Vectors")
        plt.suptitle("Initial Data")
        plt.tight_layout()
        plt.show()

    # Determine which dataset is smaller
    smaller_adata = adata_rna if n1 <= n2 else adata_prot
    larger_adata = adata_prot if n1 <= n2 else adata_rna
    smaller_is_first = n1 <= n2
    smaller_name = "RNA" if smaller_is_first else "Protein"
    larger_name = "Protein" if smaller_is_first else "RNA"
    # Get the size of the smaller dataset
    n_smaller = len(smaller_adata)
    n_larger = len(larger_adata)

    # Reshape dist_matrix if needed to ensure it's shaped as [smaller, larger]
    if not smaller_is_first:
        dist_matrix = dist_matrix.T

    if plot_flag:
        plt.figure(figsize=(10, 8))
        sns.heatmap(dist_matrix[:1000, :1000], cmap="viridis")
        plt.title("Initial Distance Matrix")
        plt.show()

    # First evaluate each cell's worst match - process in batches for very large matrices
    print("Computing best matches...")
    if n_smaller > 10000:
        smaller_best_matches = np.zeros(n_smaller)
        for i in range(0, n_smaller, batch_size):
            end_idx = min(i + batch_size, n_smaller)
            smaller_best_matches[i:end_idx] = np.min(dist_matrix[i:end_idx], axis=1)
    else:
        smaller_best_matches = np.min(dist_matrix, axis=1)

    if n_larger > 10000:
        larger_best_matches = np.zeros(n_larger)
        for i in range(0, n_larger, batch_size):
            end_idx = min(i + batch_size, n_larger)
            larger_best_matches[i:end_idx] = np.min(dist_matrix[:, i:end_idx], axis=0)
    else:
        larger_best_matches = np.min(dist_matrix, axis=0)

    # Continue with the rest of the function
    if plot_flag:
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.plot(sorted(smaller_best_matches))
        plt.title(f"sorted {smaller_name} Dataset")
        plt.subplot(122)
        plt.plot(sorted(larger_best_matches))
        plt.title(f"sorted {larger_name} Dataset")
        plt.show()

    # Print diagnostic information
    print("\nDistance Distribution Statistics:")
    print(f"cells worst match percentiles:")
    for p in [0, 25, 50, 75, 100]:
        print(f"{p}th percentile: {np.percentile(smaller_best_matches, p):.3f}")
    print(f"\nProtein cells worst match percentiles:")
    for p in [0, 25, 50, 75, 100]:
        print(f"{p}th percentile: {np.percentile(larger_best_matches, p):.3f}")

    # If threshold is auto, compute it from the distribution of worst matches
    if threshold == "auto":
        all_worst_matches = np.concatenate([smaller_best_matches, larger_best_matches])
        threshold = np.percentile(all_worst_matches, 75)  # Use 75th percentile as threshold
        print(f"\nSetting auto threshold to 75th percentile: {threshold:.3f}")

    print(f"\nUsing threshold: {threshold:.3f}")

    # Create masks for good matches in each dataset
    good_smaller_mask = smaller_best_matches <= threshold
    good_larger_mask = larger_best_matches <= threshold

    # Print how many cells pass the threshold
    print(f"\nCells passing threshold:")
    print(f"Smaller dataset cells: {np.sum(good_smaller_mask)} / {len(good_smaller_mask)}")
    print(f"Larger dataset cells: {np.sum(good_larger_mask)} / {len(good_larger_mask)}")

    # Get indices of cells that have good matches
    good_smaller_indices = np.where(good_smaller_mask)[0]
    good_larger_indices = np.where(good_larger_mask)[0]
    dist_matrix[good_smaller_indices, :] = np.inf
    # for now return the good matches without running the hungarian algorithm
    return smaller_adata[good_smaller_indices].copy(), larger_adata[good_larger_indices].copy()
    # Create filtered distance matrix with only good matches
    filtered_dist_matrix = dist_matrix[good_smaller_mask][:, good_larger_mask]

    if len(filtered_dist_matrix) == 0:
        print(
            f"No good matches found with threshold {threshold:.3f}. Try increasing the threshold."
        )
        return smaller_adata[[]].copy(), larger_adata[[]].copy()

    # Run Hungarian algorithm on filtered distance matrix
    row_ind, col_ind = linear_sum_assignment(filtered_dist_matrix)

    # Map back to original indices
    final_smaller_indices = good_smaller_indices[row_ind]
    final_larger_indices = good_larger_indices[col_ind]

    # Get match quality for reporting
    match_quality = dist_matrix[final_smaller_indices, final_larger_indices]

    if plot_flag:
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        sns.heatmap(filtered_dist_matrix, cmap="viridis")
        plt.title("Filtered Distance Matrix\n(After removing bad matches)")
        plt.subplot(122)
        plt.hist(match_quality, bins=50, alpha=0.5, color="green", label="Good matches")
        plt.axvline(x=threshold, color="red", linestyle="--", label=f"Threshold ({threshold:.3f})")
        plt.title("all Matches Quality Distribution")
        plt.xlabel("Distance")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Create DataFrame for sorting smaller dataset
    smaller_df = pd.DataFrame(
        {
            "index": final_smaller_indices,
            "major_cell_types": smaller_adata.obs["major_cell_types"].iloc[final_smaller_indices],
            "cell_types": smaller_adata.obs["cell_types"].iloc[final_smaller_indices],
        }
    )

    # Sort smaller dataset
    smaller_df = smaller_df.sort_values(["major_cell_types", "cell_types"])
    sorted_smaller_indices = smaller_df["index"].values

    # Get corresponding larger indices in the same order as sorted smaller indices
    sorted_matched_larger_indices = final_larger_indices[
        np.array([np.where(final_smaller_indices == idx)[0][0] for idx in sorted_smaller_indices])
    ]

    if plot_flag:
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        sns.heatmap(
            dist_matrix[sorted_smaller_indices][:, sorted_matched_larger_indices], cmap="viridis"
        )
        plt.title("Distance Matrix (Sorted by Cell Type)")
        plt.subplot(122)
        plt.hist(match_quality, bins=50, alpha=0.5, color="green", label="Good matches")
        plt.axvline(x=threshold, color="red", linestyle="--", label=f"Threshold ({threshold:.3f})")
        plt.title("Match Quality Distribution (After Sorting)")
        plt.xlabel("Distance")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Find which cells in the larger dataset were not matched
    all_larger_indices = np.arange(n_larger)
    unused_larger_indices = np.setdiff1d(all_larger_indices, sorted_matched_larger_indices)
    # Get unused indices that were below threshold in original matching
    unused_larger_indices_below_threshold = unused_larger_indices[
        pd.Series(unused_larger_indices).isin(np.where(good_larger_mask)[0])
    ]

    # Sort unmatched cells by cell types
    if len(unused_larger_indices_below_threshold) > 0:
        larger_df = pd.DataFrame(
            {
                "index": unused_larger_indices_below_threshold,
                "major_cell_types": larger_adata.obs["major_cell_types"].iloc[
                    unused_larger_indices_below_threshold
                ],
                "cell_types": larger_adata.obs["cell_types"].iloc[
                    unused_larger_indices_below_threshold
                ],
            }
        )
        larger_df = larger_df.sort_values(["major_cell_types", "cell_types"])
        sorted_unused_larger_indices = larger_df["index"].values

        # Combine matched and unmatched indices
        final_larger_indices = np.concatenate(
            [sorted_matched_larger_indices, sorted_unused_larger_indices]
        )
    else:
        final_larger_indices = sorted_matched_larger_indices

    # Get the final indices based on whether adata_rna or adata_prot was the smaller dataset
    if smaller_is_first:
        final_adata_rna_indices = sorted_smaller_indices
        final_adata_prot_indices = final_larger_indices
    else:
        final_adata_rna_indices = final_larger_indices
        final_adata_prot_indices = sorted_smaller_indices

    if plot_flag:
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        sns.heatmap(adata_rna[final_adata_rna_indices].obsm[obs_key1].values, cmap="viridis")
        plt.title("Final RNA Archetype Vectors")
        plt.subplot(122)
        sns.heatmap(adata_prot[final_adata_prot_indices].obsm[obs_key2].values, cmap="viridis")
        plt.title("Final Protein Archetype Vectors")
        plt.suptitle("Final Matched Data")
        plt.tight_layout()
        plt.show()

        min_len = min(len(final_adata_rna_indices), len(final_adata_prot_indices))
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        sns.heatmap(
            adata_rna[final_adata_rna_indices[:min_len]].obsm[obs_key1].values, cmap="viridis"
        )
        plt.title("Final RNA Archetype Vectors")
        plt.subplot(122)
        sns.heatmap(
            adata_prot[final_adata_prot_indices[:min_len]].obsm[obs_key2].values, cmap="viridis"
        )
        plt.title("Final Protein Archetype Vectors")
        plt.suptitle("Final Matched Data min length")
        plt.tight_layout()
        plt.show()

    # Calculate statistics
    stats = {
        "total_smaller": n_smaller,
        "total_larger": n_larger,
        "matched_cells": len(good_smaller_indices),
        "removed_smaller_modality": n_smaller - len(good_smaller_indices),
        "removed_larger_modality": n_larger - len(good_larger_indices),
        "mean_distance": match_quality.mean(),
        "median_distance": np.median(match_quality),
        "std_distance": np.std(match_quality),
        "min_distance": np.min(match_quality),
        "max_distance": np.max(match_quality),
    }

    # Print comprehensive report

    # Get the correct numbers for each dataset
    if smaller_is_first:
        smaller_kept = stats["matched_cells"]
        smaller_total = stats["total_smaller"]
        smaller_removed = stats["removed_smaller_modality"]
        larger_kept = len(good_larger_indices)
        larger_total = stats["total_larger"]
        larger_removed = larger_total - len(good_larger_indices)
    else:
        smaller_kept = stats["matched_cells"]
        smaller_total = stats["total_smaller"]
        smaller_removed = stats["removed_smaller_modality"]
        larger_kept = len(good_smaller_indices)
        larger_total = stats["total_larger"]
        larger_removed = larger_total - len(good_smaller_indices)

    print(
        "Matching Report:\n"
        f"- Kept {smaller_kept}/{smaller_total} {smaller_name} cells "
        f"({smaller_removed} removed due to poor matching)\n"
        f"- Kept {larger_kept}/{larger_total} {larger_name} cells "
        f"({larger_removed} removed with no good matches)\n"
        f"- Average match distance before matching: {matching_distance_before:.3f}\n"
        f"- Average match distance after matching: {stats['mean_distance']:.3f}\n"
        f"- Median match distance: {stats['median_distance']:.3f}\n"
        f"- Std match distance: {stats['std_distance']:.3f}\n"
        f"- Min/Max match distance: {stats['min_distance']:.3f}/{stats['max_distance']:.3f}"
    )

    if plot_flag:
        # Plot remaining visualizations if needed
        if smaller_is_first:
            unmatched_smaller = np.setdiff1d(np.arange(n_smaller), good_smaller_indices)
            unmatched_rna_indices = unmatched_smaller
            unmatched_prot_indices = []
        else:
            unmatched_smaller = np.setdiff1d(np.arange(n_smaller), good_larger_indices)
            unmatched_rna_indices = []
            unmatched_prot_indices = unmatched_smaller

        plot_merged_pca_tsne(
            adata_rna,
            adata_prot,
            unmatched_rna_indices=unmatched_rna_indices,
            unmatched_prot_indices=unmatched_prot_indices,
            pca_components=5,
        )

    # Create final matched AnnData objects
    matched_smaller = smaller_adata[sorted_smaller_indices].copy()
    matched_larger = larger_adata[final_larger_indices].copy()

    # Set metadata to indicate the datasets are ordered
    matched_smaller.uns["ordered_matching_cells"] = True
    matched_larger.uns["ordered_matching_cells"] = True
    matched_smaller.obs["index_col"] = np.arange(matched_smaller.shape[0])
    matched_larger.obs["index_col"] = np.arange(matched_larger.shape[0])

    # Verify alignment of matched cells
    n_matched = len(sorted_smaller_indices)
    if not (
        matched_smaller.obs["cell_types"].astype(str).values[:n_matched]
        == matched_larger.obs["cell_types"].astype(str).values[:n_matched]
    ).all():
        print("Warning: Cell types of matched cells are not aligned!")
        # Print some diagnostic information
        print(matched_larger.obs["cell_types"].astype(str).values[:5])
        print("\nUnique cell types in dataset 1:")
        print(np.unique(matched_smaller.obs["cell_types"].astype(str).values))
        print("\nUnique cell types in dataset 2:")
        print(np.unique(matched_larger.obs["cell_types"].astype(str).values))

    # Return in the correct order based on input
    if smaller_is_first:
        return matched_smaller, matched_larger
    else:
        return matched_larger, matched_smaller


def get_latest_file(folder, prefix):
    files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(".h5ad")]
    if not files:
        return None

    # Try to sort by timestamp if present, otherwise use file modification time
    def get_sort_key(x):
        timestamp_match = re.search(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", x)
        if timestamp_match:
            return timestamp_match.group()
        # Convert modification time to timestamp string format for consistent comparison
        mtime = os.path.getmtime(os.path.join(folder, x))
        return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(mtime))

    files.sort(key=get_sort_key, reverse=True)
    return os.path.join(folder, files[0])


def nnls_omp(basis_matrix, target_vector, tol=1e-4):
    omp = OrthogonalMatchingPursuit(tol=tol, fit_intercept=False)
    omp.fit(basis_matrix.T, target_vector)
    weights = omp.coef_
    weights = np.maximum(0, weights)  # Enforce non-negativity
    return weights


def get_cell_representations_as_archetypes_ols(count_matrix, archetype_matrix):
    """
    Compute archetype weights for each cell using Ordinary Least Squares (OLS).

    Parameters:
    -----------
    count_matrix : np.ndarray
        Matrix of cells in reduced-dimensional space (e.g., PCA),
        shape (n_cells, n_features).
    archetype_matrix : np.ndarray
        Matrix of archetypes,
        shape (n_archetypes, n_features).

    Returns:
    --------
    weights : np.ndarray
        Matrix of archetype weights for each cell,
        shape (n_cells, n_archetypes).
    """
    n_cells = count_matrix.shape[0]
    n_archetypes = archetype_matrix.shape[0]
    weights = np.zeros((n_cells, n_archetypes))

    # Transpose the archetype matrix
    A_T = archetype_matrix.T  # Shape: (n_features, n_archetypes)

    # For each cell, solve the least squares problem
    for i in range(n_cells):
        x = count_matrix[i]
        # Solve for w in A_T w = x
        w, residuals, rank, s = np.linalg.lstsq(A_T, x, rcond=None)
        weights[i] = w

    return weights


def get_cell_representations_as_archetypes_omp(count_matrix, archetype_matrix, tol=1e-4):
    # Preprocess archetype matrix

    n_cells = count_matrix.shape[0]
    n_archetypes = archetype_matrix.shape[0]
    weights = np.zeros((n_cells, n_archetypes))

    for i in range(n_cells):
        weights[i] = nnls_omp(archetype_matrix, count_matrix[i], tol=tol)

    row_sums = weights.sum(axis=1, keepdims=True)
    weights[row_sums == 0] = 1.0 / n_archetypes  # Assign uniform weights to zero rows

    weights /= weights.sum(axis=1, keepdims=True)

    return weights


def nnls_cvxpy(A, b):
    """
    Solve the NNLS problem using cvxpy.
    """
    n_features = A.shape[1]
    x = cp.Variable(n_features, nonneg=True)
    objective = cp.Minimize(cp.sum_squares(A @ x - b))
    problem = cp.Problem(objective)
    problem.solve()
    return x.value


def reorder_rows_to_maximize_diagonal(matrix):
    """
    Reorders rows of a matrix to maximize diagonal dominance by placing the highest values
    in the closest positions to the diagonal.

    Parameters:
    -----------
    matrix : np.ndarray
        An m x n matrix.

    Returns:
    --------
    reordered_matrix : np.ndarray
        The input matrix with reordered rows.
    row_order : list
        The indices of the rows in their new order.
    """
    # Track available rows
    original = None
    if isinstance(matrix, pd.DataFrame):
        original = copy.deepcopy(matrix)
        matrix = matrix.values
    available_rows = list(range(matrix.shape[0]))
    row_order = []

    # Reorder rows iteratively
    for col in range(matrix.shape[1]):
        if not available_rows:
            break

        # Find the row with the maximum value for the current column
        best_row = max(available_rows, key=lambda r: matrix[r, col])
        row_order.append(best_row)
        available_rows.remove(best_row)

    # Handle leftover rows if there are more rows than columns
    row_order += available_rows

    # Reorder the matrix
    reordered_matrix = matrix[row_order]
    if original is not None:
        reordered_matrix = pd.DataFrame(
            reordered_matrix, index=original.index, columns=original.columns
        )
    return reordered_matrix, row_order


def get_cell_representations_as_archetypes_cvxpy(count_matrix, archetype_matrix, solver=cp.ECOS):
    """
    Compute archetype weights for each cell using constrained optimization (non-negative least squares).

    Parameters:
    -----------
    count_matrix : np.ndarray
        Matrix of cells in reduced-dimensional space,
        shape (n_cells, n_features).
    archetype_matrix : np.ndarray
        Matrix of archetypes,
        shape (n_archetypes, n_features).
    solver : cvxpy solver, optional
        Solver to use for optimization. Default is cp.ECOS.

    Returns:
    --------
    weights : np.ndarray
        Non-negative weights for each cell,
        shape (n_cells, n_archetypes).
    """
    n_cells = count_matrix.shape[0]
    n_archetypes = archetype_matrix.shape[0]
    weights = np.zeros((n_cells, n_archetypes))

    A_T = archetype_matrix.T  # Shape: (n_features, n_archetypes)
    assert not np.isnan(count_matrix).any(), "count_matrix contains NaNs"
    assert not np.isinf(count_matrix).any(), "count_matrix contains infinities"
    assert not np.isnan(archetype_matrix).any(), "archetype_matrix contains NaNs"
    assert not np.isinf(archetype_matrix).any(), "archetype_matrix contains infinities"
    # make sure all values are positive
    # if not (count_matrix >= 0).all():
    #     count_matrix = count_matrix - count_matrix.min()

    for i in range(n_cells):
        x = count_matrix[i]
        w = cp.Variable(n_archetypes)
        objective = cp.Minimize(cp.sum_squares(A_T @ w - x))
        # constraints = [w >= 0] # legacy=
        constraints = [
            w >= 0,
            cp.sum(w) == 1,
        ]  # this make sure that each data points is a convex combination of the archetypes

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=solver)
        except cp.SolverError:
            problem.solve(solver=cp.SCS)  # Try SCS if the primary solver fails

        weights[i] = w.value

    return weights


def get_cell_representations_as_archetypes(count_matrix, archetype_matrix):
    """
    Compute archetype weights for each cell using cvxpy.
    """
    n_cells = count_matrix.shape[0]
    n_archetypes = archetype_matrix.shape[0]
    weights = np.zeros((n_cells, n_archetypes))
    for i in range(n_cells):
        weights[i], _ = nnls(archetype_matrix.T, count_matrix[i])
    weights /= weights.sum(axis=1, keepdims=True)  # Normalize rows
    return weights


def preprocess_rna_maxfuse(adata_1):
    sc.pp.normalize_total(adata_1)
    sc.pp.log1p(adata_1)
    sc.pp.highly_variable_genes(adata_1, n_top_genes=5000)
    # only retain highly variable genes
    adata_1 = adata_1[:, adata_1.var.highly_variable].copy()
    sc.pp.scale(adata_1)
    return adata_1


def preprocess_rna(adata_rna, min_genes=100, min_cells=20, n_top_genes=2000):
    """
    Preprocess RNA data for downstream analysis with PCA and variance tracking.
    """
    sc.pp.filter_cells(adata_rna, min_genes=min_genes)
    sc.pp.filter_genes(adata_rna, min_cells=min_cells)
    # Identify highly variable genes (for further analysis, could narrow down)

    # maxfuse addition
    sc.pp.normalize_total(adata_rna)
    if not adata_rna.X.max() < 100:
        sc.pp.log1p(adata_rna)
    # maxfuse end addition
    sc.pp.highly_variable_genes(adata_rna, n_top_genes=n_top_genes, flavor="seurat_v3")
    # sc.pp.scale(adata_rna)
    adata_rna = adata_rna[:, adata_rna.var["highly_variable"]]
    # maxfuse scale additoin
    sc.pp.scale(adata_rna)
    print(f"Selected {adata_rna.shape[1]} highly variable genes.")
    # PCA after selecting highly variable genes
    sc.pp.pca(adata_rna)
    print(
        f"Variance ratio after highly variable gene selection PCA (10 PCs): {adata_rna.uns['pca']['variance_ratio'][:10].sum():.4f}"
    )

    # Annotate mitochondrial, ribosomal, and hemoglobin genes
    adata_rna.var["mt"] = adata_rna.var_names.str.startswith("Mt-")  # Mouse data
    adata_rna.var["ribo"] = adata_rna.var_names.str.startswith(("RPS", "RPL"))
    adata_rna.var["hb"] = adata_rna.var_names.str.contains("^HB[^(P)]", regex=True)
    # Filter cells and genes (different sample)
    # Calculate QC metrics
    # todo cause crash in archetype generation real for some resean?!!?
    # sc.pp.calculate_qc_metrics(adata_rna, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)

    # Add raw counts to layers for future reference
    adata_rna.layers["counts"] = adata_rna.X.copy()

    # Log-transform the data
    # only log transform the data if it is not already log transformed

    sc.pp.pca(adata_rna)
    print(
        f"Variance ratio after log transformation PCA (10 PCs): {adata_rna.uns['pca']['variance_ratio'][:10].sum():.4f}"
    )

    # Normalize total counts
    # sc.pp.normalize_total(adata_rna, target_sum=5e3)
    sc.pp.pca(adata_rna)
    print(
        f"Variance ratio after normalization PCA (10 PCs): {adata_rna.uns['pca']['variance_ratio'][:10].sum():.4f}"
    )

    # Scale the data
    # sc.pp.scale(adata_rna, max_value=10)
    # sc.pp.pca(adata_rna)
    # print(f"Variance ratio after scaling PCA: {adata_rna.uns['pca']['variance_ratio'][:10].sum():.4f}")

    return adata_rna


def preprocess_protein(adata_prot):
    sc.pp.filter_cells(adata_prot, min_genes=30)
    sc.pp.filter_genes(adata_prot, min_cells=50)

    sc.pp.pca(adata_prot)
    print(
        f"Variance ratio after PCA (10 PCs): {adata_prot.uns['pca']['variance_ratio'][:10].sum():.4f}"
    )
    print()
    sc.pp.normalize_total(adata_prot)
    sc.pp.pca(adata_prot)
    print(
        f"Variance ratio after normalization PCA (10 PCs): {adata_prot.uns['pca']['variance_ratio'][:10].sum():.4f}"
    )
    print()
    sc.pp.log1p(adata_prot)
    sc.pp.pca(adata_prot)
    print(
        f"Variance ratio after log transformation PCA (10 PCs): {adata_prot.uns['pca']['variance_ratio'][:10].sum():.4f}"
    )
    # matrix = adata_prot.X
    # np.log1p(matrix / np.exp(np.mean(np.log1p(matrix + 1), axis=1, keepdims=True)))
    # adata_prot.X = matrix
    # sc.pp.scale(adata_prot, max_value=10)

    return adata_prot


def preprocess_protein_new_bad(adata_prot):
    sc.pp.filter_cells(adata_prot, min_genes=25)
    sc.pp.filter_genes(adata_prot, min_cells=50)
    sc.pp.pca(adata_prot)
    print(
        f"Variance ratio after PCA (10 PCs): {adata_prot.uns['pca']['variance_ratio'][:10].sum():.4f}"
    )
    adata_prot.X = adata_prot.X.toarray() if sp.issparse(adata_prot.X) else adata_prot.X
    for marker in adata_prot.var_names:
        mask = ~np.isnan(
            adata_prot[:, marker].X.todense()
            if sp.issparse(adata_prot.X)
            else adata_prot[:, marker].X
        )
        mean_val = np.mean(adata_prot[:, marker].X[mask])
        std_val = np.std(adata_prot[:, marker].X[mask])
        adata_prot[:, marker].X = (adata_prot[:, marker].X - mean_val) / std_val
    sc.pp.pca(adata_prot)
    print(
        f"Variance ratio after normalization PCA (10 PCs): {adata_prot.uns['pca']['variance_ratio'][:10].sum():.4f}"
    )
    sc.pp.scale(adata_prot, zero_center=True, max_value=10.0)
    sc.pp.pca(adata_prot)
    print(
        f"Variance ratio after scaling PCA (10 PCs): {adata_prot.uns['pca']['variance_ratio'][:10].sum():.4f}"
    )
    # sc.pp.log1p(adata_prot)
    # sc.pp.pca(adata_prot)
    # print(
    #     f"Variance ratio after log transformation PCA (10 PCs): {adata_prot.uns['pca']['variance_ratio'][:10].sum():.4f}")
    # matrix = adata_prot.X
    # np.log1p(matrix / np.exp(np.mean(np.log1p(matrix + 1), axis=1, keepdims=True)))
    # adata_prot.X = matrix
    # sc.pp.scale(adata_prot, max_value=10)

    return adata_prot


def select_gene_likelihood(adata):
    """
    Determines the appropriate gene likelihood distribution for the SCVI model
    based on the properties of the input AnnData object.

    Parameters:
    - adata: AnnData object containing single-cell RNA-seq data.

    Returns:
    - str: Selected gene likelihood distribution ("nb", "zinb", "poisson").
    """

    # Check for zero-inflation by counting the proportion of zero values in the data
    zero_proportion = (adata.X == 0).sum() / adata.X.size

    # Select likelihood based on zero inflation and count properties
    if zero_proportion > 0.4:
        gene_likelihood = "zinb"  # Zero-Inflated Negative Binomial for high zero-inflation
    elif adata.X.mean() < 5:
        gene_likelihood = "poisson"  # Poisson for low-count data
    else:
        gene_likelihood = "nb"  # Negative Binomial for typical gene expression

    print(f"Selected gene likelihood: {gene_likelihood}")
    return gene_likelihood


def add_spatial_data_to_prot(
    adata_prot_subset, major_to_minor_dict, plot_flag=False
) -> (AnnData, list, list):
    horizontal_splits = [0, 500, 1000]
    vertical_splits = [0, 333, 666, 1000]

    # Create regions as a list of coordinate grids
    regions = [
        list(
            product(
                range(horizontal_splits[i], horizontal_splits[i + 1]),
                range(vertical_splits[j], vertical_splits[j + 1]),
            )
        )
        for i in range(len(horizontal_splits) - 1)
        for j in range(len(vertical_splits) - 1)
    ]

    # Create a board for visualization (optional)
    board = np.zeros((1000, 1000))
    for idx, region in enumerate(regions):
        coords = np.array(region)
        board[coords[:, 0], coords[:, 1]] = idx + 1

    if plot_flag:
        plt.imshow(board)
        plt.title("CNs")
        plt.colorbar()
        plt.show()

    # Assign random initial coordinates to all cells
    adata_prot_subset.obs["X"] = np.random.randint(0, 1000, adata_prot_subset.n_obs)
    adata_prot_subset.obs["Y"] = np.random.randint(0, 1000, adata_prot_subset.n_obs)

    # Create a dictionary mapping tuples of (cell_type_1, cell_type_2, cell_type_3) to a region index
    minor_to_region_dict = {}
    major_B = major_to_minor_dict.get("B cells", [])
    major_CD4 = major_to_minor_dict.get("CD4 T", [])
    major_CD8 = major_to_minor_dict.get("CD8 T", [])

    for i, (cell_type_1, cell_type_2, cell_type_3) in enumerate(
        zip_longest(major_B, major_CD4, major_CD8)
    ):
        # If any of these are None, they won't match any cell, but we can still store them
        minor_to_region_dict[(cell_type_1, cell_type_2, cell_type_3)] = i

    # Now place the cells of each subgroup into their assigned region
    for (cell_type_1, cell_type_2, cell_type_3), region_index in minor_to_region_dict.items():
        coords = np.array(regions[region_index])

        # Update positions for each cell type if not None
        if cell_type_1 is not None:
            cell_indices_1 = adata_prot_subset.obs["cell_types"] == cell_type_1
            if cell_indices_1.sum() > 0:
                adata_prot_subset.obs.loc[cell_indices_1, "X"] = np.random.choice(
                    coords[:, 0], cell_indices_1.sum()
                )
                adata_prot_subset.obs.loc[cell_indices_1, "Y"] = np.random.choice(
                    coords[:, 1], cell_indices_1.sum()
                )

        if cell_type_2 is not None:
            cell_indices_2 = adata_prot_subset.obs["cell_types"] == cell_type_2
            if cell_indices_2.sum() > 0:
                adata_prot_subset.obs.loc[cell_indices_2, "X"] = np.random.choice(
                    coords[:, 0], cell_indices_2.sum()
                )
                adata_prot_subset.obs.loc[cell_indices_2, "Y"] = np.random.choice(
                    coords[:, 1], cell_indices_2.sum()
                )

        if cell_type_3 is not None:
            cell_indices_3 = adata_prot_subset.obs["cell_types"] == cell_type_3
            if cell_indices_3.sum() > 0:
                adata_prot_subset.obs.loc[cell_indices_3, "X"] = np.random.choice(
                    coords[:, 0], cell_indices_3.sum()
                )
                adata_prot_subset.obs.loc[cell_indices_3, "Y"] = np.random.choice(
                    coords[:, 1], cell_indices_3.sum()
                )

    # Store the spatial coordinates in obsm
    adata_prot_subset.obsm["X_spatial"] = adata_prot_subset.obs[["X", "Y"]].to_numpy()

    return adata_prot_subset, horizontal_splits, vertical_splits


def verify_gradients(*models):
    return
    for model in models:
        if all(param.grad is None for param in model.module.parameters()):
            print("No gradients found for any parameter in the model.")
            # raise ValueError("No gradients found for any parameter in the model.")


def compute_pairwise_kl(loc, scale):
    # Expand for broadcasting
    loc1 = loc.unsqueeze(1)
    loc2 = loc.unsqueeze(0)
    scale1 = scale.unsqueeze(1)
    scale2 = scale.unsqueeze(0)
    # Compute KL divergence for each pair
    kl_matrix = (
        torch.log(scale2 / scale1) + (scale1**2 + (loc1 - loc2) ** 2) / (2 * scale2**2) - 0.5
    ).sum(
        dim=-1
    )  # Sum over latent dimensions
    return kl_matrix


def compute_pairwise_kl_two_items(loc1, loc2, scale1, scale2):
    # Expand for broadcasting
    loc1 = loc1.unsqueeze(1)
    loc2 = loc2.unsqueeze(0)
    scale1 = scale1.unsqueeze(1)
    scale2 = scale2.unsqueeze(0)
    # Compute KL divergence for each pair
    kl_matrix = (
        torch.log(scale2 / scale1) + (scale1**2 + (loc1 - loc2) ** 2) / (2 * scale2**2) - 0.5
    ).sum(
        dim=-1
    )  # Sum over latent dimensions
    return kl_matrix


def plot_torch_normal(mean, std_dev, num_points=1000):
    """
    Plots a Normal distribution given the mean and standard deviation.

    Parameters:
        mean (float): The mean of the distribution.
        std_dev (float): The standard deviation of the distribution.
        num_points (int): Number of points to plot (default: 1000).
    """
    # Create the Normal distribution
    normal_dist = torch.distributions.Normal(mean, std_dev)
    x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, num_points)
    y = torch.exp(normal_dist.log_prob(torch.tensor(x))).numpy()
    plt.plot(x, y, label=f"Mean={mean:.2f}, Variance={std_dev ** 2:.2f}")
    plt.title("Normal Distribution (Torch)")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()


def plot_archetypes(
    data_points,
    archetype,
    samples_cell_types: List[str],
    data_point_archetype_indices: List[int],
    modality="",
    cell_type_colors: Dict[str, Any] = None,
    max_points=2000,
):
    """Plot archetypes with subsampling for large datasets.

    Parameters:
    -----------
    data_points : array
        Data points matrix
    archetype : array
        Archetype matrix
    samples_cell_types : List[str]
        Cell type for each data point
    data_point_archetype_indices : List[int]
        Archetype index for each data point
    modality : str
        Modality name for plotting
    cell_type_colors : Dict[str, Any]
        Cell type color mapping
    max_points : int
        Maximum number of points to plot
    """
    if not isinstance(samples_cell_types, List):
        raise TypeError("samples_cell_types should be a list of strings.")
    if not isinstance(data_point_archetype_indices, List):
        raise TypeError("data_point_archetype_indices should be a list of integers.")
    if len(data_points) != len(samples_cell_types) or len(data_points) != len(
        data_point_archetype_indices
    ):
        raise ValueError(
            "Length of data_points, samples_cell_types, and data_point_archetype_indices must be equal."
        )

    # Check the shapes of data_points and archetype
    print("Shape of data_points:", data_points.shape)
    print("Shape of archetype before any adjustment:", archetype.shape)

    # Ensure archetype has the same number of features as data_points
    if archetype.shape[1] != data_points.shape[1]:
        # Check if transposing helps
        if archetype.T.shape[1] == data_points.shape[1]:
            print("Transposing archetype array to match dimensions.")
            archetype = archetype.T
        else:
            raise ValueError("archetype array cannot be reshaped to match data_points dimensions.")

    print("Shape of archetype after adjustment:", archetype.shape)

    # Apply subsampling if data_points is too large
    if len(data_points) > max_points:
        print(f"Subsampling data to {max_points} points for visualization")
        # Create random indices for subsampling
        subsample_indices = np.random.choice(len(data_points), max_points, replace=False)
        data_points_subset = data_points[subsample_indices]
        samples_cell_types_subset = [samples_cell_types[i] for i in subsample_indices]
        data_point_archetype_indices_subset = [
            data_point_archetype_indices[i] for i in subsample_indices
        ]
    else:
        data_points_subset = data_points
        samples_cell_types_subset = samples_cell_types
        data_point_archetype_indices_subset = data_point_archetype_indices

    # Combine data points and archetypes
    num_archetypes = archetype.shape[0]
    data = np.concatenate((data_points_subset, archetype), axis=0)
    labels = ["data"] * len(data_points_subset) + ["archetype"] * num_archetypes
    cell_types = samples_cell_types_subset + ["archetype"] * num_archetypes

    # Perform PCA and t-SNE with limited dimensions for efficiency
    data_pca = data[:, : min(50, data.shape[1])]

    # Run t-SNE with subsampling if needed
    tsne_max_points = min(1000, data.shape[0])  # t-SNE is computationally intensive
    if data.shape[0] > tsne_max_points:
        print(
            f"Further subsampling to {tsne_max_points} points for t-SNE (computationally intensive)"
        )
        tsne_indices = np.random.choice(data.shape[0], tsne_max_points, replace=False)
        tsne_data = data_pca[tsne_indices]
        data_tsne = TSNE(n_components=2).fit_transform(tsne_data)
        # Map back to original indices
        tsne_labels = [labels[i] for i in tsne_indices]
        tsne_cell_types = [cell_types[i] for i in tsne_indices]
        tsne_data_point_arch_indices = []
        for i in tsne_indices:
            if i < len(data_point_archetype_indices_subset):
                tsne_data_point_arch_indices.append(data_point_archetype_indices_subset[i])
            else:
                tsne_data_point_arch_indices.append(np.nan)
        tsne_archetype_numbers = [np.nan] * (len(tsne_indices) - num_archetypes) + list(
            range(num_archetypes)
        )
    else:
        data_tsne = TSNE(n_components=2).fit_transform(data_pca)
        tsne_labels = labels
        tsne_cell_types = cell_types
        tsne_data_point_arch_indices = (
            data_point_archetype_indices_subset + [np.nan] * num_archetypes
        )
        tsne_archetype_numbers = [np.nan] * len(data_points_subset) + list(range(num_archetypes))

    # Create a numbering for archetypes
    archetype_numbers = [np.nan] * len(data_points_subset) + list(range(num_archetypes))

    # Create DataFrames for plotting
    df_pca = pd.DataFrame(
        {
            "PCA1": data_pca[:, 0],
            "PCA2": data_pca[:, 1],
            "type": labels,
            "cell_type": cell_types,
            "archetype_number": archetype_numbers,
            "data_point_archetype_index": data_point_archetype_indices_subset
            + [np.nan] * num_archetypes,
        }
    )

    df_tsne = pd.DataFrame(
        {
            "TSNE1": data_tsne[:, 0],
            "TSNE2": data_tsne[:, 1],
            "type": tsne_labels,
            "cell_type": tsne_cell_types,
            "archetype_number": tsne_archetype_numbers,
            "data_point_archetype_index": tsne_data_point_arch_indices,
        }
    )

    # Use the provided color mapping or generate a new one
    if cell_type_colors is not None:
        palette_dict = cell_type_colors
    else:
        # Define color palette based on unique cell types
        unique_cell_types = list(pd.unique(samples_cell_types))
        palette = sns.color_palette("tab20", len(unique_cell_types))
        palette_dict = {cell_type: color for cell_type, color in zip(unique_cell_types, palette)}
        palette_dict["archetype"] = "black"  # Assign black to archetype

    # Ensure 'archetype' color is set
    if "archetype" not in palette_dict:
        palette_dict["archetype"] = "black"

    # Plot PCA
    plt.figure(figsize=(10, 6))
    df_pca = df_pca.sort_values(by="cell_type")
    sns.scatterplot(
        data=df_pca,
        x="PCA1",
        y="PCA2",
        hue="cell_type",
        style="type",
        palette=palette_dict,
        size="type",
        sizes={"data": 40, "archetype": 500},
        legend="brief",
        alpha=1,
    )

    # Remove 'type' from the legend
    handles, labels_ = plt.gca().get_legend_handles_labels()
    cell_type_legend = [
        (h, l) for h, l in zip(handles, labels_) if l in palette_dict.keys() and l != "archetype"
    ]
    if cell_type_legend:
        handles, labels_ = zip(*cell_type_legend)
    plt.legend(handles, labels_, title="Cell Types", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Annotate archetype points with numbers
    archetype_points = df_pca[df_pca["type"] == "archetype"]
    for _, row in archetype_points.iterrows():
        plt.text(
            row["PCA1"],
            row["PCA2"],
            str(int(row["archetype_number"])),
            fontsize=12,
            fontweight="bold",
            color="red",
        )

    # Add lines from each data point to its matching archetype
    df_pca_data = df_pca[df_pca["type"] == "data"].copy()
    df_pca_archetypes = df_pca[df_pca["type"] == "archetype"].copy()

    # 'archetype_number' is already assigned to archetypes
    df_pca_archetypes["archetype_number"] = df_pca_archetypes["archetype_number"].astype(int)

    # Create a mapping from archetype_number to its PCA coordinates
    archetype_coords = df_pca_archetypes.set_index("archetype_number")[["PCA1", "PCA2"]]

    # Now for each data point, draw a line to its corresponding archetype, limiting to max_lines
    max_lines = min(1000, len(df_pca_data))  # Limit number of lines to prevent clutter
    if len(df_pca_data) > max_lines:
        print(f"Limiting connection lines to {max_lines} for visualization clarity")
        line_indices = np.random.choice(len(df_pca_data), max_lines, replace=False)
        df_pca_data_subset = df_pca_data.iloc[line_indices]
    else:
        df_pca_data_subset = df_pca_data

    for idx, row in df_pca_data_subset.iterrows():
        archetype_index = int(row["data_point_archetype_index"])
        data_point_coords = (row["PCA1"], row["PCA2"])
        try:
            archetype_point_coords = archetype_coords.loc[archetype_index]
            plt.plot(
                [data_point_coords[0], archetype_point_coords["PCA1"]],
                [data_point_coords[1], archetype_point_coords["PCA2"]],
                color="gray",
                linewidth=0.5,
                alpha=0.3,
            )
        except KeyError:
            # If archetype_index does not match any archetype_number, skip
            pass

    plt.title(f"{modality} PCA: Data Points and Archetypes\nColored by Cell Types")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.legend()
    plt.gca().set_aspect("equal")
    plt.show()

    # Plot t-SNE
    plt.figure(figsize=(10, 6))
    df_tsne = df_tsne.sort_values(by="cell_type")
    sns.scatterplot(
        data=df_tsne,
        x="TSNE1",
        y="TSNE2",
        hue="cell_type",
        style="type",
        palette=palette_dict,
        size="type",
        sizes={"data": 20, "archetype": 500},
        legend="brief",
        alpha=1,
    )

    # Remove 'type' from the legend
    handles, labels_ = plt.gca().get_legend_handles_labels()
    cell_type_legend = [
        (h, l) for h, l in zip(handles, labels_) if l in palette_dict.keys() and l != "archetype"
    ]

    if cell_type_legend:
        handles, labels_ = zip(*cell_type_legend)
    plt.legend(handles, labels_, title="Cell Types", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Annotate archetype points with numbers
    archetype_points_tsne = df_tsne[df_tsne["type"] == "archetype"]
    for _, row in archetype_points_tsne.iterrows():
        plt.text(
            row["TSNE1"],
            row["TSNE2"],
            str(int(row["archetype_number"])),
            fontsize=12,
            fontweight="bold",
            color="red",
        )

    # Add lines from each data point to its matching archetype in t-SNE plot
    df_tsne_data = df_tsne[df_tsne["type"] == "data"].copy()
    df_tsne_archetypes = df_tsne[df_tsne["type"] == "archetype"].copy()

    # Create a mapping from archetype_number to its t-SNE coordinates
    archetype_coords_tsne = df_tsne_archetypes.set_index("archetype_number")[["TSNE1", "TSNE2"]]

    # Now for each data point, draw a line to its corresponding archetype, limiting to max_lines
    if len(df_tsne_data) > max_lines:
        line_indices = np.random.choice(len(df_tsne_data), max_lines, replace=False)
        df_tsne_data_subset = df_tsne_data.iloc[line_indices]
    else:
        df_tsne_data_subset = df_tsne_data

    for idx, row in df_tsne_data_subset.iterrows():
        if pd.isna(row["data_point_archetype_index"]):
            continue
        archetype_index = int(row["data_point_archetype_index"])
        data_point_coords = (row["TSNE1"], row["TSNE2"])
        try:
            archetype_point_coords = archetype_coords_tsne.loc[archetype_index]
            plt.plot(
                [data_point_coords[0], archetype_point_coords["TSNE1"]],
                [data_point_coords[1], archetype_point_coords["TSNE2"]],
                color="gray",
                linewidth=0.2,
                alpha=0.3,
            )
        except KeyError:
            # If archetype_index does not match any archetype_number, skip
            pass

    plt.title(f"{modality} t-SNE Scatter Plot with Archetypes Numbered")
    plt.tight_layout()
    plt.show()


def plot_aligned_normalized_losses(history):
    """
    Plot aligned and normalized losses for train and validation data.

    Parameters:
        history (dict): Dictionary containing training and validation loss data as Pandas DataFrames.
    """
    # Extract all loss keys that contain 'loss'
    all_loss_keys = [k for k in history.keys() if "loss" in k]

    # Identify unique base loss names
    unique_losses = list(
        set(k.replace("train_", "").replace("validation_", "") for k in all_loss_keys)
    )

    # Filter to keep only those with both train and validation keys
    filtered_losses = [
        loss_name
        for loss_name in unique_losses
        if f"train_{loss_name}" in history.keys() and f"validation_{loss_name}" in history.keys()
    ]

    # Create figure and subplots
    fig, axes = plt.subplots(
        len(filtered_losses), 1, figsize=(8, 4 * len(filtered_losses)), sharex=True
    )

    # Handle single loss case
    if len(filtered_losses) == 1:
        axes = [axes]

    for ax, loss_name in zip(axes, filtered_losses):
        # Get train data
        train_key = f"train_{loss_name}"
        train_df = history[train_key]
        train_epochs = train_df.index.astype(float)  # Ensure numeric indices
        train_data = train_df.values.flatten().astype(float)  # Ensure numeric values

        train_min, train_max = train_data.min(), train_data.max()
        norm_train_data = (
            (train_data - train_min) / (train_max - train_min)
            if train_min != train_max
            else train_data
        )

        # Get validation data
        val_key = f"validation_{loss_name}"
        val_df = history[val_key]
        val_epochs = val_df.index.astype(float)  # Ensure numeric indices
        val_data = val_df.values.flatten().astype(float)  # Ensure numeric values

        val_min, val_max = val_data.min(), val_data.max()
        norm_val_data = (
            (val_data - val_min) / (val_max - val_min) if val_min != val_max else val_data
        )

        # Interpolate validation data to align with training epochs
        interpolated_val_data = np.interp(train_epochs, val_epochs, norm_val_data)

        # Plot both on the same subplot
        ax.plot(
            train_epochs,
            norm_train_data,
            label=f"Train {loss_name} (min: {train_min:.2f}, max: {train_max:.2f})",
        )
        ax.plot(
            train_epochs,
            interpolated_val_data,
            label=f"Val {loss_name} (min: {val_min:.2f}, max: {val_max:.2f})",
        )

        ax.set_title(loss_name)
        ax.set_ylabel("Normalized Loss")
        ax.legend()

    axes[-1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.show()


def evaluate_distance_metrics(A: np.ndarray, B: np.ndarray, metrics: List[str]) -> Dict:
    return  # keep this here for now do not remove this function or change it in any way
    """
    Evaluates multiple distance metrics to determine which one best captures the similarity
    between matching rows in matrices A and B.

    Parameters:
    - A: np.ndarray of shape (n_samples, n_features)
    - B: np.ndarray of shape (n_samples, n_features)
    - metrics: List of distance metrics to evaluate

    Returns:
    - results: Dictionary containing evaluation metrics for each distance metric
    """
    results = {}

    for metric in metrics:
        print(f"Evaluating distance metric: {metric}")

        # Compute the distance matrix between rows of A and rows of B
        distances = cdist(A, B, metric=metric)
        # For each row i, get the distances between A[i] and all rows in B
        # Then compute the rank of the matching distance
        ranks = []
        for i in range(len(A)):
            row_distances = distances[i, :]
            # Get the rank of the matching distance
            # Rank 1 means the smallest distance
            rank = np.argsort(row_distances).tolist().index(i) + 1
            ranks.append(rank)
        ranks = np.array(ranks)
        total_samples = len(A)
        # Compute evaluation metrics
        num_correct_matches = np.sum(ranks == 1)
        percentage_correct = num_correct_matches / total_samples * 100
        mean_rank = np.mean(ranks)
        mrr = np.mean(1 / ranks)
        print(f"Percentage of correct matches (rank 1): {percentage_correct:.2f}%")
        print(f"Mean rank of matching rows: {mean_rank:.2f}")
        print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
        print("")
        results[metric] = {
            "percentage_correct": percentage_correct,
            "mean_rank": mean_rank,
            "mrr": mrr,
            "ranks": ranks,
        }
    return results


def plot_archetypes_matching(data1, data2, rows=5, max_cols=20):
    """Plot archetype matching between two modalities.

    Parameters:
    -----------
    data1: DataFrame
        First modality archetype data
    data2: DataFrame
        Second modality archetype data
    rows: int
        Number of rows to plot
    max_cols: int
        Maximum number of columns to plot
    """
    # Ensure matrices aren't too wide for visualization
    if data1.shape[1] > max_cols or data2.shape[1] > max_cols:
        print(
            f"Warning: Large archetype matrices. Limiting to {max_cols} columns for visualization."
        )
        data1_plot = data1.iloc[:, :max_cols]
        data2_plot = data2.iloc[:, :max_cols]
    else:
        data1_plot = data1
        data2_plot = data2

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("RNA Archetype Weights\nAcross Cell Types")
    offset = 1
    rows = min(rows, len(data1_plot), len(data2_plot))
    for i in range(rows):
        y1 = data1_plot.iloc[i] + i * offset
        y2 = data2_plot.iloc[i] + i * offset
        plt.plot(y1, label=f"modality 1 archetype {i + 1}")
        plt.plot(y2, linestyle="--", label=f"modality 2 archetype {i + 1}")
    plt.xlabel("Columns")
    plt.ylabel("proportion of cell types accounted for an archetype")
    plt.title("Show that the archetypes are aligned by using")
    # rotate x labels
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.title("Protein Archetype Weights\nAcross Cell Types")
    plt.suptitle("Cross-Modal Archetype Weight Distribution Analysis")
    offset = 1
    rows = min(rows, len(data1_plot), len(data2_plot))
    for i in range(rows):
        y1 = data1_plot.iloc[i] + i * offset
        y2 = data2_plot.iloc[i] + i * offset
        plt.plot(y1, label=f"modality 1 archetype {i + 1}")
        plt.plot(y2, linestyle="--", label=f"modality 2 archetype {i + 1}")
    plt.xlabel("Columns")
    plt.ylabel("proportion of cell types accounted for an archetype")
    plt.title("Show that the archetypes are aligned by using")
    # rotate x labels
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


def compare_matchings(
    archetype_proportion_list_rna,
    archetype_proportion_list_protein,
    metric="correlation",
    num_trials=100,
):
    """Compare optimal matching cost with average random matching cost and plot norms."""
    # Extract the best pair based on optimal matching
    best_cost = float("inf")
    for i, (rna, protein) in enumerate(
        zip(archetype_proportion_list_rna, archetype_proportion_list_protein)
    ):
        rna = rna.values if hasattr(rna, "values") else rna
        protein = protein.values if hasattr(protein, "values") else protein
        row_ind, col_ind, cost, cost_matrix = match_rows(rna, protein, metric)
        if cost < best_cost:
            best_cost = cost
            best_rna, best_protein = rna, protein
            best_rna_archetype_order, best_protein_archetype_order = row_ind, col_ind
            best_cost_matrix = cost_matrix
    print(f"Optimal normalized matching cost: {best_cost:.4f}")

    # Compute distances for the optimal matching
    optimal_distances = best_cost_matrix[best_rna_archetype_order, best_protein_archetype_order]

    # Compute distances for a single random matching
    random_cost, random_distances = compute_random_matching_cost(best_rna, best_protein, metric)

    # Visualization of distances
    n_samples = best_rna.shape[0]
    indices = np.arange(n_samples)

    plt.figure(figsize=(10, 6))
    plt.plot(indices, np.sort(optimal_distances), label="Optimal Matching", marker="o")
    plt.plot(indices, np.sort(random_distances), label="Random Matching", marker="x")
    plt.xlabel("Sample/Archtype Index (sorted by distance)")
    plt.ylabel("Distance")
    plt.title("Comparison of Distances between Matched Rows")
    plt.legend()
    plt.show()

    # Compute average random matching cost over multiple trials
    random_costs = []
    for _ in range(num_trials):
        cost, _ = compute_random_matching_cost(best_rna, best_protein, metric)
        random_costs.append(cost)
    avg_random_cost = np.mean(random_costs)
    std_random_cost = np.std(random_costs)
    print(f"Average random matching cost over {num_trials} trials: {avg_random_cost:.4f}")
    print(f"Standard deviation: {std_random_cost:.4f}")

    # Bar plot of normalized matching costs
    labels = ["Optimal Matching", "Random Matching"]
    costs = [best_cost, avg_random_cost]
    errors = [0, std_random_cost]
    plt.figure(figsize=(8, 6))
    plt.bar(labels, costs, yerr=errors, capsize=5, color=["skyblue", "lightgreen"])
    plt.ylabel("Normalized Matching Cost")
    plt.title("Optimal vs. Random Row Matching Costs")
    plt.show()


def compute_random_matching_cost(rna, protein, metric="correlation"):
    """Compute normalized cost and distances for a random row assignment."""
    n_samples = rna.shape[0]
    random_indices = np.random.permutation(n_samples)
    protein_random = protein[random_indices]

    if metric == "euclidean":
        distances = np.linalg.norm(rna - protein_random, axis=1)
    elif metric == "cosine":
        # Normalize rows to compute cosine similarity
        rna_norm = rna / np.linalg.norm(rna, axis=1, keepdims=True)
        protein_random_norm = protein_random / np.linalg.norm(protein_random, axis=1, keepdims=True)
        cosine_similarity = np.sum(rna_norm * protein_random_norm, axis=1)
        distances = 1 - cosine_similarity  # Cosine distance
        for i in range(100):
            random_indices = np.random.permutation(n_samples)
            protein_random = protein[random_indices]
            protein_random_norm = protein_random / np.linalg.norm(
                protein_random, axis=1, keepdims=True
            )
            cosine_similarity = np.sum(rna_norm * protein_random_norm, axis=1)
            distances = np.vstack((distances, 1 - cosine_similarity))
        distances = np.mean(distances, axis=0)
    elif metric == "correlation":
        # Compute Pearson correlation distance
        rna_mean = np.mean(rna, axis=1, keepdims=True)
        protein_random_mean = np.mean(protein_random, axis=1, keepdims=True)
        rna_centered = rna - rna_mean
        protein_random_centered = protein_random - protein_random_mean

        numerator = np.sum(rna_centered * protein_random_centered, axis=1)
        denominator = np.sqrt(np.sum(rna_centered**2, axis=1)) * np.sqrt(
            np.sum(protein_random_centered**2, axis=1)
        )
        pearson_correlation = numerator / denominator
        distances = 1 - pearson_correlation  # Correlation distance

    else:
        raise ValueError("Unsupported metric. Use 'euclidean' or 'cosine'.")

    normalized_cost = np.sum(distances) / n_samples
    return normalized_cost, distances


def match_rows(matrix1, matrix2, metric="correlation"):
    """Helper function to match rows between two matrices."""
    if metric == "correlation":
        # Compute correlation matrix
        corr_matrix = np.corrcoef(matrix1, matrix2)[: matrix1.shape[0], matrix1.shape[0] :]
        # Convert correlation to distance (1 - correlation)
        dist_matrix = 1 - corr_matrix
    else:
        # Use scipy's cdist for other metrics
        dist_matrix = cdist(matrix1, matrix2, metric=metric)

    # Use Hungarian algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    total_cost = dist_matrix[row_ind, col_ind].sum()
    return row_ind, col_ind, total_cost, dist_matrix


def find_best_pair_by_row_matching(
    archetype_proportion_list_rna, archetype_proportion_list_protein, metric="correlation"
):
    """
    Find the best index in the list by matching rows using linear assignment.

    Parameters:
    -----------
    archetype_proportion_list : list of tuples
        List where each tuple contains (rna, protein) matrices.
    metric : str, optional
        Distance metric to use ('euclidean' or 'cosine').

    Returns:
    --------
    best_num_or_archetypes_index : int
        Index of the best matching pair in the list.
    best_total_cost : float
        Total cost of the best matching.
    best_rna_archetype_order : np.ndarray
        Indices of RNA rows.
    best_protein_archetype_order : np.ndarray
        Indices of Protein rows matched to RNA rows.
    """

    best_num_or_archetypes_index = None
    best_total_cost = float("inf")
    best_rna_archetype_order = None
    best_protein_archetype_order = None

    for i, (rna, protein) in enumerate(
        zip(archetype_proportion_list_rna, archetype_proportion_list_protein)
    ):
        rna = rna.values if hasattr(rna, "values") else rna
        protein = protein.values if hasattr(protein, "values") else protein

        assert rna.shape[1] == protein.shape[1], f"Mismatch in dimensions at index {i}."

        row_ind, col_ind, total_cost, _ = match_rows(rna, protein, metric=metric)
        print(f"Pair {i}: Total matching cost = {total_cost}")

        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_num_or_archetypes_index = i
            best_rna_archetype_order = row_ind
            best_protein_archetype_order = col_ind

    return (
        best_num_or_archetypes_index,
        best_total_cost,
        best_rna_archetype_order,
        best_protein_archetype_order,
    )


def clean_uns_for_h5ad(adata: AnnData):
    """
    Remove or convert non-serializable objects from the `uns` attribute of an AnnData object.
    """
    keys_to_remove = []
    for key, value in adata.uns.items():
        if isinstance(value, sns.palettes._ColorPalette):
            # Convert seaborn ColorPalette to a list of colors
            adata.uns[key] = list(value)
        elif not isinstance(value, (str, int, float, list, dict, np.ndarray)):
            # Mark non-serializable keys for removal
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del adata.uns[key]
    if hasattr(adata, "obsm"):
        adata.obsm = {str(key): value for key, value in adata.obsm.items()}

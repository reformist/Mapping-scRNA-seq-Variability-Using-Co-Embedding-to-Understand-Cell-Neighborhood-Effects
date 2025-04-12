"""Utility functions for hyperparameter search."""

import gc
import os
from datetime import datetime
from pathlib import Path
from pprint import pprint

import anndata as ad
import mlflow
import numpy as np
import psutil
import scanpy as sc
import torch
from anndata import AnnData
from plotting_functions import (
    plot_archetype_embedding,
    plot_cell_type_distributions,
    plot_combined_latent_space,
    plot_latent_pca_both_modalities_by_celltype,
    plot_latent_pca_both_modalities_cn,
    plot_normalized_losses,
    plot_rna_protein_latent_cn_cell_type_umap,
)
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_mutual_info_score
from tqdm import tqdm

from bar_nick_utils import compare_distance_distributions, mixing_score


def log_parameters(params, run_index, total_runs):
    """Log parameters in a nice format."""
    separator = "=" * 80
    print(f"\n{separator}")
    print(f"RUN {run_index+1}/{total_runs} PARAMETERS:")
    print(f"{separator}")
    for key, value in params.items():
        print(f"{key}: {value}")
    print(f"{separator}\n")


# Add logging to file functionality
class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Ensure content is written immediately

    def flush(self):
        for f in self.files:
            f.flush()


def setup_and_train_model(
    adata_rna_subset, adata_prot_subset, params, model_checkpoints_folder=None
):
    """Setup and train the VAE models with given parameters."""
    from _3_train_vae_with_archetypes_vectors import train_vae

    # Log parameters
    mlflow.log_params(
        {
            "batch_size": round(params["batch_size"], 3),
            "contrastive_weight": round(params["contrastive_weight"], 3),
            "similarity_weight": round(params["similarity_weight"], 3),
            "matching_weight": round(params["matching_weight"], 3),
            "cell_type_clustering_weight": round(params["cell_type_clustering_weight"], 3),
            "kl_weight_rna": round(params["kl_weight_rna"], 3),
            "kl_weight_prot": round(params["kl_weight_prot"], 3),
            "adv_weight": round(params.get("adv_weight", 0), 3),
            "n_hidden_rna": round(params.get("n_hidden_rna", 0), 3),
            "n_hidden_prot": round(params.get("n_hidden_prot", 0), 3),
            "n_layers": round(params.get("n_layers", 0), 3),
            "latent_dim": round(params.get("latent_dim", 0), 3),
        }
    )
    # Training setup and execution
    rna_vae, protein_vae = train_vae(
        adata_rna_subset=adata_rna_subset,
        adata_prot_subset=adata_prot_subset,
        model_checkpoints_folder=model_checkpoints_folder,
        **params,
    )

    return rna_vae, protein_vae


def process_latent_spaces(rna_adata, protein_adata):
    """Process and combine latent spaces from both modalities.

    This function assumes that the latent representations have been computed
    using vae.module() and stored in the "X_scVI" field of the AnnData objects.

    Args:
        rna_adata: RNA AnnData object with latent representation in obsm["X_scVI"]
        protein_adata: Protein AnnData object with latent representation in obsm["X_scVI"]

    Returns:
        rna_latent: RNA latent AnnData
        prot_latent: Protein latent AnnData
        combined_latent: Combined latent AnnData
    """

    # Store latent representations
    SCVI_LATENT_KEY = "X_scVI"
    # Prepare AnnData objects
    rna_latent = AnnData(rna_adata.obsm[SCVI_LATENT_KEY].copy())
    prot_latent = AnnData(protein_adata.obsm[SCVI_LATENT_KEY].copy())
    rna_latent.obs = rna_adata.obs.copy()
    prot_latent.obs = protein_adata.obs.copy()

    # Clear any existing embeddings
    rna_latent.obsm.pop("X_pca", None)
    prot_latent.obsm.pop("X_pca", None)

    # Use standard parameters for individual modalities
    sc.pp.neighbors(rna_latent, use_rep="X", n_neighbors=10)
    sc.tl.umap(rna_latent)
    sc.pp.neighbors(prot_latent, use_rep="X", n_neighbors=10)
    sc.tl.umap(prot_latent)

    # Combine latent spaces
    combined_latent = ad.concat(
        [rna_latent.copy(), prot_latent.copy()],
        join="outer",
        label="modality",
        keys=["RNA", "Protein"],
    )

    # Clear any existing neighbors data to ensure clean calculation
    combined_latent.obsm.pop("X_pca", None) if "X_pca" in combined_latent.obsm else None
    combined_latent.obsp.pop(
        "connectivities", None
    ) if "connectivities" in combined_latent.obsp else None
    combined_latent.obsp.pop("distances", None) if "distances" in combined_latent.obsp else None
    combined_latent.uns.pop("neighbors", None) if "neighbors" in combined_latent.uns else None

    # Use cosine metric and larger n_neighbors for better batch integration
    sc.pp.neighbors(combined_latent, use_rep="X")
    sc.tl.umap(combined_latent)

    return rna_latent, prot_latent, combined_latent


def batched_cdist(X, Y, batch_size=5000):
    """Calculate pairwise distances in batches to prevent memory issues."""
    n_x = X.shape[0]
    n_y = Y.shape[0]
    distances = np.zeros((n_x, n_y))

    for i in tqdm(range(0, n_x, batch_size), desc="Processing rows", total=n_x // batch_size):
        end_i = min(i + batch_size, n_x)
        batch_X = X[i:end_i]

        for j in range(0, n_y, batch_size):
            end_j = min(j + batch_size, n_y)
            batch_Y = Y[j:end_j]

            batch_distances = cdist(batch_X, batch_Y)
            distances[i:end_i, j:end_j] = batch_distances

        print(f"Processed {end_i}/{n_x} rows", end="\r")

    return distances


def match_cells_and_calculate_distances(rna_latent, prot_latent):
    """Match cells between modalities and calculate distances."""
    # Calculate pairwise distances
    latent_distances = batched_cdist(rna_latent.X, prot_latent.X)

    # Find matches
    prot_matches_in_rna = np.argmin(latent_distances, axis=0)
    matching_distances = np.min(latent_distances, axis=0)

    # Generate random matches for comparison
    rand_indices = np.random.permutation(len(rna_latent))
    rand_latent_distances = latent_distances[rand_indices, :]
    rand_prot_matches_in_rna = np.argmin(rand_latent_distances, axis=0)
    rand_matching_distances = np.min(rand_latent_distances, axis=0)

    return {
        "prot_matches_in_rna": prot_matches_in_rna,
        "matching_distances": matching_distances,
        "rand_prot_matches_in_rna": rand_prot_matches_in_rna,
        "rand_matching_distances": rand_matching_distances,
    }


def calculate_post_training_metrics(rna_adata, protein_adata, prot_matches_in_rna):
    """Calculate various metrics for model evaluation."""
    # Calculate NMI scores
    nmi_cell_types_cn_rna = adjusted_mutual_info_score(
        rna_adata.obs["cell_types"], rna_adata.obs["CN"]
    )
    nmi_cell_types_cn_prot = adjusted_mutual_info_score(
        protein_adata.obs["cell_types"], protein_adata.obs["CN"]
    )
    nmi_cell_types_modalities = adjusted_mutual_info_score(
        rna_adata.obs["cell_types"].values[prot_matches_in_rna],
        protein_adata.obs["cell_types"].values,
    )

    # Calculate accuracy
    matches = (
        rna_adata.obs["cell_types"].values[prot_matches_in_rna]
        == protein_adata.obs["cell_types"].values
    )
    accuracy = matches.sum() / len(matches)

    # Calculate mixing score
    mixing_result = mixing_score(
        rna_adata.obsm["X_scVI"],
        protein_adata.obsm["X_scVI"],
        rna_adata,
        protein_adata,
        index_rna=range(len(rna_adata)),
        index_prot=range(len(protein_adata)),
        plot_flag=True,
    )

    return {
        "nmi_cell_types_cn_rna": nmi_cell_types_cn_rna,
        "nmi_cell_types_cn_prot": nmi_cell_types_cn_prot,
        "nmi_cell_types_modalities": nmi_cell_types_modalities,
        "cell_type_matching_accuracy": accuracy,
        "mixing_score_ilisi": mixing_result["iLISI"],
        "mixing_score_clisi": mixing_result["cLISI"],
    }


def generate_visualizations(
    rna_adata, protein_adata, rna_latent, prot_latent, combined_latent, history, matching_results
):
    """Generate all visualizations for the model."""
    # Plot training results
    plot_normalized_losses(history)

    # Plot latent representations
    plot_latent_pca_both_modalities_cn(
        rna_adata.obsm["X_scVI"],
        protein_adata.obsm["X_scVI"],
        rna_adata,
        protein_adata,
        index_rna=range(len(rna_adata.obs.index)),
        index_prot=range(len(protein_adata.obs.index)),
        use_subsample=True,
    )

    plot_latent_pca_both_modalities_by_celltype(
        rna_adata,
        protein_adata,
        rna_adata.obsm["X_scVI"],
        protein_adata.obsm["X_scVI"],
        use_subsample=True,
    )

    # Plot distance distributions
    compare_distance_distributions(
        matching_results["rand_matching_distances"],
        rna_latent,
        prot_latent,
        matching_results["matching_distances"],
    )

    # Plot combined visualizations
    plot_combined_latent_space(combined_latent, use_subsample=True)
    plot_cell_type_distributions(combined_latent, 3, use_subsample=True)

    # Plot archetype and embedding visualizations
    plot_archetype_embedding(rna_adata, protein_adata, use_subsample=True)
    plot_rna_protein_latent_cn_cell_type_umap(rna_adata, protein_adata, use_subsample=True)


def save_results(rna_vae, protein_vae, save_dir):
    """Save trained models and their data."""
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save to local directory
    sc.write(Path(f"{save_dir}/rna_vae_trained_{time_stamp}.h5ad"), rna_vae.adata)
    sc.write(Path(f"{save_dir}/protein_vae_trained_{time_stamp}.h5ad"), protein_vae.adata)

    # Log to MLflow in models folder
    mlflow.log_artifact(f"{save_dir}/rna_vae_trained_{time_stamp}.h5ad", "models")
    mlflow.log_artifact(f"{save_dir}/protein_vae_trained_{time_stamp}.h5ad", "models")

    return time_stamp


def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB


def log_memory_usage(prefix=""):
    """Log current memory usage"""
    mem_usage = get_memory_usage()
    print(f"{prefix}Memory usage: {mem_usage:.2f} GB")
    return mem_usage


def clear_memory():
    """Clear memory by running garbage collection and clearing CUDA cache if available"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return get_memory_usage()


def check_memory_threshold(threshold_gb=0.9):
    """Check if memory usage is above threshold"""
    mem_usage = get_memory_usage()
    if mem_usage > threshold_gb:
        print(f"Warning: Memory usage ({mem_usage:.2f} GB) above threshold ({threshold_gb} GB)")
        return True
    return False


def handle_error(e, params, run_name):
    """Handle errors during hyperparameter search."""
    import traceback

    print("\n" + "=" * 80)
    print("❌ RUN FAILED ❌".center(80))
    print("❌ RUN FAILED ❌".center(80))
    print("❌ RUN FAILED ❌".center(80))
    print("❌ RUN FAILED ❌".center(80))
    print("❌ RUN FAILED ❌".center(80))
    print("❌ RUN FAILED ❌".center(80))
    print("❌ RUN FAILED ❌".center(80))
    print("❌ RUN FAILED ❌".center(80))
    print("❌ RUN FAILED ❌".center(80))
    print("❌ RUN FAILED ❌".center(80))
    print("=" * 80 + "\n")

    error_msg = f"""
    Error in run {run_name}:
    Error Type: {type(e).__name__}
    Error Message: {str(e)}
    Memory Usage: {get_memory_usage():.2f} GB
    Stack Trace:
    {traceback.format_exc()}
    Parameters used:
    {params}
    """
    pprint(error_msg)
    mlflow.log_param("error_type", type(e).__name__)
    mlflow.log_param("error_message", str(e))
    mlflow.log_param("error_memory_usage", f"{get_memory_usage():.2f} GB")
    mlflow.log_param("error_stack_trace", traceback.format_exc())
    clear_memory()  # Clear memory on error


def run_cell_type_clustering_loss(
    adata, latent_mean, indices, device="cuda:0" if torch.cuda.is_available() else "cpu"
):
    """Calculate cell type clustering loss to preserve cell type relationships.

    Args:
        adata: AnnData object with cell type information
        latent_mean: Latent mean representation from the VAE (already computed from module() call)
        indices: Indices of cells to use
        device: Device to use for calculations

    Returns:
        Cell type clustering loss tensor
    """
    cell_types = torch.tensor(adata[indices].obs["cell_types"].cat.codes.values).to(device)

    # Combine cell types and latent representations from both modalities
    # Calculate centroid for each cell type
    unique_cell_types = torch.unique(cell_types)
    num_cell_types = len(unique_cell_types)

    # Skip the cell type clustering loss if there's only one cell type
    cell_type_clustering_loss = torch.tensor(0.0).to(device)

    if num_cell_types > 1:
        # Calculate centroids for each cell type in latent space
        centroids = []
        cells_per_type = []
        type_to_idx = {}

        for i, cell_type in enumerate(unique_cell_types):
            mask = cell_types == cell_type
            type_to_idx[cell_type.item()] = i
            if mask.sum() > 0:
                cells = latent_mean[mask]
                centroid = cells.mean(dim=0)
                centroids.append(centroid)
                cells_per_type.append(cells)

        if len(centroids) > 1:  # Need at least 2 centroids
            centroids = torch.stack(centroids)

            # Get original structure from archetype vectors
            # Compute the structure matrix once and cache it
            all_cell_types = adata.obs["cell_types"].cat.codes.values
            all_unique_types = np.unique(all_cell_types)

            # Get centroids in archetype space for each cell type
            original_centroids = []
            for ct in all_unique_types:
                mask = all_cell_types == ct
                if mask.sum() > 0:
                    ct_archetype_vecs = adata.obsm["archetype_vec"][mask]
                    original_centroids.append(np.mean(ct_archetype_vecs, axis=0))

            # Convert to torch tensor
            original_centroids = torch.tensor(np.array(original_centroids), dtype=torch.float32).to(
                device
            )

            # Compute affinity/structure matrix (using Gaussian kernel)
            sigma = torch.cdist(original_centroids, original_centroids).mean()
            dists = torch.cdist(original_centroids, original_centroids)
            original_structure_matrix = torch.exp(-(dists**2) / (2 * sigma**2))

            # Set diagonal to 0 to focus on between-cluster relationships
            original_structure_matrix = original_structure_matrix * (
                1 - torch.eye(len(all_unique_types), device=device)
            )

            # Compute current structure matrix in latent space
            # Use same sigma as original for consistency
            sigma = torch.cdist(centroids, centroids).mean()
            latent_dists = torch.cdist(centroids, centroids)
            current_structure_matrix = torch.exp(-(latent_dists**2) / (2 * sigma**2))

            # Set diagonal to 0 to focus on between-cluster relationships
            current_structure_matrix = current_structure_matrix * (
                1 - torch.eye(len(centroids), device=device)
            )

            # Now compute the structure preservation loss
            structure_preservation_loss = 0.0
            count = 0

            # For each cell type in the batch, compare its relationships
            for i, type_i in enumerate(unique_cell_types):
                if type_i.item() < len(original_structure_matrix):
                    for j, type_j in enumerate(unique_cell_types):
                        if i != j and type_j.item() < len(original_structure_matrix):
                            # Get original and current affinity values
                            orig_affinity = original_structure_matrix[type_i.item(), type_j.item()]
                            current_affinity = current_structure_matrix[i, j]

                            # Square difference
                            diff = (orig_affinity - current_affinity) ** 2
                            structure_preservation_loss += diff
                            count += 1

            if count > 0:
                structure_preservation_loss = structure_preservation_loss / count

            # Calculate within-cluster cohesion and variance regularization
            cohesion_loss = 0.0
            total_cells = 0

            # Calculate variances for each cluster in each dimension
            valid_clusters = [cells for cells in cells_per_type if len(cells) > 1]
            if len(valid_clusters) > 1:
                # Calculate variance for each cluster in each dimension
                cluster_variances = []
                for cells in valid_clusters:
                    # Compute variance per dimension
                    vars_per_dim = torch.var(cells, dim=0)
                    cluster_variances.append(vars_per_dim)

                # Stack variances for all clusters
                cluster_variances = torch.stack(cluster_variances)

                # Calculate mean variance across all clusters for each dimension
                mean_variance_per_dim = torch.mean(cluster_variances, dim=0)

                # Log the mean variance for some dimensions
                if mean_variance_per_dim.shape[0] > 3:
                    print(
                        f"Mean variances for first 3 dimensions: {mean_variance_per_dim[:3].detach().cpu().numpy()}"
                    )
                    print(f"Dimension variance std: {torch.std(mean_variance_per_dim).item():.4f}")

                # Modify the cluster cohesion calculation to include variance regularization
                for i, cells in enumerate(cells_per_type):
                    if len(cells) > 1:
                        # Calculate distances to centroid (standard cohesion)
                        dists = torch.norm(cells - centroids[i], dim=1)

                        # Calculate variance regularization term for this cluster
                        vars_per_dim = torch.var(cells, dim=0)
                        variance_diff = torch.mean((vars_per_dim - mean_variance_per_dim) ** 2)

                        # Log the variance regularization term for the first few clusters
                        if i < 3:  # Only log for first 3 clusters to avoid too much output
                            print(f"Cluster {i} variance diff: {variance_diff.item():.4f}")
                        print(
                            f"Cluster {i} cohesion loss: {dists.mean().item():.4f} + variance diff {0.3 * variance_diff.item():.4f}"
                        )
                        # Combine standard cohesion with variance regularization
                        # Use 0.3 as weight for variance regularization within cohesion
                        cohesion_loss += dists.mean() + 0.3 * variance_diff

                        total_cells += 1
            else:
                # If we don't have enough clusters with multiple cells, just use standard cohesion
                for i, cells in enumerate(cells_per_type):
                    if len(cells) > 1:
                        dists = torch.norm(cells - centroids[i], dim=1)
                        cohesion_loss += dists.mean()
                        total_cells += 1

            if total_cells > 0:
                cohesion_loss = cohesion_loss / total_cells

            # Normalize the cohesion loss by the average inter-centroid distance
            # This makes it scale-invariant
            avg_inter_centroid_dist = torch.cdist(centroids, centroids).mean()
            if avg_inter_centroid_dist > 0:
                normalized_cohesion_loss = cohesion_loss / avg_inter_centroid_dist
            else:
                normalized_cohesion_loss = cohesion_loss

            # Combined loss: balance between structure preservation and cohesion
            # The higher weight on structure_preservation_loss (2.0) prioritizes
            # preserving the original relationships between clusters
            cell_type_clustering_loss = 2.0 * structure_preservation_loss + normalized_cohesion_loss
            return cell_type_clustering_loss

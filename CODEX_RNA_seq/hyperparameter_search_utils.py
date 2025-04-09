"""Utility functions for hyperparameter search."""

import gc
import importlib.util
import os
from pathlib import Path

import anndata as ad
import mlflow
import numpy as np
import pandas as pd
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

from bar_nick_utils import clean_uns_for_h5ad, compare_distance_distributions, mixing_score

# Import the training function using importlib
spec = importlib.util.spec_from_file_location(
    "train_vae_module",
    os.path.join(os.path.dirname(__file__), "3_train_vae_with_archetypes_vectors.py"),
)
train_vae_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_vae_module)
train_vae = train_vae_module.train_vae


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


def setup_and_train_model(adata_rna_subset, adata_prot_subset, params):
    """Setup and train the VAE models with given parameters."""

    # Training setup and execution
    rna_vae, protein_vae, latent_rna_before, latent_prot_before = train_vae(
        adata_rna_subset=adata_rna_subset, adata_prot_subset=adata_prot_subset, **params
    )

    # Log parameters
    mlflow.log_params(
        {
            "batch_size": params["batch_size"],
            "contrastive_weight": params["contrastive_weight"],
            "similarity_weight": params["similarity_weight"],
            "matching_weight": params["matching_weight"],
            "cell_type_clustering_weight": params["cell_type_clustering_weight"],
            "adv_weight": params.get("adv_weight", None),
            "n_hidden_rna": params.get("n_hidden_rna", None),
            "n_hidden_prot": params.get("n_hidden_prot", None),
            "n_layers": params.get("n_layers", None),
            "latent_dim": params.get("latent_dim", None),
        }
    )

    return rna_vae, protein_vae, latent_rna_before, latent_prot_before


def process_latent_spaces(rna_vae, protein_vae):
    """Process and combine latent spaces from both modalities."""
    # Get latent representations
    latent_rna = rna_vae.adata.obsm["X_scVI"]
    latent_prot = protein_vae.adata.obsm["X_scVI"]

    # Store latent representations
    SCVI_LATENT_KEY = "X_scVI"
    rna_vae.adata.obs["CN"] = rna_vae.adata.obs["CN"].values
    rna_vae.adata.obsm[SCVI_LATENT_KEY] = latent_rna
    protein_vae.adata.obsm[SCVI_LATENT_KEY] = latent_prot

    # Prepare AnnData objects
    rna_latent = AnnData(rna_vae.adata.obsm[SCVI_LATENT_KEY].copy())
    prot_latent = AnnData(protein_vae.adata.obsm[SCVI_LATENT_KEY].copy())
    rna_latent.obs = rna_vae.adata.obs.copy()
    prot_latent.obs = protein_vae.adata.obs.copy()

    # Run dimensionality reduction
    sc.pp.pca(rna_latent)
    sc.pp.neighbors(rna_latent)
    sc.tl.umap(rna_latent)
    sc.pp.pca(prot_latent)
    sc.pp.neighbors(prot_latent)
    sc.tl.umap(prot_latent)

    # Combine latent spaces
    combined_latent = ad.concat(
        [rna_latent.copy(), prot_latent.copy()],
        join="outer",
        label="modality",
        keys=["RNA", "Protein"],
    )
    combined_major_cell_types = pd.concat(
        (rna_vae.adata.obs["major_cell_types"], protein_vae.adata.obs["major_cell_types"]),
        join="outer",
    )
    combined_latent.obs["major_cell_types"] = combined_major_cell_types
    combined_latent.obs["cell_types"] = pd.concat(
        (rna_vae.adata.obs["cell_types"], protein_vae.adata.obs["cell_types"]), join="outer"
    )
    combined_latent.obs["CN"] = pd.concat(
        (rna_vae.adata.obs["CN"], protein_vae.adata.obs["CN"]), join="outer"
    )
    sc.pp.pca(combined_latent)
    sc.pp.neighbors(combined_latent, n_neighbors=15)
    sc.tl.umap(combined_latent, min_dist=0.1)

    return rna_latent, prot_latent, combined_latent


def batched_cdist(X, Y, batch_size=1000):
    """Calculate pairwise distances in batches to prevent memory issues."""
    n_x = X.shape[0]
    n_y = Y.shape[0]
    distances = np.zeros((n_x, n_y))

    for i in range(0, n_x, batch_size):
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


def calculate_metrics(rna_vae, protein_vae, prot_matches_in_rna):
    """Calculate various metrics for model evaluation."""
    # Calculate NMI scores
    nmi_cell_types_cn_rna = adjusted_mutual_info_score(
        rna_vae.adata.obs["cell_types"], rna_vae.adata.obs["CN"]
    )
    nmi_cell_types_cn_prot = adjusted_mutual_info_score(
        protein_vae.adata.obs["cell_types"], protein_vae.adata.obs["CN"]
    )
    nmi_cell_types_modalities = adjusted_mutual_info_score(
        rna_vae.adata.obs["cell_types"].values[prot_matches_in_rna],
        protein_vae.adata.obs["cell_types"].values,
    )

    # Calculate accuracy
    matches = (
        rna_vae.adata.obs["cell_types"].values[prot_matches_in_rna]
        == protein_vae.adata.obs["cell_types"].values
    )
    accuracy = matches.sum() / len(matches)

    # Calculate mixing score
    mixing_result = mixing_score(
        rna_vae.adata.obsm["X_scVI"],
        protein_vae.adata.obsm["X_scVI"],
        rna_vae.adata,
        protein_vae.adata,
        index_rna=range(len(rna_vae.adata)),
        index_prot=range(len(protein_vae.adata)),
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
    rna_vae, protein_vae, rna_latent, prot_latent, combined_latent, history, matching_results
):
    """Generate all visualizations for the model."""
    # Plot training results
    plot_normalized_losses(history)

    # Plot latent representations
    plot_latent_pca_both_modalities_cn(
        rna_vae.adata.obsm["X_scVI"],
        protein_vae.adata.obsm["X_scVI"],
        rna_vae.adata,
        protein_vae.adata,
        index_rna=range(len(rna_vae.adata.obs.index)),
        index_prot=range(len(protein_vae.adata.obs.index)),
        use_subsample=True,
    )

    plot_latent_pca_both_modalities_by_celltype(
        rna_vae.adata,
        protein_vae.adata,
        rna_vae.adata.obsm["X_scVI"],
        protein_vae.adata.obsm["X_scVI"],
        use_subsample=True,
    )

    # Plot distance distributions
    compare_distance_distributions(
        matching_results["rand_matching_distances"],
        rna_latent,
        prot_latent,
        matching_results["matching_distances"],
        use_subsample=True,
    )

    # Plot combined visualizations
    plot_combined_latent_space(combined_latent, use_subsample=True)
    plot_cell_type_distributions(combined_latent, 3, use_subsample=True)

    # Plot archetype and embedding visualizations
    plot_archetype_embedding(rna_vae, protein_vae, use_subsample=True)
    plot_rna_protein_latent_cn_cell_type_umap(rna_vae, protein_vae, use_subsample=True)


def save_results(rna_vae, protein_vae, save_dir):
    """Save model results and artifacts."""
    clean_uns_for_h5ad(rna_vae.adata)
    clean_uns_for_h5ad(protein_vae.adata)
    time_stamp = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(save_dir, exist_ok=True)

    sc.write(Path(f"{save_dir}/rna_vae_trained_{time_stamp}.h5ad"), rna_vae.adata)
    sc.write(Path(f"{save_dir}/protein_vae_trained_{time_stamp}.h5ad"), protein_vae.adata)

    # Log artifacts
    mlflow.log_artifact(f"{save_dir}/rna_vae_trained_{time_stamp}.h5ad")
    mlflow.log_artifact(f"{save_dir}/protein_vae_trained_{time_stamp}.h5ad")

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
    print(error_msg)
    mlflow.log_param("error_type", type(e).__name__)
    mlflow.log_param("error_message", str(e))
    mlflow.log_param("error_memory_usage", f"{get_memory_usage():.2f} GB")
    mlflow.log_param("error_stack_trace", traceback.format_exc())
    clear_memory()  # Clear memory on error

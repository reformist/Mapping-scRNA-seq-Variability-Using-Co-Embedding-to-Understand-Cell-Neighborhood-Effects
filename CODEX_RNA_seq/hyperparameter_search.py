# %%
"""Hyperparameter search for VAE training with archetypes vectors."""

import gc
import importlib.util
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import psutil
import torch


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


# Create log directory if it doesn't exist
os.makedirs("logs", exist_ok=True)
log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = open(f"logs/hyperparameter_search_{log_timestamp}.log", "w")

# Redirect stdout to both console and log file
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, log_file)

print(f"Starting hyperparameter search at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log file: logs/hyperparameter_search_{log_timestamp}.log")


# Function to log parameters in a nice format
def log_parameters(params, run_index, total_runs):
    separator = "=" * 80
    print(f"\n{separator}")
    print(f"RUN {run_index+1}/{total_runs} PARAMETERS:")
    print(f"{separator}")
    for key, value in params.items():
        print(f"{key}: {value}")
    print(f"{separator}\n")


# Set up paths once
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
import warnings
from pathlib import Path

import anndata as ad
import mlflow
import numpy as np
import pandas as pd
import plotting_functions as pf
import scanpy as sc
import torch
from anndata import AnnData
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_mutual_info_score

import bar_nick_utils

# Force reimport internal modules
importlib.reload(pf)
importlib.reload(bar_nick_utils)
import CODEX_RNA_seq.logging_functions

importlib.reload(CODEX_RNA_seq.logging_functions)


# Force reimport logging functions
import CODEX_RNA_seq.logging_functions

importlib.reload(CODEX_RNA_seq.logging_functions)

from plotting_functions import (
    plot_archetype_embedding,
    plot_cell_type_distributions,
    plot_combined_latent_space,
    plot_latent_pca_both_modalities_by_celltype,
    plot_latent_pca_both_modalities_cn,
    plot_normalized_losses,
    plot_rna_protein_latent_cn_cell_type_umap,
)

from bar_nick_utils import (
    clean_uns_for_h5ad,
    compare_distance_distributions,
    get_latest_file,
    get_umap_filtered_fucntion,
    mixing_score,
)

if not hasattr(sc.tl.umap, "_is_wrapped"):
    sc.tl.umap = get_umap_filtered_fucntion()
    sc.tl.umap._is_wrapped = True
np.random.seed(42)
torch.manual_seed(42)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pd.set_option("display.max_columns", 10)
pd.set_option("display.max_rows", 10)
warnings.filterwarnings("ignore")
pd.options.display.max_rows = 10
pd.options.display.max_columns = 10
np.set_printoptions(threshold=100)
np.random.seed(0)
# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import scanpy as sc
from sklearn.model_selection import ParameterGrid

# Import the training function using importlib
spec = importlib.util.spec_from_file_location(
    "train_vae_module",
    os.path.join(os.path.dirname(__file__), "3_train_vae_with_archetypes_vectors.py"),
)
train_vae_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_vae_module)
train_vae = train_vae_module.train_vae


# Get latest file function
def get_latest_file(directory, prefix):
    """Get the latest file in a directory with a given prefix."""
    files = list(Path(directory).glob(f"{prefix}*"))
    if not files:
        raise FileNotFoundError(f"No files found with prefix {prefix} in {directory}")
    return str(max(files, key=os.path.getctime))


def save_and_log_figure(fig, name, dpi=150):
    """Save figure to disk and log to MLflow."""
    # Create figures directory if it doesn't exist
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    # Save figure
    fig_path = figures_dir / f"{name}.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Log to MLflow
    mlflow.log_artifact(str(fig_path))
    return fig_path


# %%
# Define hyperparameter search space
param_grid = {
    "max_epochs": [3],  # Changed from n_epochs to max_epochs to match train_vae
    "batch_size": [1000, 2000, 3000],
    "lr": [1e-4],
    "contrastive_weight": [1.0, 10.0, 100.0],
    "similarity_weight": [100.0, 1000.0, 10000.0],
    "diversity_weight": [0.1],
    "matching_weight": [10000.0, 10_000.0, 1_000_000.0],  # Updated range to reflect typical values
    "cell_type_clustering_weight": [0.1, 1.0, 10.0],  # Added cell type clustering weight
    "n_hidden_rna": [64, 128],
    "n_hidden_prot": [32, 64],
    "n_layers": [2, 3],
    "latent_dim": [10],
    "kl_weight_rna": [0.1, 1.0],  # Updated range
    "kl_weight_prot": [1.0, 10.0],  # Updated range
    "adv_weight": [0.0],  # Added adversarial weight
    "train_size": [0.9],  # Added train size parameter
    "validation_size": [0.1],  # Added validation size parameter
    "check_val_every_n_epoch": [1],  # Added validation frequency parameter
    "gradient_clip_val": [1.0],  # Added gradient clipping parameter
    "plot_x_times": [1],  # Added plot frequency parameter
}

# %%
# Setup MLflow
mlflow.set_tracking_uri("file:./mlruns")
experiment_name = f"vae_hyperparameter_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
experiment_id = mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# %%
# Load data
save_dir = Path("CODEX_RNA_seq/data/processed_data").absolute()
log_memory_usage("Before loading data: ")

adata_rna_subset = sc.read_h5ad(
    get_latest_file(save_dir, "adata_rna_subset_prepared_for_training_")
)
log_memory_usage("After loading RNA data: ")

adata_prot_subset = sc.read_h5ad(
    get_latest_file(save_dir, "adata_prot_subset_prepared_for_training_")
)
log_memory_usage("After loading protein data: ")

# Subsample data if memory usage is high
if check_memory_threshold():
    print("Memory usage high, subsampling data...")
    rna_sample_size = min(len(adata_rna_subset), 1500)
    prot_sample_size = min(len(adata_prot_subset), 1500)
    adata_rna_subset = sc.pp.subsample(adata_rna_subset, n_obs=rna_sample_size, copy=True)
    adata_prot_subset = sc.pp.subsample(adata_prot_subset, n_obs=prot_sample_size, copy=True)
    log_memory_usage("After subsampling: ")

print(f"Subsampled RNA dataset shape: {adata_rna_subset.shape}")
print(f"Subsampled protein dataset shape: {adata_prot_subset.shape}")


# %%
# Run hyperparameter search
results = []
# print the number of combinations
total_combinations = len(ParameterGrid(param_grid))
print(f"Number of combinations: {total_combinations}")

# Initialize timing variables
start_time = datetime.now()
elapsed_times = []

for i, params in enumerate(ParameterGrid(param_grid)):
    iter_start_time = datetime.now()
    log_memory_usage(f"Start of iteration {i+1}: ")

    # Print progress information
    print(f"\n--- Run {i+1}/{total_combinations} ({(i+1)/total_combinations*100:.2f}%) ---")

    # Log the parameters for this run
    log_parameters(params, i, total_combinations)

    # Calculate and display time estimates if we have completed at least one iteration
    if i > 0:
        if len(elapsed_times) > 0:  # Make sure we have at least one completed iteration
            # Initialize with timedelta(0) instead of integer 0
            total_time = timedelta(0)
            for t in elapsed_times:
                total_time += t
            avg_time_per_iter = total_time / len(elapsed_times)

            remaining_iters = total_combinations - (i + 1)
            est_remaining_time = avg_time_per_iter * remaining_iters

            elapsed_total = datetime.now() - start_time
            est_total_time = elapsed_total + est_remaining_time

            print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Elapsed time: {elapsed_total}")
            print(f"Average time per iteration: {avg_time_per_iter}")
            print(f"Estimated time remaining: {est_remaining_time}")
            print(
                f"Estimated completion time: {(datetime.now() + est_remaining_time).strftime('%Y-%m-%d %H:%M:%S')}"
            )
        else:
            print("No completed iterations yet for time estimation.")

    run_name = f"vae_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(params)

        try:
            # Train model
            log_memory_usage("Before training: ")
            rna_vae, protein_vae, latent_rna_before, latent_prot_before = train_vae(
                adata_rna_subset=adata_rna_subset,
                adata_prot_subset=adata_prot_subset,
                **params,
            )
            log_memory_usage("After training: ")

            # Clear memory after training
            clear_memory()
            log_memory_usage("After clearing memory: ")

            training_kwargs = params
            # Log parameters
            history = rna_vae._training_plan.get_history()

            mlflow.log_params(
                {
                    "batch_size": training_kwargs["batch_size"],
                    "contrastive_weight": training_kwargs["contrastive_weight"],
                    "similarity_weight": training_kwargs["similarity_weight"],
                    # "diversity_weight": training_kwargs["diversity_weight"],
                    "matching_weight": training_kwargs["matching_weight"],
                    "cell_type_clustering_weight": training_kwargs["cell_type_clustering_weight"],
                    "adv_weight": training_kwargs.get("adv_weight", None),
                    "n_hidden_rna": training_kwargs.get("n_hidden_rna", None),
                    "n_hidden_prot": training_kwargs.get("n_hidden_prot", None),
                    "n_layers": training_kwargs.get("n_layers", None),
                    "latent_dim": training_kwargs.get("latent_dim", None),
                }
            )

            # Record iteration time
            iter_time = datetime.now() - iter_start_time
            elapsed_times.append(iter_time)
            print(f"\nIteration completed in: {iter_time}")

            # Get training history from the training plan
            print("\nGetting training history...")
            history = rna_vae._training_plan.get_history()
            print("✓ Training history loaded")

            # Log training history metrics
            mlflow.log_metrics(
                {
                    key: history[hist_key][-1] if history[hist_key] else float("nan")
                    for key, hist_key in {
                        "final_train_similarity_loss": "train_similarity_loss",
                        "final_train_similarity_loss_raw": "train_similarity_loss_raw",
                        "final_train_total_loss": "train_total_loss",
                        "final_val_total_loss": "val_total_loss",
                        "final_train_cell_type_clustering_loss": "train_cell_type_clustering_loss",
                    }.items()
                }
            )

            # %
            print("\Get latent representations...")
            latent_rna = rna_vae.adata.obsm["X_latent"]
            latent_prot = protein_vae.adata.obsm["X_latent"]

            # Store latent representations
            print("\nStoring latent representations...")
            SCVI_LATENT_KEY = "X_scVI"
            rna_vae.adata.obs["CN"] = rna_vae.adata.obs["CN"].values
            rna_vae.adata.obsm[SCVI_LATENT_KEY] = latent_rna
            protein_vae.adata.obsm[SCVI_LATENT_KEY] = latent_prot
            print("✓ Latent representations stored")

            # Prepare AnnData objects
            print("\nPreparing AnnData objects...")
            rna_latent = AnnData(rna_vae.adata.obsm[SCVI_LATENT_KEY].copy())
            prot_latent = AnnData(protein_vae.adata.obsm[SCVI_LATENT_KEY].copy())
            rna_latent.obs = rna_vae.adata.obs.copy()
            prot_latent.obs = protein_vae.adata.obs.copy()
            print("✓ AnnData objects prepared")

            # Run dimensionality reduction
            print("\nRunning dimensionality reduction...")
            sc.pp.pca(rna_latent)
            sc.pp.neighbors(rna_latent)
            sc.tl.umap(rna_latent)
            sc.pp.pca(prot_latent)
            sc.pp.neighbors(prot_latent)
            sc.tl.umap(prot_latent)
            print("✓ Dimensionality reduction completed")

            # Combine latent spaces
            print("\nCombining latent spaces...")
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
            print("✓ UMAP computed successfully")
            print("✓ Latent spaces combined")

            print("\nMatching cells between modalities...")
            # Calculate pairwise distances between RNA and protein cells in latent space

            # Replace direct cdist call with batched version to prevent memory issues
            def batched_cdist(X, Y, batch_size=1000):
                n_x = X.shape[0]
                n_y = Y.shape[0]
                distances = np.zeros((n_x, n_y))

                # Process in batches to avoid memory overflow
                for i in range(0, n_x, batch_size):
                    end_i = min(i + batch_size, n_x)
                    batch_X = X[i:end_i]

                    for j in range(0, n_y, batch_size):
                        end_j = min(j + batch_size, n_y)
                        batch_Y = Y[j:end_j]

                        # Calculate distances for this batch pair
                        batch_distances = cdist(batch_X, batch_Y)
                        distances[i:end_i, j:end_j] = batch_distances

                    print(f"Processed {end_i}/{n_x} rows", end="\r")

                return distances

            print(
                f"RNA latent shape: {rna_latent.X.shape}, Protein latent shape: {prot_latent.X.shape}"
            )
            print("Computing pairwise distances in batches...")
            latent_distances = batched_cdist(rna_latent.X, prot_latent.X)

            print("Generating random permutation for baseline comparison...")
            # Use only a subset for the random comparison if datasets are large
            rand_indices = np.random.permutation(len(rna_latent))
            rand_latent_distances = latent_distances[rand_indices, :]

            # Find closest matches for RNA cells to protein cells
            print("Finding closest matches...")
            prot_matches_in_rna = np.argmin(latent_distances, axis=0).astype(
                np.int32
            )  # size of prot use to index rna

            # Calculate matching distances
            matching_distances = np.min(latent_distances, axis=0)

            # Generate random matches for comparison
            n_rna = len(rna_latent)
            n_prot = len(prot_latent)
            # if rna is smaller than protin the original data then sse var rna larget to true
            if n_rna < n_prot:
                rna_larger = True
            else:
                rna_larger = False

            rand_prot_matches_in_rna = np.argmin(rand_latent_distances, axis=0)
            rand_matching_distances = np.min(rand_latent_distances, axis=0)

            # Calculate random matching distances
            # Store matching information in combined_latent.uns
            combined_latent.uns["cell_matching"] = {
                "prot_matches_in_rna": prot_matches_in_rna,
                "matching_distances": matching_distances,
                "rand_prot_matches_in_rna": rand_prot_matches_in_rna,
                "rand_matching_distances": rand_matching_distances,
            }
            print(f"✓ Matched {len(rna_latent)} RNA cells to protein cells")
            print(f"✓ Matched {len(prot_latent)} protein cells to RNA cells")
            print(f"Average random matching distance: {rand_matching_distances.mean().item():.3f}")
            print(f"Average matching distance: {matching_distances.mean().item():.3f}")
            # Calculate distances and metrics
            print("\nCalculating distances and metrics...")
            # Use the stored matching distances instead of recalculating
            distances = combined_latent.uns["cell_matching"]["matching_distances"]
            rand_distances = combined_latent.uns["cell_matching"]["rand_matching_distances"]
            print("✓ Distances calculated")

            # Plot training results
            print("\nPlotting training results...")
            plot_normalized_losses(history)
            print("✓ Training losses plotted")

            # plot_umap_visualizations_original_data(rna_vae.adata, protein_vae.adata)
            # # Plot spatial data
            # print("\nPlotting spatial data...")
            # plot_spatial_data(protein_vae.adata)
            # print("✓ Spatial data plotted")

            # Plot latent representations
            print("\nPlotting latent representations...")
            plot_latent_pca_both_modalities_cn(
                latent_rna,
                latent_prot,
                rna_vae.adata,
                protein_vae.adata,
                index_rna=range(len(rna_vae.adata.obs.index)),
                index_prot=range(len(protein_vae.adata.obs.index)),
                use_subsample=True,
            )
            plot_latent_pca_both_modalities_by_celltype(
                rna_vae.adata, protein_vae.adata, latent_rna, latent_prot, use_subsample=True
            )
            print("✓ Latent representations plotted")

            # Plot distance distributions
            print("\nPlotting distance distributions...")
            compare_distance_distributions(rand_distances, rna_latent, prot_latent, distances)
            print("✓ Distance distributions plotted")

            # Plot combined visualizations
            print("\nPlotting combined visualizations...")
            plot_combined_latent_space(combined_latent, use_subsample=True)

            plot_cell_type_distributions(combined_latent, 3, use_subsample=True)

            print("✓ Combined visualizations plotted")

            sc.pl.umap(
                combined_latent,
                color=["CN", "modality", "cell_types"],
                title=[
                    "Combined_Latent_UMAP_CN",
                    "Combined_Latent_UMAP_Modality",
                    "Combined_Latent_UMAP_CellTypes",
                ],
                alpha=0.5,
            )

            sc.pl.pca(
                combined_latent,
                color=["CN", "modality"],
                title=["Combined_Latent_PCA_CN", "Combined_Latent_PCA_Modality"],
                alpha=0.5,
            )

            print("✓ UMAP visualizations plotted")

            # Plot archetype and embedding visualizations
            print("\nPlotting archetype and embedding visualizations...")
            plot_archetype_embedding(rna_vae, protein_vae, use_subsample=True)

            plot_rna_protein_latent_cn_cell_type_umap(rna_vae, protein_vae, use_subsample=True)

            print("✓ Archetype and embedding visualizations plotted")

            # Calculate and display final metrics
            print("\nCalculating final metrics...")
            mixing_result = mixing_score(
                latent_rna,
                latent_prot,
                rna_vae.adata,
                protein_vae.adata,
                index_rna=range(len(rna_vae.adata)),
                index_prot=range(len(protein_vae.adata)),
                plot_flag=True,
            )
            print(f"✓ Mixing score: {mixing_result}")

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
            matches = (
                rna_vae.adata.obs["cell_types"].values[prot_matches_in_rna]
                == protein_vae.adata.obs["cell_types"].values
            )

            accuracy = matches.sum() / len(matches)

            print(f"\nFinal Metrics:")
            print(f"Normalized Mutual Information (RNA CN): {nmi_cell_types_cn_rna:.3f}")
            print(f"Normalized Mutual Information (Protein CN): {nmi_cell_types_cn_prot:.3f}")
            print(
                f"Normalized Mutual Information (Cross-modality): {nmi_cell_types_modalities:.3f}"
            )
            print(f"Cell Type Matching Accuracy: {accuracy:.4f}")
            print("✓ Final metrics calculated")

            # Log final metrics
            mlflow.log_metrics(
                {
                    "nmi_cell_types_cn_rna": nmi_cell_types_cn_rna,
                    "nmi_cell_types_cn_prot": nmi_cell_types_cn_prot,
                    "nmi_cell_types_modalities": nmi_cell_types_modalities,
                    "cell_type_matching_accuracy": accuracy,
                    "mixing_score_ilisi": mixing_result["iLISI"],
                    "mixing_score_clisi": mixing_result["cLISI"],
                }
            )

            # Save results
            print("\nSaving results...")
            clean_uns_for_h5ad(rna_vae.adata)
            clean_uns_for_h5ad(protein_vae.adata)
            save_dir = Path("CODEX_RNA_seq/data/trained_data").absolute()
            time_stamp = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
            os.makedirs(save_dir, exist_ok=True)

            print(
                f"\nTrained RNA VAE dimensions: {rna_vae.adata.shape[0]} samples x {rna_vae.adata.shape[1]} features"
            )
            print(
                f"Trained Protein VAE dimensions: {protein_vae.adata.shape[0]} samples x {protein_vae.adata.shape[1]} features\n"
            )

            sc.write(Path(f"{save_dir}/rna_vae_trained_{time_stamp}.h5ad"), rna_vae.adata)
            sc.write(Path(f"{save_dir}/protein_vae_trained_{time_stamp}.h5ad"), protein_vae.adata)
            print("✓ Results saved")

            # Log artifacts
            mlflow.log_artifact(f"{save_dir}/rna_vae_trained_{time_stamp}.h5ad")
            mlflow.log_artifact(f"{save_dir}/protein_vae_trained_{time_stamp}.h5ad")

            # End MLflow run
            mlflow.end_run()

            print("\nAll visualization and analysis steps completed!")

        except Exception as e:
            print(f"Error in run {run_name}: {str(e)}")
            mlflow.log_param("error", str(e))
            clear_memory()  # Clear memory on error
            continue

    # Clear memory after each iteration
    clear_memory()
    log_memory_usage(f"End of iteration {i+1}: ")

# %%
# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("hyperparameter_search_results.csv", index=False)
mlflow.log_artifact("hyperparameter_search_results.csv")

# %%
# Find best parameters
best_params = results_df.loc[results_df["final_val_loss"].idxmin()]
print("Best parameters:")
print(best_params)


# %%
# Clean up: restore original stdout and close log file
print(f"\nHyperparameter search completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log saved to: logs/hyperparameter_search_{log_timestamp}.log")
sys.stdout = original_stdout
log_file.close()

# Log the log file to MLflow
mlflow.log_artifact(f"logs/hyperparameter_search_{log_timestamp}.log")

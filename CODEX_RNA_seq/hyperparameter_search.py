# %%
"""Hyperparameter search for VAE training with archetypes vectors."""

import importlib.util
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import torch
from sklearn.model_selection import ParameterGrid

# Set up paths once
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

from CODEX_RNA_seq.training_utils import (
    Tee,
    calculate_metrics,
    clear_memory,
    generate_visualizations,
    handle_error,
    log_memory_usage,
    log_parameters,
    match_cells_and_calculate_distances,
    process_latent_spaces,
    save_results,
    setup_and_train_model,
)

# Create log directory if it doesn't exist
os.makedirs("logs", exist_ok=True)
log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = open(f"logs/hyperparameter_search_{log_timestamp}.log", "w")

# Redirect stdout to both console and log file
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, log_file)

print(f"Starting hyperparameter search at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log file: logs/hyperparameter_search_{log_timestamp}.log")

import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import scanpy as sc
import torch

import bar_nick_utils

# Force reimport internal modules
importlib.reload(bar_nick_utils)
import CODEX_RNA_seq.logging_functions

importlib.reload(CODEX_RNA_seq.logging_functions)

if not hasattr(sc.tl.umap, "_is_wrapped"):
    sc.tl.umap = bar_nick_utils.get_umap_filtered_fucntion()
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

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

# Define hyperparameter search space
param_grid = {
    "plot_x_times": [5],
    "check_val_every_n_epoch": [5],
    "max_epochs": [31],  # Changed from n_epochs to max_epochs to match train_vae
    "batch_size": [1000],
    "lr": [1e-4],
    "contrastive_weight": [0.0, 100.0, 100_000],
    "similarity_weight": [0.0, 10000.0, 1000000.0],
    "diversity_weight": [0.0],
    "matching_weight": [0, 10_000.0, 1_000_000.0],  # Updated range to reflect typical values
    "cell_type_clustering_weight": [0, 100.0, 10000.0],  # Added cell type clustering weight
    "n_hidden_rna": [64],
    "n_hidden_prot": [32],
    "n_layers": [3],
    "latent_dim": [10],
    "kl_weight_rna": [0.001, 0.01],
    "kl_weight_prot": [10, 0.1],
    "adv_weight": [0.0],
    "train_size": [0.85],
    "validation_size": [0.15],
    "gradient_clip_val": [1.0],
}

# Setup MLflow
mlflow.set_tracking_uri("file:./mlruns")
experiment_name = f"vae_hyperparameter_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
experiment_id = mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# Load data
save_dir = Path("CODEX_RNA_seq/data/processed_data").absolute()
log_memory_usage("Before loading data: ")

adata_rna_subset = sc.read_h5ad(
    bar_nick_utils.get_latest_file(save_dir, "adata_rna_subset_prepared_for_training_")
)
log_memory_usage("After loading RNA data: ")

adata_prot_subset = sc.read_h5ad(
    bar_nick_utils.get_latest_file(save_dir, "adata_prot_subset_prepared_for_training_")
)
log_memory_usage("After loading protein data: ")

# Subsample data if memory usage is high
print("Memory usage high, subsampling data...")
# rna_sample_size = min(len(adata_rna_subset), 1500)
# prot_sample_size = min(len(adata_prot_subset), 1500)
# adata_rna_subset = sc.pp.subsample(adata_rna_subset, n_obs=rna_sample_size, copy=True)
# adata_prot_subset = sc.pp.subsample(adata_prot_subset, n_obs=prot_sample_size, copy=True)
log_memory_usage("After subsampling: ")

print(f"Subsampled RNA dataset shape: {adata_rna_subset.shape}")
print(f"Subsampled protein dataset shape: {adata_prot_subset.shape}")

# Run hyperparameter search
results = []
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
    if i > 0 and len(elapsed_times) > 0:
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

    run_name = f"vae_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        try:
            # Setup and train model
            log_memory_usage("Before training: ")
            rna_vae, protein_vae, latent_rna_before, latent_prot_before = setup_and_train_model(
                adata_rna_subset, adata_prot_subset, params
            )
            log_memory_usage("After training: ")

            # Log successful run
            mlflow.log_param("run_failed", False)

            # Clear memory after training
            clear_memory()
            log_memory_usage("After clearing memory: ")

            # Get training history
            history = rna_vae._training_plan.get_history()

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

            # Process latent spaces
            # subsample the adata_rna_subset and adata_prot_subset to 1000 cells
            rna_adata = rna_vae.adata
            rna_adata = sc.pp.subsample(rna_adata, n_obs=1500, copy=True)
            prot_adata = protein_vae.adata
            prot_adata = sc.pp.subsample(prot_adata, n_obs=1500, copy=True)
            rna_latent, prot_latent, combined_latent = process_latent_spaces(rna_adata, prot_adata)

            # Match cells and calculate distances
            matching_results = match_cells_and_calculate_distances(rna_latent, prot_latent)

            # Calculate metrics
            metrics = calculate_metrics(
                rna_adata, prot_adata, matching_results["prot_matches_in_rna"]
            )

            # Log metrics
            mlflow.log_metrics(metrics)

            # Generate visualizations

            generate_visualizations(
                rna_adata,
                prot_adata,
                rna_latent,
                prot_latent,
                combined_latent,
                history,
                matching_results,
            )

            # Save results
            save_dir = Path("CODEX_RNA_seq/data/trained_data").absolute()
            save_results(rna_vae, protein_vae, save_dir)

            # Record iteration time
            iter_time = datetime.now() - iter_start_time
            elapsed_times.append(iter_time)
            print(f"\nIteration completed in: {iter_time}")

        except Exception as e:
            # Log failed run
            mlflow.log_param("run_failed", True)
            handle_error(e, params, run_name)
            time.sleep(5)  # Sleep for 5 seconds after failure
            continue

    # Clear memory after each iteration
    clear_memory()
    log_memory_usage(f"End of iteration {i+1}: ")

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("hyperparameter_search_results.csv", index=False)
mlflow.log_artifact("hyperparameter_search_results.csv")

# Find best parameters
best_params = results_df.loc[results_df["final_val_loss"].idxmin()]
print("Best parameters:")
print(best_params)

# Clean up: restore original stdout and close log file
print(f"\nHyperparameter search completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log saved to: logs/hyperparameter_search_{log_timestamp}.log")
sys.stdout = original_stdout
log_file.close()

# Log the log file to MLflow
mlflow.log_artifact(f"logs/hyperparameter_search_{log_timestamp}.log")

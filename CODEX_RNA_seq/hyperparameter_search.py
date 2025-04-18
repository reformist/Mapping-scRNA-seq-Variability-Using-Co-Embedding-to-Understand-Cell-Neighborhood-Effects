# %%
"""Hyperparameter search for VAE training with archetypes vectors."""

import importlib.util
import json
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
    calculate_post_training_metrics,
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
    "plot_x_times": [30],
    "check_val_every_n_epoch": [2],
    "max_epochs": [800],  # Changed from n_epochs to max_epochs to match train_vae
    # "plot_x_times": [1],
    # "check_val_every_n_epoch": [1],
    # "max_epochs": [3],  # Changed from n_epochs to max_epochs to match train_vae
    "batch_size": [3000],
    "contrastive_weight": [
        0
        # 10
    ],
    "similarity_weight": [100],
    "diversity_weight": [0.0],
    "matching_weight": [0.1],  # Updated range to reflect typical values
    "cell_type_clustering_weight": [
        1_000,
        # 10_000,
        # 100_000,
    ],  # Within-modality cell type clustering
    "cross_modal_cell_type_weight": [
        1_000
        # 10_000,
    ],  # Added cross-modal cell type alignment weight
    "n_hidden_rna": [512],
    "n_hidden_prot": [32],
    "n_layers": [4],
    "latent_dim": [20],
    "kl_weight_rna": [1],
    "kl_weight_prot": [1],
    "adv_weight": [0.0],
    "train_size": [0.85],
    "validation_size": [0.15],
    "gradient_clip_val": [1.0],
    "lr": [
        # 1e-2,
        1e-3,
        # ,1e-4
    ],  # 1e-3 was good
}

mlflow.set_tracking_uri("file:./mlruns")

# Get existing experiment or create new one
experiment_name = "vae_hyperparameter_search_9"  # Fixed name instead of timestamp
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
except Exception:
    experiment_id = mlflow.create_experiment(experiment_name)

mlflow.set_experiment(experiment_name)

# Get existing runs and their parameters
existing_runs = mlflow.search_runs(experiment_ids=[experiment_id])
existing_params = []
for _, run in existing_runs.iterrows():
    run_params = {}
    for param in param_grid.keys():
        param_key = f"params.{param}"
        if (
            run.status == "FINISHED"
            and param_key in run.index
            and param
            not in [
                "plot_x_times",
                "check_val_every_n_epoch",
                "max_epochs",
            ]
        ):
            run_params[param] = run[param_key]
    if run_params:  # Only add if we have parameters
        existing_params.append(run_params)

# Filter out parameter combinations that have already been tried
all_combinations = list(ParameterGrid(param_grid))
new_combinations = []
for combo in all_combinations:
    # Create a copy of the combo without the ignored keys
    combo_to_check = {
        k: v
        for k, v in combo.items()
        if k not in ["plot_x_times", "check_val_every_n_epoch", "max_epochs"]
    }
    if combo_to_check not in existing_params:
        new_combinations.append(combo)

total_combinations = len(new_combinations)
print(f"Total combinations: {len(all_combinations)}")
print(f"Already tried: {len(existing_params)}")
print(f"New combinations to try: {total_combinations}")

# Load data
# save_dir = Path("CODEX_RNA_seq/data/processed_data").absolute()
# log_memory_usage("Before loading data: ")

# adata_rna_subset = sc.read_h5ad(
#     bar_nick_utils.get_latest_file(save_dir, "adata_rna_subset_prepared_for_training_")
# )
# log_memory_usage("After loading RNA data: ")

# adata_prot_subset = sc.read_h5ad(
#     bar_nick_utils.get_latest_file(save_dir, "adata_prot_subset_prepared_for_training_")
# )

# todo this is cite-seq data temp
adata_rna_subset_cite_seq = (
    "CODEX_RNA_seq/adata_rna_subset_prepared_for_training_2025-04-10-18-05-13.h5ad"
)
adata_prot_subset_cite_seq = (
    "CODEX_RNA_seq/adata_prot_subset_prepared_for_training_2025-04-10-18-05-13.h5ad"
)
adata_rna_subset = sc.read_h5ad(adata_rna_subset_cite_seq)
adata_prot_subset = sc.read_h5ad(adata_prot_subset_cite_seq)
adata_rna_subset.obs["cell_types"] = adata_rna_subset.obs["major_cell_types"]
adata_prot_subset.obs["cell_types"] = adata_prot_subset.obs["major_cell_types"]
log_memory_usage("After loading protein data: ")

print(f"RNA dataset shape: {adata_rna_subset.shape}")
print(f"Protein dataset shape: {adata_prot_subset.shape}")

# # Subsample for debugging # todo remove!!!!
# rna_sample_size = min(len(adata_rna_subset), 1500)
# prot_sample_size = min(len(adata_prot_subset), 1500)
# adata_rna_subset = sc.pp.subsample(adata_rna_subset, n_obs=rna_sample_size, copy=True)
# adata_prot_subset = sc.pp.subsample(adata_prot_subset, n_obs=prot_sample_size, copy=True)
log_memory_usage("After subsampling: ")
model_checkpoints_folder = Path("CODEX_RNA_seq/pretrained_vae/epoch_74")  # base model
# good mixing model
model_checkpoints_folder = Path(
    "/home/barroz/projects/Mapping-scRNA-seq-Variability-Using-Co-Embedding-to-Understand-Cell-Neighborhood-Effects/mlruns/653315841334425763/59ea832431a94c608377d66fbac380f6/artifacts/epoch_59"
)
# good mixing model trained more epochs
model_checkpoints_folder = Path(
    "/home/barroz/projects/Mapping-scRNA-seq-Variability-Using-Co-Embedding-to-Understand-Cell-Neighborhood-Effects/mlruns/653315841334425763/a9dcb6cd70404a15a486ae38130ed5cb/artifacts/epoch_149"
)

model_checkpoints_folder = Path(  # best mixing so far
    "/home/barroz/projects/Mapping-scRNA-seq-Variability-Using-Co-Embedding-to-Understand-Cell-Neighborhood-Effects/mlruns/653315841334425763/9cecb1810d4e4145b35cf8651f0eec86/artifacts/epoch_149"
)
model_checkpoints_folder = Path(
    "/home/barroz/projects/Mapping-scRNA-seq-Variability-Using-Co-Embedding-to-Understand-Cell-Neighborhood-Effects/mlruns/653315841334425763/467605ebd3a6460bb60788777148a920/artifacts/epoch_399"
)
model_checkpoints_folder = Path(
    "/home/barroz/projects/Mapping-scRNA-seq-Variability-Using-Co-Embedding-to-Understand-Cell-Neighborhood-Effects/mlruns/653315841334425763/964c65b61dcf40a289a67ada1cfa8c13/artifacts/epoch_399"
)
model_checkpoints_folder = Path(
    "/home/barroz/projects/Mapping-scRNA-seq-Variability-Using-Co-Embedding-to-Understand-Cell-Neighborhood-Effects/mlruns/653315841334425763/026fec4ddfc048f388c1515a0e97ce61/artifacts/epoch_399"
)

# Estimate training time before starting
if total_combinations > 0 and len(new_combinations) > 0:
    # Use first parameter combination for estimation
    first_params = new_combinations[0]
    rna_cells = adata_rna_subset.shape[0]
    prot_cells = adata_prot_subset.shape[0]

    print(f"\n--- Time Estimation for {rna_cells} RNA cells and {prot_cells} protein cells ---")
    time_per_iter, total_time = CODEX_RNA_seq.logging_functions.estimate_training_time(
        rna_cells, prot_cells, first_params, total_combinations
    )

    print(f"Estimated time per iteration: {time_per_iter}")
    print(f"Estimated total time for {total_combinations} combinations: {total_time}")
    print(
        f"Estimated completion time: {(datetime.now() + total_time).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("Note: This is a rough estimate based on dataset size and hyperparameters")
    print("Actual times may vary based on system load and other factors")
    print("More accurate estimates will be provided after the first iteration completes")
    print("------------------------------------------------------------\n")

# Subsample data if memory usage is high
# rna_sample_size = min(len(adata_rna_subset), 1500)
# prot_sample_size = min(len(adata_prot_subset), 1500)
# adata_rna_subset = sc.pp.subsample(adata_rna_subset, n_obs=rna_sample_size, copy=True)
# adata_prot_subset = sc.pp.subsample(adata_prot_subset, n_obs=prot_sample_size, copy=True)
log_memory_usage("After subsampling: ")

print(f"Subsampled RNA dataset shape: {adata_rna_subset.shape}")
print(f"Subsampled protein dataset shape: {adata_prot_subset.shape}")

# Run hyperparameter search
results = []
print(f"Number of new combinations to try: {total_combinations}")

# Initialize timing variables
start_time = datetime.now()
elapsed_times = []

for i, params in enumerate(new_combinations):
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

    # Create run-specific log file
    run_log_dir = os.path.join("logs", "mlflow_run_logs")
    os.makedirs(run_log_dir, exist_ok=True)
    run_log_file_path = os.path.join(run_log_dir, f"{run_name}.log")
    run_log_file = open(run_log_file_path, "w")

    # Save original stdout and set up run-specific Tee
    run_original_stdout = sys.stdout
    run_tee = Tee(run_original_stdout, run_log_file)
    sys.stdout = run_tee

    with mlflow.start_run(run_name=run_name):
        try:
            # Create loss weights JSON
            loss_weights = {
                "kl_weight_rna": params["kl_weight_rna"],
                "kl_weight_prot": params["kl_weight_prot"],
                "contrastive_weight": params["contrastive_weight"],
                "similarity_weight": params["similarity_weight"],
                "matching_weight": params["matching_weight"],
                "cell_type_clustering_weight": params[
                    "cell_type_clustering_weight"
                ],  # Controls within-modality clustering
                "cross_modal_cell_type_weight": params[
                    "cross_modal_cell_type_weight"
                ],  # Controls cross-modal cell type alignment
            }

            # Save loss weights to a temporary JSON file
            loss_weights_path = "loss_weights.json"
            with open(loss_weights_path, "w") as f:
                json.dump(loss_weights, f, indent=4)

            # Log loss weights JSON as artifact
            mlflow.log_artifact(loss_weights_path)
            # Clean up temporary file
            os.remove(loss_weights_path)

            # Setup and train model
            rna_vae, protein_vae = setup_and_train_model(
                adata_rna_subset, adata_prot_subset, params, model_checkpoints_folder
            )

            # Log successful run

            # Clear memory after training
            clear_memory()

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
                        "final_train_cross_modal_cell_type_loss": "train_cross_modal_cell_type_loss",
                    }.items()
                }
            )

            # Process latent spaces
            # subsample the adata_rna_subset and adata_prot_subset to 1000 cells
            rna_adata = rna_vae.adata
            rna_adata = sc.pp.subsample(rna_adata, n_obs=min(len(rna_adata), 5000), copy=True)
            prot_adata = protein_vae.adata
            prot_adata = sc.pp.subsample(prot_adata, n_obs=min(len(prot_adata), 5000), copy=True)
            rna_latent, prot_latent, combined_latent = process_latent_spaces(rna_adata, prot_adata)

            # Match cells and calculate distances
            matching_results = match_cells_and_calculate_distances(rna_latent, prot_latent)

            # Calculate metrics
            metrics = calculate_post_training_metrics(
                rna_adata, prot_adata, matching_results["prot_matches_in_rna"]
            )

            # Log metrics
            mlflow.log_metrics({k: round(v, 3) for k, v in metrics.items()})

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

            # Close run log file and reset stdout before logging the artifact
            sys.stdout = run_original_stdout
            run_log_file.close()

            # Log the run-specific log file as an artifact
            mlflow.log_artifact(run_log_file_path, "logs")
            print(f"Logged run log file: {run_log_file_path}")
            mlflow.log_param("run_failed", False)

        except Exception as e:
            # Log failed run
            error_details = str(e)
            # Create detailed error log file
            error_log_path = os.path.join(run_log_dir, f"{run_name}_error.log")
            with open(error_log_path, "w") as error_file:
                import traceback

                error_file.write(
                    f"Error occurred at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                )
                error_file.write(f"Parameters: {json.dumps(params, indent=2)}\n\n")
                error_file.write(f"Error message: {str(e)}\n\n")
                error_file.write("Traceback:\n")
                error_file.write(traceback.format_exc())

            # Log the error file as an artifact
            mlflow.log_artifact(error_log_path, "error_logs")
            print(f"Logged detailed error information to: {error_log_path}")

            handle_error(e, params, run_name)

            # Close run log file and reset stdout
            sys.stdout = run_original_stdout
            run_log_file.close()

            # Log the run-specific log file as an artifact even in case of error
            mlflow.log_artifact(run_log_file_path, "logs")
            print(f"Logged run log file for failed run: {run_log_file_path}")
            mlflow.end_run(status="FAILED")
            time.sleep(5)  # Sleep for 5 seconds after failure
            continue
        finally:
            # Ensure stdout is restored even if an unexpected error occurs
            if sys.stdout != run_original_stdout:
                sys.stdout = run_original_stdout
                if not run_log_file.closed:
                    run_log_file.close()

    # Clear memory after each iteration
    clear_memory()
    log_memory_usage(f"End of iteration {i+1}: ")

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("hyperparameter_search_results.csv", index=False)
mlflow.log_artifact("hyperparameter_search_results.csv")

# Find best parameters


# Clean up: restore original stdout and close log file
print(f"\nHyperparameter search completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log saved to: logs/hyperparameter_search_{log_timestamp}.log")
sys.stdout = original_stdout
log_file.close()

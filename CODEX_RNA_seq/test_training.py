"""Simple test script for training VAE with archetypes vectors."""
import os
import sys

# Set up paths once
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

from pathlib import Path

import mlflow
import numpy as np
import scanpy as sc
import torch

import bar_nick_utils

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("Loading data...")
save_dir = Path("CODEX_RNA_seq/data/processed_data").absolute()

# Use get_latest_file to find the most recent files
adata_rna_subset = sc.read_h5ad(
    bar_nick_utils.get_latest_file(save_dir, "adata_rna_subset_prepared_for_training_")
)
adata_prot_subset = sc.read_h5ad(
    bar_nick_utils.get_latest_file(save_dir, "adata_prot_subset_prepared_for_training_")
)

print(f"Original RNA dataset shape: {adata_rna_subset.shape}")
print(f"Original protein dataset shape: {adata_prot_subset.shape}")

# Subsample to make the test run quickly
rna_sample_size = min(len(adata_rna_subset), 1000)
prot_sample_size = min(len(adata_prot_subset), 1000)
adata_rna_subset = sc.pp.subsample(adata_rna_subset, n_obs=rna_sample_size, copy=True)
adata_prot_subset = sc.pp.subsample(adata_prot_subset, n_obs=prot_sample_size, copy=True)

print(f"Subsampled RNA dataset shape: {adata_rna_subset.shape}")
print(f"Subsampled protein dataset shape: {adata_prot_subset.shape}")

# Import training utils
from CODEX_RNA_seq.training_utils import setup_and_train_model

# Define minimal test parameters
test_params = {
    "plot_x_times": 2,
    "max_epochs": 2,  # Just 2 epochs for quick testing
    "batch_size": 200,
    "lr": 1e-4,
    "contrastive_weight": 0,
    "similarity_weight": 100.0,
    "diversity_weight": 0.0,
    "matching_weight": 100.0,
    "cell_type_clustering_weight": 10.0,
    "n_hidden_rna": 32,
    "n_hidden_prot": 16,
    "n_layers": 2,
    "latent_dim": 8,
    "kl_weight_rna": 1,
    "kl_weight_prot": 1,
    "adv_weight": 0.0,
    "train_size": 0.8,
    "validation_size": 0.2,
    "check_val_every_n_epoch": 1,
    "gradient_clip_val": 1.0,
}

print("Starting test training...")
print(f"Parameters: {test_params}")

# Setup MLflow for tracking
mlflow.set_tracking_uri("file:./mlruns")
experiment_name = "test_training"
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
except Exception:
    experiment_id = mlflow.create_experiment(experiment_name)

mlflow.set_experiment(experiment_name)

# Run training
with mlflow.start_run(run_name="test_run"):
    # Log parameters
    mlflow.log_params(test_params)

    # Train model
    rna_vae, protein_vae, _, _ = setup_and_train_model(
        adata_rna_subset, adata_prot_subset, test_params
    )

    print("Training completed!")

    # Log basic metrics
    metrics = {
        "final_epoch": rna_vae._training_plan.current_epoch,
        "global_steps": rna_vae._training_plan.global_step,
    }

    mlflow.log_metrics(metrics)

print("Test finished!")

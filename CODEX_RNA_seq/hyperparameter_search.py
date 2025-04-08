# %%
"""Hyperparameter search for VAE training with archetypes vectors."""

import importlib.util
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from datetime import datetime

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
    "n_epochs": [5, 10],
    "batch_size": [500, 1000],
    "lr": [1e-3, 1e-4],
    "contrastive_weight": [1.0, 10.0, 100.0],
    "similarity_weight": [100.0, 1000.0, 10000.0],
    "diversity_weight": [0.1, 1.0],
    "matching_weight": [0.1, 1.0, 10.0],
    "n_hidden_rna": [64, 128, 256],
    "n_hidden_prot": [32, 64, 128],
    "n_layers": [1, 2, 3],
    "latent_dim": [10],
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
adata_rna_subset = sc.read_h5ad(
    get_latest_file(save_dir, "adata_rna_subset_prepared_for_training_")
)
adata_prot_subset = sc.read_h5ad(
    get_latest_file(save_dir, "adata_prot_subset_prepared_for_training_")
)

# %%
# Run hyperparameter search
results = []
for params in ParameterGrid(param_grid):
    run_name = f"vae_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(params)

        try:
            # Train model
            rna_vae, protein_vae = train_vae(
                adata_rna_subset=adata_rna_subset,
                adata_prot_subset=adata_prot_subset,
                use_gpu=True,
                **params,
            )

            # Get training history
            history = rna_vae._training_plan.get_history()

            # Calculate metrics
            final_train_loss = history["train_total_loss"][-1]
            final_val_loss = history["val_total_loss"][-1]
            final_similarity_loss = history["train_similarity_loss"][-1]

            # Log metrics
            mlflow.log_metrics(
                {
                    "final_train_loss": final_train_loss,
                    "final_val_loss": final_val_loss,
                    "final_similarity_loss": final_similarity_loss,
                }
            )

            # Store results
            results.append(
                {
                    **params,
                    "final_train_loss": final_train_loss,
                    "final_val_loss": final_val_loss,
                    "final_similarity_loss": final_similarity_loss,
                }
            )

            # Plot and save training curves
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(history["train_similarity_loss"], label="Similarity Loss")
            ax.plot(history["train_total_loss"], label="Total Loss")
            ax.plot(history["val_total_loss"], label="Validation Loss")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title("Training Curves")
            ax.legend()
            plt.tight_layout()
            save_and_log_figure(fig, "training_curves")

            # Plot and save latent space visualizations
            fig = sc.pl.umap(rna_vae.adata, color=["CN", "cell_types"], return_fig=True, show=False)
            save_and_log_figure(fig, "rna_umap")

            fig = sc.pl.umap(
                protein_vae.adata, color=["CN", "cell_types"], return_fig=True, show=False
            )
            save_and_log_figure(fig, "protein_umap")

            # Plot and save latent space distributions
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(rna_vae.get_latent_representation().flatten(), bins=50, alpha=0.5, label="RNA")
            ax.hist(
                protein_vae.get_latent_representation().flatten(),
                bins=50,
                alpha=0.5,
                label="Protein",
            )
            ax.set_xlabel("Latent Value")
            ax.set_ylabel("Count")
            ax.set_title("Latent Space Distributions")
            ax.legend()
            plt.tight_layout()
            save_and_log_figure(fig, "latent_distributions")

        except Exception as e:
            print(f"Error in run {run_name}: {str(e)}")
            mlflow.log_param("error", str(e))
            continue

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
# Train final model with best parameters
final_run_name = f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
with mlflow.start_run(run_name=final_run_name):
    mlflow.log_params(best_params.to_dict())

    rna_vae, protein_vae = train_vae(
        adata_rna_subset=adata_rna_subset,
        adata_prot_subset=adata_prot_subset,
        use_gpu=True,
        **best_params.to_dict(),
    )

    # Save final models
    save_dir = Path("CODEX_RNA_seq/data/trained_data").absolute()
    time_stamp = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(save_dir, exist_ok=True)

    sc.write(Path(f"{save_dir}/rna_vae_final_{time_stamp}.h5ad"), rna_vae.adata)
    sc.write(Path(f"{save_dir}/protein_vae_final_{time_stamp}.h5ad"), protein_vae.adata)

    mlflow.log_artifacts(save_dir)

    # Plot and save final model visualizations
    fig = sc.pl.umap(rna_vae.adata, color=["CN", "cell_types"], return_fig=True, show=False)
    save_and_log_figure(fig, "final_rna_umap")

    fig = sc.pl.umap(protein_vae.adata, color=["CN", "cell_types"], return_fig=True, show=False)
    save_and_log_figure(fig, "final_protein_umap")

    # Plot and save final latent space distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(rna_vae.get_latent_representation().flatten(), bins=50, alpha=0.5, label="RNA")
    ax.hist(protein_vae.get_latent_representation().flatten(), bins=50, alpha=0.5, label="Protein")
    ax.set_xlabel("Latent Value")
    ax.set_ylabel("Count")
    ax.set_title("Final Latent Space Distributions")
    ax.legend()
    plt.tight_layout()
    save_and_log_figure(fig, "final_latent_distributions")

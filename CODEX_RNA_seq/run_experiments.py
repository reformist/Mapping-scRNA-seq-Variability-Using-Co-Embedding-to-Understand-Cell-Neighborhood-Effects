# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: scvi
#     language: python
#     name: python3
# ---

import os
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.metrics import adjusted_mutual_info_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the training function from the original script
import importlib.util

from bar_nick_utils import clean_uns_for_h5ad, mixing_score
from CODEX_RNA_seq.configs.vae_training_configs import config_to_dict, generate_configs
from CODEX_RNA_seq.plotting_functions_vae import (
    plot_archetype_vectors,
    plot_cell_type_distributions,
    plot_combined_latent_space,
    plot_combined_latent_space_umap,
    plot_latent,
    plot_normalized_losses,
    plot_rna_protein_embeddings,
    plot_spatial_data,
)

spec = importlib.util.spec_from_file_location(
    "train_vae_module", "CODEX_RNA_seq/3_train_vae_with_archetypes_vectors.py"
)
train_vae_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_vae_module)
train_vae = train_vae_module.train_vae


def save_and_log_plot(plt, name, mlflow):
    """Save plot to temporary file and log to MLflow"""
    temp_path = f"/tmp/{name}.png"
    plt.savefig(temp_path, bbox_inches="tight", dpi=300)
    plt.close()
    mlflow.log_artifact(temp_path)
    os.remove(temp_path)


def run_experiment(config, experiment_dir, adata_rna, adata_prot):
    """Run a single experiment with given configuration"""
    print(f"\nStarting experiment with config: {config}")

    # Create experiment directory
    exp_dir = experiment_dir / f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Start MLflow run
    with mlflow.start_run(run_name=f"vae_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_params(config_to_dict(config))

        # Apply data subset if specified
        if hasattr(config, "data_subset") and config.data_subset < 1.0:
            print(f"Using {config.data_subset*100}% of the data for quick testing")
            # Randomly sample cells
            n_cells = int(len(adata_rna) * config.data_subset)
            indices = np.random.choice(len(adata_rna), size=n_cells, replace=False)
            adata_rna_subset = adata_rna[indices].copy()
            adata_prot_subset = adata_prot[indices].copy()
        else:
            adata_rna_subset = adata_rna
            adata_prot_subset = adata_prot

        # Train model with fixed parameters and varying loss weights
        rna_vae, protein_vae = train_vae(
            adata_rna_subset=adata_rna_subset,
            adata_prot_subset=adata_prot_subset,
            n_epochs=10,  # Reduced epochs for testing
            batch_size=256,  # Smaller batch size
            lr=1e-4,  # Lower learning rate
            use_gpu=True,
            # Varying loss weights
            matching_weight=config.matching_weight,
            similarity_weight=config.similarity_weight,
            contrastive_weight=config.contrastive_weight,
            reconstruction_weight=config.reconstruction_weight,
            # Fixed architecture parameters
            n_hidden_rna=64,  # Smaller network
            n_hidden_prot=32,
            n_layers=2,
            latent_dim=10,
        )

        # Generate metrics
        with torch.no_grad():
            latent_rna = rna_vae.get_latent_representation()
            latent_prot = protein_vae.get_latent_representation()

        # Calculate metrics
        mixing_result = mixing_score(
            latent_rna,
            latent_prot,
            rna_vae.adata,
            protein_vae.adata,
            index=range(len(rna_vae.adata)),
            plot_flag=True,
        )

        nmi_cell_types_cn_rna = adjusted_mutual_info_score(
            rna_vae.adata.obs["cell_types"], rna_vae.adata.obs["CN"]
        )
        nmi_cell_types_cn_prot = adjusted_mutual_info_score(
            protein_vae.adata.obs["cell_types"], protein_vae.adata.obs["CN"]
        )
        nmi_cell_types_modalities = adjusted_mutual_info_score(
            rna_vae.adata.obs["cell_types"], protein_vae.adata.obs["cell_types"]
        )
        matches = (
            rna_vae.adata.obs["cell_types"].values == protein_vae.adata.obs["cell_types"].values
        )
        accuracy = matches.sum() / len(matches)

        # Log metrics
        metrics = {
            "mixing_score": mixing_result,
            "nmi_cell_types_cn_rna": nmi_cell_types_cn_rna,
            "nmi_cell_types_cn_prot": nmi_cell_types_cn_prot,
            "nmi_cell_types_modalities": nmi_cell_types_modalities,
            "cell_type_matching_accuracy": accuracy,
        }
        mlflow.log_metrics(metrics)

        # Save models
        clean_uns_for_h5ad(rna_vae.adata)
        clean_uns_for_h5ad(protein_vae.adata)

        # Log artifacts
        rna_vae.adata.write(exp_dir / "rna_vae_trained.h5ad")
        protein_vae.adata.write(exp_dir / "protein_vae_trained.h5ad")
        mlflow.log_artifacts(str(exp_dir))

        # Generate and log all plots
        # 1. Training losses
        plot_normalized_losses(rna_vae._training_plan.get_history())
        save_and_log_plot(plt, "normalized_losses", mlflow)

        # 2. Spatial data
        plot_spatial_data(rna_vae.adata, protein_vae.adata)
        save_and_log_plot(plt, "spatial_data", mlflow)

        # 3. Latent representations
        plot_latent(
            latent_rna,
            latent_prot,
            rna_vae.adata,
            protein_vae.adata,
            index=range(len(protein_vae.adata.obs.index)),
        )
        save_and_log_plot(plt, "latent_representations", mlflow)

        # 4. Combined latent space
        combined_latent = ad.concat(
            [rna_vae.adata.copy(), protein_vae.adata.copy()],
            join="outer",
            label="modality",
            keys=["RNA", "Protein"],
        )
        combined_latent.obsm["X_scVI"] = np.vstack([latent_rna, latent_prot])

        sc.pp.pca(combined_latent)
        sc.pp.neighbors(combined_latent)
        sc.tl.umap(combined_latent)

        plot_combined_latent_space(combined_latent)
        save_and_log_plot(plt, "combined_latent_space", mlflow)

        plot_combined_latent_space_umap(combined_latent)
        save_and_log_plot(plt, "combined_latent_space_umap", mlflow)

        # 5. Cell type distributions
        plot_cell_type_distributions(combined_latent, 3)
        save_and_log_plot(plt, "cell_type_distributions", mlflow)

        # 6. Archetype vectors
        plot_archetype_vectors(rna_vae, protein_vae)
        save_and_log_plot(plt, "archetype_vectors", mlflow)

        # 7. RNA-Protein embeddings
        plot_rna_protein_embeddings(rna_vae, protein_vae)
        save_and_log_plot(plt, "rna_protein_embeddings", mlflow)

        # 8. Additional UMAP plots
        sc.pl.umap(
            combined_latent,
            color=["CN", "modality"],
            title=["UMAP Combined Latent space CN", "UMAP Combined Latent space modality"],
            alpha=0.5,
            show=False,
        )
        save_and_log_plot(plt, "umap_cn_modality", mlflow)

        sc.pl.umap(
            combined_latent,
            color=["CN", "modality", "cell_types"],
            title=[
                "UMAP Combined Latent space CN",
                "UMAP Combined Latent space modality",
                "UMAP Combined Latent space cell types",
            ],
            alpha=0.5,
            show=False,
        )
        save_and_log_plot(plt, "umap_all_features", mlflow)

        # Log training history
        history = rna_vae._training_plan.get_history()
        for key, values in history.items():
            for i, value in enumerate(values):
                mlflow.log_metric(f"{key}_step_{i}", value)

        return metrics


def main():
    """
    Run VAE experiments with hyperparameter search.

    Command line arguments:
        --quick-test: Run with a subset (10%) of the data for quick testing
    """
    # Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    experiment_name = f"vae_hyperparameter_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)

    # Setup paths
    data_dir = Path("CODEX_RNA_seq/data/processed_data")
    experiment_dir = Path("CODEX_RNA_seq/experiments")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Run VAE experiments")
    parser.add_argument(
        "--quick-test", action="store_true", help="Run with a subset of data for quick testing"
    )
    args = parser.parse_args()

    # Load data
    adata_rna = sc.read_h5ad(data_dir / "adata_rna_subset_prepared_for_training.h5ad")
    adata_prot = sc.read_h5ad(data_dir / "adata_prot_subset_prepared_for_training.h5ad")

    # Generate configurations
    configs = generate_configs(quick_test=args.quick_test)
    print(f"Generated {len(configs)} configurations")
    if args.quick_test:
        print("Running in quick test mode with 10% of the data")

    # Run experiments
    results = []
    for i, config in enumerate(configs):
        print(f"\nRunning experiment {i+1}/{len(configs)}")
        try:
            metrics = run_experiment(config, experiment_dir, adata_rna, adata_prot)
            results.append({**config_to_dict(config), **metrics})
        except Exception as e:
            print(f"Error in experiment {i+1}: {str(e)}")
            continue

    # Save all results
    results_df = pd.DataFrame(results)
    results_df.to_csv(experiment_dir / "all_results.csv", index=False)

    # Print summary
    print("\nExperiment Summary:")
    print(f"Total experiments: {len(configs)}")
    print(f"Successful experiments: {len(results)}")
    print(f"Failed experiments: {len(configs) - len(results)}")

    # Print best results
    print("\nBest Results:")
    for metric in ["mixing_score", "nmi_cell_types_modalities", "cell_type_matching_accuracy"]:
        best = results_df.loc[results_df[metric].idxmax()]
        print(f"\nBest {metric}:")
        print(f"Value: {best[metric]:.4f}")
        print("Configuration:")
        for key, value in best.items():
            if key != metric:
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()

# %%
"""Train VAE with archetypes vectors."""

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

# %%
""" DO NOT REMOVE THIS COMMENT!!!
TO use this script, you need to add the training plan to use the DualVAETrainingPlan (version 1.2.2.post2) class in scVI library.
in _training_mixin.py, line 131, you need to change the line:
training_plan = self._training_plan_cls(self.module, **plan_kwargs) # existing line
self._training_plan = training_plan # add this line

"""
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from pprint import pprint

import anndata
import mlflow
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.preprocessing import normalize

# Add repository root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set working directory to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from pathlib import Path

import plotting_functions as pf

import bar_nick_utils
import CODEX_RNA_seq.logging_functions
import CODEX_RNA_seq.metrics
import CODEX_RNA_seq.training_utils

importlib.reload(pf)
importlib.reload(bar_nick_utils)
importlib.reload(CODEX_RNA_seq.logging_functions)
importlib.reload(CODEX_RNA_seq.metrics)
importlib.reload(CODEX_RNA_seq.training_utils)

# Import training utilities
from CODEX_RNA_seq.training_utils import (
    Tee,
    calculate_metrics,
    clear_memory,
    generate_visualizations,
    log_memory_usage,
    log_parameters,
    match_cells_and_calculate_distances,
    process_latent_spaces,
    run_cell_type_clustering_loss,
    save_results,
    setup_and_train_model,
)


def validate_scvi_training_mixin():
    """Validate that the required line exists in scVI's _training_mixin.py file."""
    try:
        # Import scvi to get the actual module path
        import scvi

        scvi_path = scvi.__file__
        base_dir = os.path.dirname(os.path.dirname(scvi_path))
        training_mixin_path = os.path.join(base_dir, "scvi", "model", "base", "_training_mixin.py")

        if not os.path.exists(training_mixin_path):
            raise FileNotFoundError(f"Could not find _training_mixin.py at {training_mixin_path}")

        # Read the file
        with open(training_mixin_path, "r") as f:
            lines = f.readlines()

        # Check if the required line exists
        required_line = "self._training_plan = training_plan"
        line_found = any(required_line in line for line in lines)

        if not line_found:
            raise RuntimeError(
                f"Required line '{required_line}' not found in {training_mixin_path}. "
                "Please add this line after the training_plan assignment."
            )
        print("✓ scVI training mixin validation passed")
    except Exception as e:
        raise RuntimeError(
            f"Failed to validate scVI training mixin: {str(e)}\n"
            "Please ensure you have modified the scVI library as described in the comment above."
        )


# Validate scVI training mixin before proceeding
validate_scvi_training_mixin()

# Set up paths once
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

import warnings
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import plotting_functions as pf
import scanpy as sc
import scvi
import torch
from anndata import AnnData
from scipy.sparse import issparse
from scvi.model import SCVI
from scvi.train import TrainingPlan
from torch.nn.functional import normalize

import bar_nick_utils

# Force reimport internal modules
importlib.reload(pf)
importlib.reload(bar_nick_utils)
import CODEX_RNA_seq.logging_functions

importlib.reload(CODEX_RNA_seq.logging_functions)

# Force reimport logging functions
import CODEX_RNA_seq.logging_functions
from CODEX_RNA_seq.logging_functions import log_epoch_end, log_step

importlib.reload(CODEX_RNA_seq.logging_functions)

from plotting_functions import (
    plot_latent_pca_both_modalities_by_celltype,
    plot_latent_pca_both_modalities_cn,
    plot_rna_protein_matching_means_and_scale,
    plot_similarity_loss_history,
)

from bar_nick_utils import (
    clean_uns_for_h5ad,
    compute_pairwise_kl,
    compute_pairwise_kl_two_items,
    get_latest_file,
    get_umap_filtered_fucntion,
    select_gene_likelihood,
)

if not hasattr(sc.tl.umap, "_is_wrapped"):
    sc.tl.umap = get_umap_filtered_fucntion()
    sc.tl.umap._is_wrapped = True
np.random.seed(42)
torch.manual_seed(42)
pd.set_option("display.max_columns", 10)
pd.set_option("display.max_rows", 10)
warnings.filterwarnings("ignore")
pd.options.display.max_rows = 10
pd.options.display.max_columns = 10
np.set_printoptions(threshold=100)

np.random.seed(0)

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    num_rna_cells = config["subsample"]["num_rna_cells"]
    num_protein_cells = config["subsample"]["num_protein_cells"]
    plot_flag = config["plot_flag"]
else:
    num_rna_cells = num_protein_cells = 2000
    plot_flag = True
# Subsample to 20% of the cells for testing
# For reproducibility


# %%
# Define the DualVAETrainingPlan class
class DualVAETrainingPlan(TrainingPlan):
    def __init__(self, rna_module, **kwargs):
        protein_vae = kwargs.pop("protein_vae")
        rna_vae = kwargs.pop("rna_vae")
        self.plot_x_times = kwargs.pop("plot_x_times", 5)
        contrastive_weight = kwargs.pop("contrastive_weight", 1.0)
        self.batch_size = kwargs.pop("batch_size", 1000)
        max_epochs = kwargs.pop("max_epochs", 1)
        self.similarity_weight = kwargs.pop("similarity_weight")
        self.cell_type_clustering_weight = kwargs.pop("cell_type_clustering_weight", 1000.0)
        self.lr = kwargs.pop("lr", 0.001)
        self.kl_weight_rna = kwargs.pop("kl_weight_rna", 1.0)
        self.kl_weight_prot = kwargs.pop("kl_weight_prot", 1.0)
        self.matching_weight = kwargs.pop("matching_weight", 1000.0)
        train_size = kwargs.pop("train_size", 0.9)
        self.validation_step_ = 0
        validation_size = kwargs.pop("validation_size", 0.1)
        device = kwargs.pop("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        # Verify train and validation sizes sum to 1
        self.metrics_history = []

        if abs(train_size + validation_size - 1.0) > 1e-6:
            raise ValueError("train_size + validation_size must sum to 1.0")

        super().__init__(rna_module, **kwargs)
        self.rna_vae = rna_vae
        self.protein_vae = protein_vae

        # Create train/validation splits
        n_rna = len(self.rna_vae.adata)
        n_prot = len(self.protein_vae.adata)

        # Create indices for RNA data
        rna_indices = np.arange(n_rna)
        np.random.shuffle(rna_indices)
        n_train_rna = int(n_rna * train_size)
        self.train_indices_rna = rna_indices[:n_train_rna]
        self.val_indices_rna = rna_indices[n_train_rna:]

        # Create indices for protein data
        prot_indices = np.arange(n_prot)
        np.random.shuffle(prot_indices)
        n_train_prot = int(n_prot * train_size)
        self.train_indices_prot = prot_indices[:n_train_prot]
        self.val_indices_prot = prot_indices[n_train_prot:]

        num_batches = 2
        latent_dim = self.rna_vae.module.n_latent
        self.batch_classifier = torch.nn.Linear(latent_dim, num_batches)
        self.contrastive_weight = contrastive_weight
        self.protein_vae.module.to(device)
        self.rna_vae.module = self.rna_vae.module.to(device)
        self.first_step = True

        n_samples = len(self.train_indices_rna)  # Use training set size
        steps_per_epoch = int(np.ceil(n_samples / self.batch_size))
        self.total_steps = steps_per_epoch * (max_epochs)
        self.similarity_loss_history = []
        self.steady_state_window = 5
        self.steady_state_tolerance = 0.5
        self.similarity_active = True
        self.reactivation_threshold = 0.1
        self.active_similarity_loss_active_history = []
        self.train_losses = []
        self.val_losses = []
        self.similarity_losses = []  # Store similarity losses
        self.similarity_losses_raw = []  # Store raw similarity losses
        self.similarity_weight_history = []  # Store similarity weights
        self.train_rna_losses = []
        self.train_protein_losses = []
        self.train_matching_losses = []
        self.train_contrastive_losses = []
        self.train_adv_losses = []
        self.train_cell_type_clustering_losses = []  # New list for cell type clustering losses
        self.val_rna_losses = []
        self.val_protein_losses = []
        self.val_matching_losses = []
        self.val_contrastive_losses = []
        self.val_adv_losses = []
        self.val_cell_type_clustering_losses = []  # New list for cell type clustering losses
        self.early_stopping_callback = None  # Will be set by trainer

        # Setup logging

        # Create run directory for checkpoint saves
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = Path(f"CODEX_RNA_seq/data/checkpoints/run_{self.run_timestamp}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Save training parameters
        self.save_training_parameters(kwargs)

        self.pbar = None
        self.val_pbar = None

    def save_training_parameters(self, kwargs):
        """Save training parameters to a JSON file in the checkpoint directory."""
        # Add the parameters from self that aren't in kwargs
        params = {
            "batch_size": self.batch_size,
            "max_epochs": self.total_steps
            // int(np.ceil(len(self.rna_vae.adata) / self.batch_size)),
            "similarity_weight": self.similarity_weight,
            "cell_type_clustering_weight": self.cell_type_clustering_weight,
            "lr": self.lr,
            "kl_weight_rna": self.kl_weight_rna,
            "kl_weight_prot": self.kl_weight_prot,
            "contrastive_weight": self.contrastive_weight,
            "plot_x_times": self.plot_x_times,
            "steady_state_window": self.steady_state_window,
            "steady_state_tolerance": self.steady_state_tolerance,
            "reactivation_threshold": self.reactivation_threshold,
            "rna_adata_shape": list(self.rna_vae.adata.shape),
            "protein_adata_shape": list(self.protein_vae.adata.shape),
            "latent_dim": self.rna_vae.module.n_latent,
            "device": str(self.device),
            "timestamp": self.run_timestamp,
        }

        # Remove non-serializable objects
        params_to_save = {
            k: v for k, v in params.items() if isinstance(v, (str, int, float, bool, list, dict))
        }

        # Save parameters to JSON file
        with open(f"{self.checkpoint_dir}/training_parameters.json", "w") as f:
            json.dump(params_to_save, f, indent=4)

        print(f"✓ Training parameters saved to {self.checkpoint_dir}/training_parameters.json")

        # Also save parameters to a separate txt file in readable format instead of appending to the log file
        with open(f"{self.checkpoint_dir}/training_parameters.txt", "w") as f:
            f.write("Training Parameters:\n")
            for key, value in params_to_save.items():
                f.write(f"{key}: {value}\n")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.rna_vae.module.parameters()) + list(self.protein_vae.module.parameters()),
            lr=self.lr,
            weight_decay=1e-5,
        )
        d = {
            "optimizer": optimizer,
            "gradient_clip_val": 1.0,  # Critical for stability
            "gradient_clip_algorithm": "value",
        }
        return d

    def training_step(self, batch, batch_idx):
        indices = range(self.batch_size)
        indices_rna = np.random.choice(
            self.train_indices_rna,
            size=len(indices),
            replace=True if len(indices) > len(self.train_indices_rna) else False,
        )
        indices_rna = np.sort(indices_rna)
        indices_prot = np.random.choice(
            self.train_indices_prot,
            size=len(indices),
            replace=True if len(indices) > len(self.train_indices_prot) else False,
        )
        indices_prot = np.sort(indices_prot)

        rna_batch = self._get_rna_batch(batch, indices_rna)
        protein_batch = self._get_protein_batch(batch, indices_prot)

        # Calculate all losses using the new function
        losses = calculate_losses(
            rna_batch,
            protein_batch,
            self.rna_vae,
            self.protein_vae,
            self.device,
            self.similarity_weight,
            self.similarity_active,
            self.contrastive_weight,
            self.matching_weight,
            self.cell_type_clustering_weight,
            self.kl_weight_rna,
            self.kl_weight_prot,
            plot_flag=plot_flag,
            global_step=self.global_step,
            total_steps=self.total_steps,
            plot_x_times=self.plot_x_times,
        )

        # Update similarity loss history
        if len(self.similarity_loss_history) >= self.steady_state_window:
            self.similarity_loss_history.pop(0)
        self.similarity_loss_history.append(losses["similarity_loss_raw"])

        # Store losses in history
        self.similarity_losses.append(losses["similarity_loss"].item())
        self.similarity_losses_raw.append(losses["similarity_loss_raw"].item())
        self.similarity_weight_history.append(self.similarity_weight)
        self.active_similarity_loss_active_history.append(self.similarity_active)
        self.train_losses.append(losses["total_loss"].item())
        self.train_rna_losses.append(losses["rna_loss"].item())
        self.train_protein_losses.append(losses["protein_loss"].item())
        self.train_matching_losses.append(losses["matching_loss"].item())
        self.train_contrastive_losses.append(losses["contrastive_loss"].item())
        self.train_cell_type_clustering_losses.append(losses["cell_type_clustering_loss"].item())

        # Log metrics
        if plot_flag and self.global_step % (1 + int(self.total_steps / self.plot_x_times)) == 0:
            plot_similarity_loss_history(
                self.similarity_losses, self.active_similarity_loss_active_history, self.global_step
            )

        log_step(
            losses,
            metrics=None,
            global_step=self.global_step,
            current_epoch=self.current_epoch,
            is_validation=False,
            total_steps=self.total_steps,
            print_to_console=self.global_step % 50 == 0,
        )

        return losses["total_loss"]

    def validation_step(self, batch, batch_idx):
        """Validation step using the same loss calculations as training.

        Args:
            batch: The batch of data
            batch_idx: The index of the batch

        Returns:
            The total validation loss
        """
        # Get validation batches
        indices = range(self.batch_size)
        self.validation_step_ += 1
        indices_prot = np.random.choice(
            self.val_indices_prot,
            size=len(indices),
            replace=True if len(indices) > len(self.val_indices_prot) else False,
        )
        indices_prot = np.sort(indices_prot)
        indices_rna = np.random.choice(
            self.val_indices_rna,
            size=len(indices),
            replace=True if len(indices) > len(self.val_indices_rna) else False,
        )
        indices_rna = np.sort(indices_rna)

        rna_batch = self._get_rna_batch(batch, indices_rna)
        protein_batch = self._get_protein_batch(batch, indices_prot)

        # Calculate all losses using the same function as training
        losses = calculate_losses(
            rna_batch,
            protein_batch,
            self.rna_vae,
            self.protein_vae,
            self.device,
            self.similarity_weight,
            self.similarity_active,
            self.contrastive_weight,
            self.matching_weight,
            self.cell_type_clustering_weight,
            self.kl_weight_rna,
            self.kl_weight_prot,
        )

        # Store validation losses
        self.val_losses.append(losses["total_loss"].item())
        self.val_rna_losses.append(losses["rna_loss"].item())
        self.val_protein_losses.append(losses["protein_loss"].item())
        self.val_matching_losses.append(losses["matching_loss"].item())
        self.val_contrastive_losses.append(losses["contrastive_loss"].item())
        self.val_cell_type_clustering_losses.append(losses["cell_type_clustering_loss"].item())

        # Log validation metrics
        # self.log("val_total_loss", losses["total_loss"], on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val_rna_loss", losses["rna_loss"], on_step=False, on_epoch=True)
        # self.log("val_protein_loss", losses["protein_loss"], on_step=False, on_epoch=True)
        # self.log("val_matching_loss", losses["matching_loss"], on_step=False, on_epoch=True)
        # self.log("val_contrastive_loss", losses["contrastive_loss"], on_step=False, on_epoch=True)
        # self.log("val_cell_type_clustering_loss", losses["cell_type_clustering_loss"], on_step=False, on_epoch=True)
        metrics = {}

        log_step(
            losses,
            metrics=metrics,
            global_step=self.global_step,
            current_epoch=self.current_epoch,
            is_validation=True,
            total_steps=self.total_steps,
            print_to_console=self.validation_step_ % 10 == 0,
        )

        return losses["total_loss"]

    def calculate_metrics_for_data(self, rna_adata, prot_adata, prefix="", subsample_size=None):
        """Calculate metrics for given RNA and protein data.

        Args:
            rna_adata: RNA AnnData object
            prot_adata: Protein AnnData object
            prefix: Prefix for metric names (e.g., "train_" or "val_")
            subsample_size: If not None, subsample the data to this size
        """
        print(f"   Calculating {prefix}metrics...")

        # Subsample if requested
        if subsample_size is not None:
            rna_adata = sc.pp.subsample(rna_adata, n_obs=subsample_size, copy=True)
            prot_adata = sc.pp.subsample(prot_adata, n_obs=subsample_size, copy=True)
            print(f"   ✓ Subsampled to {subsample_size} cells")

        # Get latent representations
        with torch.no_grad():
            rna_data = rna_adata.X
            if issparse(rna_data):
                rna_data = rna_data.toarray()
            rna_tensor = torch.tensor(rna_data, dtype=torch.float32).to(self.device)
            rna_batch = torch.tensor(rna_adata.obs["_scvi_batch"].values, dtype=torch.long).to(
                self.device
            )

            rna_inference = self.rna_vae.module.inference(
                rna_tensor, batch_index=rna_batch, n_samples=1
            )
            rna_latent = rna_inference["qz"].mean.detach().cpu().numpy()

            prot_data = prot_adata.X
            if issparse(prot_data):
                prot_data = prot_data.toarray()
            prot_tensor = torch.tensor(prot_data, dtype=torch.float32).to(self.device)
            prot_batch = torch.tensor(prot_adata.obs["_scvi_batch"].values, dtype=torch.long).to(
                self.device
            )

            prot_inference = self.protein_vae.module.inference(
                prot_tensor, batch_index=prot_batch, n_samples=1
            )
            prot_latent = prot_inference["qz"].mean.detach().cpu().numpy()

        # Create AnnData objects
        rna_latent_adata = AnnData(rna_latent)
        prot_latent_adata = AnnData(prot_latent)
        rna_latent_adata.obs = rna_adata.obs.copy()
        prot_latent_adata.obs = prot_adata.obs.copy()

        # Calculate matching accuracy
        accuracy = CODEX_RNA_seq.metrics.matching_accuracy(rna_latent_adata, prot_latent_adata)
        print(f"   ✓ {prefix}matching accuracy calculated")

        # Calculate silhouette F1
        silhouette_f1 = CODEX_RNA_seq.metrics.compute_silhouette_f1(
            rna_latent_adata, prot_latent_adata
        )
        print(f"   ✓ {prefix}silhouette F1 calculated")

        # Calculate ARI F1
        combined_latent = anndata.concat(
            [rna_latent_adata, prot_latent_adata],
            join="outer",
            label="modality",
            keys=["RNA", "Protein"],
        )
        sc.pp.pca(combined_latent)
        sc.pp.neighbors(combined_latent, n_neighbors=10)
        ari_f1 = CODEX_RNA_seq.metrics.compute_ari_f1(combined_latent)
        print(f"   ✓ {prefix}ARI F1 calculated")

        return {
            f"{prefix}cell_type_matching_accuracy": accuracy,
            f"{prefix}silhouette_f1_score": silhouette_f1.mean(),
            f"{prefix}ari_f1_score": ari_f1,
        }

    def on_validation_epoch_end(self):
        """Calculate and store metrics at the end of each validation epoch."""
        print(f"\nProcessing validation epoch {self.current_epoch}...")
        self.validation_step_ = 0
        # Calculate validation metrics
        val_metrics = self.calculate_metrics_for_data(
            self.rna_vae.adata[self.val_indices_rna],
            self.protein_vae.adata[self.val_indices_prot],
            prefix="val_",
        )

        # Calculate training metrics with subsampling
        print("7. Calculating training metrics...")
        train_metrics = self.calculate_metrics_for_data(
            self.rna_vae.adata[self.train_indices_rna],
            self.protein_vae.adata[self.train_indices_prot],
            prefix="train_",
            subsample_size=len(self.val_indices_rna),  # Use validation set size for subsampling
        )

        # Combine metrics
        epoch_metrics = {**val_metrics, **train_metrics}

        # Store in history
        self.metrics_history.append(epoch_metrics)
        print("   ✓ Metrics stored")

        # Log metrics to MLflow
        mlflow.log_metrics(epoch_metrics, step=self.current_epoch)

        print(f"✓ Validation epoch {self.current_epoch} completed successfully!")

    def on_train_end(self, plot_flag=True):
        """Called when training ends."""
        print("\nTraining completed!")

        # Get final latent representations
        with torch.no_grad():
            # Get RNA latent
            rna_data = self.rna_vae.adata.X
            if issparse(rna_data):
                rna_data = rna_data.toarray()
            rna_tensor = torch.tensor(rna_data, dtype=torch.float32).to(self.device)
            rna_batch = torch.tensor(
                self.rna_vae.adata.obs["_scvi_batch"].values, dtype=torch.long
            ).to(self.device)

            rna_inference = self.rna_vae.module.inference(
                rna_tensor, batch_index=rna_batch, n_samples=1
            )
            rna_latent = rna_inference["qz"].mean.detach().cpu().numpy()

            # Get protein latent
            prot_data = self.protein_vae.adata.X
            if issparse(prot_data):
                prot_data = prot_data.toarray()
            prot_tensor = torch.tensor(prot_data, dtype=torch.float32).to(self.device)
            prot_batch = torch.tensor(
                self.protein_vae.adata.obs["_scvi_batch"].values, dtype=torch.long
            ).to(self.device)

            prot_inference = self.protein_vae.module.inference(
                prot_tensor, batch_index=prot_batch, n_samples=1
            )
            prot_latent = prot_inference["qz"].mean.detach().cpu().numpy()

        # Store in adata
        self.rna_vae.adata.obsm["X_scVI"] = rna_latent
        self.protein_vae.adata.obsm["X_scVI"] = prot_latent

        # Plot metrics over time
        if plot_flag and hasattr(self, "metrics_history"):
            pf.plot_training_metrics_history(
                self.metrics_history,
            )

        # Find best metrics
        if hasattr(self, "metrics_history"):
            best_metrics = {}
            for metric in self.metrics_history[0].keys():
                values = [epoch[metric] for epoch in self.metrics_history]
                if "loss" in metric or "error" in metric:
                    best_metrics[metric] = min(values)
                else:
                    best_metrics[metric] = max(values)

            # Log best metrics to MLflow
            mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()})
            print("✓ Best metrics logged to MLflow")

        # Save model checkpoints
        rna_adata_save = self.rna_vae.adata.copy()
        protein_adata_save = self.protein_vae.adata.copy()

        # Clean for h5ad saving
        clean_uns_for_h5ad(rna_adata_save)
        clean_uns_for_h5ad(protein_adata_save)

        # Save the AnnData objects
        checkpoint_path = f"{self.checkpoint_dir}/epoch_{self.current_epoch}"
        os.makedirs(checkpoint_path, exist_ok=True)

        sc.write(f"{checkpoint_path}/rna_adata_epoch_{self.current_epoch}.h5ad", rna_adata_save)
        sc.write(
            f"{checkpoint_path}/protein_adata_epoch_{self.current_epoch}.h5ad", protein_adata_save
        )

    def save_checkpoint(self):
        """Save the model checkpoint including AnnData objects with latent representations."""
        print(f"\nSaving checkpoint at epoch {self.current_epoch}...")

        # Get latent representations with the current model state
        with torch.no_grad():
            # Get RNA latent
            rna_data = self.rna_vae.adata.X
            if issparse(rna_data):
                rna_data = rna_data.toarray()
            rna_tensor = torch.tensor(rna_data, dtype=torch.float32).to(self.device)
            rna_batch = torch.tensor(
                self.rna_vae.adata.obs["_scvi_batch"].values, dtype=torch.long
            ).to(self.device)

            rna_inference = self.rna_vae.module.inference(
                rna_tensor, batch_index=rna_batch, n_samples=1
            )
            rna_latent = rna_inference["qz"].mean.detach().cpu().numpy()

            # Get protein latent
            prot_data = self.protein_vae.adata.X
            if issparse(prot_data):
                prot_data = prot_data.toarray()
            prot_tensor = torch.tensor(prot_data, dtype=torch.float32).to(self.device)
            prot_batch = torch.tensor(
                self.protein_vae.adata.obs["_scvi_batch"].values, dtype=torch.long
            ).to(self.device)

            prot_inference = self.protein_vae.module.inference(
                prot_tensor, batch_index=prot_batch, n_samples=1
            )
            prot_latent = prot_inference["qz"].mean.detach().cpu().numpy()

        # Store in adata
        self.rna_vae.adata.obsm["X_scVI"] = rna_latent
        self.protein_vae.adata.obsm["X_scVI"] = prot_latent

        # Make copies to avoid modifying the originals during saving
        rna_adata_save = self.rna_vae.adata.copy()
        protein_adata_save = self.protein_vae.adata.copy()

        # Clean for h5ad saving
        clean_uns_for_h5ad(rna_adata_save)
        clean_uns_for_h5ad(protein_adata_save)

        # Save the AnnData objects
        checkpoint_path = f"{self.checkpoint_dir}/epoch_{self.current_epoch}"
        os.makedirs(checkpoint_path, exist_ok=True)

        sc.write(f"{checkpoint_path}/rna_adata_epoch_{self.current_epoch}.h5ad", rna_adata_save)
        sc.write(
            f"{checkpoint_path}/protein_adata_epoch_{self.current_epoch}.h5ad", protein_adata_save
        )

        # Save loss history

        print(f"✓ Checkpoint saved at {checkpoint_path}")

        # Update the log file
        print(f"\nCheckpoint saved at epoch {self.current_epoch}\n")
        print(f"Location: {checkpoint_path}\n")
        print(f"RNA dataset shape: {rna_adata_save.shape}\n")
        print(f"Protein dataset shape: {protein_adata_save.shape}\n")

    def _get_protein_batch(self, batch, indices):
        protein_data = self.protein_vae.adata[indices]
        X = protein_data.X
        if issparse(X):
            X = X.toarray()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        batch_indices = torch.tensor(protein_data.obs["_scvi_batch"].values, dtype=torch.long).to(
            self.device
        )
        archetype_vec = torch.tensor(
            protein_data.obsm["archetype_vec"].values, dtype=torch.float32
        ).to(self.device)

        protein_batch = {
            "X": X,
            "batch": batch_indices,
            "labels": indices,
            "archetype_vec": archetype_vec,
        }
        return protein_batch

    def _get_rna_batch(self, batch, indices):
        rna_data = self.rna_vae.adata[indices]
        X = rna_data.X
        if issparse(X):
            X = X.toarray()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        batch_indices = torch.tensor(rna_data.obs["_scvi_batch"].values, dtype=torch.long).to(
            self.device
        )
        archetype_vec = torch.tensor(rna_data.obsm["archetype_vec"].values, dtype=torch.float32).to(
            self.device
        )

        rna_batch = {
            "X": X,
            "batch": batch_indices,
            "labels": indices,
            "archetype_vec": archetype_vec,
        }
        return rna_batch

    def get_history(self):
        """Return the training history including similarity losses"""
        return {
            "train_similarity_loss": self.similarity_losses,
            "train_similarity_loss_raw": self.similarity_losses_raw,
            "train_total_loss": self.train_losses,
            "train_rna_loss": self.train_rna_losses,
            "train_protein_loss": self.train_protein_losses,
            "train_matching_loss": self.train_matching_losses,
            "train_contrastive_loss": self.train_contrastive_losses,
            "train_adv_loss": self.train_adv_losses,
            "train_cell_type_clustering_loss": self.train_cell_type_clustering_losses,
            "val_total_loss": self.val_losses,
            "val_rna_loss": self.val_rna_losses,
        }

    def on_early_stopping(self):
        """Called when early stopping is triggered."""
        print("\nEarly stopping triggered!")

        print("✓ Early stopping artifacts saved")

    def on_epoch_end(self):
        """Called at the end of each training epoch."""
        log_epoch_end(self.log_file, self.current_epoch, self.train_losses, self.val_losses)
        print(f"\nEnd of epoch {self.current_epoch}")
        print(
            f"Average train loss: {sum(self.train_losses)/len(self.train_losses) if self.train_losses else 0:.4f}"
        )
        print(
            f"Average validation loss: {sum(self.val_losses)/len(self.val_losses) if self.val_losses else 0:.4f}"
        )


# %%
def train_vae(
    adata_rna_subset,
    adata_prot_subset,
    max_epochs=1,
    batch_size=128,
    lr=1e-3,
    contrastive_weight=1.0,
    similarity_weight=1000.0,
    diversity_weight=0.1,
    matching_weight=100.0,
    cell_type_clustering_weight=1.0,
    train_size=0.9,
    check_val_every_n_epoch=1,
    adv_weight=0.1,
    n_hidden_rna=128,
    n_hidden_prot=50,
    n_layers=3,
    latent_dim=10,
    validation_size=0.1,
    gradient_clip_val=1.0,
    accumulate_grad_batches=1,
    kl_weight_rna=1.0,
    kl_weight_prot=1.0,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    **kwargs,
):
    """Train the VAE models."""
    print("Initializing VAEs...")
    # setup adata for scvi
    if adata_rna_subset.X.min() < 0:
        adata_rna_subset.X = adata_rna_subset.X - adata_rna_subset.X.min()
    if adata_prot_subset.X.min() < 0:
        adata_prot_subset.X = adata_prot_subset.X - adata_prot_subset.X.min()
    adata_rna_subset.obs["index_col"] = range(len(adata_rna_subset.obs.index))
    adata_prot_subset.obs["index_col"] = range(len(adata_prot_subset.obs.index))
    SCVI.setup_anndata(
        adata_rna_subset,
        labels_key="index_col",
    )
    if adata_prot_subset.X.min() < 0:
        adata_prot_subset.X = adata_prot_subset.X - adata_prot_subset.X.min()

    SCVI.setup_anndata(
        adata_prot_subset,
        labels_key="index_col",
    )
    rna_vae = scvi.model.SCVI(
        adata_rna_subset,
        gene_likelihood=select_gene_likelihood(adata_rna_subset),
        n_hidden=n_hidden_rna,
        n_layers=n_layers,
        n_latent=latent_dim,
    )
    protein_vae = scvi.model.SCVI(
        adata_prot_subset,
        gene_likelihood="normal",
        n_hidden=n_hidden_prot,
        n_layers=n_layers,
        n_latent=latent_dim,
    )
    print("VAEs initialized")

    rna_vae._training_plan_cls = DualVAETrainingPlan

    # Set up training parameters

    # Create training plan with both VAEs
    plan_kwargs = {
        "protein_vae": protein_vae,
        "rna_vae": rna_vae,
        "contrastive_weight": contrastive_weight,
        "similarity_weight": similarity_weight,
        "diversity_weight": diversity_weight,
        "cell_type_clustering_weight": cell_type_clustering_weight,
        "matching_weight": matching_weight,
        "adv_weight": adv_weight,
        "plot_x_times": kwargs.pop("plot_x_times", 5),
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "lr": lr,
        "kl_weight_rna": kl_weight_rna,
        "kl_weight_prot": kl_weight_prot,
    }
    train_kwargs = {
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "train_size": train_size,
        "validation_size": validation_size,
        "check_val_every_n_epoch": check_val_every_n_epoch,
        "gradient_clip_val": gradient_clip_val,
        "accumulate_grad_batches": accumulate_grad_batches,
    }
    print("Plan parameters:")
    pprint(plan_kwargs)
    # Create training plan instance
    print("Creating training plan...")
    training_plan = DualVAETrainingPlan(rna_vae.module, **plan_kwargs)
    rna_vae._training_plan = training_plan
    print("Training plan created")

    # Train the model
    print("Starting training...")
    rna_vae.is_trained_ = True
    protein_vae.is_trained_ = True
    rna_vae.module.cpu()
    protein_vae.module.cpu()
    latent_rna_before = rna_vae.get_latent_representation()
    latent_prot_before = protein_vae.get_latent_representation()
    rna_vae.module.to(device)
    protein_vae.module.to(device)
    rna_vae.is_trained_ = False
    protein_vae.is_trained_ = False

    rna_vae.train(**train_kwargs, plan_kwargs=plan_kwargs)

    # Manually set trained flag
    rna_vae.is_trained_ = True
    protein_vae.is_trained_ = True
    print("Training flags set")

    return rna_vae, protein_vae, latent_rna_before, latent_prot_before


if __name__ == "__main__":
    save_dir = "CODEX_RNA_seq/data/processed_data"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # %%
    # read in the data
    cwd = os.getcwd()
    save_dir = Path("CODEX_RNA_seq/data/processed_data").absolute()

    # Use get_latest_file to find the most recent files
    adata_rna_subset = sc.read_h5ad(
        get_latest_file(save_dir, "adata_rna_subset_prepared_for_training_")
    )
    adata_prot_subset = sc.read_h5ad(
        get_latest_file(save_dir, "adata_prot_subset_prepared_for_training_")
    )

    # Subsample the data for faster testing
    print(f"Original RNA dataset shape: {adata_rna_subset.shape}")
    print(f"Original protein dataset shape: {adata_prot_subset.shape}")
    # Load config if exists
    rna_sample_size = min(len(adata_rna_subset), num_rna_cells)
    prot_sample_size = min(len(adata_prot_subset), num_protein_cells)
    adata_rna_subset = sc.pp.subsample(adata_rna_subset, n_obs=rna_sample_size, copy=True)
    adata_prot_subset = sc.pp.subsample(adata_prot_subset, n_obs=prot_sample_size, copy=True)

    print(f"Subsampled RNA dataset shape: {adata_rna_subset.shape}")
    print(f"Subsampled protein dataset shape: {adata_prot_subset.shape}")

    # Import utility functions
    from CODEX_RNA_seq.training_utils import (
        calculate_metrics,
        clear_memory,
        generate_visualizations,
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
    log_file = open(f"logs/vae_training_{log_timestamp}.log", "w")

    # Redirect stdout to both console and log file
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)

    print(f"Starting VAE training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: logs/vae_training_{log_timestamp}.log")

    # Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    experiment_name = f"vae_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_id = mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    log_memory_usage("Before loading data: ")

    # Load data - already done in the imports/setup section above
    print(f"RNA dataset shape: {adata_rna_subset.shape}")
    print(f"Protein dataset shape: {adata_prot_subset.shape}")

    # Define training parameters
    training_params = {
        "plot_x_times": 3,
        "max_epochs": 20,
        "batch_size": 1000,
        "lr": 1e-4,
        "contrastive_weight": 0,
        "similarity_weight": 10.0,
        "diversity_weight": 0.1,
        "matching_weight": 100000.0,
        "cell_type_clustering_weight": 1.0,
        "n_hidden_rna": 64,
        "n_hidden_prot": 32,
        "n_layers": 3,
        "latent_dim": 10,
        "kl_weight_rna": 0.1,
        "kl_weight_prot": 10.0,
        "adv_weight": 0.0,
        "train_size": 0.9,
        "validation_size": 0.1,
        "check_val_every_n_epoch": 100000,
        "gradient_clip_val": 1.0,
    }

    # Create loss weights JSON
    loss_weights = {
        "kl_weight_rna": training_params["kl_weight_rna"],
        "kl_weight_prot": training_params["kl_weight_prot"],
        "contrastive_weight": training_params["contrastive_weight"],
        "similarity_weight": training_params["similarity_weight"],
        # "diversity_weight": training_params["diversity_weight"],
        "matching_weight": training_params["matching_weight"],
        "cell_type_clustering_weight": training_params["cell_type_clustering_weight"],
        # "adv_weight": training_params["adv_weight"]
    }

    # Save loss weights to a temporary JSON file
    loss_weights_path = "loss_weights.json"
    with open(loss_weights_path, "w") as f:
        json.dump(loss_weights, f, indent=4)

    # Log parameters
    log_parameters(training_params, 0, 1)

    # Start MLflow run
    run_name = f"vae_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_name = f"vae_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        # Log loss weights JSON as artifact at the start
        mlflow.log_artifact(loss_weights_path)
        # Clean up temporary file
        os.remove(loss_weights_path)

        rna_vae, protein_vae, latent_rna_before, latent_prot_before = setup_and_train_model(
            adata_rna_subset, adata_prot_subset, training_params
        )
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
                }.items()
            }
        )

        # Process latent spaces
        rna_latent, prot_latent, combined_latent = process_latent_spaces(
            rna_vae.adata, protein_vae.adata
        )

        # Match cells and calculate distances
        matching_results = match_cells_and_calculate_distances(rna_latent, prot_latent)

        # Calculate metrics
        metrics = calculate_metrics(rna_vae, protein_vae, matching_results["prot_matches_in_rna"])

        # Log metrics
        mlflow.log_metrics(metrics)

        # Generate visualizations

        generate_visualizations(
            rna_vae,
            protein_vae,
            rna_latent,
            prot_latent,
            combined_latent,
            history,
            matching_results,
        )

        # Save results
        save_dir = Path("CODEX_RNA_seq/data/trained_data").absolute()
        save_results(rna_vae, protein_vae, save_dir)

    sys.stdout = original_stdout
    log_file.close()

    # Log the log file to MLflow
    mlflow.log_artifact(f"logs/vae_training_{log_timestamp}.log")


def calculate_losses(
    rna_batch,
    protein_batch,
    rna_vae,
    protein_vae,
    device,
    similarity_weight,
    similarity_active,
    contrastive_weight,
    matching_weight,
    cell_type_clustering_weight,
    kl_weight_rna,
    kl_weight_prot,
    plot_flag=False,
    global_step=None,
    total_steps=None,
    plot_x_times=None,
):
    """Calculate all losses for a batch of data.

    Args:
        rna_batch: Dictionary containing RNA batch data
        protein_batch: Dictionary containing protein batch data
        rna_vae: RNA VAE model
        protein_vae: Protein VAE model
        device: Device to use for calculations
        similarity_weight: Weight for similarity loss
        similarity_active: Whether similarity loss is active
        contrastive_weight: Weight for contrastive loss
        matching_weight: Weight for matching loss
        cell_type_clustering_weight: Weight for cell type clustering loss
        kl_weight_rna: Weight for RNA KL loss
        kl_weight_prot: Weight for protein KL loss
        plot_flag: Whether to plot metrics
        global_step: Current global step
        total_steps: Total number of steps
        plot_x_times: Number of times to plot during training

    Returns:
        Dictionary containing all calculated losses and metrics
    """
    # Get model outputs
    rna_inference_outputs, _, rna_loss_output_raw = rna_vae.module(rna_batch)
    protein_inference_outputs, _, protein_loss_output_raw = protein_vae.module(protein_batch)

    # Calculate base losses
    rna_loss_output = rna_loss_output_raw.loss * kl_weight_rna
    protein_loss_output = protein_loss_output_raw.loss * kl_weight_prot

    # Get latent representations
    rna_latent_mean = rna_inference_outputs["qz"].mean
    rna_latent_std = rna_inference_outputs["qz"].scale
    protein_latent_mean = protein_inference_outputs["qz"].mean
    protein_latent_std = protein_inference_outputs["qz"].scale

    # Calculate latent distances
    latent_distances = compute_pairwise_kl_two_items(
        rna_latent_mean,
        protein_latent_mean,
        rna_latent_std,
        protein_latent_std,
    )
    latent_distances = torch.clamp(latent_distances, max=torch.quantile(latent_distances, 0.90))

    # Calculate archetype distances
    archetype_dis = torch.cdist(
        normalize(rna_batch["archetype_vec"], dim=1),
        normalize(protein_batch["archetype_vec"], dim=1),
    )
    archetype_dis = torch.clamp(archetype_dis, max=torch.quantile(archetype_dis, 0.90))
    # Calculate matching loss
    archetype_dis_tensor = torch.tensor(archetype_dis, dtype=torch.float, device=device)
    threshold = 0.0005
    archetype_dis_tensor = (archetype_dis_tensor - archetype_dis_tensor.min()) / (
        archetype_dis_tensor.max() - archetype_dis_tensor.min()
    )
    latent_distances = (latent_distances - latent_distances.min()) / (
        latent_distances.max() - latent_distances.min()
    )

    squared_diff = (latent_distances - archetype_dis_tensor) ** 2
    acceptable_range_mask = (archetype_dis_tensor < threshold) & (latent_distances < threshold)
    stress_loss = squared_diff.mean()
    num_cells = squared_diff.numel()
    num_acceptable = acceptable_range_mask.sum()
    exact_pairs = 10 * torch.diag(latent_distances).mean()

    reward_strength = 0
    reward = reward_strength * (num_acceptable.float() / num_cells)
    matching_loss = (stress_loss - reward + exact_pairs) * matching_weight

    # Calculate contrastive loss
    rna_distances = compute_pairwise_kl(rna_latent_mean, rna_latent_std)
    prot_distances = compute_pairwise_kl(protein_latent_mean, protein_latent_std)
    distances = prot_distances + rna_distances

    # Get cell type and neighborhood info
    cell_neighborhood_info_protein = torch.tensor(
        protein_vae.adata[protein_batch["labels"]].obs["CN"].cat.codes.values
    ).to(device)
    cell_neighborhood_info_rna = torch.tensor(
        rna_vae.adata[rna_batch["labels"]].obs["CN"].cat.codes.values
    ).to(device)

    rna_major_cell_type = (
        torch.tensor(rna_vae.adata[rna_batch["labels"]].obs["major_cell_types"].values.codes)
        .to(device)
        .squeeze()
    )
    protein_major_cell_type = (
        torch.tensor(
            protein_vae.adata[protein_batch["labels"]].obs["major_cell_types"].values.codes
        )
        .to(device)
        .squeeze()
    )

    # Create masks for different cell type and neighborhood combinations
    num_cells = rna_batch["X"].shape[0]
    same_cn_mask = cell_neighborhood_info_rna.unsqueeze(
        0
    ) == cell_neighborhood_info_protein.unsqueeze(1)
    same_major_cell_type = rna_major_cell_type.unsqueeze(0) == protein_major_cell_type.unsqueeze(1)
    diagonal_mask = torch.eye(num_cells, dtype=torch.bool, device=device)

    distances = distances.masked_fill(diagonal_mask, 0)

    same_major_type_same_cn_mask = (same_major_cell_type * same_cn_mask).type(torch.bool)
    same_major_type_different_cn_mask = (same_major_cell_type * ~same_cn_mask).type(torch.bool)
    different_major_type_same_cn_mask = (~same_major_cell_type * same_cn_mask).type(torch.bool)
    different_major_type_different_cn_mask = (~same_major_cell_type * ~same_cn_mask).type(
        torch.bool
    )

    same_major_type_same_cn_mask.masked_fill_(diagonal_mask, 0)
    same_major_type_different_cn_mask.masked_fill_(diagonal_mask, 0)
    different_major_type_same_cn_mask.masked_fill_(diagonal_mask, 0)
    different_major_type_different_cn_mask.masked_fill_(diagonal_mask, 0)

    # Calculate contrastive loss
    contrastive_loss = 0.0
    if same_major_type_same_cn_mask.sum() > 0:
        contrastive_loss += -distances.masked_select(same_major_type_same_cn_mask).mean()
    if different_major_type_same_cn_mask.sum() > 0:
        contrastive_loss += distances.masked_select(different_major_type_same_cn_mask).mean()
    if different_major_type_different_cn_mask.sum() > 0:
        contrastive_loss += (
            distances.masked_select(different_major_type_different_cn_mask).mean() * 0.5
        )
    if same_major_type_different_cn_mask.sum() > 0:
        contrastive_loss += -distances.masked_select(same_major_type_different_cn_mask).mean() * 0.5
    contrastive_loss = contrastive_loss * contrastive_weight

    # Calculate cell type clustering loss
    rna_raw_cell_type_clustering_loss = run_cell_type_clustering_loss(
        rna_vae.adata, rna_latent_mean, rna_batch["labels"]
    )
    prot_raw_cell_type_clustering_loss = run_cell_type_clustering_loss(
        protein_vae.adata, protein_latent_mean, protein_batch["labels"]
    )
    cell_type_clustering_loss = (
        rna_raw_cell_type_clustering_loss + prot_raw_cell_type_clustering_loss
    ) * cell_type_clustering_weight

    # Calculate similarity loss
    rna_dis = torch.cdist(rna_latent_mean, rna_latent_mean)
    prot_dis = torch.cdist(protein_latent_mean, protein_latent_mean)
    rna_prot_dis = torch.cdist(rna_latent_mean, protein_latent_mean)
    similarity_loss_raw = torch.abs(
        ((rna_dis.abs().mean() + prot_dis.abs().mean()) / 2) - rna_prot_dis.abs().mean()
    )
    similarity_loss = (
        similarity_loss_raw * similarity_weight
        if similarity_active
        else torch.tensor(0.0).to(device)
    )

    # Calculate total loss
    total_loss = (
        rna_loss_output
        + protein_loss_output
        + contrastive_loss
        + matching_loss
        + similarity_loss
        + cell_type_clustering_loss
    )

    # Prepare metrics for plotting if needed
    if (
        plot_flag
        and global_step is not None
        and total_steps is not None
        and plot_x_times is not None
    ):
        should_plot = global_step > -1 and global_step % (1 + int(total_steps / plot_x_times)) == 0
        if should_plot:
            rna_latent_mean_numpy = rna_latent_mean.detach().cpu().numpy()
            rna_latent_std_numpy = rna_latent_std.detach().cpu().numpy()
            protein_latent_mean_numpy = protein_latent_mean.detach().cpu().numpy()
            protein_latent_std_numpy = protein_latent_std.detach().cpu().numpy()

            plot_latent_pca_both_modalities_cn(
                rna_latent_mean_numpy,
                protein_latent_mean_numpy,
                rna_vae.adata,
                protein_vae.adata,
                index_rna=rna_batch["labels"],
                index_prot=protein_batch["labels"],
                global_step=global_step,
            )
            plot_latent_pca_both_modalities_by_celltype(
                rna_vae.adata,
                protein_vae.adata,
                rna_latent_mean_numpy,
                protein_latent_mean_numpy,
                index_rna=rna_batch["labels"],
                index_prot=protein_batch["labels"],
                global_step=global_step,
            )
            plot_rna_protein_matching_means_and_scale(
                rna_latent_mean_numpy,
                protein_latent_mean_numpy,
                rna_latent_std_numpy,
                protein_latent_std_numpy,
                archetype_dis,
                global_step=global_step,
            )

    # Create losses dictionary
    losses = {
        "total_loss": total_loss,
        "rna_loss": rna_loss_output,
        "protein_loss": protein_loss_output,
        "contrastive_loss": contrastive_loss,
        "matching_loss": matching_loss,
        "similarity_loss": similarity_loss,
        "similarity_loss_raw": similarity_loss_raw,
        "cell_type_clustering_loss": cell_type_clustering_loss,
        "latent_distances": latent_distances.mean(),
        "num_acceptable": num_acceptable,
        "num_cells": num_cells,
        "exact_pairs": exact_pairs,
    }

    return losses

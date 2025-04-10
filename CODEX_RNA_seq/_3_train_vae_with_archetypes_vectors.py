# %%

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

import anndata
import mlflow
import numpy as np
import pandas as pd
import scanpy as sc

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

import anndata as ad
import mlflow
import numpy as np
import pandas as pd
import plotting_functions as pf
import scanpy as sc
import scvi
import torch
import torch.nn.functional as F
from anndata import AnnData
from pytorch_lightning.loggers import TensorBoardLogger
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
from CODEX_RNA_seq.logging_functions import (
    log_epoch_end,
    log_step_metrics,
    log_training_metrics,
    log_validation_metrics,
    print_training_metrics,
    setup_logging,
    update_log,
)

importlib.reload(CODEX_RNA_seq.logging_functions)

from plotting_functions import (
    plot_inference_outputs,
    plot_latent_pca_both_modalities_by_celltype,
    plot_latent_pca_both_modalities_cn,
    plot_rna_protein_matching_means_and_scale,
    plot_similarity_loss_history,
)

from bar_nick_utils import (
    calculate_iLISI,
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

        # Print initial protein VAE parameters
        print("\nInitial Protein VAE Parameters:")
        for name, param in protein_vae.module.named_parameters():
            if param.requires_grad:
                print(f"{name}:")
                print(f"  Mean: {param.data.mean().item():.4f}")
                print(f"  Std: {param.data.std().item():.4f}")
                print(f"  Min: {param.data.min().item():.4f}")
                print(f"  Max: {param.data.max().item():.4f}")
                print("---")

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
        validation_size = kwargs.pop("validation_size", 0.1)
        device = kwargs.pop("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        # Verify train and validation sizes sum to 1
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
        self.total_steps = steps_per_epoch * max_epochs
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
        self.similarity_weights = []  # Store similarity weights
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
        self.early_stopping_callback = None  # Will be set by trainer

        # Setup logging
        self.log_file = setup_logging()

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
        # Print protein VAE parameters every 100 steps
        if self.global_step % 100 == 0:
            print("\nProtein VAE Parameters:")
            for name, param in self.protein_vae.module.named_parameters():
                if param.requires_grad:
                    print(f"{name}:")
                    print(f"  Mean: {param.data.mean().item():.4f}")
                    print(f"  Std: {param.data.std().item():.4f}")
                    print(f"  Min: {param.data.min().item():.4f}")
                    print(f"  Max: {param.data.max().item():.4f}")
                    # print(f"  Grad Mean: {param.grad.mean().item() if param.grad is not None else 'None':.4f}")
                    print("---")

        indices = range(self.batch_size)
        indices_rna = np.random.choice(
            self.train_indices_rna,  # Use training indices
            size=len(indices),
            replace=True if len(indices) > len(self.train_indices_rna) else False,
        )
        indices_rna = np.sort(indices_rna)
        indices_prot = np.random.choice(
            self.train_indices_prot,  # Use training indices
            size=len(indices),
            replace=True if len(indices) > len(self.train_indices_prot) else False,
        )
        indices_prot = np.sort(indices_prot)
        rna_batch = self._get_rna_batch(batch, indices_rna)
        if rna_batch["X"].shape[0] != self.batch_size:
            raise ValueError("RNA batch size is not equal to the batch size")
        rna_inference_outputs, _, rna_loss_output_raw = self.rna_vae.module(rna_batch)
        rna_loss_output = rna_loss_output_raw.loss * self.kl_weight_rna
        protein_batch = self._get_protein_batch(batch, indices_prot)
        protein_inference_outputs, _, protein_loss_output_raw = self.protein_vae.module(
            protein_batch
        )
        protein_loss_output = protein_loss_output_raw.loss * self.kl_weight_prot
        # rna_inference_outputs = self.rna_vae.module.inference(
        #     rna_batch["X"], batch_index=rna_batch["batch"], n_samples=1
        # )
        index = rna_batch["labels"]
        rna_latent_mean = rna_inference_outputs["qz"].mean
        rna_latent_std = rna_inference_outputs["qz"].scale
        rna_latent_mean_numpy = rna_latent_mean.detach().cpu().numpy()
        rna_latent_std_numpy = rna_latent_std.detach().cpu().numpy()
        # protein_inference_outputs = self.protein_vae.module.inference(
        #     protein_batch["X"], batch_index=protein_batch["batch"], n_samples=1
        # )
        protein_latent_mean = protein_inference_outputs["qz"].mean
        protein_latent_std = protein_inference_outputs["qz"].scale
        protein_latent_mean_numpy = protein_latent_mean.detach().cpu().numpy()
        protein_latent_std_numpy = protein_latent_std.detach().cpu().numpy()

        archetype_dis = torch.cdist(
            normalize(rna_batch["archetype_vec"], dim=1),
            normalize(protein_batch["archetype_vec"], dim=1),
        )

        latent_distances = compute_pairwise_kl_two_items(
            rna_latent_mean,
            protein_latent_mean,
            rna_latent_std,
            protein_latent_std,
        )
        latent_distances = torch.clamp(latent_distances, max=torch.quantile(latent_distances, 0.90))

        should_plot = (
            self.global_step > -1
            and self.global_step % (1 + int(self.total_steps / (self.plot_x_times))) == 0
            and plot_flag
        )

        if should_plot:
            plot_latent_pca_both_modalities_cn(
                rna_latent_mean_numpy,
                protein_latent_mean_numpy,
                self.rna_vae.adata,
                self.protein_vae.adata,
                index_rna=indices_rna,
                index_prot=indices_prot,
                global_step=self.global_step,
            )
            plot_latent_pca_both_modalities_by_celltype(
                self.rna_vae.adata,
                self.protein_vae.adata,
                rna_latent_mean_numpy,
                protein_latent_mean_numpy,
                index_rna=indices_rna,
                index_prot=indices_prot,
                global_step=self.global_step,
            )
            plot_rna_protein_matching_means_and_scale(
                rna_latent_mean_numpy,
                protein_latent_mean_numpy,
                rna_latent_std_numpy,
                protein_latent_std_numpy,
                archetype_dis,
                global_step=self.global_step,
            )
            print(f"min latent distances: {round(latent_distances.min().item(),3)}")
            print(f"max latent distances: {round(latent_distances.max().item(),3)}")
            print(f"mean latent distances: {round(latent_distances.mean().item(),3)}")

        archetype_dis_tensor = torch.tensor(
            archetype_dis, dtype=torch.float, device=latent_distances.device
        )
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
        matching_loss = stress_loss - reward + exact_pairs
        matching_loss = matching_loss * self.matching_weight
        rna_distances = compute_pairwise_kl(rna_latent_mean, rna_latent_std)
        prot_distances = compute_pairwise_kl(protein_latent_mean, protein_latent_std)

        # # Ensure both tensors have the same dimensions
        # if rna_distances.shape != prot_distances.shape:
        #     min_size = min(rna_distances.shape[0], prot_distances.shape[0])
        #     rna_distances = rna_distances[:min_size, :min_size]
        #     prot_distances = prot_distances[:min_size, :min_size]

        distances = prot_distances + rna_distances

        rna_size = prot_size = rna_batch["X"].shape[0]
        mixed_latent = torch.cat([rna_latent_mean, protein_latent_mean], dim=0)
        batch_labels = torch.cat([torch.zeros(rna_size), torch.ones(prot_size)]).to(self.device)
        batch_pred = self.batch_classifier(mixed_latent)
        adv_loss = -F.cross_entropy(batch_pred, batch_labels.long())

        if self.first_step and plot_flag and False:
            plot_inference_outputs(
                rna_inference_outputs,
                protein_inference_outputs,
                latent_distances,
                rna_distances,
                prot_distances,
            )
            self.first_step = False
        cell_neighborhood_info_protein = torch.tensor(
            self.protein_vae.adata[indices_prot].obs["CN"].cat.codes.values
        ).to(self.device)
        cell_neighborhood_info_rna = torch.tensor(
            self.rna_vae.adata[indices_rna].obs["CN"].cat.codes.values
        ).to(self.device)
        cell_neighborhood_info_prot = torch.tensor(
            self.protein_vae.adata[indices_prot].obs["CN"].cat.codes.values
        ).to(self.device)
        rna_major_cell_type = (
            torch.tensor(self.rna_vae.adata[indices_rna].obs["major_cell_types"].values.codes)
            .to(self.device)
            .squeeze()
        )
        protein_major_cell_type = (
            torch.tensor(self.protein_vae.adata[indices_prot].obs["major_cell_types"].values.codes)
            .to(self.device)
            .squeeze()
        )

        num_cells = self.rna_vae.adata[index].shape[0]
        same_cn_mask = cell_neighborhood_info_rna.unsqueeze(
            0
        ) == cell_neighborhood_info_prot.unsqueeze(1)
        same_major_cell_type = rna_major_cell_type.unsqueeze(
            0
        ) == protein_major_cell_type.unsqueeze(1)
        diagonal_mask = torch.eye(
            num_cells, dtype=torch.bool, device=cell_neighborhood_info_rna.device
        )

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

        diversity_loss = 0.0

        if same_major_type_same_cn_mask.sum() > 0:
            diversity_loss += distances.masked_select(same_major_type_same_cn_mask).mean() * 0.1
        if different_major_type_same_cn_mask.sum() > 0:
            diversity_loss += (
                distances.masked_select(different_major_type_same_cn_mask).mean() * 0.1
            )
        if different_major_type_different_cn_mask.sum() > 0:
            diversity_loss += (
                distances.masked_select(different_major_type_different_cn_mask).mean() * 0.1
            )
        if same_major_type_different_cn_mask.sum() > 0:
            diversity_loss += (
                -distances.masked_select(same_major_type_different_cn_mask).mean() * 0.1
            )

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
            contrastive_loss += (
                -distances.masked_select(same_major_type_different_cn_mask).mean() * 0.5
            )
        contrastive_loss = contrastive_loss * self.contrastive_weight

        # Add cell type clustering loss calculation after the diversity_loss and contrastive_loss calculation
        # and before the in_steady_state calculation

        # Compute cell type clustering loss to keep cell types in distinct clusters
        rna_raw_cell_type_clustering_loss = run_cell_type_clustering_loss(
            self.rna_vae.adata, rna_latent_mean, indices_rna
        )
        prot_raw_cell_type_clustering_loss = run_cell_type_clustering_loss(
            self.protein_vae.adata, protein_latent_mean, indices_prot
        )
        cell_type_clustering_loss = (
            rna_raw_cell_type_clustering_loss + prot_raw_cell_type_clustering_loss
        ) * self.cell_type_clustering_weight
        # Existing loss calculations
        in_steady_state = False
        coeff_of_variation = 0  # default value
        if len(self.similarity_loss_history) == self.steady_state_window:
            # Calculate mean and standard deviation over the window
            mean_loss = sum(self.similarity_loss_history) / self.steady_state_window
            std_loss = (
                sum((x - mean_loss) ** 2 for x in self.similarity_loss_history)
                / self.steady_state_window
            ) ** 0.5

            # Check if variation is small enough to be considered steady state
            coeff_of_variation = std_loss / mean_loss
            if coeff_of_variation < self.steady_state_tolerance:
                in_steady_state = True

        # Determine if loss has increased significantly from steady state
        rna_dis = torch.cdist(rna_latent_mean, rna_latent_mean)
        prot_dis = torch.cdist(protein_latent_mean, protein_latent_mean)
        rna_prot_dis = torch.cdist(rna_latent_mean, protein_latent_mean)
        similarity_loss_raw = torch.abs(
            ((rna_dis.abs().mean() + prot_dis.abs().mean()) / 2) - rna_prot_dis.abs().mean()
        )
        loss_increased = False
        if not self.similarity_active and len(self.similarity_loss_history) > 0:
            recent_loss = similarity_loss_raw.item()
            min_steady_loss = min(self.similarity_loss_history)
            if recent_loss > min_steady_loss * (1 + self.reactivation_threshold):
                loss_increased = True
                if self.global_step % 10 == 0:
                    # print(
                    #     f"[Step {self.global_step}] Loss increased significantly: recent={recent_loss:.4f}, min_steady={min_steady_loss:.4f}, ratio={recent_loss/min_steady_loss:.4f}, threshold={1 + self.reactivation_threshold}"
                    # )
                    pass

        # Update the weight based on steady state detection
        if in_steady_state and self.similarity_active:
            current_similarity_weight = (
                self.similarity_weight / 1000
            )  # Zero out weight when in steady state
            self.similarity_active = False
            # print(
            #     f"[Step {self.global_step}] DEACTIVATING similarity loss - Entering steady state (CV={coeff_of_variation:.4f})"
            # )
            # print(f"  - Window values: {[f'{x:.4f}' for x in self.similarity_loss_history]}")
            # print(f"  - Tolerance threshold: {self.steady_state_tolerance}")
        elif loss_increased and not self.similarity_active:
            current_similarity_weight = self.similarity_weight  # Reactivate with full weight
            self.similarity_active = True
            # print(
            #     f"[Step {self.global_step}] REACTIVATING similarity loss - Loss increased significantly"
            # )
        else:
            current_similarity_weight = self.similarity_weight if self.similarity_active else 0
            if self.global_step % 50 == 0:
                # print(
                #     f"[Step {self.global_step}] Similarity status: active={self.similarity_active}, window_size={len(self.similarity_loss_history)}/{self.steady_state_window}"
                # )
                if len(self.similarity_loss_history) == self.steady_state_window:
                    mean_loss = sum(self.similarity_loss_history) / self.steady_state_window
                    std_loss = (
                        sum((x - mean_loss) ** 2 for x in self.similarity_loss_history)
                        / self.steady_state_window
                    ) ** 0.5
                    cv = std_loss / mean_loss
                    # print(
                    #     f"  - Window values: {[f'{x:.4f}' for x in self.similarity_loss_history]}"
                    # )
                    # print(
                    #     f"  - CV={cv:.4f}, threshold={self.steady_state_tolerance}, steady_state={cv < self.steady_state_tolerance}"
                    # )

        # Initialize iLISI_score with a default value
        iLISI_score = 0.0

        if self.global_step % self.steady_state_window == 0:
            combined_latent = ad.concat(
                [
                    AnnData(rna_latent_mean.detach().cpu().numpy()),
                    AnnData(protein_latent_mean.detach().cpu().numpy()),
                ],
                join="outer",
                label="modality",
                keys=["RNA", "Protein"],
            )
            sc.pp.pca(combined_latent, n_comps=5)
            sc.pp.neighbors(combined_latent, use_rep="X_pca", n_neighbors=10)
            iLISI_score = calculate_iLISI(combined_latent, "modality", plot_flag=False)
            if iLISI_score < 1.9 and self.similarity_weight > 1e8:
                self.similarity_weight = self.similarity_weight * 10
            elif self.similarity_weight > 100:  # make it smaller only if it is not too small
                self.similarity_weight = self.similarity_weight / 10

        # Store similarity metrics
        similarity_loss = current_similarity_weight * similarity_loss_raw

        # Update the history window (maintain fixed size)
        if len(self.similarity_loss_history) >= self.steady_state_window:
            self.similarity_loss_history.pop(0)  # Remove oldest value
        self.similarity_loss_history.append(similarity_loss_raw.item())

        self.similarity_losses.append(similarity_loss.item())
        self.similarity_losses_raw.append(similarity_loss_raw.item())
        self.similarity_weights.append(self.similarity_weight)
        self.active_similarity_loss_active_history.append(self.similarity_active)

        # Add cell type clustering loss to the total loss
        total_loss = (
            rna_loss_output
            + protein_loss_output
            + contrastive_loss
            # + adv_loss # dont remove comment for now
            + matching_loss
            + similarity_loss
            + cell_type_clustering_loss
            # + diversity_loss # dont remove comment for now
        )

        # Store the clustering loss
        self.train_cell_type_clustering_losses.append(cell_type_clustering_loss.item())

        if should_plot:
            plot_similarity_loss_history(
                self.similarity_losses, self.active_similarity_loss_active_history, self.global_step
            )

            print_training_metrics(
                self.global_step,
                self.current_epoch,
                rna_loss_output,
                protein_loss_output,
                contrastive_loss,
                adv_loss,
                matching_loss,
                similarity_loss,
                diversity_loss,
                total_loss,
                latent_distances,
                similarity_loss_raw,
                self.similarity_weight,
                self.similarity_active,
                num_acceptable,
                num_cells,
                exact_pairs,
                cell_type_clustering_loss=cell_type_clustering_loss,
            )

            print(f"min latent distances: {round(latent_distances.min().item(),3)}")
            print(f"max latent distances: {round(latent_distances.max().item(),3)}")
            print(f"mean latent distances: {round(latent_distances.mean().item(),3)}")
            print(f"cell type clustering loss: {round(cell_type_clustering_loss.item(),3)}")

        update_log(self.log_file, "train_similarity_loss_raw", similarity_loss_raw.item())
        update_log(self.log_file, "train_similarity_weighted", similarity_loss.item())
        update_log(self.log_file, "train_similarity_weight", self.similarity_weight)

        log_training_metrics(
            self.log_file,
            rna_loss_output,
            protein_loss_output,
            contrastive_loss,
            matching_loss,
            similarity_loss,
            total_loss,
            adv_loss,
            diversity_loss,
            cell_type_clustering_loss=self.cell_type_clustering_weight * cell_type_clustering_loss,
        )

        log_step_metrics(
            self.log_file,
            self.global_step,
            total_loss,
            rna_loss_output,
            protein_loss_output,
            contrastive_loss,
            matching_loss,
            similarity_loss,
        )

        self.train_losses.append(total_loss.item())
        self.train_rna_losses.append(rna_loss_output.item())
        self.train_protein_losses.append(protein_loss_output.item())
        self.train_matching_losses.append(matching_loss.item())
        self.train_contrastive_losses.append(contrastive_loss.item())
        self.train_adv_losses.append(adv_loss.item())

        # Save checkpoint every 100 epochs
        if self.current_epoch % 100 == 0 and self.current_epoch > 0:
            self.save_checkpoint()

        # self.pbar.update(1)
        # self.pbar.set_postfix(
        #     {
        #         "train_loss": f"{total_loss.item():.4f}",
        #         "similarity_loss": f"{similarity_loss.item():.4f}",
        #         "cell_type_loss": f"{cell_type_clustering_loss.item():.4f}",
        #     }
        # )

        return total_loss

    def validation_step(self, batch, batch_idx):
        # if self.val_pbar is None:
        #     self.val_pbar = tqdm(
        #         total=len(self.val_indices_prot) // self.batch_size,  # Use validation set size
        #         desc="Validation",
        #         leave=True
        #     )

        indices = range(self.batch_size)
        indices_prot = np.random.choice(
            self.val_indices_prot,  # Use validation indices
            size=len(indices),
            replace=True if len(indices) > len(self.val_indices_prot) else False,
        )
        indices_prot = np.sort(indices_prot)
        indices_rna = np.random.choice(
            self.val_indices_rna,  # Use validation indices
            size=len(indices),
            replace=True if len(indices) > len(self.val_indices_rna) else False,
        )
        indices_rna = np.sort(indices_rna)
        rna_batch = self._get_rna_batch(batch, indices_rna)
        protein_batch = self._get_protein_batch(batch, indices_prot)

        rna_inference_outputs, _, rna_loss_output_raw = self.rna_vae.module(rna_batch)
        protein_inference_outputs, _, protein_loss_output_raw = self.protein_vae.module(
            protein_batch
        )
        rna_loss_output = rna_loss_output_raw.loss * self.kl_weight_rna
        protein_loss_output = protein_loss_output_raw.loss * self.kl_weight_prot
        rna_latent_mean = rna_inference_outputs["qz"].mean
        rna_latent_std = rna_inference_outputs["qz"].scale
        protein_latent_mean = protein_inference_outputs["qz"].mean
        protein_latent_std = protein_inference_outputs["qz"].scale

        latent_distances = compute_pairwise_kl_two_items(
            rna_latent_mean,
            protein_latent_mean,
            rna_latent_std,
            protein_latent_std,
        )

        # Calculate matching loss
        archetype_dis = torch.cdist(
            normalize(rna_batch["archetype_vec"], dim=1),
            normalize(protein_batch["archetype_vec"], dim=1),
        )
        archetype_dis_tensor = torch.tensor(
            archetype_dis, dtype=torch.float, device=latent_distances.device
        )
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
        matching_loss = stress_loss - reward + exact_pairs

        # Calculate contrastive loss
        rna_distances = compute_pairwise_kl(rna_latent_mean, rna_latent_std)
        prot_distances = compute_pairwise_kl(protein_latent_mean, protein_latent_std)
        distances = 5 * prot_distances + rna_distances

        rna_size = prot_size = rna_batch["X"].shape[0]
        mixed_latent = torch.cat([rna_latent_mean, protein_latent_mean], dim=0)
        batch_labels = torch.cat([torch.zeros(rna_size), torch.ones(prot_size)]).to(self.device)
        batch_pred = self.batch_classifier(mixed_latent)
        adv_loss = -F.cross_entropy(batch_pred, batch_labels.long())

        # Calculate similarity loss
        cell_neighborhood_info_protein = torch.tensor(
            self.protein_vae.adata[indices_prot].obs["CN"].cat.codes.values
        ).to(self.device)
        cell_neighborhood_info_rna = torch.tensor(
            self.rna_vae.adata[indices_rna].obs["CN"].cat.codes.values
        ).to(self.device)
        same_cn_mask = cell_neighborhood_info_rna.unsqueeze(
            0
        ) == cell_neighborhood_info_protein.unsqueeze(1)

        if self.similarity_active:
            similarity_loss_raw = torch.sum(distances * (-same_cn_mask.float())) / (
                torch.sum(-same_cn_mask.float()) + 1e-10
            )
            similarity_loss = similarity_loss_raw * self.similarity_weight
        else:
            similarity_loss = torch.tensor(0.0).to(self.device)

        rna_cell_type_clustering_loss = run_cell_type_clustering_loss(
            self.rna_vae.adata, rna_latent_mean, indices_rna
        )
        prot_cell_type_clustering_loss = run_cell_type_clustering_loss(
            self.protein_vae.adata, protein_latent_mean, indices_prot
        )
        cell_type_clustering_loss = (
            rna_cell_type_clustering_loss + prot_cell_type_clustering_loss
        ) * self.cell_type_clustering_weight

        # Calculate total validation loss with same components as training
        validation_total_loss = (
            rna_loss_output
            + protein_loss_output
            + self.contrastive_weight * distances.mean()
            + matching_loss
            + similarity_loss
            + cell_type_clustering_loss
        )

        # Log the validation loss metric
        self.log(
            "val_total_loss", validation_total_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        log_validation_metrics(
            self.log_file,
            rna_loss_output,
            protein_loss_output,
            self.contrastive_weight * distances.mean(),
            validation_total_loss,
            latent_distances,
            cell_type_clustering_loss=self.cell_type_clustering_weight * cell_type_clustering_loss,
        )

        self.val_losses.append(validation_total_loss.item())
        self.val_rna_losses.append(rna_loss_output.item())
        self.val_protein_losses.append(protein_loss_output.item())
        self.val_matching_losses.append(matching_loss.item())
        self.val_adv_losses.append(adv_loss.item())

        # self.val_pbar.update(1)
        # self.val_pbar.set_postfix({"val_loss": f"{validation_total_loss.item():.4f}"})

        return validation_total_loss

    def on_validation_epoch_end(self):
        """Calculate and store metrics at the end of each validation epoch."""
        print(f"\nProcessing validation epoch {self.current_epoch}...")
        # Get latent representations from both VAEs for validation data only
        print("1. Getting latent representations...")
        with torch.no_grad():
            # Get RNA latent for validation data
            print("   Getting RNA latent representations...")
            rna_data = self.rna_vae.adata[self.val_indices_rna].X
            if issparse(rna_data):
                rna_data = rna_data.toarray()
            rna_tensor = torch.tensor(rna_data, dtype=torch.float32).to(self.device)
            rna_batch = torch.tensor(
                self.rna_vae.adata[self.val_indices_rna].obs["_scvi_batch"].values, dtype=torch.long
            ).to(self.device)

            rna_inference = self.rna_vae.module.inference(
                rna_tensor, batch_index=rna_batch, n_samples=1
            )
            rna_latent = rna_inference["qz"].mean.detach().cpu().numpy()
            print("   ✓ RNA latent representations obtained")

            # Get protein latent for validation data
            print("   Getting protein latent representations...")
            prot_data = self.protein_vae.adata[self.val_indices_prot].X
            if issparse(prot_data):
                prot_data = prot_data.toarray()
            prot_tensor = torch.tensor(prot_data, dtype=torch.float32).to(self.device)
            prot_batch = torch.tensor(
                self.protein_vae.adata[self.val_indices_prot].obs["_scvi_batch"].values,
                dtype=torch.long,
            ).to(self.device)

            prot_inference = self.protein_vae.module.inference(
                prot_tensor, batch_index=prot_batch, n_samples=1
            )
            prot_latent = prot_inference["qz"].mean.detach().cpu().numpy()
            print("   ✓ Protein latent representations obtained")

        print("2. Storing latent representations in adata...")
        # Store in adata
        self.rna_vae.adata.obsm["X_scVI_val"] = np.zeros(
            (len(self.rna_vae.adata), rna_latent.shape[1])
        )
        self.rna_vae.adata.obsm["X_scVI_val"][self.val_indices_rna] = rna_latent

        self.protein_vae.adata.obsm["X_scVI_val"] = np.zeros(
            (len(self.protein_vae.adata), prot_latent.shape[1])
        )
        self.protein_vae.adata.obsm["X_scVI_val"][self.val_indices_prot] = prot_latent
        print("   ✓ Latent representations stored")

        print("3. Preparing validation data...")
        # Calculate basic metrics using validation data only
        val_rna_adata = self.rna_vae.adata[self.val_indices_rna]
        val_prot_adata = self.protein_vae.adata[self.val_indices_prot]

        embedding_key = "X_scVI_val"
        assert (
            embedding_key in val_rna_adata.obsm
        ), f"No embeddings found in adata_rna.obsm['{embedding_key}']."
        assert (
            embedding_key in val_prot_adata.obsm
        ), f"No embeddings found in adata_prot.obsm['{embedding_key}']."

        val_rna_latent = AnnData(val_rna_adata.obsm[embedding_key].copy())
        val_prot_latent = AnnData(val_prot_adata.obsm[embedding_key].copy())
        val_rna_latent.obs = val_rna_adata.obs.copy()
        val_prot_latent.obs = val_prot_adata.obs.copy()
        print("   ✓ Validation data prepared")

        print("4. Combining latent spaces...")
        combined_latent = anndata.concat(
            [val_rna_latent, val_prot_latent],
            join="outer",
            label="modality",
            keys=["RNA", "Protein"],
        )
        # sc.pp.pca(combined_latent)
        # sc.pp.neighbors(combined_latent, n_neighbors=8)
        # print("   ✓ Latent spaces combined")

        # print("5. Calculating metrics...")
        # print("   Calculating silhouette score...")
        # silhouette = CODEX_RNA_seq.metrics.silhouette_score_calc(combined_latent)
        # print("   ✓ Silhouette score calculated")

        # print("   Calculating F1 score...")
        # f1 = CODEX_RNA_seq.metrics.f1_score_calc(val_rna_latent, val_prot_latent)
        # print("   ✓ F1 score calculated")

        # print("   Calculating ARI score...")
        # ari = CODEX_RNA_seq.metrics.ari_score_calc(val_rna_latent, val_prot_latent)
        # print("   ✓ ARI score calculated")

        print("   Calculating matching accuracy...")
        accuracy = CODEX_RNA_seq.metrics.matching_accuracy(val_rna_latent, val_prot_latent)
        print("   ✓ Matching accuracy calculated")

        print("   Calculating silhouette F1...")
        silhouette_f1 = CODEX_RNA_seq.metrics.compute_silhouette_f1(val_rna_latent, val_prot_latent)
        print("   ✓ Silhouette F1 calculated")

        print("   Calculating ARI F1...")
        sc.pp.pca(combined_latent)
        sc.pp.neighbors(combined_latent, n_neighbors=10)
        ari_f1 = CODEX_RNA_seq.metrics.compute_ari_f1(combined_latent)
        print("   ✓ ARI F1 calculated")

        # print("   Calculating mixing scores...")
        # mixing_result = bar_nick_utils.mixing_score(
        #     val_rna_latent,
        #     val_prot_latent,
        #     val_rna_adata,
        #     val_prot_adata,
        #     plot_flag=False,
        # )
        # print("   ✓ Mixing scores calculated")

        # print("   Calculating NMI scores...")
        # nmi_cell_types_cn_rna = adjusted_mutual_info_score(
        #     val_rna_adata.obs["cell_types"],
        #     val_rna_adata.obs["CN"],
        # )
        # nmi_cell_types_cn_prot = adjusted_mutual_info_score(
        #     val_prot_adata.obs["cell_types"],
        #     val_prot_adata.obs["CN"],
        # )
        # print("   ✓ NMI scores calculated")

        # print("   Calculating cross-modality NMI...")
        # nn_celltypes_prot = CODEX_RNA_seq.metrics.calc_dist(val_rna_latent, val_prot_latent)
        # nmi_cell_types_modalities = adjusted_mutual_info_score(
        #     val_rna_adata.obs["cell_types"].values,
        #     nn_celltypes_prot,
        # )
        # print("   ✓ Cross-modality NMI calculated")

        # print("   Calculating Leiden clustering metrics...")
        # sc.tl.leiden(combined_latent)
        # Extract cluster labels
        # leiden_clusters = combined_latent.obs["leiden"].astype(int).values
        # print("   ✓ Leiden clustering completed")

        # print("   Calculating normalized silhouette scores...")
        # silhouette_vals = silhouette_samples(combined_latent, leiden_clusters)
        # normalized_silhouette = (np.mean(silhouette_vals) + 1) / 2
        # print("   ✓ Normalized silhouette scores calculated")

        print("6. Storing metrics...")
        # Store metrics for this epoch
        epoch_metrics = {
            # "silhouette_score": silhouette,
            # "f1_score": f1,
            # "ari_score": ari,
            "val_cell_type_matching_accuracy": accuracy,
            # "mixing_score_ilisi": mixing_result["iLISI"],
            # "mixing_score_clisi": mixing_result["cLISI"],
            # "nmi_cell_types_cn_rna": nmi_cell_types_cn_rna,
            # "nmi_cell_types_cn_prot": nmi_cell_types_cn_prot,
            # "nmi_cell_types_modalities": nmi_cell_types_modalities,
            # "normalized_silhouette": normalized_silhouette,
            # "num_leiden_clusters": len(np.unique(leiden_clusters)),
            "val_silhouette_f1_score": silhouette_f1.mean(),
            "val_ari_f1_score": ari_f1,
        }

        # Store in history
        if not hasattr(self, "metrics_history"):
            self.metrics_history = []
        self.metrics_history.append(epoch_metrics)
        print("   ✓ Metrics stored")

        print(f"✓ Validation epoch {self.current_epoch} completed successfully!")

    def on_train_end(self, plot_flag=True):
        """Called when training ends."""
        print("\nTraining completed!")

        # Print final protein VAE parameters
        print("\nFinal Protein VAE Parameters:")
        for name, param in self.protein_vae.module.named_parameters():
            if param.requires_grad:
                print(f"{name}:")
                print(f"  Mean: {param.data.mean().item():.4f}")
                print(f"  Std: {param.data.std().item():.4f}")
                print(f"  Min: {param.data.min().item():.4f}")
                print(f"  Max: {param.data.max().item():.4f}")
                print("---")

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
        try:
            with open(self.log_file, "a") as f:
                f.write(f"\nCheckpoint saved at epoch {self.current_epoch}\n")
                f.write(f"Location: {checkpoint_path}\n")
                f.write(f"RNA dataset shape: {rna_adata_save.shape}\n")
                f.write(f"Protein dataset shape: {protein_adata_save.shape}\n")
        except Exception as e:
            print(f"Warning: Could not update log file: {e}")

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
        # if self.pbar is not None:
        #     self.pbar.close()
        #     self.pbar = None
        # if self.val_pbar is not None:
        #     self.val_pbar.close()
        #     self.val_pbar = None

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

    print("Setting up TensorBoard logger...")
    logger = TensorBoardLogger(
        save_dir="my_logs", name=f"experiment_name_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    print("TensorBoard logger setup complete")

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
    print("Plan parameters:", plan_kwargs)

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

    # Log parameters
    log_parameters(training_params, 0, 1)

    # Start MLflow run
    run_name = f"vae_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_name = f"vae_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        # Setup and train model
        log_memory_usage("Before training: ")
        rna_vae, protein_vae, latent_rna_before, latent_prot_before = setup_and_train_model(
            adata_rna_subset, adata_prot_subset, training_params
        )
        log_memory_usage("After training: ")

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
        rna_latent, prot_latent, combined_latent = process_latent_spaces(rna_vae, protein_vae)

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

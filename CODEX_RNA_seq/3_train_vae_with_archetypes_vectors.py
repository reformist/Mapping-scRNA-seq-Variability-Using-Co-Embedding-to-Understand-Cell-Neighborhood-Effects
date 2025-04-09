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

import mlflow
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_mutual_info_score, silhouette_samples
from tqdm import tqdm

# Add repository root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set working directory to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plotting_functions as pf

import bar_nick_utils
import CODEX_RNA_seq.logging_functions
import CODEX_RNA_seq.metrics

importlib.reload(pf)
importlib.reload(bar_nick_utils)
importlib.reload(CODEX_RNA_seq.logging_functions)
importlib.reload(CODEX_RNA_seq.metrics)


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
from scipy.spatial.distance import cdist
from scvi.model import SCVI
from scvi.train import TrainingPlan
from sklearn.metrics import adjusted_mutual_info_score
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
    plot_archetype_embedding,
    plot_cell_type_distributions,
    plot_combined_latent_space,
    plot_inference_outputs,
    plot_latent_pca_both_modalities_by_celltype,
    plot_latent_pca_both_modalities_cn,
    plot_normalized_losses,
    plot_rna_protein_latent_cn_cell_type_umap,
    plot_rna_protein_matching_means_and_scale,
    plot_similarity_loss_history,
    plot_spatial_data,
    plot_umap_visualizations_original_data,
)

from bar_nick_utils import (
    calculate_iLISI,
    clean_uns_for_h5ad,
    compare_distance_distributions,
    compute_pairwise_kl,
    compute_pairwise_kl_two_items,
    get_latest_file,
    get_umap_filtered_fucntion,
    mixing_score,
    select_gene_likelihood,
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
save_dir = "CODEX_RNA_seq/data/processed_data"


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

rna_sample_size = min(len(adata_rna_subset), num_rna_cells)
prot_sample_size = min(len(adata_prot_subset), num_protein_cells)
adata_rna_subset = sc.pp.subsample(adata_rna_subset, n_obs=rna_sample_size, copy=True)
adata_prot_subset = sc.pp.subsample(adata_prot_subset, n_obs=prot_sample_size, copy=True)


print(f"Subsampled RNA dataset shape: {adata_rna_subset.shape}")
print(f"Subsampled protein dataset shape: {adata_prot_subset.shape}")


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
        self.similarity_weight = kwargs.pop(
            "similarity_weight"
        )  # Remove default to use config value
        self.cell_type_clustering_weight = kwargs.pop("cell_type_clustering_weight", 1000.0)
        self.lr = kwargs.pop("lr", 0.001)
        self.kl_weight_rna = kwargs.pop("kl_weight_rna", 1.0)
        self.kl_weight_prot = kwargs.pop("kl_weight_prot", 1.0)
        super().__init__(rna_module, **kwargs)
        self.rna_vae = rna_vae
        self.protein_vae = protein_vae
        num_batches = 2
        latent_dim = self.rna_vae.module.n_latent
        self.batch_classifier = torch.nn.Linear(latent_dim, num_batches)
        self.contrastive_weight = contrastive_weight
        self.protein_vae.module.to(device)
        self.rna_vae.module = self.rna_vae.module.to(device)
        self.first_step = True
        n_samples = len(self.rna_vae.adata)
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
            "device": str(device),
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
        d = {  # maybe add this?
            "optimizer": optimizer,
            "gradient_clip_val": 1.0,  # Critical for stability
            "gradient_clip_algorithm": "value",
        }
        return d

    def training_step(self, batch, batch_idx):
        if self.pbar is None:
            self.pbar = tqdm(total=self.total_steps, desc="Training", leave=True)

        indices = range(self.batch_size)
        indices_rna = np.random.choice(
            range(len(self.rna_vae.adata)),
            size=len(indices),
            replace=True if len(indices) > len(self.rna_vae.adata) else False,
        )
        indices_rna = np.sort(indices_rna)
        indices_prot = np.random.choice(
            range(len(self.protein_vae.adata)),
            size=len(indices),
            replace=True if len(indices) > len(self.protein_vae.adata) else False,
        )
        indices_prot = np.sort(indices_prot)
        rna_batch = self._get_rna_batch(batch, indices_rna)
        if rna_batch["X"].shape[0] != self.batch_size:
            raise ValueError("RNA batch size is not equal to the batch size")
        rna_inference_outputs, _, rna_loss_output = self.rna_vae.module(
            rna_batch, loss_kwargs={"kl_weight": self.kl_weight_rna}
        )
        protein_batch = self._get_protein_batch(batch, indices_prot)
        protein_inference_outputs, _, protein_loss_output = self.protein_vae.module(
            protein_batch, loss_kwargs={"kl_weight": self.kl_weight_prot}
        )

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
        rna_distances = compute_pairwise_kl(rna_latent_mean, rna_latent_std)
        prot_distances = compute_pairwise_kl(protein_latent_mean, protein_latent_std)

        # # Ensure both tensors have the same dimensions
        # if rna_distances.shape != prot_distances.shape:
        #     min_size = min(rna_distances.shape[0], prot_distances.shape[0])
        #     rna_distances = rna_distances[:min_size, :min_size]
        #     prot_distances = prot_distances[:min_size, :min_size]

        distances = 5 * prot_distances + rna_distances

        rna_size = prot_size = rna_batch["X"].shape[0]
        mixed_latent = torch.cat([rna_latent_mean, protein_latent_mean], dim=0)
        batch_labels = torch.cat([torch.zeros(rna_size), torch.ones(prot_size)]).to(device)
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
        ).to(device)
        cell_neighborhood_info_rna = torch.tensor(
            self.rna_vae.adata[indices_rna].obs["CN"].cat.codes.values
        ).to(device)
        cell_neighborhood_info_prot = torch.tensor(
            self.protein_vae.adata[indices_prot].obs["CN"].cat.codes.values
        ).to(device)
        rna_major_cell_type = (
            torch.tensor(self.rna_vae.adata[indices_rna].obs["major_cell_types"].values.codes)
            .to(device)
            .squeeze()
        )
        protein_major_cell_type = (
            torch.tensor(self.protein_vae.adata[indices_prot].obs["major_cell_types"].values.codes)
            .to(device)
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
        rna_cell_types = torch.tensor(
            self.rna_vae.adata[indices_rna].obs["cell_types"].cat.codes.values
        ).to(device)
        protein_cell_types = torch.tensor(
            self.protein_vae.adata[indices_prot].obs["cell_types"].cat.codes.values
        ).to(device)

        # Combine cell types and latent representations from both modalities
        combined_cell_types = torch.cat([rna_cell_types, protein_cell_types])
        combined_latent_means = torch.cat([rna_latent_mean, protein_latent_mean])

        # Calculate centroid for each cell type
        unique_cell_types = torch.unique(combined_cell_types)
        num_cell_types = len(unique_cell_types)

        # Skip the cell type clustering loss if there's only one cell type
        cell_type_clustering_loss = torch.tensor(0.0).to(device)

        if num_cell_types > 1:
            # Calculate centroids for each cell type in latent space
            centroids = []
            cells_per_type = []
            type_to_idx = {}

            for i, cell_type in enumerate(unique_cell_types):
                mask = combined_cell_types == cell_type
                type_to_idx[cell_type.item()] = i
                if mask.sum() > 0:
                    cells = combined_latent_means[mask]
                    centroid = cells.mean(dim=0)
                    centroids.append(centroid)
                    cells_per_type.append(cells)

            if len(centroids) > 1:  # Need at least 2 centroids
                centroids = torch.stack(centroids)

                # Get original structure from archetype vectors
                if not hasattr(self, "original_structure_matrix"):
                    # Compute the structure matrix once and cache it
                    all_cell_types = self.rna_vae.adata.obs["cell_types"].cat.codes.values
                    all_unique_types = np.unique(all_cell_types)

                    # Get centroids in archetype space for each cell type
                    original_centroids = []
                    for ct in all_unique_types:
                        mask = all_cell_types == ct
                        if mask.sum() > 0:
                            ct_archetype_vecs = self.rna_vae.adata.obsm["archetype_vec"][mask]
                            original_centroids.append(np.mean(ct_archetype_vecs, axis=0))

                    # Convert to torch tensor
                    original_centroids = torch.tensor(
                        np.array(original_centroids), dtype=torch.float32
                    ).to(device)

                    # Compute affinity/structure matrix (using Gaussian kernel)
                    sigma = torch.cdist(original_centroids, original_centroids).mean()
                    dists = torch.cdist(original_centroids, original_centroids)
                    self.original_structure_matrix = torch.exp(-(dists**2) / (2 * sigma**2))

                    # Set diagonal to 0 to focus on between-cluster relationships
                    self.original_structure_matrix = self.original_structure_matrix * (
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
                    if type_i.item() < len(self.original_structure_matrix):
                        for j, type_j in enumerate(unique_cell_types):
                            if i != j and type_j.item() < len(self.original_structure_matrix):
                                # Get original and current affinity values
                                orig_affinity = self.original_structure_matrix[
                                    type_i.item(), type_j.item()
                                ]
                                current_affinity = current_structure_matrix[i, j]

                                # Square difference
                                diff = (orig_affinity - current_affinity) ** 2
                                structure_preservation_loss += diff
                                count += 1

                if count > 0:
                    structure_preservation_loss = structure_preservation_loss / count

                # Calculate within-cluster cohesion
                cohesion_loss = 0.0
                total_cells = 0

                for i, cells in enumerate(cells_per_type):
                    if len(cells) > 1:
                        # Calculate distances to centroid
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
                cell_type_clustering_loss = (
                    2.0 * structure_preservation_loss + normalized_cohesion_loss
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
            rna_loss_output.loss
            + protein_loss_output.loss
            # + contrastive_loss
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
        self.train_rna_losses.append(rna_loss_output.loss.item())
        self.train_protein_losses.append(protein_loss_output.loss.item())
        self.train_matching_losses.append(matching_loss.item())
        self.train_contrastive_losses.append(contrastive_loss.item())
        self.train_adv_losses.append(adv_loss.item())

        # Save checkpoint every 100 epochs
        if self.current_epoch % 100 == 0 and self.current_epoch > 0:
            self.save_checkpoint()

        self.pbar.update(1)
        self.pbar.set_postfix(
            {
                "train_loss": f"{total_loss.item():.4f}",
                "similarity_loss": f"{similarity_loss.item():.4f}",
                "cell_type_loss": f"{cell_type_clustering_loss.item():.4f}",
            }
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        if self.val_pbar is None:
            self.val_pbar = tqdm(
                total=len(self.protein_vae.adata) // self.batch_size, desc="Validation", leave=True
            )

        indices = range(self.batch_size)
        indices_prot = np.random.choice(
            range(len(self.protein_vae.adata)),
            size=len(indices),
            replace=True if len(indices) > len(self.protein_vae.adata) else False,
        )
        indices_prot = np.sort(indices_prot)
        indices_rna = np.random.choice(
            range(len(self.rna_vae.adata)),
            size=len(indices),
            replace=True if len(indices) > len(self.rna_vae.adata) else False,
        )
        indices_rna = np.sort(indices_rna)
        rna_batch = self._get_rna_batch(batch, indices_rna)
        protein_batch = self._get_protein_batch(batch, indices_prot)

        rna_inference_outputs, _, rna_loss_output = self.rna_vae.module(rna_batch)
        protein_inference_outputs, _, protein_loss_output = self.protein_vae.module(protein_batch)

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
        batch_labels = torch.cat([torch.zeros(rna_size), torch.ones(prot_size)]).to(device)
        batch_pred = self.batch_classifier(mixed_latent)
        adv_loss = -F.cross_entropy(batch_pred, batch_labels.long())

        # Calculate similarity loss
        cell_neighborhood_info_protein = torch.tensor(
            self.protein_vae.adata[indices_prot].obs["CN"].cat.codes.values
        ).to(device)
        cell_neighborhood_info_rna = torch.tensor(
            self.rna_vae.adata[indices_rna].obs["CN"].cat.codes.values
        ).to(device)
        same_cn_mask = cell_neighborhood_info_rna.unsqueeze(
            0
        ) == cell_neighborhood_info_protein.unsqueeze(1)

        if self.similarity_active:
            similarity_loss_raw = torch.sum(distances * (-same_cn_mask.float())) / (
                torch.sum(-same_cn_mask.float()) + 1e-10
            )
            similarity_loss = similarity_loss_raw * self.similarity_weight
        else:
            similarity_loss = torch.tensor(0.0).to(device)

        # Calculate cell type clustering loss
        rna_cell_types = torch.tensor(
            self.rna_vae.adata[indices_rna].obs["cell_types"].cat.codes.values
        ).to(device)
        protein_cell_types = torch.tensor(
            self.protein_vae.adata[indices_prot].obs["cell_types"].cat.codes.values
        ).to(device)

        # Combine cell types and latent representations from both modalities
        combined_cell_types = torch.cat([rna_cell_types, protein_cell_types])
        combined_latent_means = torch.cat([rna_latent_mean, protein_latent_mean])

        # Calculate centroid for each cell type
        unique_cell_types = torch.unique(combined_cell_types)
        num_cell_types = len(unique_cell_types)

        # Skip the cell type clustering loss if there's only one cell type
        cell_type_clustering_loss = torch.tensor(0.0).to(device)

        if num_cell_types > 1:
            # Calculate centroids for each cell type in latent space
            centroids = []
            cells_per_type = []
            type_to_idx = {}

            for i, cell_type in enumerate(unique_cell_types):
                mask = combined_cell_types == cell_type
                type_to_idx[cell_type.item()] = i
                if mask.sum() > 0:
                    cells = combined_latent_means[mask]
                    centroid = cells.mean(dim=0)
                    centroids.append(centroid)
                    cells_per_type.append(cells)

            if len(centroids) > 1:  # Need at least 2 centroids
                centroids = torch.stack(centroids)

                # Get original structure from archetype vectors
                if not hasattr(self, "original_structure_matrix"):
                    # Compute the structure matrix once and cache it
                    all_cell_types = self.rna_vae.adata.obs["cell_types"].cat.codes.values
                    all_unique_types = np.unique(all_cell_types)

                    # Get centroids in archetype space for each cell type
                    original_centroids = []
                    for ct in all_unique_types:
                        mask = all_cell_types == ct
                        if mask.sum() > 0:
                            ct_archetype_vecs = self.rna_vae.adata.obsm["archetype_vec"][mask]
                            original_centroids.append(np.mean(ct_archetype_vecs, axis=0))

                    # Convert to torch tensor
                    original_centroids = torch.tensor(
                        np.array(original_centroids), dtype=torch.float32
                    ).to(device)

                    # Compute affinity/structure matrix (using Gaussian kernel)
                    sigma = torch.cdist(original_centroids, original_centroids).mean()
                    dists = torch.cdist(original_centroids, original_centroids)
                    self.original_structure_matrix = torch.exp(-(dists**2) / (2 * sigma**2))

                    # Set diagonal to 0 to focus on between-cluster relationships
                    self.original_structure_matrix = self.original_structure_matrix * (
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
                    if type_i.item() < len(self.original_structure_matrix):
                        for j, type_j in enumerate(unique_cell_types):
                            if i != j and type_j.item() < len(self.original_structure_matrix):
                                # Get original and current affinity values
                                orig_affinity = self.original_structure_matrix[
                                    type_i.item(), type_j.item()
                                ]
                                current_affinity = current_structure_matrix[i, j]

                                # Square difference
                                diff = (orig_affinity - current_affinity) ** 2
                                structure_preservation_loss += diff
                                count += 1

                if count > 0:
                    structure_preservation_loss = structure_preservation_loss / count

                # Calculate within-cluster cohesion
                cohesion_loss = 0.0
                total_cells = 0

                for i, cells in enumerate(cells_per_type):
                    if len(cells) > 1:
                        # Calculate distances to centroid
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
                cell_type_clustering_loss = (
                    2.0 * structure_preservation_loss + normalized_cohesion_loss
                ) * self.cell_type_clustering_weight

        # Calculate total validation loss with same components as training
        validation_total_loss = (
            rna_loss_output.loss
            + protein_loss_output.loss
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
        self.val_rna_losses.append(rna_loss_output.loss.item())
        self.val_protein_losses.append(protein_loss_output.loss.item())
        self.val_matching_losses.append(matching_loss.item())
        self.val_adv_losses.append(adv_loss.item())

        self.val_pbar.update(1)
        self.val_pbar.set_postfix({"val_loss": f"{validation_total_loss.item():.4f}"})

        return validation_total_loss

    def on_validation_epoch_end(self):
        """Calculate and store metrics at the end of each validation epoch."""
        # Get latent representations from both VAEs
        with torch.no_grad():
            # Get RNA latent
            rna_data = self.rna_vae.adata.X
            if issparse(rna_data):
                rna_data = rna_data.toarray()
            rna_tensor = torch.tensor(rna_data, dtype=torch.float32).to(device)
            rna_batch = torch.tensor(
                self.rna_vae.adata.obs["_scvi_batch"].values, dtype=torch.long
            ).to(device)

            rna_inference = self.rna_vae.module.inference(
                rna_tensor, batch_index=rna_batch, n_samples=1
            )
            rna_latent = rna_inference["qz"].mean.detach().cpu().numpy()

            # Get protein latent
            prot_data = self.protein_vae.adata.X
            if issparse(prot_data):
                prot_data = prot_data.toarray()
            prot_tensor = torch.tensor(prot_data, dtype=torch.float32).to(device)
            prot_batch = torch.tensor(
                self.protein_vae.adata.obs["_scvi_batch"].values, dtype=torch.long
            ).to(device)

            prot_inference = self.protein_vae.module.inference(
                prot_tensor, batch_index=prot_batch, n_samples=1
            )
            prot_latent = prot_inference["qz"].mean.detach().cpu().numpy()

        # Store in adata
        self.rna_vae.adata.obsm["X_scVI"] = rna_latent
        self.protein_vae.adata.obsm["X_scVI"] = prot_latent

        # Calculate basic metrics
        silhouette = CODEX_RNA_seq.metrics.silhouette_score_calc(
            self.rna_vae.adata, self.protein_vae.adata
        )
        f1 = CODEX_RNA_seq.metrics.f1_score_calc(self.rna_vae.adata, self.protein_vae.adata)
        ari = CODEX_RNA_seq.metrics.ari_score_calc(self.rna_vae.adata, self.protein_vae.adata)
        accuracy = CODEX_RNA_seq.metrics.matching_accuracy(
            self.rna_vae.adata, self.protein_vae.adata
        )

        # Calculate advanced metrics
        try:
            silhouette_f1 = CODEX_RNA_seq.metrics.compute_silhouette_f1(
                self.rna_vae.adata, self.protein_vae.adata
            )
            ari_f1 = CODEX_RNA_seq.metrics.compute_ari_f1(
                self.rna_vae.adata, self.protein_vae.adata
            )
            has_advanced_metrics = True
        except Exception as e:
            print(f"Warning: Could not calculate advanced metrics: {e}")
            has_advanced_metrics = False

        # Calculate mixing scores
        mixing_result = bar_nick_utils.mixing_score(
            rna_latent,
            prot_latent,
            self.rna_vae.adata,
            self.protein_vae.adata,
            index_rna=range(len(self.rna_vae.adata)),
            index_prot=range(len(self.protein_vae.adata)),
            plot_flag=False,
        )

        # Calculate NMI scores
        nmi_cell_types_cn_rna = adjusted_mutual_info_score(
            self.rna_vae.adata.obs["cell_types"],
            self.rna_vae.adata.obs["CN"],
        )
        nmi_cell_types_cn_prot = adjusted_mutual_info_score(
            self.protein_vae.adata.obs["cell_types"],
            self.protein_vae.adata.obs["CN"],
        )

        # Calculate cross-modality NMI
        nn_celltypes_prot = CODEX_RNA_seq.metrics.calc_dist(
            self.rna_vae.adata, self.protein_vae.adata
        )
        nmi_cell_types_modalities = adjusted_mutual_info_score(
            self.rna_vae.adata.obs["cell_types"].values,
            nn_celltypes_prot,
        )

        # Calculate Leiden clustering metrics
        combined_latent = np.concatenate([rna_latent, prot_latent], axis=0)
        leiden_clusters = CODEX_RNA_seq.metrics.leiden_from_embeddings(combined_latent)

        # Calculate normalized silhouette scores
        silhouette_vals = silhouette_samples(combined_latent, leiden_clusters)
        normalized_silhouette = CODEX_RNA_seq.metrics.normalize_silhouette(silhouette_vals)

        # Store metrics for this epoch
        epoch_metrics = {
            "silhouette_score": silhouette,
            "f1_score": f1,
            "ari_score": ari,
            "matching_accuracy": accuracy,
            "mixing_score_ilisi": mixing_result["iLISI"],
            "mixing_score_clisi": mixing_result["cLISI"],
            "nmi_cell_types_cn_rna": nmi_cell_types_cn_rna,
            "nmi_cell_types_cn_prot": nmi_cell_types_cn_prot,
            "nmi_cell_types_modalities": nmi_cell_types_modalities,
            "normalized_silhouette": normalized_silhouette,
            "num_leiden_clusters": len(np.unique(leiden_clusters)),
            "val_silhouette_f1_score": silhouette_f1.mean(),
            "val_ari_f1_score": ari_f1,
        }

        # Store in history
        if not hasattr(self, "metrics_history"):
            self.metrics_history = []
        self.metrics_history.append(epoch_metrics)

        print(f"✓ Validation metrics calculated for epoch {self.current_epoch}")

    def on_train_end(self, plot_flag=True):
        """Called when training ends."""
        print("\nTraining completed!")

        # Get final latent representations
        with torch.no_grad():
            # Get RNA latent
            rna_data = self.rna_vae.adata.X
            if issparse(rna_data):
                rna_data = rna_data.toarray()
            rna_tensor = torch.tensor(rna_data, dtype=torch.float32).to(device)
            rna_batch = torch.tensor(
                self.rna_vae.adata.obs["_scvi_batch"].values, dtype=torch.long
            ).to(device)

            rna_inference = self.rna_vae.module.inference(
                rna_tensor, batch_index=rna_batch, n_samples=1
            )
            rna_latent = rna_inference["qz"].mean.detach().cpu().numpy()

            # Get protein latent
            prot_data = self.protein_vae.adata.X
            if issparse(prot_data):
                prot_data = prot_data.toarray()
            prot_tensor = torch.tensor(prot_data, dtype=torch.float32).to(device)
            prot_batch = torch.tensor(
                self.protein_vae.adata.obs["_scvi_batch"].values, dtype=torch.long
            ).to(device)

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
            rna_tensor = torch.tensor(rna_data, dtype=torch.float32).to(device)
            rna_batch = torch.tensor(
                self.rna_vae.adata.obs["_scvi_batch"].values, dtype=torch.long
            ).to(device)

            rna_inference = self.rna_vae.module.inference(
                rna_tensor, batch_index=rna_batch, n_samples=1
            )
            rna_latent = rna_inference["qz"].mean.detach().cpu().numpy()

            # Get protein latent
            prot_data = self.protein_vae.adata.X
            if issparse(prot_data):
                prot_data = prot_data.toarray()
            prot_tensor = torch.tensor(prot_data, dtype=torch.float32).to(device)
            prot_batch = torch.tensor(
                self.protein_vae.adata.obs["_scvi_batch"].values, dtype=torch.long
            ).to(device)

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
        X = torch.tensor(X, dtype=torch.float32).to(device)
        batch_indices = torch.tensor(protein_data.obs["_scvi_batch"].values, dtype=torch.long).to(
            device
        )
        archetype_vec = torch.tensor(
            protein_data.obsm["archetype_vec"].values, dtype=torch.float32
        ).to(device)

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
        X = torch.tensor(X, dtype=torch.float32).to(device)
        batch_indices = torch.tensor(rna_data.obs["_scvi_batch"].values, dtype=torch.long).to(
            device
        )
        archetype_vec = torch.tensor(rna_data.obsm["archetype_vec"].values, dtype=torch.float32).to(
            device
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
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
        if self.val_pbar is not None:
            self.val_pbar.close()
            self.val_pbar = None

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

    print("Training completed")

    # Manually set trained flag
    rna_vae.is_trained_ = True
    protein_vae.is_trained_ = True
    print("Training flags set")

    return rna_vae, protein_vae, latent_rna_before, latent_prot_before


if __name__ == "__main__":
    # %%
    # Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    experiment_name = "vae_training"
    try:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    except:
        experiment_id = mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # Setup VAEs and training parameters

    # %%
    # Train the model
    print("\nStarting training...")
    print("Current working directory:", os.getcwd())
    print("Python path:", sys.path)
    adata_rna_subset_original = adata_rna_subset.copy()
    # Create new AnnData with PCA values
    # adata_rna_subset = sc.AnnData(
    #     X=adata_rna_subset_original.obsm["X_pca"] - np.min(adata_rna_subset_original.obsm["X_pca"]),
    #     obs=adata_rna_subset_original.obs.copy(),
    #     var=pd.DataFrame(
    #         index=[f"PC_{i}" for i in range(adata_rna_subset_original.obsm["X_pca"].shape[1])]
    #     ),
    #     obsm=adata_rna_subset_original.obsm.copy(),
    #     uns=adata_rna_subset_original.uns.copy(),
    # )
    # # Create new AnnData with PCA values for protein data
    # adata_prot_subset_original = adata_prot_subset.copy()
    # adata_prot_subset = sc.AnnData(
    #     X=adata_prot_subset_original.obsm["X_pca"] - np.min(adata_prot_subset_original.obsm["X_pca"]),
    #     obs=adata_prot_subset_original.obs.copy(),
    #     var=pd.DataFrame(
    #         index=[f"PC_{i}" for i in range(adata_prot_subset_original.obsm["X_pca"].shape[1])]
    #     ),
    #     obsm=adata_prot_subset_original.obsm.copy(),
    #     uns=adata_prot_subset_original.uns.copy(),
    # )

    training_kwargs = {
        "max_epochs": 4,
        "batch_size": 1200,
        "train_size": 0.9,
        "validation_size": 0.1,
        "check_val_every_n_epoch": 1,
        "early_stopping": True,
        "early_stopping_patience": 20,
        "early_stopping_monitor": "val_total_loss",
        "devices": 1,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 1,
        "lr": 1e-4,
        "use_gpu": True,
        "plot_x_times": 3,
        "contrastive_weight": 10.0,
        "similarity_weight": 10000.0,
        "matching_weight": 1000000.0,
        "cell_type_clustering_weight": 0.1,
        "kl_weight_rna": 0.1,
        "kl_weight_prot": 10.0,
    }
    # %%
    rna_vae, protein_vae, latent_rna_before, latent_prot_before = train_vae(
        adata_rna_subset=adata_rna_subset,
        adata_prot_subset=adata_prot_subset,
        **training_kwargs,
    )
    print("Training completed")
    rna_vae_new = rna_vae

    # %%
    # Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("vae_training")
    run_name = f"vae_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Log parameters
    mlflow.log_params(
        {
            "batch_size": training_kwargs["batch_size"],
            "use_gpu": training_kwargs["use_gpu"],
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

    # %%
    # Get training history from the training plan
    print("\nGetting training history...")
    history = rna_vae_new._training_plan.get_history()
    print("✓ Training history loaded")

    # Log training history metrics
    mlflow.log_metrics(
        {
            "final_train_similarity_loss": history["train_similarity_loss"][-1],
            "final_train_similarity_loss_raw": history["train_similarity_loss_raw"][-1],
            # "final_train_similarity_weight": history["train_similarity_weight"][-1],
            "final_train_total_loss": history["train_total_loss"][-1],
            "final_val_total_loss": history["val_total_loss"][-1],
            "final_train_cell_type_clustering_loss": history["train_cell_type_clustering_loss"][-1],
        }
    )

    # %%
    print("\Get latent representations...")
    latent_rna = rna_vae_new.adata.obsm["X_scVI"]
    latent_prot = protein_vae.adata.obsm["X_scVI"]

    # Store latent representations
    print("\nStoring latent representations...")
    SCVI_LATENT_KEY = "X_scVI"
    rna_vae_new.adata.obs["CN"] = rna_vae.adata.obs["CN"].values
    rna_vae_new.adata.obsm[SCVI_LATENT_KEY] = latent_rna
    protein_vae.adata.obsm[SCVI_LATENT_KEY] = latent_prot
    print("✓ Latent representations stored")

    # Prepare AnnData objects
    print("\nPreparing AnnData objects...")
    rna_latent = AnnData(rna_vae_new.adata.obsm[SCVI_LATENT_KEY].copy())
    prot_latent = AnnData(protein_vae.adata.obsm[SCVI_LATENT_KEY].copy())
    rna_latent.obs = rna_vae_new.adata.obs.copy()
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
        (rna_vae_new.adata.obs["major_cell_types"], protein_vae.adata.obs["major_cell_types"]),
        join="outer",
    )
    combined_latent.obs["major_cell_types"] = combined_major_cell_types
    combined_latent.obs["cell_types"] = pd.concat(
        (rna_vae_new.adata.obs["cell_types"], protein_vae.adata.obs["cell_types"]), join="outer"
    )
    combined_latent.obs["CN"] = pd.concat(
        (rna_vae_new.adata.obs["CN"], protein_vae.adata.obs["CN"]), join="outer"
    )
    sc.pp.pca(combined_latent)
    sc.pp.neighbors(combined_latent, n_neighbors=15)
    try:
        sc.tl.umap(combined_latent, min_dist=0.1)
        print("✓ UMAP computed successfully")
    except Exception as e:
        print(f"Warning: UMAP computation failed: {str(e)}")
        print("Continuing with other visualizations...")
    print("✓ Latent spaces combined")
    # %%
    print("\nMatching cells between modalities...")
    # Calculate pairwise distances between RNA and protein cells in latent space

    latent_distances = cdist(rna_latent.X, prot_latent.X)
    rand_latent_distances = latent_distances[np.random.permutation(len(rna_latent)), :]

    # Find closest matches for RNA cells to protein cells
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

    # %%

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
    # %%
    # Plot training results
    print("\nPlotting training results...")
    plot_normalized_losses(history)
    print("✓ Training losses plotted")

    plot_umap_visualizations_original_data(rna_vae_new.adata, protein_vae.adata)
    # Plot spatial data
    print("\nPlotting spatial data...")
    plot_spatial_data(protein_vae.adata)
    print("✓ Spatial data plotted")

    # Plot latent representations
    print("\nPlotting latent representations...")
    plot_latent_pca_both_modalities_cn(
        latent_rna,
        latent_prot,
        rna_vae_new.adata,
        protein_vae.adata,
        index_rna=range(len(rna_vae_new.adata.obs.index)),
        index_prot=range(len(protein_vae.adata.obs.index)),
    )
    plot_latent_pca_both_modalities_by_celltype(
        rna_vae_new.adata, protein_vae.adata, latent_rna, latent_prot
    )
    print("✓ Latent representations plotted")

    # Plot distance distributions
    print("\nPlotting distance distributions...")
    compare_distance_distributions(rand_distances, rna_latent, prot_latent, distances)
    print("✓ Distance distributions plotted")

    # Plot combined visualizations
    print("\nPlotting combined visualizations...")
    plot_combined_latent_space(combined_latent)

    plot_cell_type_distributions(combined_latent, 3)

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
    plot_archetype_embedding(rna_vae_new, protein_vae)

    plot_rna_protein_latent_cn_cell_type_umap(rna_vae_new, protein_vae)

    print("✓ Archetype and embedding visualizations plotted")

    # Calculate and display final metrics
    print("\nCalculating final metrics...")
    mixing_result = mixing_score(
        latent_rna,
        latent_prot,
        rna_vae_new.adata,
        protein_vae.adata,
        index_rna=range(len(rna_vae_new.adata)),
        index_prot=range(len(protein_vae.adata)),
        plot_flag=True,
    )
    print(f"✓ Mixing score: {mixing_result}")

    nmi_cell_types_cn_rna = adjusted_mutual_info_score(
        rna_vae_new.adata.obs["cell_types"], rna_vae_new.adata.obs["CN"]
    )
    nmi_cell_types_cn_prot = adjusted_mutual_info_score(
        protein_vae.adata.obs["cell_types"], protein_vae.adata.obs["CN"]
    )
    # %%
    nmi_cell_types_modalities = adjusted_mutual_info_score(
        rna_vae_new.adata.obs["cell_types"].values[prot_matches_in_rna],
        protein_vae.adata.obs["cell_types"].values,
    )
    matches = (
        rna_vae_new.adata.obs["cell_types"].values[prot_matches_in_rna]
        == protein_vae.adata.obs["cell_types"].values
    )

    accuracy = matches.sum() / len(matches)

    print(f"\nFinal Metrics:")
    print(f"Normalized Mutual Information (RNA CN): {nmi_cell_types_cn_rna:.3f}")
    print(f"Normalized Mutual Information (Protein CN): {nmi_cell_types_cn_prot:.3f}")
    print(f"Normalized Mutual Information (Cross-modality): {nmi_cell_types_modalities:.3f}")
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
    clean_uns_for_h5ad(rna_vae_new.adata)
    clean_uns_for_h5ad(protein_vae.adata)
    save_dir = Path("CODEX_RNA_seq/data/trained_data").absolute()
    time_stamp = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(save_dir, exist_ok=True)

    print(
        f"\nTrained RNA VAE dimensions: {rna_vae_new.adata.shape[0]} samples x {rna_vae_new.adata.shape[1]} features"
    )
    print(
        f"Trained Protein VAE dimensions: {protein_vae.adata.shape[0]} samples x {protein_vae.adata.shape[1]} features\n"
    )

    sc.write(Path(f"{save_dir}/rna_vae_trained_{time_stamp}.h5ad"), rna_vae_new.adata)
    sc.write(Path(f"{save_dir}/protein_vae_trained_{time_stamp}.h5ad"), protein_vae.adata)
    print("✓ Results saved")

    # Log artifacts
    mlflow.log_artifact(f"{save_dir}/rna_vae_trained_{time_stamp}.h5ad")
    mlflow.log_artifact(f"{save_dir}/protein_vae_trained_{time_stamp}.h5ad")

    # End MLflow run
    mlflow.end_run()

    print("\nAll visualization and analysis steps completed!")

    # %%

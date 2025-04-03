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
"""
TO use this script, you need to add the training plan to use the DualVAETrainingPlan class in scVI library.
in _training_mixin.py, line 131, you need to change the line:
training_plan = self._training_plan_cls(self.module, **plan_kwargs) # existing line
self._training_plan = training_plan # add this line

"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set working directory to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import scvi
import seaborn as sns
from anndata import AnnData
from matplotlib.patches import Arc
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.optimize import linear_sum_assignment
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# import torch F
from torch.nn.functional import normalize

# Add repository root to Python path without changing working directory

importlib.reload(scvi)
import re

import torch
import torch.nn.functional as F
from scvi.model import SCVI
from scvi.train import TrainingPlan
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

import bar_nick_utils

importlib.reload(bar_nick_utils)

# Import plotting functions
import CODEX_RNA_seq.plotting_functions_vae
from bar_nick_utils import (
    clean_uns_for_h5ad,
    compare_distance_distributions,
    compute_pairwise_kl,
    compute_pairwise_kl_two_items,
    get_umap_filtered_fucntion,
    mixing_score,
    select_gene_likelihood,
)

importlib.reload(CODEX_RNA_seq.plotting_functions_vae)
# Import logging functions
import CODEX_RNA_seq.logging_functions
from CODEX_RNA_seq.plotting_functions_vae import (
    plot_archetype_vectors,
    plot_archetype_vs_latent_distances,
    plot_cell_type_distributions,
    plot_combined_latent_space,
    plot_combined_latent_space_umap,
    plot_cosine_distance,
    plot_inference_outputs,
    plot_latent,
    plot_latent_distances,
    plot_latent_mean_std,
    plot_normalized_losses,
    plot_rna_protein_embeddings,
    plot_rna_protein_matching_means_and_scale,
    plot_spatial_data,
)

importlib.reload(CODEX_RNA_seq.logging_functions)
from CODEX_RNA_seq.logging_functions import (
    load_history,
    log_epoch_end,
    log_extra_metrics,
    log_step_metrics,
    log_training_metrics,
    log_validation_metrics,
    print_distance_metrics,
    print_training_metrics,
    setup_logging,
    update_log,
)

# reimport the two above too

if not hasattr(sc.tl.umap, "_is_wrapped"):
    sc.tl.umap = get_umap_filtered_fucntion()
    sc.tl.umap._is_wrapped = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pd.set_option("display.max_columns", 10)
pd.set_option("display.max_rows", 10)
warnings.filterwarnings("ignore")
pd.options.display.max_rows = 10
pd.options.display.max_columns = 10
np.set_printoptions(threshold=100)
np.random.seed(0)
save_dir = "CODEX_RNA_seq/data/processed_data"

plot_flag = True

# %%
# read in the data
cwd = os.getcwd()
save_dir = Path("CODEX_RNA_seq/data/processed_data").absolute()

adata_rna_subset = sc.read_h5ad(f"{save_dir}/adata_rna_subset_prepared_for_training.h5ad")
adata_prot_subset = sc.read_h5ad(f"{save_dir}/adata_prot_subset_prepared_for_training.h5ad")

# Subsample the data for faster testing
print(f"Original RNA dataset shape: {adata_rna_subset.shape}")
print(f"Original protein dataset shape: {adata_prot_subset.shape}")

# Subsample to 20% of the cells for testing
np.random.seed(42)  # For reproducibility
frac = 1
sc.pp.subsample(adata_rna_subset, fraction=frac)
sc.pp.subsample(adata_prot_subset, fraction=frac)

print(f"Subsampled RNA dataset shape: {adata_rna_subset.shape}")
print(f"Subsampled protein dataset shape: {adata_prot_subset.shape}")

adata_rna_subset.X = adata_rna_subset.X.astype(np.float32)
adata_prot_subset.X = adata_prot_subset.X.astype(np.float32)
adata_prot_subset.obs["CN"] = adata_prot_subset.obs["CN"].astype(int)
adata_rna_subset.obs["CN"] = adata_rna_subset.obs["CN"].astype(int)
adata_prot_subset.obs["CN"] = pd.Categorical(adata_prot_subset.obs["CN"])
adata_rna_subset.obs["CN"] = pd.Categorical(adata_rna_subset.obs["CN"])


# %%
# Define the DualVAETrainingPlan class
class DualVAETrainingPlan(TrainingPlan):
    def __init__(self, rna_module, **kwargs):
        protein_vae = kwargs.pop("protein_vae")
        rna_vae = kwargs.pop("rna_vae")
        self.plot_x_times = kwargs.pop("plot_x_times", 5)
        contrastive_weight = kwargs.pop("contrastive_weight", 1.0)
        self.batch_size = kwargs.pop("batch_size", 1000)
        n_epochs = kwargs.pop("n_epochs", 1)
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
        self.total_steps = steps_per_epoch * n_epochs
        self.similarity_loss_history = []
        self.steady_state_window = 50
        self.steady_state_tolerance = 0.5
        self.similarity_weight = 100000
        self.similarity_active = True
        self.reactivation_threshold = 0.1
        self.active_similarity_loss_active_history = []
        self.train_losses = []
        self.val_losses = []
        self.similarity_losses = []  # Store similarity losses
        self.similarity_losses_raw = []  # Store raw similarity losses
        self.similarity_weights = []  # Store similarity weights
        self.similarity_ratios = []  # Store similarity ratios

        # Setup logging
        self.log_file = setup_logging()

    def training_step(self, batch, batch_idx):
        rna_batch = self._get_rna_batch(batch)
        kl_weight = 2
        self.loss_kwargs.update({"kl_weight": kl_weight})
        _, _, rna_loss_output = self.rna_vae.module(rna_batch, loss_kwargs=self.loss_kwargs)
        protein_batch = self._get_protein_batch(batch)
        _, _, protein_loss_output = self.protein_vae.module(
            protein_batch, loss_kwargs=self.loss_kwargs
        )

        rna_inference_outputs = self.rna_vae.module.inference(
            rna_batch["X"], batch_index=rna_batch["batch"], n_samples=1
        )
        index = rna_batch["labels"]

        protein_inference_outputs = self.protein_vae.module.inference(
            protein_batch["X"], batch_index=protein_batch["batch"], n_samples=1
        )

        archetype_dis = torch.cdist(
            normalize(rna_batch["archetype_vec"], dim=1),
            normalize(protein_batch["archetype_vec"], dim=1),
        )

        latent_distances = compute_pairwise_kl_two_items(
            rna_inference_outputs["qz"].mean,
            protein_inference_outputs["qz"].mean,
            rna_inference_outputs["qz"].scale,
            protein_inference_outputs["qz"].scale,
        )
        latent_distances = torch.clamp(latent_distances, max=torch.quantile(latent_distances, 0.90))

        should_plot = (
            self.global_step > -1
            and self.global_step % (1 + int(self.total_steps / (self.plot_x_times))) == 0
            and plot_flag
        )

        if should_plot:
            plot_latent_mean_std(
                rna_inference_outputs,
                protein_inference_outputs,
                adata_rna_subset,
                adata_prot_subset,
                index=protein_batch["labels"],
            )

            plot_rna_protein_matching_means_and_scale(
                rna_inference_outputs, protein_inference_outputs
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
        rna_distances = compute_pairwise_kl(
            rna_inference_outputs["qz"].mean, rna_inference_outputs["qz"].scale
        )
        prot_distances = compute_pairwise_kl(
            protein_inference_outputs["qz"].mean, protein_inference_outputs["qz"].scale
        )
        distances = 5 * prot_distances + rna_distances

        rna_size = prot_size = rna_batch["X"].shape[0]
        mixed_latent = torch.cat(
            [rna_inference_outputs["qz"].mean, protein_inference_outputs["qz"].mean], dim=0
        )
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

        cell_neighborhood_info = torch.tensor(self.protein_vae.adata[index].obs["CN"].values).to(
            device
        )
        rna_major_cell_type = (
            torch.tensor(self.rna_vae.adata[index].obs["major_cell_types"].values.codes)
            .to(device)
            .squeeze()
        )
        protein_major_cell_type = (
            torch.tensor(self.protein_vae.adata[index].obs["major_cell_types"].values.codes)
            .to(device)
            .squeeze()
        )

        num_cells = self.rna_vae.adata[index].shape[0]
        same_cn_mask = cell_neighborhood_info.unsqueeze(0) == cell_neighborhood_info.unsqueeze(1)
        same_major_cell_type = rna_major_cell_type.unsqueeze(
            0
        ) == protein_major_cell_type.unsqueeze(1)
        diagonal_mask = torch.eye(num_cells, dtype=torch.bool, device=cell_neighborhood_info.device)

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

        # Calculate similarity loss
        is_steady_state = False
        if len(self.similarity_loss_history) > self.steady_state_window:
            recent_history = self.similarity_loss_history[-self.steady_state_window :]
            variation = np.std(recent_history) / (np.mean(recent_history) + 1e-10)
            is_steady_state = variation < self.steady_state_tolerance

        if is_steady_state and self.similarity_active:
            self.similarity_active = False
        elif not self.similarity_active and self.global_step % 50 == 0:
            if self.global_step > 50:
                recent_loss = np.mean(self.similarity_loss_history[-50:])
                if recent_loss > self.reactivation_threshold:
                    self.similarity_active = True
                    self.similarity_weight = 1000

        if self.similarity_active:
            similarity_loss_raw = torch.sum(distances * (-same_cn_mask.float())) / (
                torch.sum(-same_cn_mask.float()) + 1e-10
            )
            ratio = self.similarity_weight / 1000
            similarity_loss = similarity_loss_raw * self.similarity_weight
        else:
            similarity_loss_raw = torch.tensor(0.0).to(device)
            ratio = 0.0
            similarity_loss = torch.tensor(0.0).to(device)

        # Store similarity metrics
        self.similarity_losses.append(similarity_loss.item())
        self.similarity_losses_raw.append(similarity_loss_raw.item())
        self.similarity_weights.append(self.similarity_weight)
        self.similarity_ratios.append(ratio)

        self.similarity_loss_history.append(similarity_loss_raw.item())
        self.active_similarity_loss_active_history.append(self.similarity_active)

        total_loss = (
            rna_loss_output.loss
            + protein_loss_output.loss
            + contrastive_loss
            + adv_loss
            + matching_loss
            + similarity_loss
            + diversity_loss
        )

        should_plot = (
            self.global_step > -1
            and self.global_step % (1 + int(self.total_steps / (self.plot_x_times))) == 0
            and plot_flag
        )

        if should_plot:
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
                ratio,
                self.similarity_active,
                num_acceptable,
                num_cells,
                exact_pairs,
            )

            plot_latent_mean_std(
                rna_inference_outputs,
                protein_inference_outputs,
                adata_rna_subset,
                adata_prot_subset,
                index=protein_batch["labels"],
            )

            plot_rna_protein_matching_means_and_scale(
                rna_inference_outputs, protein_inference_outputs
            )
            print(f"min latent distances: {round(latent_distances.min().item(),3)}")
            print(f"max latent distances: {round(latent_distances.max().item(),3)}")
            print(f"mean latent distances: {round(latent_distances.mean().item(),3)}")

        update_log(self.log_file, "train_similarity_loss_raw", similarity_loss_raw.item())
        update_log(self.log_file, "train_similarity_weighted", similarity_loss.item())
        update_log(self.log_file, "train_similarity_weight", self.similarity_weight)
        update_log(self.log_file, "train_similarity_ratio", ratio)

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
        return total_loss

    def validation_step(self, batch, batch_idx):
        rna_batch = self._get_rna_batch(batch)
        protein_batch = self._get_protein_batch(batch)

        _, _, rna_loss_output = self.rna_vae.module(rna_batch)
        _, _, protein_loss_output = self.protein_vae.module(protein_batch)

        rna_inference_outputs = self.rna_vae.module.inference(
            rna_batch["X"], batch_index=rna_batch["batch"], n_samples=1
        )
        protein_inference_outputs = self.protein_vae.module.inference(
            protein_batch["X"], batch_index=protein_batch["batch"], n_samples=1
        )

        latent_distances = compute_pairwise_kl_two_items(
            rna_inference_outputs["qz"].mean,
            protein_inference_outputs["qz"].mean,
            rna_inference_outputs["qz"].scale,
            protein_inference_outputs["qz"].scale,
        )

        validation_total_loss = rna_loss_output.loss + protein_loss_output.loss

        log_validation_metrics(
            self.log_file,
            rna_loss_output,
            protein_loss_output,
            torch.tensor(0.0),  # contrastive loss not used in validation
            validation_total_loss,
            latent_distances,
        )

        self.val_losses.append(validation_total_loss.item())
        return validation_total_loss

    def on_epoch_end(self):
        log_epoch_end(self.log_file, self.current_epoch, self.train_losses, self.val_losses)
        self.train_losses = []
        self.val_losses = []

    def _get_protein_batch(self, batch):
        indices = batch["labels"].detach().cpu().numpy().flatten()
        indices = np.sort(indices)

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

    def _get_rna_batch(self, batch):
        indices = batch["labels"].detach().cpu().numpy().flatten()
        indices = np.sort(indices)
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
            "train_similarity_weight": self.similarity_weights,
            "train_similarity_ratio": self.similarity_ratios,
            "train_total_loss": self.train_losses,
            "val_total_loss": self.val_losses,
        }


# %%
def train_vae(
    adata_rna_subset, adata_prot_subset, n_epochs=1, batch_size=128, lr=1e-3, use_gpu=True, **kwargs
):
    """Train the VAE models."""
    print("Initializing VAEs...")
    # setup adata for scvi
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
        n_hidden=128,
        n_layers=3,
    )
    protein_vae = scvi.model.SCVI(
        adata_prot_subset, gene_likelihood="normal", n_hidden=50, n_layers=3
    )
    print("VAEs initialized")

    rna_vae._training_plan_cls = DualVAETrainingPlan

    print("Setting up TensorBoard logger...")
    logger = TensorBoardLogger(
        save_dir="my_logs", name=f"experiment_name_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    print("TensorBoard logger setup complete")

    # Set up training parameters with smaller batch size
    train_kwargs = {
        "max_epochs": n_epochs,
        "batch_size": batch_size,
        "train_size": 0.9,
        "validation_size": 0.1,
        "early_stopping": False,
        "check_val_every_n_epoch": 1,
        "accelerator": "gpu" if use_gpu and torch.cuda.is_available() else "cpu",
        "devices": 1,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 4,
    }
    print("Training parameters:", train_kwargs)

    # Create training plan with both VAEs
    plan_kwargs = {
        "protein_vae": protein_vae,
        "rna_vae": rna_vae,
        "contrastive_weight": kwargs.pop("contrastive_weight", 1.0),
        "plot_x_times": kwargs.pop("plot_x_times", 5),
        "batch_size": batch_size,
        "n_epochs": n_epochs,
    }
    print("Plan parameters:", plan_kwargs)

    # Create training plan instance
    print("Creating training plan...")
    training_plan = DualVAETrainingPlan(rna_vae.module, **plan_kwargs)
    print("Training plan created")

    # Train the model
    print("Starting training...")
    rna_vae.train(**train_kwargs, plan_kwargs=plan_kwargs)
    print("Training completed")

    # Manually set trained flag
    rna_vae.is_trained_ = True
    protein_vae.is_trained_ = True
    print("Training flags set")

    return rna_vae, protein_vae


# %%
# Setup VAEs and training parameters
training_kwargs = {
    "contrastive_weight": 10.0,
    "plot_x_times": 5,
}

# %%
# Train the model
print("\nStarting training...")
print("Current working directory:", os.getcwd())
print("Python path:", sys.path)

rna_vae, protein_vae = train_vae(
    adata_rna_subset=adata_rna_subset,
    adata_prot_subset=adata_prot_subset,
    n_epochs=10,
    batch_size=1000,
    lr=1e-3,
    use_gpu=True,
    **training_kwargs,
)
print("Training completed")
rna_vae_new = rna_vae

# Get training history from the training plan
print("\nGetting training history...")
history = rna_vae_new._training_plan.get_history()
print("✓ Training history loaded")

print("\nPreparing models for visualization...")
rna_vae_new.module.to(device)
protein_vae.module.to(device)
rna_vae_new.module.eval()
protein_vae.module.eval()
print("✓ Models prepared")

# Generate latent representations
print("\nGenerating latent representations...")
with torch.no_grad():
    latent_rna = rna_vae_new.get_latent_representation()
    latent_prot = protein_vae.get_latent_representation()
print("✓ Latent representations generated")

# Store latent representations
print("\nStoring latent representations...")
SCVI_LATENT_KEY = "X_scVI"
rna_vae_new.adata.obs["CN"] = protein_vae.adata.obs["CN"].values
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
    [rna_latent.copy(), prot_latent.copy()], join="outer", label="modality", keys=["RNA", "Protein"]
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
sc.pp.neighbors(combined_latent)
sc.tl.umap(combined_latent, min_dist=0.1)
print("✓ Latent spaces combined")

# Calculate distances and metrics
print("\nCalculating distances and metrics...")
distances = np.linalg.norm(rna_latent.X - prot_latent.X, axis=1)
rand_rna_latent = rna_latent.copy()
shuffled_indices = np.random.permutation(rand_rna_latent.obs.index)
rand_rna_latent = rand_rna_latent[shuffled_indices].copy()
rand_distances = np.linalg.norm(rand_rna_latent.X - prot_latent.X, axis=1)
print("✓ Distances calculated")

# Plot training results
print("\nPlotting training results...")
plot_normalized_losses(history)
print("✓ Training losses plotted")

# Plot spatial data
print("\nPlotting spatial data...")
plot_spatial_data(rna_vae_new.adata, protein_vae.adata)
print("✓ Spatial data plotted")

# Plot latent representations
print("\nPlotting latent representations...")
plot_latent(
    latent_rna,
    latent_prot,
    rna_vae_new.adata,
    protein_vae.adata,
    index=range(len(protein_vae.adata.obs.index)),
)
print("✓ Latent representations plotted")

# Plot distance distributions
print("\nPlotting distance distributions...")
plot_latent_distances(rna_latent, prot_latent, distances)
compare_distance_distributions(rand_distances, rna_latent, prot_latent, distances)
print("✓ Distance distributions plotted")

# Plot combined visualizations
print("\nPlotting combined visualizations...")
plot_combined_latent_space(combined_latent)
plot_combined_latent_space_umap(combined_latent)
plot_cell_type_distributions(combined_latent, 3)
print("✓ Combined visualizations plotted")

# Plot UMAP visualizations
print("\nPlotting UMAP visualizations...")
sc.pl.umap(
    combined_latent,
    color=["CN", "modality"],
    title=["UMAP Combined Latent space CN", "UMAP Combined Latent space modality"],
    alpha=0.5,
)
sc.pl.umap(
    combined_latent,
    color=["CN", "modality", "cell_types"],
    title=[
        "UMAP Combined Latent space CN",
        "UMAP Combined Latent space modality",
        "UMAP Combined Latent space cell types",
    ],
    alpha=0.5,
)
sc.pl.pca(
    combined_latent,
    color=["CN", "modality"],
    title=["PCA Combined Latent space CN", "PCA Combined Latent space modality"],
    alpha=0.5,
)
print("✓ UMAP visualizations plotted")

# Plot archetype and embedding visualizations
print("\nPlotting archetype and embedding visualizations...")
plot_archetype_vectors(rna_vae_new, protein_vae)
plot_rna_protein_embeddings(rna_vae_new, protein_vae)
print("✓ Archetype and embedding visualizations plotted")

# Calculate and display final metrics
print("\nCalculating final metrics...")
mixing_result = mixing_score(
    latent_rna,
    latent_prot,
    rna_vae_new.adata,
    protein_vae.adata,
    index=range(len(rna_vae_new.adata)),
    plot_flag=True,
)
print(f"✓ Mixing score: {mixing_result}")

nmi_cell_types_cn_rna = adjusted_mutual_info_score(
    rna_vae_new.adata.obs["cell_types"], rna_vae_new.adata.obs["CN"]
)
nmi_cell_types_cn_prot = adjusted_mutual_info_score(
    protein_vae.adata.obs["cell_types"], protein_vae.adata.obs["CN"]
)
nmi_cell_types_modalities = adjusted_mutual_info_score(
    rna_vae_new.adata.obs["cell_types"], protein_vae.adata.obs["cell_types"]
)
matches = rna_vae_new.adata.obs["cell_types"].values == protein_vae.adata.obs["cell_types"].values
accuracy = matches.sum() / len(matches)

print(f"\nFinal Metrics:")
print(f"Normalized Mutual Information (RNA CN): {nmi_cell_types_cn_rna:.3f}")
print(f"Normalized Mutual Information (Protein CN): {nmi_cell_types_cn_prot:.3f}")
print(f"Normalized Mutual Information (Cross-modality): {nmi_cell_types_modalities:.3f}")
print(f"Cell Type Matching Accuracy: {accuracy:.4f}")
print("✓ Final metrics calculated")

# Save results
print("\nSaving results...")
clean_uns_for_h5ad(rna_vae_new.adata)
clean_uns_for_h5ad(protein_vae.adata)
save_dir = Path("CODEX_RNA_seq/data/trained_data").absolute()
sc.write(Path(f"{save_dir}/rna_vae_trained.h5ad"), rna_vae_new.adata)
sc.write(Path(f"{save_dir}/protein_vae_trained.h5ad"), protein_vae.adata)
print("✓ Results saved")

print("\nAll visualization and analysis steps completed!")

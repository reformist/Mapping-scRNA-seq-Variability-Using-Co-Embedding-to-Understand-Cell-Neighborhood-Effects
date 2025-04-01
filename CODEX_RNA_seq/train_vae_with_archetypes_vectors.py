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

from bar_nick_utils import (
    archetype_vs_latent_distances_plot,
    calculate_cLISI,
    calculate_iLISI,
    clean_uns_for_h5ad,
    compare_distance_distributions,
    compute_pairwise_kl,
    compute_pairwise_kl_two_items,
    get_latest_file,
    get_umap_filtered_fucntion,
    match_datasets,
    mixing_score,
    plot_cosine_distance,
    plot_inference_outputs,
    plot_latent,
    plot_latent_mean_std,
    plot_normalized_losses,
    plot_rna_protein_matching_means_and_scale,
    plot_similarity_loss_history,
    select_gene_likelihood,
    verify_gradients,
)

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
        self.batch_size = kwargs.pop("batch_size", 128)
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
        # if self.protein_vae.adata.uns.get("ordered_matching_cells") is not True:
        #     raise ValueError("The cells are not aligned across modalities, make sure ")
        n_samples = len(self.rna_vae.adata)
        steps_per_epoch = int(np.ceil(n_samples / self.batch_size))
        self.total_steps = steps_per_epoch * n_epochs
        self.similarity_loss_history = []
        self.steady_state_window = 50  # Number of steps to check for steady state
        self.steady_state_tolerance = 0.5  # Tolerance for determining steady state
        self.similarity_weight = 100000  # Initial weight
        self.similarity_active = True  # Flag to track if similarity loss is active
        self.reactivation_threshold = 0.1  # Threshold to reactivate similarity loss
        self.active_similarity_loss_active_history = []
        self.similarity_loss_all_history = []

        # Initialize history dictionary with step tracking
        self.history = {
            "step": [],  # Track step number
            "timestamp": [],  # Track timestamp for each step
            "epoch": [],  # Track current epoch
            "train_total_loss": [],
            "train_rna_reconstruction_loss": [],
            "train_protein_reconstruction_loss": [],
            "train_contrastive_loss": [],
            "train_matching_rna_protein_loss": [],
            "train_similarity_loss": [],
            "train_similarity_loss_raw": [],
            "train_similarity_weighted": [],
            "train_similarity_weight": [],
            "train_similarity_ratio": [],
            "train_adv_loss": [],
            "train_diversity_loss": [],
            "validation_total_loss": [],
            "validation_rna_loss": [],
            "validation_protein_loss": [],
            "validation_contrastive_loss": [],
            "validation_matching_latent_distances": [],
            "learning_rate": [],  # Track learning rate
            "batch_size": [],  # Track batch size
            "gradient_norm": [],  # Track gradient norm
        }
        print("Initialized history dictionary with keys:", self.history.keys())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.rna_vae.module.parameters()) + list(self.protein_vae.module.parameters()),
            lr=0.001,
            weight_decay=1e-5,
        )
        # d = { # maybe add this?
        # "optimizer": optimizer,
        # "gradient_clip_val": 1.0,  # Critical for stability
        # "gradient_clip_algorithm": "value"
        # }
        return optimizer

    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        rna_batch = self._get_rna_batch(batch)
        _, _, rna_loss_output = self.rna_vae.module(rna_batch, loss_kwargs=self.loss_kwargs)
        protein_batch = self._get_protein_batch(batch)
        _, _, protein_loss_output = self.protein_vae.module(
            protein_batch, loss_kwargs=self.loss_kwargs
        )

        # Inference outputs
        rna_inference_outputs = self.rna_vae.module.inference(
            rna_batch["X"], batch_index=rna_batch["batch"], n_samples=1
        )
        protein_inference_outputs = self.protein_vae.module.inference(
            protein_batch["X"], batch_index=protein_batch["batch"], n_samples=1
        )

        # Compute latent distances
        matching_rna_protein_latent_distances = torch.distributions.kl_divergence(
            rna_inference_outputs["qz"], protein_inference_outputs["qz"]
        )

        # Compute pairwise distances for RNA and protein
        rna_distances = compute_pairwise_kl(
            rna_inference_outputs["qz"].mean, rna_inference_outputs["qz"].scale
        )
        prot_distances = compute_pairwise_kl(
            protein_inference_outputs["qz"].mean, protein_inference_outputs["qz"].scale
        )

        # Retrieve cell neighborhood information
        index = rna_batch["labels"]
        cell_neighborhood_info = torch.tensor(
            self.protein_vae.adata[index].obs["CN"].values, device=device
        )
        rna_major_cell_type = torch.tensor(
            self.rna_vae.adata[index].obs["major_cell_types"].values.codes, device=device
        ).squeeze()
        protein_major_cell_type = torch.tensor(
            self.rna_vae.adata[index].obs["major_cell_types"].values.codes, device=device
        ).squeeze()

        num_cells = cell_neighborhood_info.shape[0]
        diagonal_mask = torch.eye(num_cells, dtype=torch.bool, device=device)

        same_cn_mask = cell_neighborhood_info.unsqueeze(0) == cell_neighborhood_info.unsqueeze(1)
        same_major_cell_type_mask = rna_major_cell_type.unsqueeze(
            0
        ) == protein_major_cell_type.unsqueeze(1)

        # Use protein distances or a combination as needed
        distances = prot_distances + rna_distances
        distances = distances.masked_fill(diagonal_mask, 0)

        # Define loss masks
        same_major_type_same_cn_loss = (distances**2) * (same_major_cell_type_mask & same_cn_mask)
        same_major_type_different_cn_loss = ((10 - distances).clamp(min=0) ** 2) * (
            same_major_cell_type_mask & ~same_cn_mask
        )
        different_major_type_same_cn_loss = ((10 - distances).clamp(min=0) ** 2) * (
            ~same_major_cell_type_mask & same_cn_mask
        )
        different_major_type_different_cn_loss = ((10 - distances).clamp(min=0) ** 2) * (
            ~same_major_cell_type_mask & ~same_cn_mask
        )

        cn_loss = (
            same_major_type_same_cn_loss.mean()
            + same_major_type_different_cn_loss.mean()
            + different_major_type_same_cn_loss.mean()
            + different_major_type_different_cn_loss.mean()
        ) / (num_cells * (num_cells - 1))

        validation_total_loss = (
            1 * self.contrastive_weight * cn_loss
            + 0.1 * rna_loss_output.loss
            + 0.1 * protein_loss_output.loss
            + 10 * matching_rna_protein_latent_distances.mean()
        )

        # Log metrics
        self.log(
            "validation_rna_loss", rna_loss_output.loss, on_epoch=True, sync_dist=self.use_sync_dist
        )
        self.log(
            "validation_protein_loss",
            protein_loss_output.loss,
            on_epoch=True,
            sync_dist=self.use_sync_dist,
        )
        self.log(
            "validation_contrastive_loss", cn_loss, on_epoch=True, sync_dist=self.use_sync_dist
        )
        self.log(
            "validation_total_loss",
            validation_total_loss,
            on_epoch=True,
            sync_dist=self.use_sync_dist,
        )
        self.log(
            "validation_matching_latent_distances",
            matching_rna_protein_latent_distances.mean(),
            on_epoch=True,
            sync_dist=self.use_sync_dist,
        )

        # Compute and log additional metrics
        self.compute_and_log_metrics(rna_loss_output, self.val_metrics, "validation")
        self.compute_and_log_metrics(protein_loss_output, self.val_metrics, "validation")

        # Log validation metrics and update history
        self.log(
            "validation_rna_loss", rna_loss_output.loss, on_epoch=True, sync_dist=self.use_sync_dist
        )
        self.history["validation_rna_loss"].append(rna_loss_output.loss.item())

        self.log(
            "validation_protein_loss",
            protein_loss_output.loss,
            on_epoch=True,
            sync_dist=self.use_sync_dist,
        )
        self.history["validation_protein_loss"].append(protein_loss_output.loss.item())

        self.log(
            "validation_contrastive_loss", cn_loss, on_epoch=True, sync_dist=self.use_sync_dist
        )
        self.history["validation_contrastive_loss"].append(cn_loss.item())

        self.log(
            "validation_total_loss",
            validation_total_loss,
            on_epoch=True,
            sync_dist=self.use_sync_dist,
        )
        self.history["validation_total_loss"].append(validation_total_loss.item())

        self.log(
            "validation_matching_latent_distances",
            matching_rna_protein_latent_distances.mean(),
            on_epoch=True,
            sync_dist=self.use_sync_dist,
        )
        self.history["validation_matching_latent_distances"].append(
            matching_rna_protein_latent_distances.mean().item()
        )

        if batch_idx == 0:  # Print validation metrics for first batch of each epoch
            print(f"\nValidation metrics at epoch {self.current_epoch}:")
            print(f"Total validation loss: {validation_total_loss.item():.4f}")
            print(f"RNA validation loss: {rna_loss_output.loss.item():.4f}")
            print(f"Protein validation loss: {protein_loss_output.loss.item():.4f}")
            print(f"Contrastive validation loss: {cn_loss.item():.4f}")

        return validation_total_loss

    def training_step(self, batch, batch_idx):
        rna_batch = self._get_rna_batch(batch)
        kl_weight = 2  # maybe make sure this is proper
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
        # assert len(set(self.protein_vae.adata[index].obs['CN'].values)) != 1# should this be commented out?

        protein_inference_outputs = self.protein_vae.module.inference(
            protein_batch["X"], batch_index=protein_batch["batch"], n_samples=1
        )
        # here we assume that the cells have been aligned in the same order to their best match across modalities (check adata_prot_subset.uns['ordered_matching_cells'])

        archetype_dis = scipy.spatial.distance.cdist(
            rna_batch["archetype_vec"], protein_batch["archetype_vec"], metric="cosine"
        )

        latent_distances = compute_pairwise_kl_two_items(
            rna_inference_outputs["qz"].mean,
            protein_inference_outputs["qz"].mean,
            rna_inference_outputs["qz"].scale,
            protein_inference_outputs["qz"].scale,
        )
        latent_distances = torch.clamp(latent_distances, max=torch.quantile(latent_distances, 0.90))

        if (
            self.global_step > -1
            and self.global_step % (1 + int(self.total_steps / (self.plot_x_times))) == 0
        ):
            latent_distances_temp = torch.cdist(
                rna_inference_outputs["qz"].mean, protein_inference_outputs["qz"].mean, p=2
            )
            plot_latent_mean_std(rna_inference_outputs, protein_inference_outputs)
            plot_rna_protein_matching_means_and_scale(
                rna_inference_outputs, protein_inference_outputs
            )
            print(f"min laten distances is {round(latent_distances.min().item(),3)}")
            print(f"max laten distances is {round(latent_distances.max().item(),3)}")
            print(f"mean laten distances is {round(latent_distances.mean().item(),3)}\n\n")

        archetype_dis_tensor = torch.tensor(
            archetype_dis, dtype=torch.float, device=latent_distances.device
        )
        threshold = 0.0005
        # normlize distances to [0,1] since we are using the same threshold for both archetype and latent distances
        archetype_dis_tensor = (archetype_dis_tensor - archetype_dis_tensor.min()) / (
            archetype_dis_tensor.max() - archetype_dis_tensor.min()
        )
        latent_distances = (latent_distances - latent_distances.min()) / (
            latent_distances.max() - latent_distances.min()
        )

        squared_diff = (latent_distances - archetype_dis_tensor) ** 2
        # Identify pairs that are close in the original space and remain close in the latent space
        acceptable_range_mask = (archetype_dis_tensor < threshold) & (latent_distances < threshold)
        stress_loss = squared_diff.mean()
        num_cells = squared_diff.numel()
        num_acceptable = acceptable_range_mask.sum()
        exact_pairs = 10 * torch.diag(latent_distances).mean()

        reward_strength = 0  # should be zero, if it is positive I think it cause all the sampel to be as close as possible into one central point which is not good
        # Apply the reward by subtracting from the loss based on how many acceptable pairs we have
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

        if (
            self.first_step and plot_flag and False
        ):  # show the mask only for the first batch to make sure it is working as expected
            plot_inference_outputs(
                rna_inference_outputs,
                protein_inference_outputs,
                latent_distances,
                rna_distances,
                prot_distances,
            )
            self.first_step = False
        if (
            self.global_step > -1
            and self.global_step % (1 + int(self.total_steps / (self.plot_x_times))) == 0
        ):
            print("mean prot distances is ", round(prot_distances.mean().item(), 3))
            print("mean rna distances is ", round(rna_distances.mean().item(), 3))
            print("after I multiply the prot distances by 5")

            verify_gradients(self.rna_vae.module, self.protein_vae)  # no funcitonal
            print("acceptable ratio", round(num_acceptable.float().item() / num_cells, 3))
            print("stress_loss", round(stress_loss.item(), 3))
            print("reward", round(reward.item(), 3))
            print("exact_pairs_loss", round(exact_pairs.item(), 3))
            print("matching_loss", round(matching_loss.item(), 3), "\n\n")
            "sssssssssssssssssssssssssssssssssssssssssssss"
            if plot_flag:
                plot_latent(
                    rna_inference_outputs["qz"].mean.clone().detach().cpu().numpy(),
                    protein_inference_outputs["qz"].mean.clone().detach().cpu().numpy(),
                    self.rna_vae.adata,
                    self.protein_vae.adata,
                    index=protein_batch["labels"],
                )
            mixing_score_ = mixing_score(
                rna_inference_outputs["qz"].mean,
                protein_inference_outputs["qz"].mean,
                adata_rna_subset,
                adata_prot_subset,
                index,
                plot_flag,
            )
            print(f"mixing score is {mixing_score_}\n\n")

            self.log(
                "extra_metric_acceptable_ratio",
                num_acceptable.float().item() / num_cells,
                on_epoch=False,
                on_step=True,
            )
            self.log("extra_metric_stress_loss", stress_loss.item(), on_epoch=False, on_step=True)
            self.log("extra_metric_reward", reward.item(), on_epoch=False, on_step=True)
            self.log(
                "extra_metric_exact_pairs_loss", exact_pairs.item(), on_epoch=False, on_step=True
            )
            self.log("extra_metric_iLISI", mixing_score_["iLISI"], on_epoch=False, on_step=True)
            self.log("extra_metric_cLISI", mixing_score_["cLISI"], on_epoch=False, on_step=True)

            # price accuracy for diversity
            accuracy = (batch_pred.argmax(dim=1) == batch_labels).float().mean()
            print(f"accuracy is {accuracy}")
            self.log("extra_metric_accuracy", accuracy, on_epoch=False, on_step=True)

            # plt.figure()
            # archetype_dis_tensor_ = archetype_dis_tensor.detach().cpu().numpy()
            # plt.hist(np.sort(archetype_dis_tensor_.flatten()),bins=100)
            # plt.hist(np.sort(archetype_dis_tensor_)[latent_distances.detach().cpu().numpy() < threshold].flatten(),bins=100)
            # plt.title(f'num of below threshold {np.sum(latent_distances.detach().cpu().numpy() < threshold)}')
            # plt.show()

            if plot_flag:
                archetype_vs_latent_distances_plot(
                    archetype_dis_tensor, latent_distances, threshold
                )
                plot_cosine_distance(rna_batch, protein_batch)
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
        # this will give us each row represents a item in the array, and each col is whether it is the same as the items in that index of the col
        # this way we get for each cell(a row) which other cells (index of each item in the row, which is the col) are matching
        # so if we got 1,2,1, we will get [[1,0,1],[0,1,0],[1,0,1]]
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

        same_major_type_same_cn_loss = (distances**2) * same_major_type_same_cn_mask
        same_major_type_different_cn_loss = (
            (10 - distances).clamp(min=0) ** 2
        ) * same_major_type_different_cn_mask
        different_major_type_same_cn_loss = (
            (10 - distances).clamp(min=0) ** 2
        ) * different_major_type_same_cn_mask
        different_major_type_different_cn_loss = (
            (10 - distances).clamp(min=0) ** 2
        ) * different_major_type_different_cn_mask
        # for debugging only: #
        same_cn_loss = (distances**2) * same_cn_mask
        same_major_type_loss = (distances**2) * same_major_cell_type
        # end of debugging
        positive_loss = same_major_type_same_cn_loss

        # negative_loss = different_major_type_different_cn_loss + different_major_type_same_cn_loss + 10* same_major_type_different_cn_loss
        negative_loss = same_major_type_different_cn_loss  # try to simplify loss
        cn_loss = (positive_loss.mean() + negative_loss.mean()) / (num_cells * (num_cells - 1))

        matching_loss = 1000 * matching_loss.mean()
        reconstruction_loss = rna_loss_output.loss * 000.1 + protein_loss_output.loss * 000.1
        # Calculate silhouette score for the CN clusters
        # if self.global_step % 50 ==0:
        #     cn_labels = cell_neighborhood_info.cpu().numpy()
        #     silhouette_avg_rna = silhouette_score(rna_inference_outputs["qz"].mean.detach().cpu().numpy(), cn_labels)
        #     silhouette_avg_prot = silhouette_score(protein_inference_outputs["qz"].mean.detach().cpu().numpy(), cn_labels)
        #     silhouette_avg = (silhouette_avg_rna + silhouette_avg_prot) / 2
        #
        #     if silhouette_avg < 0.1:
        #         self.contrastive_weight = self.contrastive_weight * 10
        #     else :
        #         self.contrastive_weight =  self.contrastive_weight / 10
        contrastive_loss = self.contrastive_weight * cn_loss * 1000000

        probs = F.softmax(batch_pred, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        adv_loss = -entropy.mean()  # Negative entropy to maximize uncertainty

        adv_loss = 1 * adv_loss

        # In training_step:
        all_latent = torch.cat(
            [rna_inference_outputs["qz"].mean, protein_inference_outputs["qz"].mean], dim=0
        )
        rna_dis = torch.cdist(rna_inference_outputs["qz"].mean, rna_inference_outputs["qz"].mean)
        prot_dis = torch.cdist(
            protein_inference_outputs["qz"].mean, protein_inference_outputs["qz"].mean
        )
        rna_prot_dis = torch.cdist(
            rna_inference_outputs["qz"].mean, protein_inference_outputs["qz"].mean
        )
        #
        # apply loss on similarity between rna and prot vs internal similarity
        # Calculate the similarity loss as the absolute difference between the average of the mean absolute clamped RNA and protein distances
        # and the mean absolute clamped RNA-protein distances. This helps in measuring the similarity between RNA and protein distances
        # while avoiding the influence of outliers by using clamped distances.
        similarity_loss_raw = torch.abs(
            ((rna_dis.abs().mean() + prot_dis.abs().mean()) / 2) - rna_prot_dis.abs().mean()
        )
        # Store the current loss value

        self.similarity_loss_history.append(similarity_loss_raw.item())

        # Only keep the most recent window of values
        if len(self.similarity_loss_history) > self.steady_state_window:
            self.similarity_loss_history.pop(0)

        # Determine if we're in steady state
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
        loss_increased = False
        if not self.similarity_active and len(self.similarity_loss_history) > 0:
            recent_loss = similarity_loss_raw.item()
            min_steady_loss = min(self.similarity_loss_history)
            if recent_loss > min_steady_loss * (1 + self.reactivation_threshold):
                loss_increased = True
        # Update the weight based on steady state detection
        if in_steady_state and self.similarity_active:
            current_similarity_weight = (
                self.similarity_weight / 1000
            )  # Zero out weight when in steady state
            self.similarity_active = False
        elif loss_increased and not self.similarity_active:
            current_similarity_weight = self.similarity_weight  # Reactivate with full weight
            self.similarity_active = True
        else:
            current_similarity_weight = self.similarity_weight if self.similarity_active else 0
        # if the mixing score is lower than 1.8 activaet the similarity loss
        # Apply weight to loss

        combined_latent = ad.concat(
            [
                AnnData(rna_inference_outputs["qz"].mean.detach().cpu().numpy()),
                AnnData(protein_inference_outputs["qz"].mean.detach().cpu().numpy()),
            ],
            join="outer",
            label="modality",
            keys=["RNA", "Protein"],
        )

        if self.global_step % 50 == 0:
            sc.pp.pca(combined_latent, n_comps=5)
            sc.pp.neighbors(combined_latent, use_rep="X_pca", n_neighbors=10)
            iLISI_score = calculate_iLISI(combined_latent, "modality", plot_flag=False)
            if iLISI_score < 1.9 and self.similarity_weight > 1e8:
                print()
                self.similarity_weight = self.similarity_weight * 10
            elif self.similarity_weight > 100:  # make it smaller only if it is not too small
                self.similarity_weight = self.similarity_weight / 10
        similarity_loss = current_similarity_weight * similarity_loss_raw
        self.active_similarity_loss_active_history.append(self.similarity_active)
        self.similarity_loss_all_history.append(similarity_loss.item())
        if (
            self.global_step > -1
            and self.global_step % (1 + int(self.total_steps / (self.plot_x_times))) == 0
        ):
            plot_similarity_loss_history(
                self.similarity_loss_all_history, self.active_similarity_loss_active_history
            )

        dis = torch.cdist(all_latent, all_latent)
        dis1 = dis[:rna_size, rna_size:]
        dis2 = dis[rna_size:, rna_size:]
        diversity_loss = torch.abs(dis1.mean() - dis2.mean())
        # print('diversity_loss',diversity_loss)
        diversity_loss = diversity_loss * 1000000
        total_loss = (
            reconstruction_loss
            + contrastive_loss
            + matching_loss
            + similarity_loss
            # + adv_loss
            # + diversity_loss
        )
        # Log losses and update history
        self.log(
            "train_similarity_loss_raw", similarity_loss_raw.item(), on_epoch=False, on_step=True
        )
        self.history["train_similarity_loss_raw"].append(similarity_loss_raw.item())

        self.log("train_similarity_weight", current_similarity_weight, on_epoch=False, on_step=True)
        self.history["train_similarity_weight"].append(current_similarity_weight)

        self.log("train_similarity_weighted", similarity_loss.item(), on_epoch=False, on_step=True)
        self.history["train_similarity_weighted"].append(similarity_loss.item())

        self.log("train_similarity_loss", similarity_loss.item(), on_epoch=False, on_step=True)
        self.history["train_similarity_loss"].append(similarity_loss.item())

        similarity_ratio = similarity_loss.item() / (total_loss.item() + 1e-8)
        self.log("train_similarity_ratio", similarity_ratio, on_epoch=False, on_step=True)
        self.history["train_similarity_ratio"].append(similarity_ratio)

        self.log(
            "train_rna_reconstruction_loss", rna_loss_output.loss, on_epoch=False, on_step=True
        )
        self.history["train_rna_reconstruction_loss"].append(rna_loss_output.loss.item())

        self.log(
            "train_protein_reconstruction_loss",
            protein_loss_output.loss,
            on_epoch=False,
            on_step=True,
        )
        self.history["train_protein_reconstruction_loss"].append(protein_loss_output.loss.item())

        self.log("train_contrastive_loss", contrastive_loss, on_epoch=False, on_step=True)
        self.history["train_contrastive_loss"].append(contrastive_loss.item())

        self.log("train_matching_rna_protein_loss", matching_loss, on_epoch=False, on_step=True)
        self.history["train_matching_rna_protein_loss"].append(matching_loss.item())

        self.log("train_total_loss", total_loss, on_epoch=False, on_step=True)
        self.history["train_total_loss"].append(total_loss.item())

        self.log("train_adv_loss", adv_loss, on_epoch=False, on_step=True)
        self.history["train_adv_loss"].append(adv_loss.item())

        self.log("train_diversity_loss", diversity_loss, on_epoch=False, on_step=True)
        self.history["train_diversity_loss"].append(diversity_loss.item())

        if self.global_step % 10 == 0:  # Print every 10 steps
            print(f"\nStep {self.global_step} - Current losses:")
            print(f"Total loss: {total_loss.item():.4f}")
            print(f"RNA reconstruction loss: {rna_loss_output.loss.item():.4f}")
            print(f"Protein reconstruction loss: {protein_loss_output.loss.item():.4f}")
            print(f"Contrastive loss: {contrastive_loss.item():.4f}")
            print(f"Matching loss: {matching_loss.item():.4f}")
            print(f"Similarity loss: {similarity_loss.item():.4f}")
            print(f"History size: {len(self.history['train_total_loss'])}")

        # self.saved_model = False if (self.current_epoch % 49 == 0) else True
        # if self.current_epoch % 50 == 0 and self.saved_model:
        #     print('sved model')
        #     rna_vae.save(save_dir, prefix=f'batch_{self.current_epoch}_', save_anndata=False, overwrite=True)
        #     self.saved_model =True
        return total_loss

    def _get_protein_batch(self, batch):
        indices = (
            batch["labels"].detach().cpu().numpy().flatten()
        )  # Assuming batch contains indices
        indices = np.sort(indices)

        protein_data = self.protein_vae.adata[indices]
        protein_batch = {
            "X": torch.tensor(
                protein_data.X.toarray() if issparse(protein_data.X) else protein_data.X
            ).to(device),
            "batch": torch.tensor(protein_data.obs["_scvi_batch"].values, dtype=torch.long).to(
                device
            ),
            "labels": indices,
            "archetype_vec": protein_data.obsm["archetype_vec"],
        }
        return protein_batch

    def _get_rna_batch(self, batch):
        indices = batch["labels"].detach().cpu().numpy().flatten()
        indices = np.sort(indices)
        rna_data = self.rna_vae.adata[indices]
        rna_batch = {
            "X": torch.tensor(rna_data.X.toarray() if issparse(rna_data.X) else rna_data.X).to(
                device
            ),
            "batch": torch.tensor(rna_data.obs["_scvi_batch"].values, dtype=torch.long).to(device),
            "labels": indices,
            "archetype_vec": rna_data.obsm["archetype_vec"],
        }
        return rna_batch

    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        print(f"\nEpoch {self.current_epoch} completed")
        print(
            f"Average training loss: {np.mean(self.history['train_total_loss'][-self.trainer.num_training_batches:]):.4f}"
        )
        if len(self.history["validation_total_loss"]) > 0:
            print(f"Latest validation loss: {self.history['validation_total_loss'][-1]:.4f}")
        print(f"History sizes: {[k + ': ' + str(len(v)) for k, v in self.history.items()]}")

    def get_history(self):
        """Return the training history."""
        return self.history


# %%
# Setup SCVI data
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

# %%


# %%
def train_vae(rna_vae, protein_vae, n_epochs=1, batch_size=128, lr=1e-3, use_gpu=True, **kwargs):
    """Train the VAE models."""

    SCVI.setup_anndata(
        rna_vae.adata,  # Pass the AnnData object, not the SCVI model
        labels_key="index_col",
    )
    SCVI.setup_anndata(
        protein_vae.adata,  # Pass the AnnData object, not the SCVI model
        labels_key="index_col",
    )

    # Create new VAE models with matched data
    rna_vae_new = scvi.model.SCVI(rna_vae.adata, gene_likelihood="zinb", n_hidden=50, n_layers=3)
    protein_vae_new = scvi.model.SCVI(
        protein_vae.adata, gene_likelihood="normal", n_hidden=50, n_layers=3
    )

    # Copy over the trained weights if they exist
    if hasattr(rna_vae.module, "state_dict"):
        rna_vae_new.module.load_state_dict(rna_vae.module.state_dict())
    if hasattr(protein_vae.module, "state_dict"):
        protein_vae_new.module.load_state_dict(protein_vae.module.state_dict())

    # Set training plan
    rna_vae_new._training_plan_cls = DualVAETrainingPlan
    protein_vae_new._training_plan_cls = DualVAETrainingPlan
    logger = TensorBoardLogger(
        save_dir="my_logs", name=f"experiment_name_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Set up training parameters
    train_kwargs = {
        "max_epochs": n_epochs,
        "batch_size": batch_size,
        "train_size": 0.9,
        "validation_size": 0.1,
        "early_stopping": False,
        "check_val_every_n_epoch": 1,
        "logger": logger,
        "accelerator": "gpu" if use_gpu and torch.cuda.is_available() else "cpu",
        "devices": 1,
    }

    # Create training plan with both VAEs
    plan_kwargs = {
        "protein_vae": protein_vae_new,
        "rna_vae": rna_vae_new,
        "contrastive_weight": kwargs.pop("contrastive_weight", 1.0),
        "plot_x_times": kwargs.pop("plot_x_times", 5),
        "batch_size": batch_size,
        "n_epochs": n_epochs,
    }

    # Create training plan instance to access history
    training_plan = DualVAETrainingPlan(rna_vae_new.module, **plan_kwargs)

    # Train the model
    rna_vae_new.train(**train_kwargs, plan_kwargs=plan_kwargs)

    # Store the training history
    rna_vae_new.__dict__["history"] = training_plan.history

    # Manually set trained flag
    rna_vae_new.is_trained_ = True
    protein_vae_new.is_trained_ = True

    return rna_vae_new, protein_vae_new


# %%
def setup_vaes():
    """Set up the VAE models."""
    # Select gene likelihood
    gene_likelihood = select_gene_likelihood(adata_rna_subset)
    print(f"Selected gene likelihood: {gene_likelihood}")

    # Initialize VAEs
    rna_vae = scvi.model.SCVI(
        adata_rna_subset, gene_likelihood=gene_likelihood, n_hidden=50, n_layers=3
    )
    protein_vae = scvi.model.SCVI(
        adata_prot_subset, gene_likelihood="normal", n_hidden=50, n_layers=3
    )

    # Store initial weights
    initial_weights = {name: param.clone() for name, param in rna_vae.module.named_parameters()}

    # Set training plan and device
    rna_vae._training_plan_cls = DualVAETrainingPlan
    protein_vae._training_plan_cls = DualVAETrainingPlan
    protein_vae.module.to("cpu")
    rna_vae.module.to("cpu")
    rna_vae.is_trained = protein_vae.is_trained = False

    return rna_vae, protein_vae


# %%
# Setup VAEs and training parameters
rna_vae, protein_vae = setup_vaes()
training_kwargs = {
    "contrastive_weight": 10.0,
    "plot_x_times": 10,
}

# %%
# Train the model
print("\nStarting training...")
rna_vae_new, protein_vae_new = train_vae(
    rna_vae=rna_vae,
    protein_vae=protein_vae,
    n_epochs=1,  # You can modify these parameters as needed
    batch_size=128,
    lr=1e-3,
    use_gpu=True,
    **training_kwargs,
)

# %%
# Check if history is available in the VAE's __dict__
if "history" not in rna_vae_new.__dict__:
    raise Exception("No training history found. Make sure training completed successfully.")

print("\nTraining completed successfully!")
print("Available metrics:", list(rna_vae_new.__dict__["history"].keys()))
print("Number of training steps:", len(rna_vae_new.__dict__["history"]["train_total_loss"]))
# Compare rna_vae_new and rna_vae
print("\nComparing RNA VAE models:")


# Check if models are identical
models_identical = all(
    torch.equal(p1.detach().cpu(), p2.detach().cpu())
    for (n1, p1), (n2, p2) in zip(
        rna_vae_new.module.named_parameters(), rna_vae.module.named_parameters()
    )
)
print(f"Models have identical parameters: {models_identical}")

# Compare key attributes
print("\nKey differences:")
print(f"is_trained: rna_vae={rna_vae.is_trained}, rna_vae_new={rna_vae_new.is_trained}")

# Check if history exists
has_history_old = hasattr(rna_vae, "history")
has_history_new = hasattr(rna_vae_new, "history")
print(f"Has training history: rna_vae={has_history_old}, rna_vae_new={has_history_new}")

# Compare model states
print("\nModel states:")
print(f"rna_vae device: {next(rna_vae.module.parameters()).device}")
print(f"rna_vae_new device: {next(rna_vae_new.module.parameters()).device}")
print(f"rna_vae training mode: {rna_vae.module.training}")
print(f"rna_vae_new training mode: {rna_vae_new.module.training}")

# %%
# Plot training results
print("\nPlotting normalized losses...")
plot_normalized_losses(rna_vae_new.__dict__["history"])

# %%
# Add visualization code
SCVI_LATENT_KEY = "X_scVI"
rna_vae_new.module.to(device)
protein_vae.module.to(device)
rna_vae_new.module.eval()
protein_vae.module.eval()

protein_vae.is_trained = True
with torch.no_grad():
    latent_rna = rna_vae_new.get_latent_representation()
    latent_prot = protein_vae.get_latent_representation()

    # Plot latent representation
    plot_latent(
        latent_rna,
        latent_prot,
        rna_vae_new.adata,
        protein_vae.adata,
        index=range(len(protein_vae.adata.obs.index)),
    )

# Store latent representations in AnnData objects
rna_vae_new.adata.obs["CN"] = protein_vae.adata.obs["CN"].values
rna_vae_new.adata.obsm[SCVI_LATENT_KEY] = latent_rna
protein_vae.adata.obsm[SCVI_LATENT_KEY] = latent_prot

# Set up neighbors and UMAP for RNA and protein subsets
sc.pp.neighbors(rna_vae_new.adata, key_added="latent_space_neighbors", use_rep=SCVI_LATENT_KEY)
rna_vae_new.adata.obsm["X_scVI"] = rna_vae_new.adata.obsm["X_scVI"]
sc.tl.umap(rna_vae_new.adata, neighbors_key="latent_space_neighbors")
rna_vae_new.adata.obsm["X_umap_scVI"] = rna_vae_new.adata.obsm["X_umap"]
rna_vae_new.adata.obsm.pop("X_umap")

# Plot the scVI UMAP embedding
sc.pl.embedding(
    rna_vae_new.adata,
    basis="X_umap_scVI",
    color=["cell_types", "CN"],
)

sc.pp.neighbors(protein_vae.adata, key_added="latent_space_neighbors", use_rep=SCVI_LATENT_KEY)
sc.tl.umap(protein_vae.adata, neighbors_key="latent_space_neighbors")
protein_vae.adata.obsm["X_umap_scVI"] = protein_vae.adata.obsm["X_umap"]
# renmove umap from adata
protein_vae.adata.obsm.pop("X_umap")


clean_uns_for_h5ad(rna_vae_new.adata)
clean_uns_for_h5ad(protein_vae.adata)
save_dir = Path("CODEX_RNA_seq/data/trained_data").absolute()

sc.write(Path(f"{save_dir}/rna_vae_trained.h5ad"), rna_vae_new.adata)
sc.write(Path(f"{save_dir}/protein_vae_trained.h5ad"), protein_vae.adata)


# PCA and UMAP for latent representations
rna_latent = AnnData(rna_vae_new.adata.obsm[SCVI_LATENT_KEY].copy())
prot_latent = AnnData(protein_vae.adata.obsm[SCVI_LATENT_KEY].copy())
rna_latent.obs = rna_vae_new.adata.obs.copy()
prot_latent.obs = protein_vae.adata.obs.copy()

# Combine latent spaces
combined_latent = ad.concat(
    [rna_latent, prot_latent], join="outer", label="modality", keys=["RNA", "Protein"]
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

# Run dimensionality reduction
sc.pp.pca(combined_latent)
sc.pp.neighbors(combined_latent)
sc.tl.umap(combined_latent)

# Process archetype vectors
rna_archtype = AnnData(rna_vae_new.adata.obsm["archetype_vec"])
rna_archtype.obs = rna_vae_new.adata.obs
sc.pp.neighbors(rna_archtype)
sc.tl.umap(rna_archtype)

prot_archtype = AnnData(protein_vae.adata.obsm["archetype_vec"])
prot_archtype.obs = protein_vae.adata.obs
sc.pp.neighbors(prot_archtype)
sc.tl.umap(prot_archtype)

# %%
# Visualize RNA and protein embeddings
sc.pl.embedding(
    rna_vae_new.adata,
    color=["CN", "cell_types"],
    basis="X_scVI",
    title=["Latent space, CN RNA", "Latent space, minor cell types RNA"],
)
sc.pl.embedding(
    protein_vae.adata,
    color=["CN", "cell_types"],
    basis="X_scVI",
    title=["Latent space, CN Protein", "Latent space, minor cell types Protein"],
)

# %%
# Visualize combined latent space
sc.tl.umap(combined_latent, min_dist=0.1)
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

# %%
# Analyze distances between modalities in the combined latent space
rna_latent = combined_latent[combined_latent.obs["modality"] == "RNA"]
rna_latent.obs["major_cell_types"] = rna_vae_new.adata.obs["major_cell_types"].values

prot_latent = combined_latent[combined_latent.obs["modality"] == "Protein"]
distances = np.linalg.norm(rna_latent.X - prot_latent.X, axis=1)

# Randomize RNA latent space to compare distances
rand_rna_latent = rna_latent.copy()
shuffled_indices = np.random.permutation(rand_rna_latent.obs.index)
rand_rna_latent = rand_rna_latent[shuffled_indices].copy()
rand_distances = np.linalg.norm(rand_rna_latent.X - prot_latent.X, axis=1)

# Plot randomized latent space distances
rand_rna_latent.obs["latent_dis"] = np.log(distances)
sc.pl.umap(
    rand_rna_latent,
    cmap="coolwarm",
    color="latent_dis",
    title="Latent space distances between RNA and Protein cells",
)

# %%
# Compare distance distributions and calculate mixing scores
compare_distance_distributions(rand_distances, rna_latent, prot_latent, distances)
mixing_result = mixing_score(
    latent_rna,
    latent_prot,
    rna_vae_new.adata,
    protein_vae.adata,
    index=range(len(rna_vae_new.adata)),
    plot_flag=True,
)
print(mixing_result)

# %%
# Identify the top 3 most common cell types
top_3_cell_types = combined_latent.obs["cell_types"].value_counts().index[:3]

# Plot UMAP for each of the top 3 most common cell types separately
for cell_type in top_3_cell_types:
    cell_type_data = combined_latent[combined_latent.obs["cell_types"] == cell_type]
    sc.pl.umap(
        cell_type_data,
        color=["CN", "modality", "cell_types"],
        title=[
            f"UMAP {cell_type} CN",
            f"UMAP {cell_type} modality",
            f"UMAP {cell_type} cell types",
        ],
        alpha=0.5,
    )

# %%
# Display AnnData info
print(protein_vae.adata)

# %%
# Plot spatial data
plt.figure(figsize=(12, 6))

# Plot with cell type as color
plt.subplot(1, 3, 1)
sns.scatterplot(
    x=protein_vae.adata.obs["X"],
    y=protein_vae.adata.obs["Y"],
    hue=protein_vae.adata.obs["cell_types"],
    palette="tab10",
    s=10,
)
plt.title("Protein cells colored by cell type")
plt.legend(loc="upper right", fontsize="small", title_fontsize="small")
plt.xlabel("X")
plt.ylabel("Y")

# Copy spatial coordinates from protein to RNA data
rna_vae_new.adata.obs["X"] = protein_vae.adata.obs["X"].values
rna_vae_new.adata.obs["Y"] = protein_vae.adata.obs["Y"].values

# Plot RNA cells with cell types as color
plt.subplot(1, 3, 2)
sns.scatterplot(
    x=rna_vae_new.adata.obs["X"],
    y=rna_vae_new.adata.obs["Y"],
    hue=rna_vae_new.adata.obs["cell_types"],
    s=10,
)
plt.title("RNA cells colored by cell types")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend([], [], frameon=False)

# Plot with CN as color
plt.subplot(1, 3, 3)
sns.scatterplot(
    x=protein_vae.adata.obs["X"],
    y=protein_vae.adata.obs["Y"],
    hue=protein_vae.adata.obs["CN"],
    s=10,
)
plt.title("Protein cells colored by CN")
plt.xlabel("X")
plt.ylabel("Y")

plt.tight_layout()
plt.show()

# %%
# Calculate metrics for evaluating the correspondence between modalities
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

# Normalized Mutual Information between cell types and CN labels
nmi_cell_types_cn_rna = adjusted_mutual_info_score(
    rna_vae_new.adata.obs["cell_types"], rna_vae_new.adata.obs["CN"]
)
print(
    f"Normalized Mutual Information between cell types and CN labels RNA: {nmi_cell_types_cn_rna:.3f}"
)

nmi_cell_types_cn_prot = adjusted_mutual_info_score(
    protein_vae.adata.obs["cell_types"], protein_vae.adata.obs["CN"]
)
print(
    f"Normalized Mutual Information between cell types and CN labels protein: {nmi_cell_types_cn_prot:.3f}"
)

# Normalized Mutual Information between cell types across modalities
nmi_cell_types_modalities = adjusted_mutual_info_score(
    rna_vae_new.adata.obs["cell_types"], protein_vae.adata.obs["cell_types"]
)
print(
    f"Normalized Mutual Information between cell types across modalities: {nmi_cell_types_modalities:.3f}"
)

# Calculate accuracy of cell type matching between modalities
matches = rna_vae_new.adata.obs["cell_types"].values == protein_vae.adata.obs["cell_types"].values
accuracy = matches.sum() / len(matches)
print(f"Accuracy of cell type matching between modalities: {accuracy:.4f}")

# %%
# Additional visualizations using PCA
sc.pp.pca(combined_latent)
sc.pl.pca(combined_latent, color="modality", projection="3d")
sc.pl.pca(combined_latent, color="modality", components=["3,4,5"], projection="3d")
sc.pl.pca(combined_latent, color="modality", components=["1,2,3"], projection="3d")
sc.pl.pca(combined_latent, color="modality", components=["6,7,8"], projection="3d")

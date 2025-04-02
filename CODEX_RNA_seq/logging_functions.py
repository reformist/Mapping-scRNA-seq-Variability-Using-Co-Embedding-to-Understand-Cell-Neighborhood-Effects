# %%
# Setup paths
# %%
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# %%
# Imports
# %%
import numpy as np
import torch
import torch.nn.functional as F


def log_training_metrics(
    self,
    rna_loss_output,
    protein_loss_output,
    contrastive_loss,
    matching_loss,
    similarity_loss,
    total_loss,
    adv_loss,
    diversity_loss,
):
    """Log training metrics"""
    self.history_["train_total_loss"].append(total_loss.item())
    self.history_["train_rna_reconstruction_loss"].append(rna_loss_output.loss.item())
    self.history_["train_protein_reconstruction_loss"].append(protein_loss_output.loss.item())
    self.history_["train_contrastive_loss"].append(contrastive_loss.item())
    self.history_["train_matching_rna_protein_loss"].append(matching_loss.item())
    self.history_["train_similarity_loss"].append(similarity_loss.item())
    self.history_["train_adv_loss"].append(adv_loss.item())
    self.history_["train_diversity_loss"].append(diversity_loss.item())


def log_validation_metrics(
    self,
    rna_loss_output,
    protein_loss_output,
    contrastive_loss,
    validation_total_loss,
    matching_rna_protein_latent_distances,
):
    """Log validation metrics"""
    self.history_["validation_total_loss"].append(validation_total_loss.item())
    self.history_["validation_rna_loss"].append(rna_loss_output.loss.item())
    self.history_["validation_protein_loss"].append(protein_loss_output.loss.item())
    self.history_["validation_contrastive_loss"].append(contrastive_loss.item())
    self.history_["validation_matching_latent_distances"].append(
        matching_rna_protein_latent_distances.mean().item()
    )


def log_batch_metrics(
    self,
    batch_idx,
    validation_total_loss,
    rna_loss_output,
    protein_loss_output,
    contrastive_loss,
):
    """Log batch metrics"""
    if batch_idx == 0:
        self.history_["batch_total_loss"].append(validation_total_loss.item())
        self.history_["batch_rna_loss"].append(rna_loss_output.loss.item())
        self.history_["batch_protein_loss"].append(protein_loss_output.loss.item())
        self.history_["batch_contrastive_loss"].append(contrastive_loss.item())


def log_step_metrics(
    self,
    global_step,
    total_loss,
    rna_loss_output,
    protein_loss_output,
    contrastive_loss,
    matching_loss,
    similarity_loss,
):
    """Log step metrics"""
    if global_step % 10 == 0:
        self.history_["step"].append(global_step)
        self.history_["step_total_loss"].append(total_loss.item())
        self.history_["step_rna_loss"].append(rna_loss_output.loss.item())
        self.history_["step_protein_loss"].append(protein_loss_output.loss.item())
        self.history_["step_contrastive_loss"].append(contrastive_loss.item())
        self.history_["step_matching_loss"].append(matching_loss.item())
        self.history_["step_similarity_loss"].append(similarity_loss.item())


def print_distance_metrics(
    self, prot_distances, rna_distances, num_acceptable, num_cells, stress_loss, matching_loss
):
    """Log distance metrics during training"""
    self.history_["distance_metrics/mean_protein_distances"].append(prot_distances.mean().item())
    self.history_["distance_metrics/mean_rna_distances"].append(rna_distances.mean().item())
    self.history_["distance_metrics/acceptable_ratio"].append(
        num_acceptable.float().item() / num_cells
    )
    self.history_["distance_metrics/stress_loss"].append(stress_loss.item())
    self.history_["distance_metrics/matching_loss"].append(matching_loss.item())


def log_extra_metrics(
    self,
    num_acceptable,
    num_cells,
    stress_loss,
    reward,
    exact_pairs,
    mixing_score_,
    batch_pred,
    batch_labels,
):
    """Log extra metrics during training."""
    self.history_["extra_metrics/acceptable_ratio"].append(
        num_acceptable.float().item() / num_cells
    )
    self.history_["extra_metrics/stress_loss"].append(stress_loss.item())
    self.history_["extra_metrics/reward"].append(reward.item())
    self.history_["extra_metrics/exact_pairs_loss"].append(exact_pairs.item())
    self.history_["extra_metrics/iLISI"].append(mixing_score_["iLISI"])
    self.history_["extra_metrics/cLISI"].append(mixing_score_["cLISI"])

    # Log accuracy
    accuracy = (batch_pred.argmax(dim=1) == batch_labels).float().mean()
    self.history_["extra_metrics/accuracy"].append(accuracy.item())


def log_epoch_end_(self):
    """Log epoch end metrics"""
    # if the mean is nan print the last 10 values
    if np.isnan(np.mean(self.history_["train_total_loss"][-len(self.history_["step"]) :])):
        self.history_["epoch_nan_detected"].append(True)

    epoch_avg_train_loss = np.mean(self.history_["train_total_loss"][-len(self.history_["step"]) :])
    epoch_avg_val_loss = np.mean(
        self.history_["validation_total_loss"][-len(self.history_["step"]) :]
    )

    self.history_["epoch"].append(self.current_epoch)
    self.history_["epoch_avg_train_loss"].append(epoch_avg_train_loss)
    self.history_["epoch_avg_val_loss"].append(epoch_avg_val_loss)


def setup_history(self):
    """Initialize history dictionary with all metric keys"""
    self.history_ = {
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
        "batch_total_loss": [],
        "batch_rna_loss": [],
        "batch_protein_loss": [],
        "batch_contrastive_loss": [],
        "step_total_loss": [],
        "step_rna_loss": [],
        "step_protein_loss": [],
        "step_contrastive_loss": [],
        "step_matching_loss": [],
        "step_similarity_loss": [],
        "distance_metrics/mean_protein_distances": [],
        "distance_metrics/mean_rna_distances": [],
        "distance_metrics/acceptable_ratio": [],
        "distance_metrics/stress_loss": [],
        "distance_metrics/matching_loss": [],
        "extra_metrics/acceptable_ratio": [],
        "extra_metrics/stress_loss": [],
        "extra_metrics/reward": [],
        "extra_metrics/exact_pairs_loss": [],
        "extra_metrics/iLISI": [],
        "extra_metrics/cLISI": [],
        "extra_metrics/accuracy": [],
        "epoch_nan_detected": [],
        "epoch_avg_train_loss": [],
        "epoch_avg_val_loss": [],
    }

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
    self.log("train_total_loss", total_loss.item(), on_epoch=False, on_step=True)
    self.log(
        "train_rna_reconstruction_loss", rna_loss_output.loss.item(), on_epoch=False, on_step=True
    )
    self.log(
        "train_protein_reconstruction_loss",
        protein_loss_output.loss.item(),
        on_epoch=False,
        on_step=True,
    )
    self.log("train_contrastive_loss", contrastive_loss.item(), on_epoch=False, on_step=True)
    self.log("train_matching_rna_protein_loss", matching_loss.item(), on_epoch=False, on_step=True)
    self.log("train_similarity_loss", similarity_loss.item(), on_epoch=False, on_step=True)
    self.log("train_adv_loss", adv_loss.item(), on_epoch=False, on_step=True)
    self.log("train_diversity_loss", diversity_loss.item(), on_epoch=False, on_step=True)


def log_validation_metrics(
    self,
    rna_loss_output,
    protein_loss_output,
    contrastive_loss,
    validation_total_loss,
    matching_rna_protein_latent_distances,
):
    """Log validation metrics"""
    self.log("validation_total_loss", validation_total_loss.item(), on_epoch=True, on_step=False)
    self.log("validation_rna_loss", rna_loss_output.loss.item(), on_epoch=True, on_step=False)
    self.log(
        "validation_protein_loss", protein_loss_output.loss.item(), on_epoch=True, on_step=False
    )
    self.log("validation_contrastive_loss", contrastive_loss.item(), on_epoch=True, on_step=False)
    self.log(
        "validation_matching_latent_distances",
        matching_rna_protein_latent_distances.mean().item(),
        on_epoch=True,
        on_step=False,
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
        self.log("batch_total_loss", validation_total_loss.item(), on_epoch=False, on_step=True)
        self.log("batch_rna_loss", rna_loss_output.loss.item(), on_epoch=False, on_step=True)
        self.log(
            "batch_protein_loss", protein_loss_output.loss.item(), on_epoch=False, on_step=True
        )
        self.log("batch_contrastive_loss", contrastive_loss.item(), on_epoch=False, on_step=True)


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
        print(f"\nStep {global_step}:")
        print(f"Total Loss: {total_loss.item():.4f}")
        print(f"RNA Loss: {rna_loss_output.loss.item():.4f}")
        print(f"Protein Loss: {protein_loss_output.loss.item():.4f}")
        print(f"Contrastive Loss: {contrastive_loss.item():.4f}")
        print(f"Matching Loss: {matching_loss.item():.4f}")
        print(f"Similarity Loss: {similarity_loss.item():.4f}")


def log_epoch_end_(self):
    """Log epoch end metrics"""
    # if the mean is nan print the last 10 values
    if np.isnan(np.mean(self.history_["train_total_loss"][-len(self.history_["step"]) :])):
        print("NAN")
    print(f"\nEpoch {self.current_epoch} completed:")
    print(
        f"Average training loss: {np.mean(self.history_['train_total_loss'][-len(self.history_['step']):]):.4f}"
    )
    print(
        f"Average validation loss: {np.mean(self.history_['validation_total_loss'][-len(self.history_['step']):]):.4f}"
    )


def print_distance_metrics(
    prot_distances, rna_distances, num_acceptable, num_cells, stress_loss, matching_loss
):
    """Print distance metrics during training"""
    print("\nDistance metrics:")
    print(f"Mean protein distances: {round(prot_distances.mean().item(), 3)}")
    print(f"Mean RNA distances: {round(rna_distances.mean().item(), 3)}")
    print(f"Acceptable ratio: {round(num_acceptable.float().item() / num_cells, 3)}")
    print(f"Stress loss: {round(stress_loss.item(), 3)}")
    print(f"Matching loss: {round(matching_loss.item(), 3)}")


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
    """Log extra metrics during training.

    Args:
        self: Training plan instance
        num_acceptable: Number of acceptable pairs
        num_cells: Total number of cells
        stress_loss: Stress loss value
        reward: Reward value
        exact_pairs: Exact pairs loss value
        mixing_score_: Dictionary containing iLISI and cLISI scores
        batch_pred: Batch predictions
        batch_labels: Batch labels
    """
    self.log(
        "extra_metric_acceptable_ratio",
        num_acceptable.float().item() / num_cells,
        on_epoch=False,
        on_step=True,
    )
    self.log("extra_metric_stress_loss", stress_loss.item(), on_epoch=False, on_step=True)
    self.log("extra_metric_reward", reward.item(), on_epoch=False, on_step=True)
    self.log("extra_metric_exact_pairs_loss", exact_pairs.item(), on_epoch=False, on_step=True)
    self.log("extra_metric_iLISI", mixing_score_["iLISI"], on_epoch=False, on_step=True)
    self.log("extra_metric_cLISI", mixing_score_["cLISI"], on_epoch=False, on_step=True)

    # Print accuracy
    accuracy = (batch_pred.argmax(dim=1) == batch_labels).float().mean()
    print(f"Accuracy: {accuracy}")
    self.log("extra_metric_accuracy", accuracy, on_epoch=False, on_step=True)

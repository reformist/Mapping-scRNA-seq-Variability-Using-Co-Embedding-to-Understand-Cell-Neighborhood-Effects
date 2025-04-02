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
    # Log losses
    self.log("train_rna_reconstruction_loss", rna_loss_output.loss, on_epoch=False, on_step=True)
    self.history["train_rna_reconstruction_loss"].append(rna_loss_output.loss.item())

    self.log(
        "train_protein_reconstruction_loss", protein_loss_output.loss, on_epoch=False, on_step=True
    )
    self.history["train_protein_reconstruction_loss"].append(protein_loss_output.loss.item())

    self.log("train_contrastive_loss", contrastive_loss, on_epoch=False, on_step=True)
    self.history["train_contrastive_loss"].append(contrastive_loss.item())

    self.log("train_matching_rna_protein_loss", matching_loss, on_epoch=False, on_step=True)
    self.history["train_matching_rna_protein_loss"].append(matching_loss.item())

    self.log("train_similarity_loss", similarity_loss, on_epoch=False, on_step=True)
    self.history["train_similarity_loss"].append(similarity_loss.item())

    self.log("train_total_loss", total_loss, on_epoch=False, on_step=True)
    self.history["train_total_loss"].append(total_loss.item())

    self.log("train_adv_loss", adv_loss, on_epoch=False, on_step=True)
    self.history["train_adv_loss"].append(adv_loss.item())

    self.log("train_diversity_loss", diversity_loss, on_epoch=False, on_step=True)
    self.history["train_diversity_loss"].append(diversity_loss.item())


def log_validation_metrics(
    self,
    rna_loss_output,
    protein_loss_output,
    contrastive_loss,
    validation_total_loss,
    matching_rna_protein_latent_distances,
):
    """Log validation metrics"""
    # Log validation metrics
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
        "validation_contrastive_loss", contrastive_loss, on_epoch=True, sync_dist=self.use_sync_dist
    )
    self.history["validation_contrastive_loss"].append(contrastive_loss.item())

    self.log(
        "validation_total_loss", validation_total_loss, on_epoch=True, sync_dist=self.use_sync_dist
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


def log_epoch_end(self):
    """Log epoch end metrics"""
    print(f"\nEpoch {self.current_epoch} completed")
    print(
        f"Average training loss: {np.mean(self.history['train_total_loss'][-self.trainer.num_training_batches:]):.4f}"
    )
    if len(self.history["validation_total_loss"]) > 0:
        print(f"Latest validation loss: {self.history['validation_total_loss'][-1]:.4f}")


def log_step_metrics(
    self,
    step,
    total_loss,
    rna_loss_output,
    protein_loss_output,
    contrastive_loss,
    matching_loss,
    similarity_loss,
):
    """Log step metrics"""
    if step % 10 == 0:  # Print every 10 steps
        print(f"\nStep {step} - Current losses:")
        print(f"Total loss: {total_loss.item():.4f}")
        print(f"RNA reconstruction loss: {rna_loss_output.loss.item():.4f}")
        print(f"Protein reconstruction loss: {protein_loss_output.loss.item():.4f}")
        print(f"Contrastive loss: {contrastive_loss.item():.4f}")
        print(f"Matching loss: {matching_loss.item():.4f}")
        print(f"Similarity loss: {similarity_loss.item():.4f}")


def log_batch_metrics(
    self, batch_idx, validation_total_loss, rna_loss_output, protein_loss_output, contrastive_loss
):
    """Log batch metrics"""
    if batch_idx == 0:  # Print validation metrics for first batch of each epoch
        print(f"\nValidation metrics at epoch {self.current_epoch}:")
        print(f"Total validation loss: {validation_total_loss.item():.4f}")
        print(f"RNA validation loss: {rna_loss_output.loss.item():.4f}")
        print(f"Protein validation loss: {protein_loss_output.loss.item():.4f}")
        print(f"Contrastive validation loss: {contrastive_loss.item():.4f}")

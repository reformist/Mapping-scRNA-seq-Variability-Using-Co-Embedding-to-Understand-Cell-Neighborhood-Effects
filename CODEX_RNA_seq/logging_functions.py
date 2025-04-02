# %%
# Setup paths
# %%
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# %%
# Imports
# %%
import numpy as np
import torch
import torch.nn.functional as F


def setup_logging(log_dir="logs"):
    """Initialize logging directory and return log file path"""
    print("Setting up logging...")
    # Get the directory of the current file
    current_dir = Path(__file__).parent
    print(f"Current directory: {current_dir}")
    log_dir = current_dir / log_dir
    print(f"Log directory: {log_dir}")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_log_{timestamp}.json"
    print(f"Log file path: {log_file}")

    # Initialize empty history dictionary
    history = {
        "step": [],
        "timestamp": [],
        "epoch": [],
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
        "learning_rate": [],
        "batch_size": [],
        "gradient_norm": [],
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

    # Save initial history
    print("Saving initial history...")
    with open(log_file, "w") as f:
        json.dump(history, f)
    print("Logging setup complete")

    return log_file


def update_log(log_file, key, value):
    """Update log file with new value for given key"""
    with open(log_file, "r") as f:
        history = json.load(f)

    if key not in history:
        history[key] = []

    history[key].append(value)

    with open(log_file, "w") as f:
        json.dump(history, f)


def log_training_metrics(
    log_file,
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
    update_log(log_file, "train_total_loss", total_loss.item())
    update_log(log_file, "train_rna_reconstruction_loss", rna_loss_output.loss.item())
    update_log(log_file, "train_protein_reconstruction_loss", protein_loss_output.loss.item())
    update_log(log_file, "train_contrastive_loss", contrastive_loss.item())
    update_log(log_file, "train_matching_rna_protein_loss", matching_loss.item())
    update_log(log_file, "train_similarity_loss", similarity_loss.item())
    update_log(log_file, "train_adv_loss", adv_loss.item())
    update_log(log_file, "train_diversity_loss", diversity_loss.item())


def log_validation_metrics(
    log_file,
    rna_loss_output,
    protein_loss_output,
    contrastive_loss,
    validation_total_loss,
    matching_rna_protein_latent_distances,
):
    """Log validation metrics"""
    update_log(log_file, "validation_total_loss", validation_total_loss.item())
    update_log(log_file, "validation_rna_loss", rna_loss_output.loss.item())
    update_log(log_file, "validation_protein_loss", protein_loss_output.loss.item())
    update_log(log_file, "validation_contrastive_loss", contrastive_loss.item())
    update_log(
        log_file,
        "validation_matching_latent_distances",
        matching_rna_protein_latent_distances.mean().item(),
    )


def log_batch_metrics(
    log_file,
    batch_idx,
    validation_total_loss,
    rna_loss_output,
    protein_loss_output,
    contrastive_loss,
):
    """Log batch metrics"""
    if batch_idx == 0:
        update_log(log_file, "batch_total_loss", validation_total_loss.item())
        update_log(log_file, "batch_rna_loss", rna_loss_output.loss.item())
        update_log(log_file, "batch_protein_loss", protein_loss_output.loss.item())
        update_log(log_file, "batch_contrastive_loss", contrastive_loss.item())


def log_step_metrics(
    log_file,
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
        update_log(log_file, "step", global_step)
        update_log(log_file, "step_total_loss", total_loss.item())
        update_log(log_file, "step_rna_loss", rna_loss_output.loss.item())
        update_log(log_file, "step_protein_loss", protein_loss_output.loss.item())
        update_log(log_file, "step_contrastive_loss", contrastive_loss.item())
        update_log(log_file, "step_matching_loss", matching_loss.item())
        update_log(log_file, "step_similarity_loss", similarity_loss.item())


def print_distance_metrics(
    log_file, prot_distances, rna_distances, num_acceptable, num_cells, stress_loss, matching_loss
):
    """Log distance metrics during training"""
    update_log(log_file, "distance_metrics/mean_protein_distances", prot_distances.mean().item())
    update_log(log_file, "distance_metrics/mean_rna_distances", rna_distances.mean().item())
    update_log(
        log_file, "distance_metrics/acceptable_ratio", num_acceptable.float().item() / num_cells
    )
    update_log(log_file, "distance_metrics/stress_loss", stress_loss.item())
    update_log(log_file, "distance_metrics/matching_loss", matching_loss.item())


def log_extra_metrics(
    log_file,
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
    update_log(
        log_file, "extra_metrics/acceptable_ratio", num_acceptable.float().item() / num_cells
    )
    update_log(log_file, "extra_metrics/stress_loss", stress_loss.item())
    update_log(log_file, "extra_metrics/reward", reward.item())
    update_log(log_file, "extra_metrics/exact_pairs_loss", exact_pairs.item())
    update_log(log_file, "extra_metrics/iLISI", mixing_score_["iLISI"])
    update_log(log_file, "extra_metrics/cLISI", mixing_score_["cLISI"])

    # Log accuracy
    accuracy = (batch_pred.argmax(dim=1) == batch_labels).float().mean()
    update_log(log_file, "extra_metrics/accuracy", accuracy.item())


def log_epoch_end(log_file, current_epoch, train_losses, val_losses):
    """Log epoch end metrics"""
    # Calculate epoch averages
    epoch_avg_train_loss = sum(train_losses) / len(train_losses)
    epoch_avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")

    update_log(log_file, "epoch", current_epoch)
    update_log(log_file, "epoch_avg_train_loss", epoch_avg_train_loss)
    update_log(log_file, "epoch_avg_val_loss", epoch_avg_val_loss)


def load_history(log_file):
    """Load history from log file"""
    with open(log_file, "r") as f:
        return json.load(f)

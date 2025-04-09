# %%
# Setup paths
# %%
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# %%
# Imports
# %%


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
    cell_type_clustering_loss=None,
):
    """Log training metrics to a JSON file."""
    with open(log_file, "r") as f:
        logs = json.load(f)

    if "train_metrics" not in logs:
        logs["train_metrics"] = []

    metrics_dict = {
        "rna_loss": rna_loss_output.loss.item(),
        "protein_loss": protein_loss_output.loss.item(),
        "contrastive_loss": contrastive_loss.item(),
        "matching_loss": matching_loss.item(),
        "similarity_loss": similarity_loss.item(),
        "total_loss": total_loss.item(),
        "adv_loss": adv_loss.item(),
        "diversity_loss": diversity_loss.item(),
    }

    if cell_type_clustering_loss is not None:
        if isinstance(cell_type_clustering_loss, torch.Tensor):
            metrics_dict["cell_type_clustering_loss"] = cell_type_clustering_loss.item()
        else:
            metrics_dict["cell_type_clustering_loss"] = cell_type_clustering_loss

    logs["train_metrics"].append(metrics_dict)

    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2)


def log_validation_metrics(
    log_file,
    rna_loss_output,
    protein_loss_output,
    contrastive_loss,
    validation_total_loss,
    matching_rna_protein_latent_distances,
    cell_type_clustering_loss=None,
):
    """Log validation metrics to a file."""
    with open(log_file, "r") as f:
        logs = json.load(f)

    if "validation_metrics" not in logs:
        logs["validation_metrics"] = []

    metrics_dict = {
        "val_total_loss": validation_total_loss.item(),
        "val_rna_loss": rna_loss_output.loss.item(),
        "val_protein_loss": protein_loss_output.loss.item(),
        "val_contrastive_loss": contrastive_loss.item(),
        "val_matching_distances_mean": matching_rna_protein_latent_distances.mean().item(),
        "val_matching_distances_min": matching_rna_protein_latent_distances.min().item(),
        "val_matching_distances_max": matching_rna_protein_latent_distances.max().item(),
    }

    if cell_type_clustering_loss is not None:
        if isinstance(cell_type_clustering_loss, torch.Tensor):
            metrics_dict["val_cell_type_clustering_loss"] = cell_type_clustering_loss.item()
        else:
            metrics_dict["val_cell_type_clustering_loss"] = cell_type_clustering_loss

    logs["validation_metrics"].append(metrics_dict)

    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2)


def log_batch_metrics(
    log_file,
    batch_idx,
    validation_total_loss,
    rna_loss_output,
    protein_loss_output,
    contrastive_loss,
):
    """Log batch metrics"""
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


def print_training_metrics(
    global_step,
    current_epoch,
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
    similarity_weight,
    similarity_active,
    num_acceptable,
    num_cells,
    exact_pairs,
    cell_type_clustering_loss=None,
):
    """Print training metrics in a structured format."""
    print("\n" + "=" * 80)
    print(f"Step {global_step}, Epoch {current_epoch}")
    print("=" * 80)

    # Helper function to format loss with percentage
    def format_loss(loss, total):
        abs_total = abs(total)
        abs_loss = abs(loss)
        if abs_loss > 100:
            return f"{round(loss, 2)} ({round(abs_loss/abs_total*100, 1)}%)"
        return f"{loss:.3f} ({round(abs_loss/abs_total*100, 1)}%)"

    print("\nLosses:")
    print("-" * 40)
    print(f"RNA Loss: {format_loss(rna_loss_output.loss.item(), total_loss.item())}")
    print(f"Protein Loss: {format_loss(protein_loss_output.loss.item(), total_loss.item())}")
    print(f"Contrastive Loss: {format_loss(contrastive_loss.item(), total_loss.item())}")
    print(f"Adversarial Loss: {format_loss(adv_loss.item(), total_loss.item())}")
    print(f"Matching Loss: {format_loss(matching_loss.item(), total_loss.item())}")
    print(f"Similarity Loss: {format_loss(similarity_loss.item(), total_loss.item())}")
    print(f"Diversity Loss: {format_loss(diversity_loss.item(), total_loss.item())}")

    if cell_type_clustering_loss is not None:
        if isinstance(cell_type_clustering_loss, torch.Tensor):
            loss_value = cell_type_clustering_loss.item()
        else:
            loss_value = cell_type_clustering_loss
        print(f"Cell Type Clustering Loss: {format_loss(loss_value, total_loss.item())}")

    print(f"Total Loss: {total_loss.item():.3f}")

    print("\nDistance Metrics:")
    print("-" * 40)
    print(f"Min Latent Distances: {round(latent_distances.min().item(),3)}")
    print(f"Max Latent Distances: {round(latent_distances.max().item(),3)}")
    print(f"Mean Latent Distances: {round(latent_distances.mean().item(),3)}")

    print("\nSimilarity Metrics:")
    print("-" * 40)
    print(f"Similarity Loss Raw: {similarity_loss_raw.item():.3f}")
    print(f"Similarity Weight: {similarity_weight}")
    print(f"Similarity Active: {similarity_active}")

    print("\nMatching Metrics:")
    print("-" * 40)
    print(f"Number of Acceptable Pairs: {num_acceptable.item()}")
    print(f"Total Pairs: {num_cells}")
    print(f"Acceptable Ratio: {num_acceptable.item()/num_cells:.3f}")
    print(f"Exact Pairs Loss: {exact_pairs.item():.3f}")
    print("=" * 80 + "\n")

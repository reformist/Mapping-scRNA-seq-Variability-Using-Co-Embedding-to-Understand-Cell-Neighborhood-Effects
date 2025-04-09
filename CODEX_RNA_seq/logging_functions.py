# %%
# Setup paths
# %%
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
    print(f"Current directory: {current_dir.relative_to(Path.cwd())}")
    log_dir = current_dir / log_dir
    print(f"Log directory: {log_dir.relative_to(Path.cwd())}")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_log_{timestamp}.txt"
    print(f"Log file path: {log_file.relative_to(Path.cwd())}")

    # Create an empty log file
    with open(log_file, "w") as f:
        f.write(f"Training log started at {datetime.now()}\n")

    print("Logging setup complete")
    return log_file


def update_log(log_file, key, value):
    """Update log file with new value for given key"""
    if not log_file or not os.path.exists(log_file):
        return

    with open(log_file, "a") as f:
        f.write(f"{key}: {value}\n")


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
    """Log training metrics to a file."""
    if not log_file or not os.path.exists(log_file):
        return

    with open(log_file, "a") as f:
        f.write("\n--- TRAINING METRICS ---\n")
        f.write(f"RNA loss: {rna_loss_output.loss.item()}\n")
        f.write(f"Protein loss: {protein_loss_output.loss.item()}\n")
        f.write(f"Contrastive loss: {contrastive_loss.item()}\n")
        f.write(f"Matching loss: {matching_loss.item()}\n")
        f.write(f"Similarity loss: {similarity_loss.item()}\n")
        f.write(f"Total loss: {total_loss.item()}\n")
        f.write(f"Adversarial loss: {adv_loss.item()}\n")
        f.write(f"Diversity loss: {diversity_loss.item()}\n")

        if cell_type_clustering_loss is not None:
            if isinstance(cell_type_clustering_loss, torch.Tensor):
                f.write(f"Cell type clustering loss: {cell_type_clustering_loss.item()}\n")
            else:
                f.write(f"Cell type clustering loss: {cell_type_clustering_loss}\n")


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
    if not log_file or not os.path.exists(log_file):
        return

    with open(log_file, "a") as f:
        f.write("\n--- VALIDATION METRICS ---\n")
        f.write(f"Validation total loss: {validation_total_loss.item()}\n")
        f.write(f"Validation RNA loss: {rna_loss_output.loss.item()}\n")
        f.write(f"Validation protein loss: {protein_loss_output.loss.item()}\n")
        f.write(f"Validation contrastive loss: {contrastive_loss.item()}\n")
        f.write(
            f"Validation matching distances mean: {matching_rna_protein_latent_distances.mean().item()}\n"
        )
        f.write(
            f"Validation matching distances min: {matching_rna_protein_latent_distances.min().item()}\n"
        )
        f.write(
            f"Validation matching distances max: {matching_rna_protein_latent_distances.max().item()}\n"
        )

        if cell_type_clustering_loss is not None:
            if isinstance(cell_type_clustering_loss, torch.Tensor):
                f.write(
                    f"Validation cell type clustering loss: {cell_type_clustering_loss.item()}\n"
                )
            else:
                f.write(f"Validation cell type clustering loss: {cell_type_clustering_loss}\n")


def log_batch_metrics(
    log_file,
    batch_idx,
    validation_total_loss,
    rna_loss_output,
    protein_loss_output,
    contrastive_loss,
):
    """Log batch metrics"""
    if not log_file or not os.path.exists(log_file):
        return

    with open(log_file, "a") as f:
        f.write(f"\n--- BATCH {batch_idx} METRICS ---\n")
        f.write(f"Batch total loss: {validation_total_loss.item()}\n")
        f.write(f"Batch RNA loss: {rna_loss_output.loss.item()}\n")
        f.write(f"Batch protein loss: {protein_loss_output.loss.item()}\n")
        f.write(f"Batch contrastive loss: {contrastive_loss.item()}\n")


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
    if not log_file or not os.path.exists(log_file):
        return

    with open(log_file, "a") as f:
        f.write(f"\n--- STEP {global_step} METRICS ---\n")
        f.write(f"Step total loss: {total_loss.item()}\n")
        f.write(f"Step RNA loss: {rna_loss_output.loss.item()}\n")
        f.write(f"Step protein loss: {protein_loss_output.loss.item()}\n")
        f.write(f"Step contrastive loss: {contrastive_loss.item()}\n")
        f.write(f"Step matching loss: {matching_loss.item()}\n")
        f.write(f"Step similarity loss: {similarity_loss.item()}\n")


def print_distance_metrics(
    log_file, prot_distances, rna_distances, num_acceptable, num_cells, stress_loss, matching_loss
):
    """Log distance metrics during training"""
    if not log_file or not os.path.exists(log_file):
        return

    with open(log_file, "a") as f:
        f.write("\n--- DISTANCE METRICS ---\n")
        f.write(f"Mean protein distances: {prot_distances.mean().item()}\n")
        f.write(f"Mean RNA distances: {rna_distances.mean().item()}\n")
        f.write(f"Acceptable ratio: {num_acceptable.float().item() / num_cells}\n")
        f.write(f"Stress loss: {stress_loss.item()}\n")
        f.write(f"Matching loss: {matching_loss.item()}\n")


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
    if not log_file or not os.path.exists(log_file):
        return

    # Log accuracy
    accuracy = (batch_pred.argmax(dim=1) == batch_labels).float().mean()

    with open(log_file, "a") as f:
        f.write("\n--- EXTRA METRICS ---\n")
        f.write(f"Acceptable ratio: {num_acceptable.float().item() / num_cells}\n")
        f.write(f"Stress loss: {stress_loss.item()}\n")
        f.write(f"Reward: {reward.item()}\n")
        f.write(f"Exact pairs loss: {exact_pairs.item()}\n")
        f.write(f"iLISI: {mixing_score_['iLISI']}\n")
        f.write(f"cLISI: {mixing_score_['cLISI']}\n")
        f.write(f"Accuracy: {accuracy.item()}\n")


def log_epoch_end(log_file, current_epoch, train_losses, val_losses):
    """Log epoch end metrics"""
    if not log_file or not os.path.exists(log_file):
        return

    # Calculate epoch averages
    epoch_avg_train_loss = sum(train_losses) / len(train_losses)
    epoch_avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")

    with open(log_file, "a") as f:
        f.write(f"\n--- EPOCH {current_epoch} SUMMARY ---\n")
        f.write(f"Average train loss: {epoch_avg_train_loss}\n")
        f.write(f"Average validation loss: {epoch_avg_val_loss}\n")


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

import os
import sys
from datetime import timedelta

import mlflow
import torch
from tabulate import tabulate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# %%
# Imports
# %%


def print_distance_metrics(
    prot_distances, rna_distances, num_acceptable, num_cells, stress_loss, matching_loss
):
    print("\n--- DISTANCE METRICS ---\n")
    table_data = [
        ["Metric", "Value"],
        ["Mean protein distances", f"{prot_distances.mean().item():.4f}"],
        ["Mean RNA distances", f"{rna_distances.mean().item():.4f}"],
        ["Acceptable ratio", f"{num_acceptable.float().item() / num_cells:.4f}"],
        ["Stress loss", f"{stress_loss.item():.4f}"],
        ["Matching loss", f"{matching_loss.item():.4f}"],
    ]
    print(tabulate(table_data, headers="firstrow", tablefmt="fancy_grid"))


def log_epoch_end(current_epoch, train_losses, val_losses):
    # Calculate epoch averages
    epoch_avg_train_loss = sum(train_losses) / len(train_losses)
    epoch_avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")
    print(f"\n--- EPOCH {current_epoch} SUMMARY ---\n")

    table_data = [
        ["Metric", "Value"],
        ["Average train loss", f"{epoch_avg_train_loss:.4f}"],
        ["Average validation loss", f"{epoch_avg_val_loss:.4f}"],
    ]
    print(tabulate(table_data, headers="firstrow", tablefmt="fancy_grid"))

    mlflow.log_metrics(
        {"epoch_avg_train_loss": epoch_avg_train_loss, "epoch_avg_val_loss": epoch_avg_val_loss},
        step=current_epoch,
    )


def estimate_training_time(rna_cells, prot_cells, params, total_combinations):
    """
    Estimates training time based on dataset sizes and hyperparameters.

    Args:
        rna_cells: Number of RNA cells
        prot_cells: Number of protein cells
        params: Dictionary of hyperparameters
        total_combinations: Total number of parameter combinations to try

    Returns:
        Tuple of (estimated_time_per_iter, total_estimated_time)
    """
    # Base time in seconds based on actual observed runs
    # ~5 minutes per iteration with dataset sizes of ~13k RNA and ~40k protein cells
    base_seconds = 295  # 4min 55sec observed for full dataset with 80 epochs

    # Scale factors based on dataset size
    # Using much gentler scaling based on actual observations
    rna_scale = (rna_cells / 13000) ** 0.5  # Use square root scaling
    prot_scale = (prot_cells / 40000) ** 0.5

    # Hyperparameter scaling factors
    # Actual timing shows epochs have less impact than we initially thought
    epoch_scale = (params["max_epochs"] / 80) ** 0.7  # Reduced impact
    plot_scale = 0.8 + (0.2 * params["plot_x_times"] / 5)  # Minimal impact from plotting
    check_val_scale = 1.0 if params["check_val_every_n_epoch"] > 0 else 0.95  # Minimal impact

    # Scale based on hardware (GPU vs CPU)
    hardware_scale = 1.0 if torch.cuda.is_available() else 3.0

    # Calculate total scale factor
    # Cell scales have less impact than originally thought
    total_scale = (
        ((rna_scale * prot_scale) ** 0.5)
        * epoch_scale
        * plot_scale
        * check_val_scale
        * hardware_scale
    )

    # Estimated time per iteration in seconds
    estimated_seconds_per_iter = base_seconds * total_scale

    # Convert to timedelta
    estimated_time_per_iter = timedelta(seconds=estimated_seconds_per_iter)
    total_estimated_time = estimated_time_per_iter * total_combinations

    return estimated_time_per_iter, total_estimated_time


def save_tabulate_to_txt(losses, global_step, total_steps):
    """Save losses as a formatted table and log it to MLflow.

    Args:
        losses: Dictionary containing loss values
        global_step: Current global step
        total_steps: Total number of steps
    """
    # Convert tensor values to Python scalars
    losses_to_save = {
        k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.copy().items()
    }

    # Determine filename based on step
    if global_step is not None:
        last_step = global_step == total_steps - 1 if total_steps is not None else False
    if last_step:
        losses_file = "final_losses.txt"
    else:
        losses_file = f"losses_{global_step:05d}.txt"

    # Get total loss for percentage calculations
    total_loss = losses_to_save.get("total_loss", 0)

    # Create tabulate table with only main losses
    table_data = [["Loss Type", "Value"]]

    # Define main losses in order
    main_losses = [
        "total_loss",
        "rna_loss",
        "protein_loss",
        "contrastive_loss",
        "matching_loss",
        "similarity_loss",
        "cell_type_clustering_loss",
    ]
    # Format main losses with percentages
    for loss_name in main_losses:
        value = losses_to_save.get(loss_name, 0)

        if loss_name == "total_loss":
            table_data.append([loss_name, f"{value:.4f}"])
        else:
            # Calculate percentage of total
            percentage = (value / total_loss) * 100 if total_loss != 0 else 0
            formatted_value = f"{value:.3f} ({percentage:.1f}%)"
            table_data.append([loss_name, formatted_value])

    # Save formatted table to text file
    with open(losses_file, "w") as f:
        f.write(tabulate(table_data, headers="firstrow", tablefmt="fancy_grid"))

    # Log to MLflow and clean up
    mlflow.log_artifact(losses_file, "losses")
    os.remove(losses_file)


def log_step(
    losses,
    metrics=None,
    global_step=None,
    current_epoch=None,
    is_validation=False,
    similarity_weight=None,
    similarity_active=None,
    num_acceptable=None,
    num_cells=None,
    latent_distances=None,
    print_to_console=True,
    total_steps=None,
):
    """Unified function to log and print metrics for both training and validation steps.

    Args:
        losses: Dictionary containing all loss values
        metrics: Dictionary containing additional metrics (iLISI, cLISI, accuracy, etc.)
        global_step: Current global step (optional)
        current_epoch: Current epoch (optional)
        is_validation: Whether this is validation or training
        similarity_weight: Weight for similarity loss (optional for training)
        similarity_active: Whether similarity loss is active (optional for training)
        num_acceptable: Number of acceptable matches (optional for training)
        num_cells: Number of cells (optional for training)
        exact_pairs: Exact pairs metric (optional for training)
        latent_distances: Latent distances (optional)
        print_to_console: Whether to print metrics to console
    """
    prefix = "Validation " if is_validation else ""
    metrics = metrics or {}

    # Convert tensor values to Python scalars for logging
    def get_value(x):
        if x is None:
            return 0
        return round(x.item(), 4) if isinstance(x, torch.Tensor) else x

    # Extract loss values
    total_loss = get_value(losses.get("total_loss", float("nan")))
    rna_loss = get_value(losses.get("rna_loss", float("nan")))
    protein_loss = get_value(losses.get("protein_loss", float("nan")))
    contrastive_loss = get_value(losses.get("contrastive_loss", float("nan")))
    matching_loss = get_value(losses.get("matching_loss", float("nan")))
    similarity_loss = get_value(losses.get("similarity_loss", float("nan")))
    similarity_loss_raw = get_value(losses.get("similarity_loss_raw", float("nan")))
    cell_type_clustering_loss = get_value(losses.get("cell_type_clustering_loss", float("nan")))
    adv_loss = get_value(losses.get("adversarial_loss", float("nan")))
    diversity_loss = get_value(losses.get("diversity_loss", float("nan")))
    stress_loss = get_value(losses.get("stress_loss", float("nan")))
    reward = get_value(losses.get("reward", float("nan")))

    # Handle parameters that might be in losses dict or passed directly
    exact_pairs = get_value(metrics.get("exact_pairs", float("nan")))
    num_acceptable = get_value(metrics.get("num_acceptable", float("nan")))
    num_cells = get_value(metrics.get("num_cells", float("nan")))

    # Extract additional metrics
    ilisi = get_value(metrics.get("iLISI", float("nan")))
    clisi = get_value(metrics.get("cLISI", float("nan")))
    accuracy = get_value(metrics.get("accuracy", float("nan")))

    def format_loss(loss, total):
        if loss is None:
            return
        percentage = (loss / total) * 100 if total != 0 else 0
        return f"{loss:.3f} ({percentage:.1f}%)"

    def format_loss_mlflow(loss_dict, total=None):
        loss_dict = loss_dict.copy()
        loss_dict = {
            k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()
        }
        return {
            k: round(v, 4) if isinstance(v, (int, float)) else v
            for k, v in loss_dict.items()
            if v is not None
        }

    # Format metrics for printing to console
    if print_to_console:
        save_tabulate_to_txt(format_loss_mlflow(losses), global_step, total_steps)

        print("\n" + "=" * 80)
        step_info = ""
        if global_step is not None:
            step_info += f"Step {global_step}"
        if current_epoch is not None:
            step_info += f", Epoch {current_epoch}" if step_info else f"Epoch {current_epoch}"
        if is_validation:
            print(f"VALIDATION {step_info}")
        else:
            print(f"{step_info}")
        print("=" * 80)

        # Prepare loss data for tabulate
        losses_table = []
        losses_table.append(["Loss Type", "Value"])

        losses_to_print = {
            f"{prefix}RNA Loss": format_loss(rna_loss, total_loss),
            f"{prefix}Protein Loss": format_loss(protein_loss, total_loss),
            f"{prefix}Contrastive Loss": format_loss(contrastive_loss, total_loss),
            f"{prefix}Matching Loss": format_loss(matching_loss, total_loss),
            f"{prefix}Similarity Loss": format_loss(similarity_loss, total_loss),
            f"{prefix}Cell Type Clustering Loss": format_loss(
                cell_type_clustering_loss, total_loss
            ),
            f"{prefix}Total Loss": total_loss,
        }

        for loss_name, value in losses_to_print.items():
            if value is not None:
                losses_table.append([loss_name, value])

        print("\nLosses:")
        print(tabulate(losses_table, headers="firstrow", tablefmt="fancy_grid"))

        # Print additional metrics for training
        if not is_validation:
            similarity_metrics = []
            similarity_metrics.append(["Metric", "Value"])

            similarity_metrics_to_print = {
                f"{prefix}Similarity Loss Raw": similarity_loss_raw,
                f"{prefix}Similarity Weight": similarity_weight,
                f"{prefix}Similarity Active": similarity_active,
                f"{prefix}Num Acceptable": num_acceptable,
                f"{prefix}Num Cells": num_cells,
                f"{prefix}Exact Pairs": exact_pairs,
                f"{prefix}Latent Distances": get_value(latent_distances),
            }

            for metric_name, value in similarity_metrics_to_print.items():
                if value is not None:
                    similarity_metrics.append([metric_name, value])

            print("\nSimilarity Metrics:")
            print(tabulate(similarity_metrics, headers="firstrow", tablefmt="fancy_grid"))

        # Print validation-specific metrics
        if is_validation and latent_distances is not None:
            distance_metrics = []
            distance_metrics.append(["Statistic", "Value"])

            mean_val = get_value(latent_distances)
            distance_metrics.append(["Mean", mean_val])

            if isinstance(latent_distances, torch.Tensor):
                distance_metrics.append(["Min", f"{latent_distances.min().item():.4f}"])
                distance_metrics.append(["Max", f"{latent_distances.max().item():.4f}"])

            print("\nMatching Distances:")
            print(tabulate(distance_metrics, headers="firstrow", tablefmt="fancy_grid"))

        # Print extra metrics if available
        if any(x != 0 for x in [stress_loss, reward, exact_pairs, ilisi, clisi, accuracy]):
            extra_metrics = []
            extra_metrics.append(["Metric", "Value"])

            extra_metrics_to_print = {
                f"{prefix}Stress Loss": stress_loss,
                f"{prefix}Reward": reward,
                f"{prefix}Exact Pairs": exact_pairs,
                f"{prefix}iLISI": ilisi,
                f"{prefix}cLISI": clisi,
                f"{prefix}Accuracy": accuracy,
            }

            for metric_name, value in extra_metrics_to_print.items():
                if value is not None:
                    extra_metrics.append([metric_name, value])
            # skip for now
            # print("\nExtra Metrics:")
            # print(tabulate(extra_metrics, headers="firstrow", tablefmt="fancy_grid"))

        print("=" * 80 + "\n")

    # Log to MLflow
    step = global_step if global_step is not None else None
    prefix = "val_" if is_validation else ""
    items_to_log = {
        f"{prefix}total_loss": total_loss,
        f"{prefix}rna_loss": rna_loss,
        f"{prefix}protein_loss": protein_loss,
        f"{prefix}contrastive_loss": contrastive_loss,
        f"{prefix}matching_loss": matching_loss,
        f"{prefix}similarity_loss": similarity_loss,
        f"{prefix}cell_type_clustering_loss": cell_type_clustering_loss,
        f"{prefix}adversarial_loss": adv_loss,
        f"{prefix}diversity_loss": diversity_loss,
    }

    # Add training-specific metrics
    if not is_validation and similarity_loss_raw is not None:
        items_to_log["similarity_loss_raw"] = similarity_loss_raw

    if not is_validation and all(
        x is not None for x in [num_acceptable, num_cells, exact_pairs, latent_distances]
    ):
        items_to_log["acceptable_ratio"] = num_acceptable / num_cells if num_cells > 0 else 0
        items_to_log["exact_pairs"] = exact_pairs
        items_to_log["latent_distances"] = get_value(latent_distances)

    # Add extra metrics if available
    items_to_log[f"{prefix}iLISI"] = ilisi
    items_to_log[f"{prefix}cLISI"] = clisi
    items_to_log[f"{prefix}accuracy"] = accuracy
    items_to_log[f"{prefix}stress_loss"] = stress_loss
    items_to_log[f"{prefix}reward"] = reward

    mlflow.log_metrics(format_loss_mlflow(losses), step=step)

    return items_to_log

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

# %% Plotting Functions
# This file contains various plotting functions used in the analysis.

# %%
# Setup paths
# %%
import os
import sys

# Add repository root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# %%
# Imports
# %%

import importlib

import cell_lists
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData, concat
from scipy.sparse import issparse
from sklearn.decomposition import PCA

import bar_nick_utils

importlib.reload(cell_lists)
importlib.reload(bar_nick_utils)


def safe_mlflow_log_figure(fig, file_path):
    """Safely log a figure to MLflow if an experiment is active."""
    try:
        # If file_path starts with step_, save to train folder in MLflow artifacts
        if file_path.startswith("step_"):
            # Extract step number and pad with leading zeros
            step_num = file_path.split("_")[1].split(".")[0]
            padded_step = f"{int(step_num):05d}"
            new_filename = f"step_{padded_step}_{'_'.join(file_path.split('_')[2:])}"
            # Log to MLflow in train folder with padded step number
            mlflow.log_figure(fig, f"train/{new_filename}")
        else:
            # Regular logging for non-step files
            mlflow.log_figure(fig, file_path)
    except Exception as e:
        print(f"Warning: Could not log figure to MLflow: {str(e)}")
        print("Continuing without MLflow logging...")


# %% MaxFuse Plotting Functions
# This module contains functions for plotting MaxFuse-specific visualizations.


def plot_data_overview(adata_1, adata_2):
    """Plot overview of RNA and protein data"""
    print("\nPlotting data overview...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # RNA data
    sc.pl.pca(adata_1, color="cell_types", show=False, ax=axes[0])
    axes[0].set_title("RNA PCA")

    # Protein data
    sc.pl.pca(adata_2, color="cell_types", show=False, ax=axes[1])
    axes[1].set_title("Protein PCA")

    plt.tight_layout()
    plt.show()


def plot_cell_type_distribution(adata_1, adata_2, use_subsample=True):
    """Plot cell type distribution for both datasets"""
    print("\nPlotting cell type distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # RNA data
    if use_subsample:
        # subsample the data for plotting
        subsample_n_obs = min(1000, len(adata_1))
        adata_1_sub = adata_1[np.random.choice(len(adata_1), 1000, replace=False)]
        sns.countplot(data=adata_1_sub.obs, x="cell_types", ax=axes[0])
    else:
        sns.countplot(data=adata_1.obs, x="cell_types", ax=axes[0])
    axes[0].set_title("RNA Cell Types")
    axes[0].tick_params(axis="x", rotation=45)

    # Protein data
    if use_subsample:
        # subsample the data for plotting
        subsample_n_obs = min(1000, len(adata_2))
        adata_2_sub = adata_2[np.random.choice(len(adata_2), subsample_n_obs, replace=False)]
        sns.countplot(data=adata_2_sub.obs, x="cell_types", ax=axes[1])
    else:
        sns.countplot(data=adata_2.obs, x="cell_types", ax=axes[1])
    axes[1].set_title("Protein Cell Types")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def plot_spatial_data(adata_prot):
    """Plot spatial data for protein dataset"""
    print("\nPlotting spatial data...")
    plt.figure(figsize=(10, 10))

    # Get unique cell types and create a color map
    unique_cell_types = adata_prot.obs["cell_types"].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cell_types)))
    color_dict = dict(zip(unique_cell_types, colors))

    # Create scatter plot
    for cell_type in unique_cell_types:
        mask = adata_prot.obs["cell_types"] == cell_type
        plt.scatter(
            adata_prot.obsm["spatial"][mask, 0],
            adata_prot.obsm["spatial"][mask, 1],
            c=[color_dict[cell_type]],
            label=cell_type,
            s=1.5,
            alpha=0.6,
        )

    plt.title("Protein Spatial Data")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()
    safe_mlflow_log_figure(plt.gcf(), "protein_spatial_data.png")
    plt.show()


def plot_preprocessing_results(adata_1, adata_2):
    """Plot results after preprocessing"""
    print("\nPlotting preprocessing results...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    # RNA data
    sc.pl.pca(adata_1, color="cell_types", show=False, ax=axes[0, 0])
    axes[0, 0].set_title("RNA PCA")

    sc.pl.umap(adata_1, color="cell_types", show=False, ax=axes[0, 1])
    axes[0, 1].set_title("RNA UMAP")

    # Protein data
    sc.pl.pca(adata_2, color="cell_types", show=False, ax=axes[1, 0])
    axes[1, 0].set_title("Protein PCA")

    sc.pl.umap(adata_2, color="cell_types", show=False, ax=axes[1, 1])
    axes[1, 1].set_title("Protein UMAP")

    plt.tight_layout()
    plt.show()


def plot_spatial_data_comparison(adata_rna, adata_prot):
    """Plot spatial data comparison between RNA and protein datasets"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # RNA data
    axes[0].scatter(
        adata_rna.obsm["spatial"][:, 0],
        adata_rna.obsm["spatial"][:, 1],
        c=adata_rna.obs["CN"],
        cmap="tab10",
        alpha=0.6,
    )
    axes[0].set_title("RNA Spatial Data")
    axes[0].set_xlabel("X coordinate")
    axes[0].set_ylabel("Y coordinate")

    # Protein data
    axes[1].scatter(
        adata_prot.obsm["spatial"][:, 0],
        adata_prot.obsm["spatial"][:, 1],
        c=adata_prot.obs["CN"],
        cmap="tab10",
        alpha=0.6,
    )
    axes[1].set_title("Protein Spatial Data")
    axes[1].set_xlabel("X coordinate")
    axes[1].set_ylabel("Y coordinate")

    plt.tight_layout()
    plt.show()


# %% VAE Plotting Functions
# This module contains functions for plotting VAE-specific visualizations.
def plot_train_val_normalized_losses(history):
    """Plot training and validation losses normalized in the same figure.

    Args:
        history: Dictionary containing training and validation loss histories

    This function:
    1. Uses epoch-wise means of the losses
    2. Normalizes the losses for better visualization (0-1 range)
    3. Plots training and validation losses in separate subplots for easier comparison
    4. Uses consistent colors for the same loss type across both plots
    """
    try:
        # Debug print of the history keys we received
        print("DEBUG: plot_train_val_normalized_losses received history with keys:")
        for k, v in history.items():
            if isinstance(v, list):
                print(f"  {k}: {len(v)} items")
                if k.startswith("val_") and len(v) > 0:
                    print(f"    First few values: {v[:min(3, len(v))]}")

        # Get all loss keys from history
        loss_keys = [k for k in history.keys() if "loss" in k.lower() and len(history[k]) > 0]

        # Split into train and validation losses
        train_loss_keys = [k for k in loss_keys if k.startswith("train_")]
        val_loss_keys = [k for k in loss_keys if k.startswith("val_") and k != "val_epochs"]

        # Skip if we don't have both train and val losses
        if not train_loss_keys:
            print("Not enough data to plot train losses")
            return

        # Create figure with two vertically stacked subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 11), sharex=True)

        # Dictionary to hold normalized losses
        train_normalized_losses = {}
        val_normalized_losses = {}

        # Get the validation epochs information
        val_epochs = history.get("val_epochs", [])

        # If val_epochs is not available or empty, create a simple range based on validation data
        val_data_lengths = [len(history.get(k, [])) for k in val_loss_keys]
        max_val_length = max(val_data_lengths) if val_data_lengths else 0

        if not val_epochs and max_val_length > 0:
            val_epochs = list(range(max_val_length))
            print(f"Created {len(val_epochs)} validation epochs (one per validation point)")
        else:
            print(f"Using {len(val_epochs)} validation epochs from history")

        # Create epochs array for training data
        # Take the length of the longest train loss array
        train_data_lengths = [len(history.get(k, [])) for k in train_loss_keys]
        max_train_length = max(train_data_lengths) if train_data_lengths else 0
        train_epochs = list(range(max_train_length))
        print(f"Using {len(train_epochs)} training epochs")

        # Format value helper function
        def format_value(value):
            """Format numeric values - round to integer if >= 10"""
            if abs(value) >= 10:
                return f"{int(round(value))}"
            else:
                return f"{value:.2f}"

        # Get all unique loss types (without train/val prefix)
        all_loss_types = set()
        for key in train_loss_keys + val_loss_keys:
            loss_type = key.replace("train_", "").replace("val_", "")
            all_loss_types.add(loss_type)

        # Create a fixed color mapping using the standard matplotlib color cycle
        color_map = {}
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

        # Ensure the same loss type gets the same color in both plots
        for i, loss_type in enumerate(sorted(all_loss_types)):
            color_idx = i % len(colors)
            color_map[loss_type] = colors[color_idx]

        # Process training losses
        for train_key in train_loss_keys:
            if train_key in history and len(history[train_key]) > 0:
                train_values = np.array(history[train_key], dtype=np.float64)
                # Remove inf and nan
                train_values = train_values[~np.isinf(train_values) & ~np.isnan(train_values)]

                if len(train_values) > 0:
                    loss_type = train_key.replace("train_", "")
                    # Normalize to 0-1 range
                    min_val = np.min(train_values)
                    max_val = np.max(train_values)
                    if max_val > min_val:  # Avoid division by zero
                        train_normalized_losses[loss_type] = {
                            "values": (train_values - min_val) / (max_val - min_val),
                            "min": min_val,
                            "max": max_val,
                        }

        # Plot training losses
        for loss_type, data in train_normalized_losses.items():
            label = f"{loss_type.replace('_', ' ').title()} (min:{format_value(data['min'])}, max:{format_value(data['max'])})"
            axes[0].plot(
                train_epochs[: len(data["values"])],
                data["values"],
                label=label,
                alpha=0.8,
                color=color_map.get(loss_type),
                linewidth=2,
            )

        # Process validation losses - only if we have validation data
        has_val_data = False
        for val_key in val_loss_keys:
            if val_key in history and len(history[val_key]) > 0:
                val_values = np.array(history[val_key], dtype=np.float64)
                # Remove inf and nan
                val_values = val_values[~np.isinf(val_values) & ~np.isnan(val_values)]

                if len(val_values) > 0:
                    has_val_data = True
                    loss_type = val_key.replace("val_", "")
                    # Normalize to 0-1 range
                    min_val = np.min(val_values)
                    max_val = np.max(val_values)
                    if max_val > min_val:  # Avoid division by zero
                        val_normalized_losses[loss_type] = {
                            "values": (val_values - min_val) / (max_val - min_val),
                            "min": min_val,
                            "max": max_val,
                        }

        # Plot validation losses - only if we have data
        if has_val_data:
            for loss_type, data in val_normalized_losses.items():
                val_epochs_plot = val_epochs[: len(data["values"])]

                label = f"{loss_type.replace('_', ' ').title()} (min:{format_value(data['min'])}, max:{format_value(data['max'])})"
                axes[1].plot(
                    val_epochs_plot,
                    data["values"],
                    label=label,
                    alpha=0.8,
                    marker="o",
                    markersize=5,
                    color=color_map.get(loss_type),
                    linewidth=2,
                )
        else:
            # If we have no validation data, add a message
            axes[1].text(
                0.5,
                0.5,
                "No validation data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axes[1].transAxes,
                fontsize=14,
            )

        # Set titles and labels
        axes[0].set_title("Normalized Training Losses (0-1 scale)")
        axes[1].set_title("Normalized Validation Losses (0-1 scale)")
        axes[1].set_xlabel("Epoch")
        axes[0].set_ylabel("Normalized Loss")
        axes[1].set_ylabel("Normalized Loss")

        # Add legends outside the plots
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        if has_val_data:
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Add grid
        axes[0].grid(True)
        axes[1].grid(True)

        # Add a shared y-axis limit from 0 to 1
        axes[0].set_ylim(0, 1.05)
        axes[1].set_ylim(0, 1.05)

        # Set a fixed x-axis limit to match training epochs
        max_epoch = max(len(train_epochs) - 1, max(val_epochs) if val_epochs else 0)
        axes[0].set_xlim(-0.1, max_epoch + 0.1)

        # Adjust layout to fit the legend outside
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)  # Make space for legend

        safe_mlflow_log_figure(plt.gcf(), "train_val_normalized_losses.png")
        plt.close()
    except Exception as e:
        print(f"Error plotting normalized losses: {str(e)}")
        import traceback

        traceback.print_exc()
        # Don't let plotting errors disrupt training


def plot_latent_pca_both_modalities_cn(
    rna_mean,
    protein_mean,
    adata_rna_subset,
    adata_prot_subset,
    index_rna,
    index_prot,
    global_step=None,
    use_subsample=True,
):
    # Subsample if requested - use separate sampling for RNA and protein
    if use_subsample:
        # Sample RNA data
        n_subsample_rna = min(700, len(index_rna))
        rna_subsample_idx = np.random.choice(len(index_rna), n_subsample_rna, replace=False)
        index_rna = np.array(index_rna)[rna_subsample_idx]
        rna_mean = rna_mean[rna_subsample_idx]

        # Sample protein data (separately)
        n_subsample_prot = min(700, len(index_prot))
        prot_subsample_idx = np.random.choice(len(index_prot), n_subsample_prot, replace=False)
        index_prot = np.array(index_prot)[prot_subsample_idx]
        protein_mean = protein_mean[prot_subsample_idx]

    plt.figure(figsize=(10, 5))
    pca = PCA(n_components=3)
    # concatenate the means
    combined_mean = np.concatenate([rna_mean, protein_mean], axis=0)
    pca.fit(combined_mean)
    combined_pca = pca.transform(combined_mean)
    num_rna = len(rna_mean)
    plt.subplot(1, 3, 1)
    sns.scatterplot(
        x=combined_pca[:num_rna, 0],
        y=combined_pca[:num_rna, 1],
        hue=adata_rna_subset[index_rna].obs["CN"],
    )
    plt.title("RNA")

    plt.subplot(1, 3, 2)
    sns.scatterplot(
        x=combined_pca[num_rna:, 0],
        y=combined_pca[num_rna:, 1],
        hue=adata_prot_subset[index_prot].obs["CN"],
    )
    plt.title("protein")
    plt.suptitle("PCA of latent space during training\nColor by CN label")

    ax = plt.subplot(1, 3, 3, projection="3d")
    ax.scatter(
        combined_pca[:num_rna, 0],
        combined_pca[:num_rna, 1],
        combined_pca[:num_rna, 2],
        c="red",
        label="RNA",
    )
    ax.scatter(
        combined_pca[num_rna:, 0],
        combined_pca[num_rna:, 1],
        combined_pca[num_rna:, 2],
        c="blue",
        label="protein",
        alpha=0.5,
    )

    # Only draw lines if we have equal numbers of points
    if len(rna_mean) == len(protein_mean):
        for i, (rna_point, prot_point) in enumerate(
            zip(combined_pca[:num_rna], combined_pca[num_rna:])
        ):
            if i < min(num_rna, len(combined_pca) - num_rna):  # Ensure we don't go out of bounds
                ax.plot(
                    [rna_point[0], prot_point[0]],
                    [rna_point[1], prot_point[1]],
                    [rna_point[2], prot_point[2]],
                    "k--",
                    alpha=0.6,
                    lw=0.5,
                )

    ax.set_title("merged RNA and protein")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    plt.tight_layout()

    if global_step is not None:
        safe_mlflow_log_figure(plt.gcf(), f"step_{global_step}_latent_pca_both_modalities.png")
    else:
        safe_mlflow_log_figure(plt.gcf(), "latent_pca_both_modalities.png")
    plt.show()


def plot_latent_pca_both_modalities_by_celltype(
    adata_rna_subset,
    adata_prot_subset,
    latent_rna,
    latent_prot,
    index_rna=None,
    index_prot=None,
    global_step=None,
    use_subsample=True,
):
    """Plot PCA of latent space colored by cell type."""
    if index_rna is None:
        index_rna = range(len(latent_rna))
    if index_prot is None:
        index_prot = range(len(latent_prot))

    # Ensure indices are within bounds

    # Subsample if requested - use separate sampling for RNA and protein
    if use_subsample:
        # Sample RNA data
        n_subsample_rna = min(1000, len(index_rna))
        rna_subsample_idx = np.random.choice(len(index_rna), n_subsample_rna, replace=False)
        index_rna = np.array(index_rna)[rna_subsample_idx]
        latent_rna = latent_rna[rna_subsample_idx]
        # Sample protein data (separately)
        n_subsample_prot = min(1000, len(index_prot))
        prot_subsample_idx = np.random.choice(len(index_prot), n_subsample_prot, replace=False)
        index_prot = np.array(index_prot)[prot_subsample_idx]
        latent_prot = latent_prot[prot_subsample_idx]

    # Ensure all indices are valid
    if len(index_rna) == 0 or len(index_prot) == 0:
        print("Warning: No valid indices for plotting. Skipping plot.")
        return

    num_rna = len(index_rna)

    combined_latent = np.vstack([latent_rna, latent_prot])
    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined_latent)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    if "cell_type" in adata_rna_subset.obs:
        hue_col = "cell_type"
    elif "cell_types" in adata_rna_subset.obs:
        hue_col = "cell_types"
    else:
        hue_col = "major_cell_types"

    sns.scatterplot(
        x=combined_pca[:num_rna, 0],
        y=combined_pca[:num_rna, 1],
        hue=adata_rna_subset[index_rna].obs[hue_col],
    )
    plt.title("RNA")

    plt.subplot(1, 2, 2)
    if "cell_type" in adata_prot_subset.obs:
        hue_col = "cell_type"
    elif "cell_types" in adata_prot_subset.obs:
        hue_col = "cell_types"
    else:
        hue_col = "major_cell_types"

    sns.scatterplot(
        x=combined_pca[num_rna:, 0],
        y=combined_pca[num_rna:, 1],
        hue=adata_prot_subset[index_prot].obs[hue_col],
    )
    plt.title("protein")
    plt.suptitle("PCA of latent space during training\nColor by cell type")
    plt.tight_layout()

    if global_step is not None:
        safe_mlflow_log_figure(plt.gcf(), f"step_{global_step}_latent_pca_celltype.png")
    else:
        safe_mlflow_log_figure(plt.gcf(), "latent_pca_celltype.png")
    plt.show()


def plot_latent_mean_std_legacy(
    rna_inference_outputs,
    protein_inference_outputs,
    adata_rna,
    adata_prot,
    index_rna=None,
    index_prot=None,
    use_subsample=True,
):
    """Plot latent space visualization combining heatmaps and PCA plots.

    Args:
        rna_inference_outputs: RNA inference outputs containing qz means and scales
        protein_inference_outputs: Protein inference outputs containing qz means and scales
        adata_rna: RNA AnnData object
        adata_prot: Protein AnnData object
        index_rna: Indices for RNA data (optional)
        index_prot: Indices for protein data (optional)
        use_subsample: Whether to subsample to 700 points (default: True)
    """
    if index_rna is None:
        index_rna = range(len(adata_rna.obs.index))
    if index_prot is None:
        index_prot = range(len(adata_prot.obs.index))

    # Convert tensors to numpy if needed
    rna_mean = rna_inference_outputs["qz"].mean.detach().cpu().numpy()
    protein_mean = protein_inference_outputs["qz"].mean.detach().cpu().numpy()
    rna_std = rna_inference_outputs["qz"].scale.detach().cpu().numpy()
    protein_std = protein_inference_outputs["qz"].scale.detach().cpu().numpy()

    # Subsample if requested - use separate sampling for RNA and protein
    if use_subsample:
        # Sample RNA data
        n_subsample_rna = min(700, len(index_rna))
        rna_subsample_idx = np.random.choice(len(index_rna), n_subsample_rna, replace=False)
        index_rna = np.array(index_rna)[rna_subsample_idx]
        rna_mean = rna_mean[rna_subsample_idx]
        rna_std = rna_std[rna_subsample_idx]

        # Sample protein data (separately)
        n_subsample_prot = min(700, len(index_prot))
        prot_subsample_idx = np.random.choice(len(index_prot), n_subsample_prot, replace=False)
        index_prot = np.array(index_prot)[prot_subsample_idx]
        protein_mean = protein_mean[prot_subsample_idx]
        protein_std = protein_std[prot_subsample_idx]

    # Plot heatmaps
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    sns.heatmap(rna_mean)
    plt.title("RNA Mean Latent Space")

    plt.subplot(122)
    sns.heatmap(protein_mean)
    plt.title("Protein Mean Latent Space")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    sns.heatmap(rna_std)
    plt.title("RNA Std Latent Space")

    plt.subplot(122)
    sns.heatmap(protein_std)
    plt.title("Protein Std Latent Space")
    plt.tight_layout()
    plt.show()

    # Create AnnData objects for PCA visualization
    rna_ann = AnnData(X=rna_mean, obs=adata_rna.obs.iloc[index_rna].copy())
    protein_ann = AnnData(X=protein_mean, obs=adata_prot.obs.iloc[index_prot].copy())

    # Plot PCA and distributions
    plt.figure(figsize=(15, 5))

    # RNA PCA
    plt.subplot(131)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(rna_ann.X)

    df = pd.DataFrame(
        {
            "PC1": pca_result[:, 0],
            "PC2": pca_result[:, 1],
            "CN": rna_ann.obs["CN"],  # Add the CN column
        }
    )
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="CN")
    plt.title("RNA Latent Space PCA")

    # Protein PCA
    plt.subplot(132)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(protein_ann.X)
    df = pd.DataFrame(
        {
            "PC1": pca_result[:, 0],
            "PC2": pca_result[:, 1],
            "CN": protein_ann.obs["CN"],  # Add the CN column
        }
    )
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="CN")
    plt.title("Protein Latent Space PCA")

    # Standard deviation distributions
    plt.subplot(133)
    plt.hist(rna_std.flatten(), bins=50, alpha=0.5, label="RNA", density=True)
    plt.hist(protein_std.flatten(), bins=50, alpha=0.5, label="Protein", density=True)
    plt.title("Latent Space Standard Deviations")
    plt.xlabel("Standard Deviation")
    plt.ylabel("Density")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_rna_protein_matching_means_and_scale(
    rna_latent_mean,
    protein_latent_mean,
    rna_latent_std,
    protein_latent_std,
    archetype_dis_mat,
    use_subsample=True,
    global_step=None,
):
    """
    Plot the means and scales as halo  and lines between the best matches
    of the RNA and protein
    Args:
        rna_inference_outputs: the output of the RNA inference
        protein_inference_outputs: the output of the protein inference
        archetype_dis_mat: the archetype distance matrix
        use_subsample: whether to use subsampling
        global_step: the current training step, if None then not during training
    """
    if use_subsample:
        subsample_indexes = np.random.choice(rna_latent_mean.shape[0], 700, replace=False)
    else:
        subsample_indexes = np.arange(rna_latent_mean.shape[0])
    prot_new_order = archetype_dis_mat.argmin(axis=0).detach().cpu().numpy()

    rna_means = rna_latent_mean[subsample_indexes]
    rna_scales = rna_latent_std[subsample_indexes]
    protein_means = protein_latent_mean[prot_new_order][subsample_indexes]
    protein_scales = protein_latent_std[prot_new_order][subsample_indexes]
    # match the order of the means to the archetype_dis
    # Combine means for PCA
    combined_means = np.concatenate([rna_means, protein_means], axis=0)

    # Fit PCA on means
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined_means)

    # Transform scales using the same PCA transformation
    combined_scales = np.concatenate([rna_scales, protein_scales], axis=0)
    scales_transformed = pca.transform(combined_scales)

    # Plot with halos
    plt.figure(figsize=(8, 6))

    # Plot RNA points and halos
    for i in range(rna_means.shape[0]):
        # Add halo using scale information
        circle = plt.Circle(
            (pca_result[i, 0], pca_result[i, 1]),
            radius=np.linalg.norm(scales_transformed[i]) * 0.05,
            color="blue",
            alpha=0.1,
        )
        plt.gca().add_patch(circle)
    # Plot Protein points and halos
    for i in range(protein_means.shape[0]):
        # Add halo using scale information
        circle = plt.Circle(
            (pca_result[rna_means.shape[0] + i, 0], pca_result[rna_means.shape[0] + i, 1]),
            radius=np.linalg.norm(scales_transformed[rna_means.shape[0] + i]) * 0.05,
            color="orange",
            alpha=0.1,
        )
        plt.gca().add_patch(circle)

    # Add connecting lines
    for i in range(rna_means.shape[0]):
        color = "red" if (i % 2 == 0) else "green"
        plt.plot(
            [pca_result[i, 0], pca_result[rna_means.shape[0] + i, 0]],
            [pca_result[i, 1], pca_result[rna_means.shape[0] + i, 1]],
            "k-",
            alpha=0.2,
            color=color,
        )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA of RNA and Protein with Scale Halos")
    plt.legend()
    plt.gca().set_aspect("equal")
    plt.tight_layout()

    if global_step is not None:
        safe_mlflow_log_figure(
            plt.gcf(), f"step_{global_step}_rna_protein_matching_means_and_scale.png"
        )
    else:
        safe_mlflow_log_figure(plt.gcf(), "rna_protein_matching_means_and_scale.png")
    plt.show()


def plot_inference_outputs(
    rna_inference_outputs,
    protein_inference_outputs,
    latent_distances,
    rna_distances,
    prot_distances,
):
    """Plot inference outputs"""
    print("\nPlotting inference outputs...")
    fig, axes = plt.subplots(2, 3)

    # Plot latent distances
    axes[0, 0].hist(latent_distances.detach().cpu().numpy().flatten(), bins=50)
    axes[0, 0].set_title("Latent Distances")

    # Plot RNA distances
    axes[0, 1].hist(rna_distances.detach().cpu().numpy().flatten(), bins=50)
    axes[0, 1].set_title("RNA Distances")

    # Plot protein distances
    axes[0, 2].hist(prot_distances.detach().cpu().numpy().flatten(), bins=50)
    axes[0, 2].set_title("Protein Distances")

    # Plot latent vs RNA distances
    axes[1, 0].scatter(
        rna_distances.detach().cpu().numpy().flatten(),
        latent_distances.detach().cpu().numpy().flatten(),
        alpha=0.1,
    )
    axes[1, 0].set_title("Latent vs RNA Distances")

    # Plot latent vs protein distances
    axes[1, 1].scatter(
        prot_distances.detach().cpu().numpy().flatten(),
        latent_distances.detach().cpu().numpy().flatten(),
        alpha=0.1,
    )
    axes[1, 1].set_title("Latent vs Protein Distances")

    # Plot RNA vs protein distances
    axes[1, 2].scatter(
        rna_distances.detach().cpu().numpy().flatten(),
        prot_distances.detach().cpu().numpy().flatten(),
        alpha=0.1,
    )
    axes[1, 2].set_title("RNA vs Protein Distances")

    plt.tight_layout()
    safe_mlflow_log_figure(plt.gcf(), "inference_outputs.png")
    plt.show()


def plot_similarity_loss_history(
    similarity_loss_all_history, active_similarity_loss_active_history, global_step
):
    """
    Plot the similarity loss history and highlight active steps
    """
    if len(similarity_loss_all_history) < 10:
        return
    plt.figure()
    colors = [
        "red" if active else "blue" for active in active_similarity_loss_active_history[-1000:]
    ]
    num_samples = len(similarity_loss_all_history[-1000:])
    dot_size = max(1, 1000 // num_samples)  # Adjust dot size based on the number of samples
    plt.scatter(np.arange(num_samples), similarity_loss_all_history[-1000:], c=colors, s=dot_size)
    plt.title(f"step_{global_step} Similarity loss history (last 1000 steps)")
    plt.xlabel("Step")
    plt.ylabel("Similarity Loss")
    plt.xticks(
        np.arange(0, num_samples, step=max(1, num_samples // 10)),
        np.arange(
            max(0, len(similarity_loss_all_history) - 1000),
            len(similarity_loss_all_history),
            step=max(1, num_samples // 10),
        ),
    )
    red_patch = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="red",
        markersize=10,
        label="Active",
        alpha=0.5,
    )
    blue_patch = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="blue",
        markersize=10,
        label="Inactive",
        alpha=0.5,
    )
    plt.legend(handles=[red_patch, blue_patch])
    plt.tight_layout()
    safe_mlflow_log_figure(plt.gcf(), f"step_{global_step}_similarity_loss_history.png")
    plt.show()


def plot_normalized_losses(history):
    """Plot normalized training and validation losses in separate figures."""
    # Get all loss keys from history
    loss_keys = [k for k in history.keys() if "loss" in k.lower() and len(history[k]) > 0]

    # Split into train and validation losses
    train_loss_keys = [k for k in loss_keys if k.startswith("train_") and "adv" not in k.lower()]
    val_loss_keys = [
        k
        for k in loss_keys
        if (k.startswith("val_") or k.startswith("validation_")) and "adv" not in k.lower()
    ]

    # Function to normalize and plot losses
    def plot_losses(keys, title):
        plt.figure(figsize=(10, 5))
        normalized_losses = {}
        labels = {}

        for key in keys:
            values = history[key]
            if len(values) > 1:  # Only process if we have more than 1 value
                values = np.array(values[1:])  # Skip first step
                # Remove inf and nan
                values = values[~np.isinf(values) & ~np.isnan(values)]
                if len(values) > 0:  # Check again after filtering
                    min_val = np.min(values)
                    max_val = np.max(values)
                    label = f"{key} min: {min_val:.0f} max: {max_val:.0f}"
                    labels[key] = label
                    if max_val > min_val:  # Avoid division by zero
                        normalized_losses[key] = (values - min_val) / (max_val - min_val)

        # Plot each normalized loss
        for key, values in normalized_losses.items():
            if "total" in key.lower():
                plt.plot(values, label=labels[key], alpha=0.7, linestyle="--")
            else:
                plt.plot(values, label=labels[key], alpha=0.7)

        plt.title(title)
        plt.xlabel("Step")
        plt.ylabel("Normalized Loss")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        safe_mlflow_log_figure(plt.gcf(), f"{title.lower().replace(' ', '_')}.png")
        plt.show()

    # Plot training losses
    if train_loss_keys:
        plot_losses(train_loss_keys, "Normalized Training Losses")

    # Plot validation losses
    if val_loss_keys:
        plot_losses(val_loss_keys, "Normalized Validation Losses")


def plot_end_of_epoch_umap_latent_space(prefix, combined_latent, epoch):
    sc.tl.umap(combined_latent)

    # Create a figure with the UMAP visualizations colored by different factors
    fig = plt.figure(figsize=(15, 5))

    # Plot UMAP colored by modality
    ax1 = fig.add_subplot(1, 3, 1)
    sc.pl.umap(
        combined_latent,
        color="modality",
        ax=ax1,
        show=False,
        title=f"{prefix}Combined Latent UMAP by Modality",
    )

    # Plot UMAP colored by cell type
    ax2 = fig.add_subplot(1, 3, 2)
    sc.pl.umap(
        combined_latent,
        color="cell_types",
        ax=ax2,
        show=False,
        title=f"{prefix}Combined Latent UMAP by Cell Type",
    )

    # Plot UMAP colored by neighborhood
    ax3 = fig.add_subplot(1, 3, 3)
    sc.pl.umap(
        combined_latent, color="CN", ax=ax3, show=False, title=f"{prefix}Combined Latent UMAP by CN"
    )

    plt.tight_layout()

    # Save figure for MLflow logging
    umap_file = f"{prefix}combined_latent_umap_epoch_{epoch:03d}.png"
    plt.savefig(umap_file, dpi=200, bbox_inches="tight")
    if hasattr(mlflow, "active_run") and mlflow.active_run():
        mlflow.log_artifact(umap_file, artifact_path="train")
    plt.close(fig)

    print(f"   ✓ {prefix}combined latent UMAP visualized and saved")


def plot_cosine_distance(rna_batch, protein_batch):
    umap_model = UMAP(n_components=2, random_state=42).fit(rna_batch["archetype_vec"], min_dist=5)
    # Transform both modalities using the same UMAP model
    rna_archetype_2pc = umap_model.transform(rna_batch["archetype_vec"])
    prot_archetype_2pc = umap_model.transform(protein_batch["archetype_vec"])

    rna_norm = rna_archetype_2pc / np.linalg.norm(rna_archetype_2pc, axis=1)[:, None]
    scale = 1.2
    prot_norm = scale * prot_archetype_2pc / np.linalg.norm(prot_archetype_2pc, axis=1)[:, None]
    plt.scatter(rna_norm[:, 0], rna_norm[:, 1], label="RNA", alpha=0.7)
    plt.scatter(prot_norm[:, 0], prot_norm[:, 1], label="Protein", alpha=0.7)

    for rna, prot in zip(rna_norm, prot_norm):
        plt.plot([rna[0], prot[0]], [rna[1], prot[1]], "k--", alpha=0.6, lw=0.5)

    # Add unit circle for reference
    theta = np.linspace(0, 2 * np.pi, 100)

    plt.plot(np.cos(theta), np.sin(theta), "grey", linestyle="--", alpha=0.3)
    plt.axis("equal")
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(scale * np.cos(theta), scale * np.sin(theta), "grey", linestyle="--", alpha=0.3)
    plt.axis("equal")

    plt.title("Normalized Vector Alignment\n(Euclidean Distance ∝ Cosine Distance)")
    plt.xlabel("PC1 (Normalized)")
    plt.ylabel("PC2 (Normalized)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_archetype_vs_latent_distances(archetype_dis_tensor, latent_distances, threshold):
    """Plot archetype vs latent distances"""
    print("\nPlotting archetype vs latent distances...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot archetype distances
    axes[0].hist(archetype_dis_tensor.detach().cpu().numpy().flatten(), bins=50)
    axes[0].axvline(x=threshold, color="r", linestyle="--", label=f"Threshold: {threshold}")
    axes[0].set_title("Archetype Distances")
    axes[0].legend()

    # Plot latent distances
    axes[1].hist(latent_distances.detach().cpu().numpy().flatten(), bins=50)
    axes[1].axvline(x=threshold, color="r", linestyle="--", label=f"Threshold: {threshold}")
    axes[1].set_title("Latent Distances")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_latent_distances(latent_distances, threshold):
    """Plot latent distances and threshold"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot latent distances heatmap
    sns.heatmap(latent_distances.detach().cpu().numpy(), ax=axes[0])
    axes[0].set_title("Latent Distances")
    axes[0].legend()

    # Plot latent distances
    axes[1].hist(latent_distances.detach().cpu().numpy().flatten(), bins=50)
    axes[1].axvline(x=threshold, color="r", linestyle="--", label=f"Threshold: {threshold}")
    axes[1].set_title("Latent Distances")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_combined_latent_space(combined_latent, use_subsample=True):
    """Plot combined latent space visualizations"""
    # Subsample if requested
    if use_subsample:
        subsample_n_obs = min(1000, combined_latent.shape[0])
        subsample_idx = np.random.choice(combined_latent.shape[0], subsample_n_obs, replace=False)
        combined_latent_plot = combined_latent[subsample_idx].copy()
    else:
        combined_latent_plot = combined_latent.copy()

    # Plot UMAP
    sc.tl.umap(combined_latent_plot, min_dist=0.1)
    sc.pl.umap(
        combined_latent_plot,
        color=["CN", "modality", "cell_types"],
        title=[
            "UMAP Combined Latent space CN",
            "UMAP Combined Latent space modality",
            "UMAP Combined Latent space cell types",
        ],
        alpha=0.5,
    )
    plt.tight_layout()
    safe_mlflow_log_figure(plt.gcf(), "combined_latent_space_umap.png")

    # Plot PCA
    sc.pl.pca(
        combined_latent_plot,
        color=["CN", "modality"],
        title=["PCA Combined Latent space CN", "PCA Combined Latent space modality"],
        alpha=0.5,
    )
    plt.tight_layout()
    safe_mlflow_log_figure(plt.gcf(), "combined_latent_space_pca.png")


def plot_cell_type_distributions(combined_latent, top_n=3, use_subsample=True):
    """Plot UMAP for top N most common cell types"""
    top_cell_types = combined_latent.obs["cell_types"].value_counts().index[:top_n]

    for cell_type in top_cell_types:
        cell_type_data = combined_latent[combined_latent.obs["cell_types"] == cell_type]

        # Subsample if requested
        if use_subsample and cell_type_data.shape[0] > 700:
            n_subsample = min(700, cell_type_data.shape[0])
            subsample_idx = np.random.choice(cell_type_data.shape[0], n_subsample, replace=False)
            cell_type_data_plot = cell_type_data[subsample_idx].copy()
        else:
            cell_type_data_plot = cell_type_data.copy()

        sc.pl.umap(
            cell_type_data_plot,
            color=["CN", "modality", "cell_types"],
            title=[
                f"Combined latent space UMAP {cell_type}, CN",
                f"Combined latent space UMAP {cell_type}, modality",
                f"Combined latent space UMAP {cell_type}, cell types",
            ],
            alpha=0.5,
        )
        plt.tight_layout()
        safe_mlflow_log_figure(
            plt.gcf(), f"cell_type_distribution/cell_type_distribution_{cell_type}.png"
        )


def plot_rna_protein_latent_cn_cell_type_umap(rna_adata, protein_adata, use_subsample=True):
    """Plot RNA and protein embeddings"""
    # Create copies to avoid modifying the original data
    if use_subsample:
        # Subsample RNA data
        n_subsample_rna = min(700, rna_adata.shape[0])
        subsample_idx_rna = np.random.choice(rna_adata.shape[0], n_subsample_rna, replace=False)
        rna_adata_plot = rna_adata[subsample_idx_rna].copy()

        # Subsample protein data
        n_subsample_prot = min(700, protein_adata.shape[0])
        subsample_idx_prot = np.random.choice(
            protein_adata.shape[0], n_subsample_prot, replace=False
        )
        prot_adata_plot = protein_adata[subsample_idx_prot].copy()
    else:
        rna_adata_plot = rna_adata.copy()
        prot_adata_plot = protein_adata.copy()

    sc.pl.embedding(
        rna_adata_plot,
        color=["CN", "cell_types"],
        basis="X_scVI",
        title=["RNA_latent_CN", "RNA_Latent_CellTypes"],
    )
    plt.tight_layout()
    safe_mlflow_log_figure(plt.gcf(), "rna_latent_embeddings.png")

    sc.pl.embedding(
        prot_adata_plot,
        color=["CN", "cell_types"],
        basis="X_scVI",
        title=["Protein_latent_CN", "Protein_Laten_CellTypes"],
    )
    plt.tight_layout()
    safe_mlflow_log_figure(plt.gcf(), "protein_latent_embeddings.png")


def plot_archetype_embedding(rna_adata, protein_adata, use_subsample=True):
    """Plot archetype embedding"""
    # Create AnnData objects from archetype vectors
    rna_archtype = AnnData(rna_adata.obsm["archetype_vec"])
    rna_archtype.obs = rna_adata.obs.copy()

    prot_archtype = AnnData(protein_adata.obsm["archetype_vec"])
    prot_archtype.obs = protein_adata.obs.copy()

    # Apply subsampling if requested
    if use_subsample:
        # Subsample RNA data
        n_subsample_rna = min(700, rna_archtype.shape[0])
        subsample_idx_rna = np.random.choice(rna_archtype.shape[0], n_subsample_rna, replace=False)
        rna_archtype_plot = rna_archtype[subsample_idx_rna].copy()

        # Subsample protein data
        n_subsample_prot = min(700, prot_archtype.shape[0])
        subsample_idx_prot = np.random.choice(
            prot_archtype.shape[0], n_subsample_prot, replace=False
        )
        prot_archtype_plot = prot_archtype[subsample_idx_prot].copy()
    else:
        rna_archtype_plot = rna_archtype.copy()
        prot_archtype_plot = prot_archtype.copy()

    # Calculate neighbors and UMAP
    sc.pp.neighbors(rna_archtype_plot)
    sc.tl.umap(rna_archtype_plot)

    sc.pp.neighbors(prot_archtype_plot)
    sc.tl.umap(prot_archtype_plot)

    # Plot archetype vectors
    sc.pl.umap(
        rna_archtype_plot,
        color=["CN", "cell_types"],
        title=["RNA_Archetype_UMAP_CN", "RNA_Archetype_UMAP_CellTypes"],
    )
    plt.tight_layout()
    safe_mlflow_log_figure(plt.gcf(), "rna_archetype_umap.png")

    sc.pl.umap(
        prot_archtype_plot,
        color=["CN", "cell_types"],
        title=[
            "Protein_Archetype_UMAP_CN",
            "Protein_Archetype_UMAP_CellTypes",
        ],
    )
    plt.tight_layout()
    safe_mlflow_log_figure(plt.gcf(), "protein_archetype_umap.png")


# %%


def plot_neighbor_means(adata_2_prot, neighbor_means, max_cells=2000):
    """Plot neighbor means and raw protein expression."""
    # Apply subsampling if too many cells
    if adata_2_prot.shape[0] > max_cells:
        print(f"Subsampling to {max_cells} cells for neighbor means plot")
        idx = np.random.choice(adata_2_prot.shape[0], max_cells, replace=False)
        neighbor_means_plot = neighbor_means[idx]
        x_plot = (
            adata_2_prot.X[idx] if not issparse(adata_2_prot.X) else adata_2_prot.X[idx].toarray()
        )
    else:
        neighbor_means_plot = neighbor_means
        x_plot = adata_2_prot.X if not issparse(adata_2_prot.X) else adata_2_prot.X.toarray()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Mean Protein Expression of Cell Neighborhoods")
    sns.heatmap(neighbor_means_plot)
    plt.subplot(1, 2, 2)
    plt.title("Raw Protein Expression per Cell")
    sns.heatmap(x_plot)
    plt.show()


def plot_spatial_clusters(adata_2_prot, neighbor_means, max_cells=2000):
    """Plot spatial clusters and related visualizations."""

    # if the color pallete does not match the number of categories, add more colors
    if "CN_colors" in adata_2_prot.uns:
        if len(adata_2_prot.uns["CN_colors"]) < len(adata_2_prot.obs["CN"].cat.categories):
            new_palette = sns.color_palette("tab10", len(adata_2_prot.obs["CN"].cat.categories))
            adata_2_prot.uns["CN_colors"] = new_palette.as_hex()

    # Apply subsampling for spatial plot if too many cells
    if adata_2_prot.shape[0] > max_cells:
        print(f"Subsampling to {max_cells} cells for spatial cluster plot")
        adata_plot = adata_2_prot.copy()
        sc.pp.subsample(adata_plot, n_obs=max_cells)
    else:
        adata_plot = adata_2_prot

    fig, ax = plt.subplots()

    sc.pl.scatter(
        adata_plot,
        x="X",
        y="Y",
        color="CN",
        title="Cluster cells by their CN, can see the different CN in different regions, \nthanks to the different B cell types in each region",
        ax=ax,
        show=False,
    )

    # Apply subsampling for heatmaps if too many cells
    if adata_2_prot.shape[0] > max_cells:
        idx = np.random.choice(adata_2_prot.shape[0], max_cells, replace=False)
        neighbor_means_plot = neighbor_means[idx]
        x_plot = (
            adata_2_prot.X[idx] if not issparse(adata_2_prot.X) else adata_2_prot.X[idx].toarray()
        )
    else:
        neighbor_means_plot = neighbor_means
        x_plot = adata_2_prot.X if not issparse(adata_2_prot.X) else adata_2_prot.X.toarray()

    neighbor_adata = AnnData(neighbor_means_plot)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(neighbor_adata.X)
    plt.title("convet sqrt")
    plt.subplot(1, 2, 2)
    sns.heatmap(x_plot)
    plt.title("Proteins expressions of each cell")
    plt.show()

    # Use subsampled data for all downstream analyses
    if adata_2_prot.shape[0] > max_cells:
        sampled_indices = np.random.choice(adata_2_prot.shape[0], max_cells, replace=False)
        neighbor_adata = AnnData(neighbor_means[sampled_indices])
        neighbor_adata.obs["CN"] = pd.Categorical(adata_2_prot.obs["CN"].values[sampled_indices])
    else:
        neighbor_adata = AnnData(neighbor_means)
        neighbor_adata.obs["CN"] = pd.Categorical(adata_2_prot.obs["CN"])

    sc.pp.pca(neighbor_adata)
    sc.pp.neighbors(neighbor_adata)
    sc.tl.umap(neighbor_adata)
    sc.pl.umap(neighbor_adata, color="CN", title="UMAP of CN embedding")

    # Combine protein and CN data with subsampling
    if adata_2_prot.shape[0] > max_cells:
        # Create subsampled versions
        adata_prot_subset = adata_2_prot[sampled_indices].copy()
        adata_prot_cn_concat = concat(
            [adata_prot_subset, neighbor_adata],
            join="outer",
            label="modality",
            keys=["Protein", "CN"],
        )
    else:
        adata_prot_cn_concat = concat(
            [adata_2_prot, neighbor_adata], join="outer", label="modality", keys=["Protein", "CN"]
        )

    X = (
        adata_prot_cn_concat.X.toarray()
        if issparse(adata_prot_cn_concat.X)
        else adata_prot_cn_concat.X
    )
    X = np.nan_to_num(X)
    adata_prot_cn_concat.X = X
    sc.pp.pca(adata_prot_cn_concat)
    sc.pp.neighbors(adata_prot_cn_concat)
    sc.tl.umap(adata_prot_cn_concat)
    sc.pl.umap(
        adata_prot_cn_concat,
        color=["CN", "modality"],
        title=[
            "UMAP of CN embedding to make sure they are not mixed",
            "UMAP of CN embedding to make sure they are not mixed",
        ],
    )
    sc.pl.pca(
        adata_prot_cn_concat,
        color=["CN", "modality"],
        title=[
            "PCA of CN embedding to make sure they are not mixed",
            "PCA of CN embedding to make sure they are not mixed",
        ],
    )


def plot_modality_embeddings(adata_1_rna, adata_2_prot, max_cells=2000):
    """Plot PCA and UMAP embeddings for both modalities."""
    # Subsample data if needed
    if adata_1_rna.shape[0] > max_cells or adata_2_prot.shape[0] > max_cells:
        print(f"Subsampling to at most {max_cells} cells for modality embeddings plot")
        rna_plot = adata_1_rna.copy()
        prot_plot = adata_2_prot.copy()

        if rna_plot.shape[0] > max_cells:
            sc.pp.subsample(rna_plot, n_obs=max_cells)
            # Recalculate PCA and UMAP if needed
            sc.pp.pca(rna_plot)
            sc.pp.neighbors(rna_plot, key_added="original_neighbors", use_rep="X_pca")
            sc.tl.umap(rna_plot, neighbors_key="original_neighbors")
            rna_plot.obsm["X_original_umap"] = rna_plot.obsm["X_umap"]

        if prot_plot.shape[0] > max_cells:
            sc.pp.subsample(prot_plot, n_obs=max_cells)
            # Recalculate PCA and UMAP if needed
            sc.pp.pca(prot_plot)
            sc.pp.neighbors(prot_plot, key_added="original_neighbors", use_rep="X_pca")
            sc.tl.umap(prot_plot, neighbors_key="original_neighbors")
            prot_plot.obsm["X_original_umap"] = prot_plot.obsm["X_umap"]
    else:
        rna_plot = adata_1_rna
        prot_plot = adata_2_prot

    sc.tl.umap(prot_plot, neighbors_key="original_neighbors")
    sc.pl.pca(
        rna_plot,
        color=["cell_types", "major_cell_types"],
        title=["RNA pca minor cell types", "RNA pca major cell types"],
    )
    sc.pl.pca(
        prot_plot,
        color=["cell_types", "major_cell_types"],
        title=["Protein pca minor cell types", "Protein pca major cell types"],
    )
    sc.pl.embedding(
        rna_plot,
        basis="X_umap",
        color=["major_cell_types", "cell_types"],
        title=["RNA UMAP major cell types", "RNA UMAP major cell types"],
    )
    sc.pl.embedding(
        prot_plot,
        basis="X_original_umap",
        color=["major_cell_types", "cell_types"],
        title=["Protein UMAp major cell types", "Protein UMAP major cell types"],
    )


def plot_elbow_method(evs_protein, evs_rna, max_points=300):
    """Plot elbow method results."""
    # Limit the number of points if too many
    if len(evs_protein) > max_points or len(evs_rna) > max_points:
        print(f"Limiting elbow plot to first {max_points} points")
        plot_protein = evs_protein[:max_points]
        plot_rna = evs_rna[:max_points]
    else:
        plot_protein = evs_protein
        plot_rna = evs_rna

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(plot_protein)), plot_protein, marker="o", label="Protein")
    plt.plot(range(len(plot_rna)), plot_rna, marker="s", label="RNA")
    plt.xlabel("Number of Archetypes (k)")
    plt.ylabel("Explained Variance")
    plt.title("Elbow Plot: Explained Variance vs Number of Archetypes")
    plt.legend()
    plt.grid()


def plot_archetype_proportions(
    archetype_proportion_list_rna, archetype_proportion_list_protein, max_size=20
):
    """Plot archetype proportions."""
    # Ensure matrices aren't too large
    rna_prop = archetype_proportion_list_rna[0]
    prot_prop = archetype_proportion_list_protein[0]

    # If either has too many rows/columns, print warning and subsample
    if (
        rna_prop.shape[0] > max_size
        or rna_prop.shape[1] > max_size
        or prot_prop.shape[0] > max_size
        or prot_prop.shape[1] > max_size
    ):
        print(
            f"Warning: Large archetype proportion matrices. Limiting to {max_size} rows/columns for visualization."
        )

        # Subsample if needed
        if rna_prop.shape[0] > max_size:
            rna_prop = rna_prop.iloc[:max_size, :]
        if rna_prop.shape[1] > max_size:
            rna_prop = rna_prop.iloc[:, :max_size]
        if prot_prop.shape[0] > max_size:
            prot_prop = prot_prop.iloc[:max_size, :]
        if prot_prop.shape[1] > max_size:
            prot_prop = prot_prop.iloc[:, :max_size]

    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(rna_prop, cbar=False)
    plt.xticks()
    plt.title("RNA Archetypes")
    plt.subplot(1, 2, 2)
    plt.title("Protein Archetypes")
    sns.heatmap(prot_prop, cbar=False)
    plt.suptitle("Non-Aligned Archetypes Profiles")
    plt.yticks([])
    plt.show()


def plot_archetype_weights(
    best_archetype_rna_prop, best_archetype_prot_prop, row_order, max_size=20
):
    """Plot archetype weights."""
    # Ensure matrices aren't too large
    rna_prop = pd.DataFrame(best_archetype_rna_prop)
    prot_prop = pd.DataFrame(best_archetype_prot_prop)

    # If row_order is too large, limit it
    if len(row_order) > max_size:
        print(f"Warning: Limiting row_order to first {max_size} elements for visualization")
        row_order = row_order[:max_size]

    # If matrices are too large, subsample them
    if (
        rna_prop.shape[0] > max_size
        or rna_prop.shape[1] > max_size
        or prot_prop.shape[0] > max_size
        or prot_prop.shape[1] > max_size
    ):
        print(f"Warning: Large archetype weight matrices. Limiting to {max_size} rows/columns.")

        # Limit rows based on row_order if possible
        if rna_prop.shape[0] > max_size:
            # If row_order is already limited, use it directly
            if len(row_order) <= max_size:
                rna_prop = rna_prop.iloc[row_order, :]
                prot_prop = prot_prop.iloc[row_order, :]
            else:
                # Otherwise use the first max_size rows
                rna_prop = rna_prop.iloc[:max_size, :]
                prot_prop = prot_prop.iloc[:max_size, :]
                row_order = row_order[:max_size]

        # Limit columns if needed
        if rna_prop.shape[1] > max_size:
            rna_prop = rna_prop.iloc[:, :max_size]
        if prot_prop.shape[1] > max_size:
            prot_prop = prot_prop.iloc[:, :max_size]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("RNA Archetype Weights vs Cell Types")
    plt.ylabel("Archetypes")
    sns.heatmap(rna_prop.iloc[row_order], cbar=False)
    plt.yticks([])
    plt.ylabel("Archetypes")
    plt.subplot(1, 2, 2)
    plt.ylabel("Archetypes")
    plt.title("Protein Archetype Weights vs Cell Types")
    sns.heatmap(prot_prop.iloc[row_order], cbar=False)
    plt.ylabel("Archetypes")
    plt.suptitle(
        "Archetype Weight Distribution Across Cell Types (Higher Similarity = Better Alignment)"
    )
    plt.yticks([])
    plt.xticks(rotation=45)
    plt.show()


def plot_archetype_visualizations(
    adata_archetype_rna, adata_archetype_prot, adata_1_rna, adata_2_prot, max_cells=2000
):
    """Plot archetype visualizations."""
    # Apply subsampling if datasets are too large
    if adata_archetype_rna.shape[0] > max_cells:
        print(f"Subsampling RNA archetype data to {max_cells} cells for visualization")
        rna_arch_plot = sc.pp.subsample(adata_archetype_rna, n_obs=max_cells, copy=True)
    else:
        rna_arch_plot = adata_archetype_rna.copy()

    if adata_archetype_prot.shape[0] > max_cells:
        print(f"Subsampling protein archetype data to {max_cells} cells for visualization")
        prot_arch_plot = sc.pp.subsample(adata_archetype_prot, n_obs=max_cells, copy=True)
    else:
        prot_arch_plot = adata_archetype_prot.copy()

    if adata_1_rna.shape[0] > max_cells:
        print(f"Subsampling RNA data to {max_cells} cells for visualization")
        rna_plot = sc.pp.subsample(adata_1_rna, n_obs=max_cells, copy=True)
    else:
        rna_plot = adata_1_rna.copy()

    if adata_2_prot.shape[0] > max_cells:
        print(f"Subsampling protein data to {max_cells} cells for visualization")
        prot_plot = sc.pp.subsample(adata_2_prot, n_obs=max_cells, copy=True)
    else:
        prot_plot = adata_2_prot.copy()

    # Calculate PCA and plot for archetype data
    sc.pp.pca(rna_arch_plot)
    sc.pp.pca(prot_arch_plot)
    sc.pl.pca(rna_arch_plot, color=["major_cell_types", "archetype_label", "cell_types"])
    sc.pl.pca(prot_arch_plot, color=["major_cell_types", "archetype_label", "cell_types"])

    # Calculate neighbors and UMAP for archetype data
    sc.pp.neighbors(rna_arch_plot)
    sc.pp.neighbors(prot_arch_plot)
    sc.tl.umap(rna_arch_plot)
    sc.tl.umap(prot_arch_plot)
    sc.pl.umap(rna_arch_plot, color=["major_cell_types", "archetype_label", "cell_types"])
    sc.pl.umap(prot_arch_plot, color=["major_cell_types", "archetype_label", "cell_types"])

    # Calculate neighbors and UMAP for original data
    sc.pp.neighbors(rna_plot)
    sc.pp.neighbors(prot_plot)
    sc.tl.umap(rna_plot)
    sc.tl.umap(prot_plot)
    sc.pl.umap(
        rna_plot, color="archetype_label", title="RNA UMAP Embedding Colored by Archetype Labels"
    )
    sc.pl.umap(
        prot_plot,
        color="archetype_label",
        title="Protein UMAP Embedding Colored by Archetype Labels",
    )


def plot_umap_visualizations_original_data(adata_rna_subset, adata_prot_subset):
    """Generate UMAP visualizations for original RNA and protein data"""
    print("\nGenerating UMAP visualizations...")
    if "connectivities" not in adata_rna_subset.obsm:
        sc.pp.neighbors(adata_rna_subset)
    if "connectivities" not in adata_prot_subset.obsm:
        sc.pp.neighbors(adata_prot_subset)
    sc.tl.umap(adata_rna_subset)
    sc.tl.umap(adata_prot_subset)
    sc.pl.umap(
        adata_rna_subset,
        color=["CN", "cell_types", "archetype_label"],
        title=["RNA exp UMAP, CN", "RNA exp UMAP, cell types", "RNA exp UMAP, archetype label"],
        show=False,
    )

    sc.pl.umap(
        adata_prot_subset,
        color=["CN", "cell_types", "archetype_label"],
        title=[
            "Protein exp UMAP, CN",
            "Protein exp UMAP, cell types",
            "Protein exp UMAP, archetype label",
        ],
        show=False,
    )
    plt.tight_layout()
    plt.show()


def plot_archetype_heatmaps(adata_rna_subset, adata_prot_subset, archetype_distances):
    """Plot heatmaps of archetype coordinates"""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(np.log1p(adata_rna_subset.obsm["archetype_vec"].values), cbar=False)
    plt.title("RNA Archetype Vectors")
    plt.ylabel("RNA cell index")
    plt.xlabel("Archetype Betas")
    plt.subplot(1, 2, 2)
    sns.heatmap(np.log1p(adata_prot_subset.obsm["archetype_vec"].values), cbar=False)
    plt.xlabel("Archetype Betas")
    plt.ylabel("Protein cell index")
    plt.title("Protein Archetype Vectors")
    plt.show()
    # this is the heatmap of the archetype distances
    plt.figure(figsize=(10, 5))
    plt.title("Archetype Distances")
    plt.subplot(1, 2, 1)
    sns.heatmap(np.log1p(archetype_distances[::5, ::5].T))
    plt.xlabel("RNA cell index")
    plt.ylabel("Protein cell index")
    plt.gca().invert_yaxis()

    plt.subplot(1, 2, 2)
    plt.title("minimum Archetype Distances between RNA and Protein cells")
    plt.scatter(
        np.arange(len(archetype_distances.argmin(axis=1))),
        archetype_distances.argmin(axis=1),
        s=1,
        rasterized=True,
    )
    plt.xlabel("RNA cell index")
    plt.ylabel("Protein cell index")
    plt.show()


def plot_pca_and_umap(adata_rna_subset, adata_prot_subset):
    """Plot PCA and UMAP visualizations"""
    sc.pl.pca(adata_rna_subset, color=["cell_types", "major_cell_types"])
    sc.pl.pca(adata_prot_subset, color=["cell_types", "major_cell_types"])
    sc.pl.embedding(
        adata_rna_subset, basis="X_original_umap", color=["cell_types", "major_cell_types"]
    )
    sc.pl.embedding(
        adata_prot_subset, basis="X_original_umap", color=["cell_types", "major_cell_types"]
    )


def plot_b_cells_analysis(adata_rna_subset):
    """Plot analysis for B cells"""
    adata_B_cells = adata_rna_subset[
        adata_rna_subset.obs["major_cell_types"] == adata_rna_subset.obs["major_cell_types"][0]
    ]
    sc.pp.pca(adata_B_cells)
    sc.pp.neighbors(adata_B_cells, use_rep="X_pca")
    sc.tl.umap(adata_B_cells)
    if "tissue" in adata_B_cells.obs:
        sc.pl.umap(
            adata_B_cells, color=["tissue"], title="verifying tissue does not give a major effect"
        )
    else:
        sc.pl.umap(
            adata_B_cells, color=["cell_types"], title="verifying cell types are well separated"
        )


def plot_protein_umap(adata_prot_subset):
    """Plot protein UMAP visualizations"""
    sc.pp.neighbors(adata_prot_subset, use_rep="X_pca", key_added="X_neighborhood")
    sc.tl.umap(adata_prot_subset, neighbors_key="X_neighborhood")
    adata_prot_subset.obsm["X_original_umap"] = adata_prot_subset.obsm["X_umap"]
    sc.pl.umap(
        adata_prot_subset,
        color="CN",
        title="Protein UMAP of CN vectors colored by CN label",
        neighbors_key="original_neighbors",
    )
    one_cell_type = adata_prot_subset.obs["major_cell_types"][0]
    sc.pl.umap(
        adata_prot_subset[adata_prot_subset.obs["major_cell_types"] == one_cell_type],
        color="cell_types",
        title="Protein UMAP of CN vectors colored by minor cell type label",
    )
    return one_cell_type


def plot_cell_type_distribution_single(adata_prot_subset, one_cell_type):
    """Plot cell type distribution for a single dataset"""
    sns.histplot(
        adata_prot_subset[adata_prot_subset.obs["cell_types"] == one_cell_type].obs,
        x="cell_types",
        hue="CN",
        multiple="fill",
        stat="proportion",
    )


def plot_original_data_visualizations(adata_rna_subset, adata_prot_subset):
    """Plot original data visualizations"""

    sc.pl.embedding(
        adata_rna_subset,
        color=["CN", "cell_types", "archetype_label"],
        basis="X_original_umap",
        title=[
            "Original rna data CN",
            "Original rna data cell types",
            "Original rna data archetype label",
        ],
    )

    sc.pl.embedding(
        adata_prot_subset,
        color=[
            "CN",
            "archetype_label",
            "cell_types",
        ],
        basis="X_original_umap",
        title=[
            "Original protein data CN",
            "Original protein data archetype label",
            "Original protein data cell types",
        ],
    )


# %%


def plot_latent_single(means, adata, index, color_label="CN", title=""):
    plt.figure()
    pca = PCA(n_components=3)
    means_cpu = means.detach().cpu().numpy()
    index_cpu = index.detach().cpu().numpy().flatten()
    pca.fit(means_cpu)
    rna_pca = pca.transform(means_cpu)
    plt.subplot(1, 1, 1)
    plt.scatter(
        rna_pca[:, 0],
        rna_pca[:, 1],
        c=pd.Categorical(adata[index_cpu].obs[color_label].values).codes,
        cmap="jet",
    )
    plt.title(title)
    plt.show()


# %%


def test_plot_latent_pca_both_modalities_by_celltype():
    """Test function for plot_latent_pca_both_modalities_by_celltype with synthetic data."""
    import numpy as np
    from anndata import AnnData

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate synthetic data
    n_cells = 2000
    n_cell_types = 5
    latent_dim = 20

    # Create latent representations for RNA and protein
    latent_rna = np.random.normal(0, 1, size=(n_cells, latent_dim))
    latent_prot = np.random.normal(0, 1, size=(n_cells, latent_dim))

    # Create cell type labels
    cell_type_names = [f"CellType_{i}" for i in range(n_cell_types)]
    cell_types = np.random.choice(cell_type_names, size=n_cells)

    # Create CN (neighborhood) labels
    cn_names = [f"CN_{i}" for i in range(3)]
    cn_labels = np.random.choice(cn_names, size=n_cells)

    # Create AnnData objects
    adata_rna = AnnData(X=np.random.lognormal(0, 1, size=(n_cells, 100)))
    adata_prot = AnnData(X=np.random.lognormal(0, 1, size=(n_cells, 50)))

    # Add cell type and CN information to obs
    adata_rna.obs["cell_types"] = cell_types
    adata_rna.obs["CN"] = cn_labels
    adata_prot.obs["cell_types"] = cell_types
    adata_prot.obs["CN"] = cn_labels

    # Plot with default settings
    print("Testing plot_latent_pca_both_modalities_by_celltype with default settings...")
    plot_latent_pca_both_modalities_by_celltype(
        adata_rna_subset=adata_rna,
        adata_prot_subset=adata_prot,
        latent_rna=latent_rna,
        latent_prot=latent_prot,
        use_subsample=True,
    )

    # Plot without subsampling
    print("Testing plot_latent_pca_both_modalities_by_celltype without subsampling...")
    plot_latent_pca_both_modalities_by_celltype(
        adata_rna_subset=adata_rna,
        adata_prot_subset=adata_prot,
        latent_rna=latent_rna,
        latent_prot=latent_prot,
        use_subsample=False,
    )

    # Plot with specified indices
    print("Testing plot_latent_pca_both_modalities_by_celltype with specific indices...")
    index_rna = np.random.choice(n_cells, size=500, replace=False)
    index_prot = np.random.choice(n_cells, size=500, replace=False)
    plot_latent_pca_both_modalities_by_celltype(
        adata_rna_subset=adata_rna,
        adata_prot_subset=adata_prot,
        latent_rna=latent_rna,
        latent_prot=latent_prot,
        index_rna=index_rna,
        index_prot=index_prot,
        use_subsample=True,
    )

    print("All tests completed!")


def plot_training_metrics_history(metrics_history):
    """Plot training metrics history over epochs.

    Args:
        metrics_history (list): List of dictionaries containing metrics for each epoch
        save_path (str): Path to save the plot

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Convert metrics history to DataFrame
    metrics_df = pd.DataFrame(metrics_history)
    metrics_df["epoch"] = range(len(metrics_df))

    # Create subplots based on number of metrics
    n_metrics = len(metrics_df.columns) - 1  # Subtract 1 for epoch column
    n_rows = (n_metrics + 2) // 3  # 3 plots per row, round up

    fig = plt.figure(figsize=(15, 5 * n_rows))
    for i, metric in enumerate(metrics_df.columns.drop("epoch")):
        plt.subplot(n_rows, 3, i + 1)
        sns.lineplot(data=metrics_df, x="epoch", y=metric)
        plt.title(metric)
        plt.tight_layout()

    plt.tight_layout()
    safe_mlflow_log_figure(fig, "training_metrics.png")
    plt.close()

    return fig

#!/usr/bin/env python
# %%
"""Calculate metrics on archetype vector embeddings instead of latent space."""

import os
import sys
from pathlib import Path

import anndata
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# Set up paths once
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)


# %%
# Custom function for batched cosine distance
def batched_cosine_dist(X, Y, batch_size=5000):
    """Calculate pairwise cosine distances in batches to prevent memory issues."""
    from scipy.spatial.distance import cdist

    n_x = X.shape[0]
    n_y = Y.shape[0]
    distances = np.zeros((n_x, n_y))

    for i in tqdm(range(0, n_x, batch_size), desc="Processing rows", total=n_x // batch_size + 1):
        end_i = min(i + batch_size, n_x)
        batch_X = X[i:end_i]

        for j in range(0, n_y, batch_size):
            end_j = min(j + batch_size, n_y)
            batch_Y = Y[j:end_j]

            # Use cosine distance
            batch_distances = cdist(batch_X, batch_Y, metric="cosine")
            distances[i:end_i, j:end_j] = batch_distances

        print(f"Processed {end_i}/{n_x} rows", end="\r")

    return distances


# %%
# Import custom modules
import bar_nick_utils
import CODEX_RNA_seq.metrics
from CODEX_RNA_seq.training_utils import Tee, log_memory_usage, mixing_score


# %%
# Define utility functions
def plot_archetype_umap(adata_combined, save_path=None):
    """Generate UMAP visualization of archetype vectors."""
    print("Generating UMAP visualization...")

    # Compute UMAP with cosine distance
    sc.pp.neighbors(adata_combined, use_rep="X", metric="cosine", n_neighbors=15)
    sc.tl.umap(adata_combined)

    # Plot by modality
    plt.figure(figsize=(12, 10))
    ax1 = plt.subplot(2, 2, 1)
    sc.pl.umap(adata_combined, color="modality", ax=ax1, show=False, title="Modality")

    # Plot by cell type
    ax2 = plt.subplot(2, 2, 2)
    sc.pl.umap(adata_combined, color="cell_types", ax=ax2, show=False, title="Cell Types")

    # Plot by neighborhood
    ax3 = plt.subplot(2, 2, 3)
    sc.pl.umap(adata_combined, color="CN", ax=ax3, show=False, title="Cell Neighborhood")

    # Plot by major cell type if available
    if "major_cell_types" in adata_combined.obs:
        ax4 = plt.subplot(2, 2, 4)
        sc.pl.umap(
            adata_combined, color="major_cell_types", ax=ax4, show=False, title="Major Cell Types"
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("✓ UMAP visualization completed")


def plot_archetype_heatmap(rna_adata, prot_adata, n_samples=50, save_path=None):
    """Plot heatmap of archetype vectors for a sample of cells."""
    print("Generating archetype vector heatmaps...")

    # Subsample for visualization
    if len(rna_adata) > n_samples:
        rna_sample = sc.pp.subsample(rna_adata, n_obs=n_samples, copy=True)
    else:
        rna_sample = rna_adata.copy()

    if len(prot_adata) > n_samples:
        prot_sample = sc.pp.subsample(prot_adata, n_obs=n_samples, copy=True)
    else:
        prot_sample = prot_adata.copy()

    # Plot heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # RNA heatmap
    sns.heatmap(rna_sample.X, ax=axes[0], cmap="viridis", xticklabels=False, yticklabels=False)
    axes[0].set_title(f"RNA Archetype Vectors (n={len(rna_sample)})")

    # Protein heatmap
    sns.heatmap(prot_sample.X, ax=axes[1], cmap="viridis", xticklabels=False, yticklabels=False)
    axes[1].set_title(f"Protein Archetype Vectors (n={len(prot_sample)})")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("✓ Archetype heatmaps generated")


def plot_matching_accuracy_by_cell_type(rna_adata, prot_adata, prot_matches_in_rna, save_path=None):
    """Plot matching accuracy by cell type."""
    print("Generating cell type matching accuracy plot...")

    # Calculate matching by cell type
    matches = (
        rna_adata.obs["cell_types"].values[prot_matches_in_rna]
        == prot_adata.obs["cell_types"].values
    )

    # Group by cell type
    prot_cell_types = prot_adata.obs["cell_types"].values
    unique_cell_types = np.unique(prot_cell_types)

    accuracies = []
    for cell_type in unique_cell_types:
        cell_type_indices = prot_cell_types == cell_type
        type_matches = matches[cell_type_indices]
        accuracy = type_matches.sum() / len(type_matches) if len(type_matches) > 0 else 0
        accuracies.append((cell_type, accuracy, np.sum(cell_type_indices)))

    # Sort by accuracy
    accuracies.sort(key=lambda x: x[1], reverse=True)

    # Create DataFrame for plotting
    df = pd.DataFrame(accuracies, columns=["Cell Type", "Accuracy", "Count"])

    # Plot
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x="Cell Type", y="Accuracy", data=df, palette="viridis")

    # Add count labels
    for i, row in enumerate(df.itertuples()):
        ax.text(
            i, 0.05, f"n={row.Count}", ha="center", rotation=90, color="white", fontweight="bold"
        )

    plt.xticks(rotation=90)
    plt.title("Matching Accuracy by Cell Type")
    plt.ylim(0, 1.0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("✓ Cell type matching accuracy plot generated")


def plot_distance_comparison(matching_distances, rand_matching_distances, save_path=None):
    """Plot comparison of actual vs random matching distances."""
    print("Generating distance comparison plot...")

    plt.figure(figsize=(12, 6))

    # Plot distance distributions
    plt.subplot(1, 2, 1)
    sns.histplot(matching_distances, kde=True, color="blue", label="Actual matches")
    sns.histplot(rand_matching_distances, kde=True, color="red", label="Random matches")
    plt.title("Distribution of Matching Distances")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.legend()

    # Plot cumulative distributions
    plt.subplot(1, 2, 2)
    sns.ecdfplot(matching_distances, label="Actual matches")
    sns.ecdfplot(rand_matching_distances, label="Random matches")
    plt.title("Cumulative Distribution of Distances")
    plt.xlabel("Distance")
    plt.ylabel("Proportion")
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("✓ Distance comparison plot generated")


def create_tsne_visualization(rna_archetype, prot_archetype, save_path=None):
    """Create t-SNE visualization of RNA and protein archetype vectors."""
    print("Generating t-SNE visualization...")

    # Subsample if very large
    max_cells = 2000
    if len(rna_archetype) > max_cells:
        rna_sample = sc.pp.subsample(rna_archetype, n_obs=max_cells, copy=True)
    else:
        rna_sample = rna_archetype.copy()

    if len(prot_archetype) > max_cells:
        prot_sample = sc.pp.subsample(prot_archetype, n_obs=max_cells, copy=True)
    else:
        prot_sample = prot_archetype.copy()

    # Combine data
    combined_data = np.vstack([rna_sample.X, prot_sample.X])

    # Create labels for modality and cell type
    modality_labels = np.array(["RNA"] * len(rna_sample) + ["Protein"] * len(prot_sample))
    cell_type_labels = np.concatenate(
        [rna_sample.obs["cell_types"].values, prot_sample.obs["cell_types"].values]
    )

    # Run t-SNE with cosine distance
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, metric="cosine")
    embedding = tsne.fit_transform(combined_data)

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # By modality
    scatter1 = axes[0].scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[0 if m == "RNA" else 1 for m in modality_labels],
        cmap="viridis",
        alpha=0.7,
        s=5,
    )
    axes[0].set_title("t-SNE by Modality (Cosine Distance)")
    legend1 = axes[0].legend(
        handles=scatter1.legend_elements()[0], labels=["RNA", "Protein"], loc="upper right"
    )
    axes[0].add_artist(legend1)

    # By cell type
    unique_types = np.unique(cell_type_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))
    cell_type_colors = {t: colors[i] for i, t in enumerate(unique_types)}

    for cell_type in unique_types:
        mask = cell_type_labels == cell_type
        axes[1].scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            color=cell_type_colors[cell_type],
            label=cell_type,
            alpha=0.7,
            s=5,
        )

    axes[1].set_title("t-SNE by Cell Type (Cosine Distance)")
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("✓ t-SNE visualization completed")


# %%
def match_cells_using_archetypes(rna_adata, prot_adata):
    """Match cells between modalities using archetype vectors with cosine distance."""
    # Since we already converted the objects to have archetype vectors as X,
    # we can directly use their X matrices

    # Calculate pairwise distances using cosine distance
    print("Calculating pairwise cosine distances between archetype vectors...")
    latent_distances = batched_cosine_dist(rna_adata.X, prot_adata.X)

    # Find matches
    prot_matches_in_rna = np.argmin(latent_distances, axis=0)
    matching_distances = np.min(latent_distances, axis=0)

    # Generate random matches for comparison
    rand_indices = np.random.permutation(len(rna_adata))
    rand_latent_distances = latent_distances[rand_indices, :]
    rand_prot_matches_in_rna = np.argmin(rand_latent_distances, axis=0)
    rand_matching_distances = np.min(rand_latent_distances, axis=0)

    return {
        "prot_matches_in_rna": prot_matches_in_rna,
        "matching_distances": matching_distances,
        "rand_prot_matches_in_rna": rand_prot_matches_in_rna,
        "rand_matching_distances": rand_matching_distances,
    }


# %%
def calculate_post_training_metrics_on_archetypes(rna_adata, protein_adata, prot_matches_in_rna):
    """Calculate various metrics for model evaluation using archetype vectors."""
    # Calculate NMI scores
    nmi_cell_types_cn_rna = adjusted_mutual_info_score(
        rna_adata.obs["cell_types"], rna_adata.obs["CN"]
    )
    nmi_cell_types_cn_prot = adjusted_mutual_info_score(
        protein_adata.obs["cell_types"], protein_adata.obs["CN"]
    )
    nmi_cell_types_modalities = adjusted_mutual_info_score(
        rna_adata.obs["cell_types"].values[prot_matches_in_rna],
        protein_adata.obs["cell_types"].values,
    )

    # Calculate accuracy
    matches = (
        rna_adata.obs["cell_types"].values[prot_matches_in_rna]
        == protein_adata.obs["cell_types"].values
    )
    accuracy = matches.sum() / len(matches)

    # Calculate mixing score with cosine distance
    # Use X directly since it now contains the archetype vectors
    mixing_result = mixing_score(
        rna_adata.X,
        protein_adata.X,
        rna_adata,
        protein_adata,
        index_rna=range(len(rna_adata)),
        index_prot=range(len(protein_adata)),
        plot_flag=True,
        # metric='cosine'  # Use cosine distance for mixing score
    )

    return {
        "nmi_cell_types_cn_rna_archetypes": nmi_cell_types_cn_rna,
        "nmi_cell_types_cn_prot_archetypes": nmi_cell_types_cn_prot,
        "nmi_cell_types_modalities_archetypes": nmi_cell_types_modalities,
        "cell_type_matching_accuracy_archetypes": accuracy,
        "mixing_score_ilisi_archetypes": mixing_result["iLISI"],
        "mixing_score_clisi_archetypes": mixing_result["cLISI"],
    }


# %%
def process_archetype_spaces(rna_adata, prot_adata):
    """Process archetype spaces from RNA and protein data."""
    print("Processing archetype spaces...")

    # Since we now have archetype vectors as X, we can use the objects directly
    rna_archetype = rna_adata.copy()
    prot_archetype = prot_adata.copy()

    # Combine for visualization
    combined_archetype = anndata.concat(
        [rna_archetype, prot_archetype],
        join="outer",
        label="modality",
        keys=["RNA", "Protein"],
    )

    print("✓ Archetype spaces processed")

    return rna_archetype, prot_archetype, combined_archetype


# %%
def calculate_metrics_for_archetypes(rna_adata, prot_adata, prefix="", subsample_size=None):
    """Calculate metrics using archetype vectors instead of latent space.

    Args:
        rna_adata: RNA AnnData object
        prot_adata: Protein AnnData object
        prefix: Prefix for metric names (e.g., "train_" or "val_")
        subsample_size: If not None, subsample the data to this size
    """
    print(f"Calculating {prefix}metrics on archetype vectors...")

    # Subsample if requested
    if subsample_size is not None:
        rna_adata = sc.pp.subsample(rna_adata, n_obs=subsample_size, copy=True)
        prot_adata = sc.pp.subsample(prot_adata, n_obs=subsample_size, copy=True)
        print(f"Subsampled to {subsample_size} cells")

    # Since we already have archetype vectors as X, we can directly use the objects
    rna_archetype_adata = rna_adata
    prot_archetype_adata = prot_adata

    # Calculate matching accuracy
    # Check if we can modify the metrics functions to use cosine
    # For now, we use the existing functions which likely use Euclidean
    accuracy = CODEX_RNA_seq.metrics.matching_accuracy(rna_archetype_adata, prot_archetype_adata)
    print(f"✓ {prefix}matching accuracy calculated")

    # Calculate silhouette F1
    silhouette_f1 = CODEX_RNA_seq.metrics.compute_silhouette_f1(
        rna_archetype_adata, prot_archetype_adata
    )
    print(f"✓ {prefix}silhouette F1 calculated")

    # Calculate ARI F1
    combined_archetype = anndata.concat(
        [rna_archetype_adata, prot_archetype_adata],
        join="outer",
        label="modality",
        keys=["RNA", "Protein"],
    )

    # Skip PCA since archetype vectors are already low-dimensional (only ~7 dimensions)
    # sc.pp.pca(combined_archetype)

    # Use cosine distance directly on archetype vectors for neighbors
    sc.pp.neighbors(combined_archetype, n_neighbors=10, metric="cosine", use_rep="X")
    ari_f1 = CODEX_RNA_seq.metrics.compute_ari_f1(combined_archetype)
    print(f"✓ {prefix}ARI F1 calculated")

    return {
        f"{prefix}cell_type_matching_accuracy_archetypes": accuracy,
        f"{prefix}silhouette_f1_score_archetypes": silhouette_f1.mean(),
        f"{prefix}ari_f1_score_archetypes": ari_f1,
    }


# %%
# Main execution block
if __name__ == "__main__":
    # Create log directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    log_file = open(
        f"logs/archetype_metrics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log", "w"
    )

    # Redirect stdout to both console and log file
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)

    print(f"Starting calculation of metrics on archetype vectors at {pd.Timestamp.now()}")

    # %%
    # Load data
    save_dir = Path("CODEX_RNA_seq/data/processed_data").absolute()
    log_memory_usage("Before loading data: ")

    # Find latest RNA and protein files
    rna_file = bar_nick_utils.get_latest_file(save_dir, "adata_rna_subset_prepared_for_training")
    prot_file = bar_nick_utils.get_latest_file(save_dir, "adata_prot_subset_prepared_for_training")
    if not rna_file or not prot_file:
        print("Error: Could not find trained data files.")
        sys.exit(1)

    print(f"Using RNA file: {os.path.basename(rna_file)}")
    print(f"Using Protein file: {os.path.basename(prot_file)}")

    # %%
    # Load data
    print("\nLoading data...")
    adata_rna = sc.read_h5ad(rna_file)
    adata_prot = sc.read_h5ad(prot_file)
    print("✓ Data loaded")
    log_memory_usage("After loading data: ")

    # Verify that archetype vectors exist
    if "archetype_vec" not in adata_rna.obsm or "archetype_vec" not in adata_prot.obsm:
        print("Error: Archetype vectors not found in data.")
        sys.exit(1)

    print(f"RNA dataset shape: {adata_rna.shape}")
    print(f"Protein dataset shape: {adata_prot.shape}")

    # %%
    # Convert to archetype-based AnnData objects
    print("\nConverting to archetype-based AnnData objects...")
    # Create new AnnData objects with archetype vectors as X
    adata_rna_arch = anndata.AnnData(X=adata_rna.obsm["archetype_vec"])
    adata_prot_arch = anndata.AnnData(X=adata_prot.obsm["archetype_vec"])

    # Copy observations and other attributes
    adata_rna_arch.obs = adata_rna.obs.copy()
    adata_prot_arch.obs = adata_prot.obs.copy()

    # Normalize RNA archetype vectors
    rna_scaler = MinMaxScaler()
    adata_rna_arch.X = rna_scaler.fit_transform(adata_rna_arch.X)

    # Normalize protein archetype vectors
    prot_scaler = MinMaxScaler()
    adata_prot_arch.X = prot_scaler.fit_transform(adata_prot_arch.X)

    # Copy uns, obsm (except archetype_vec), and obsp if they exist
    if hasattr(adata_rna, "uns"):
        adata_rna_arch.uns = adata_rna.uns.copy()
    if hasattr(adata_prot, "uns"):
        adata_prot_arch.uns = adata_prot.uns.copy()

    for key in adata_rna.obsm.keys():
        if key != "archetype_vec":
            adata_rna_arch.obsm[key] = adata_rna.obsm[key].copy()

    for key in adata_prot.obsm.keys():
        if key != "archetype_vec":
            adata_prot_arch.obsm[key] = adata_prot.obsm[key].copy()

    if hasattr(adata_rna, "obsp") and len(adata_rna.obsp) > 0:
        for key in adata_rna.obsp.keys():
            adata_rna_arch.obsp[key] = adata_rna.obsp[key].copy()

    if hasattr(adata_prot, "obsp") and len(adata_prot.obsp) > 0:
        for key in adata_prot.obsp.keys():
            adata_prot_arch.obsp[key] = adata_prot.obsp[key].copy()

    # Replace original adata with archetype-based ones
    adata_rna = adata_rna_arch
    adata_prot = adata_prot_arch

    print(f"New RNA archetype dataset shape: {adata_rna.shape}")
    print(f"New Protein archetype dataset shape: {adata_prot.shape}")
    print("✓ Converted to archetype-based AnnData objects")
    log_memory_usage("After archetype conversion: ")

    # %%
    # Normalize archetype vectors to [0,1] range
    print("\nNormalizing archetype vectors to [0,1] range...")

    # Get the dimensions of both datasets
    n_rna_dims = adata_rna.X.shape[1]
    n_prot_dims = adata_prot.X.shape[1]

    # Check if dimensions match
    if n_rna_dims != n_prot_dims:
        print(
            f"Warning: RNA and protein archetype vectors have different dimensions ({n_rna_dims} vs {n_prot_dims})"
        )

    # Verify normalization worked
    print(
        f"RNA min values: {adata_rna.X.min(axis=0).min():.4f}, max: {adata_rna.X.max(axis=0).max():.4f}"
    )
    print(
        f"Protein min values: {adata_prot.X.min(axis=0).min():.4f}, max: {adata_prot.X.max(axis=0).max():.4f}"
    )
    print("✓ Archetype vectors normalized")

    # Create a heatmap to visualize the normalized archetype vectors
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sample_size = min(50, len(adata_rna), len(adata_prot))

    # RNA heatmap
    sns.heatmap(
        adata_rna.X[:sample_size],
        ax=axes[0],
        cmap="viridis",
        vmin=0,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
    )
    axes[0].set_title(f"Normalized RNA Archetype Vectors (n={sample_size})")

    # Protein heatmap
    sns.heatmap(
        adata_prot.X[:sample_size],
        ax=axes[1],
        cmap="viridis",
        vmin=0,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
    )
    axes[1].set_title(f"Normalized Protein Archetype Vectors (n={sample_size})")

    plt.tight_layout()
    plt.show()

    # %%
    # Subsample for faster execution
    max_cells = 5000
    print(f"\nSubsampling data to max {max_cells} cells per modality for faster execution...")
    if len(adata_rna) > max_cells:
        adata_rna = sc.pp.subsample(adata_rna, n_obs=max_cells, copy=True)
    if len(adata_prot) > max_cells:
        adata_prot = sc.pp.subsample(adata_prot, n_obs=max_cells, copy=True)
    print(f"Subsampled RNA dataset shape: {adata_rna.shape}")
    print(f"Subsampled protein dataset shape: {adata_prot.shape}")
    log_memory_usage("After subsampling: ")

    # %%
    # Process archetype spaces
    rna_archetype, prot_archetype, combined_archetype = process_archetype_spaces(
        adata_rna, adata_prot
    )

    # %%
    # Create visualization of archetype vectors
    # Create a plots directory if it doesn't exist
    plots_dir = Path("CODEX_RNA_seq/plots").absolute()
    plots_dir.mkdir(exist_ok=True)
    # plot pca
    sc.pl.pca(combined_archetype, color="cell_types", show=False)
    plt.savefig(plots_dir / "pca.png", dpi=300, bbox_inches="tight")
    plt.show()
    # plot pca for each modality
    sc.pl.pca(rna_archetype, color="cell_types", show=False)
    plt.savefig(plots_dir / "pca_rna.png", dpi=300, bbox_inches="tight")
    plt.show()
    sc.pp.pca(prot_archetype)
    sc.pl.pca(prot_archetype, color="cell_types", show=False)
    plt.savefig(plots_dir / "pca_prot.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot archetype heatmaps
    plot_archetype_heatmap(
        rna_archetype, prot_archetype, n_samples=50, save_path=plots_dir / "archetype_heatmaps.png"
    )

    # %%
    # Create t-SNE visualization
    create_tsne_visualization(
        rna_archetype, prot_archetype, save_path=plots_dir / "archetype_tsne.png"
    )

    # %%
    # Match cells and calculate distances using archetype vectors
    matching_results = match_cells_using_archetypes(adata_rna, adata_prot)

    # %%
    # Plot distance comparison
    plot_distance_comparison(
        matching_results["matching_distances"],
        matching_results["rand_matching_distances"],
        save_path=plots_dir / "distance_comparison.png",
    )

    # %%
    # Calculate metrics
    metrics = calculate_post_training_metrics_on_archetypes(
        adata_rna, adata_prot, matching_results["prot_matches_in_rna"]
    )

    # %%
    # Plot matching accuracy by cell type
    plot_matching_accuracy_by_cell_type(
        adata_rna,
        adata_prot,
        matching_results["prot_matches_in_rna"],
        save_path=plots_dir / "matching_accuracy_by_cell_type.png",
    )

    # %%
    # Calculate additional metrics
    additional_metrics = calculate_metrics_for_archetypes(adata_rna, adata_prot)
    metrics.update(additional_metrics)

    # %%
    # Generate UMAP visualization
    plot_archetype_umap(combined_archetype, save_path=plots_dir / "archetype_umap.png")

    # %%
    # Print results
    print("\nMetrics on Archetype Vectors:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # Create a summary visualization of metrics
    plt.figure(figsize=(12, 6))
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    plt.barh(metric_names, metric_values, color="skyblue")
    plt.xlabel("Value")
    plt.title("Archetype Vector Metrics Summary")
    plt.tight_layout()
    plt.savefig(plots_dir / "metrics_summary.png", dpi=300, bbox_inches="tight")
    plt.show()

    # %%
    # Calculate metrics with MLflow if available
    try:
        mlflow.log_metrics({k: round(v, 4) for k, v in metrics.items()})
        # Log plots as artifacts
        for plot_file in plots_dir.glob("*.png"):
            mlflow.log_artifact(str(plot_file))
        print("✓ Metrics and plots logged to MLflow")
    except Exception as e:
        print(f"Warning: Could not log to MLflow: {e}")

    # %%
    # Clean up: restore original stdout and close log file
    print(f"\nArchetype metrics calculation completed at {pd.Timestamp.now()}")
    sys.stdout = original_stdout
    log_file.close()


# %%

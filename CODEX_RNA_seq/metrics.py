# %% Metrics Functions
# This module contains functions for calculating various metrics.


# %% Imports and Setup
import importlib
import os
import sys
from datetime import datetime

import anndata
import numpy as np
import scanpy as sc
from anndata import AnnData, concat
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, f1_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import LabelEncoder

# Add repository root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set working directory to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plotting_functions as pf

import bar_nick_utils

importlib.reload(pf)
importlib.reload(bar_nick_utils)


def silhouette_score_calc(adata_rna, adata_prot):
    embedding_key = "X_scVI"
    assert (
        embedding_key in adata_rna.obsm
    ), f"No embeddings found in adata_rna.obsm['{embedding_key}']."
    assert (
        embedding_key in adata_prot.obsm
    ), f"No embeddings found in adata_prot.obsm['{embedding_key}']."

    rna_latent = AnnData(adata_rna.obsm[embedding_key].copy())
    prot_latent = AnnData(adata_prot.obsm[embedding_key].copy())
    rna_latent.obs = adata_rna.obs.copy()
    prot_latent.obs = adata_prot.obs.copy()

    combined_latent = concat(
        [rna_latent, prot_latent], join="outer", label="modality", keys=["RNA", "Protein"]
    )

    silhouette_avg = silhouette_score(combined_latent.X, combined_latent.obs["cell_types"])
    # print(f"Silhouette Score (RNA+Protein): {silhouette_avg}")
    return silhouette_avg


# returns list of indices of proteins that are most aligned with adata_rna.
# for example, first item in return array (adata_prot) is closest match to adata_rna
def calc_dist(adata_rna, adata_prot):
    embedding_key = "X_scVI"
    assert (
        embedding_key in adata_rna.obsm
    ), f"No embeddings found in adata_rna.obsm['{embedding_key}']."
    assert (
        embedding_key in adata_prot.obsm
    ), f"No embeddings found in adata_prot.obsm['{embedding_key}']."

    distances = cdist(adata_rna.obsm["X_scVI"], adata_prot.obsm["X_scVI"], metric="euclidean")
    nearest_indices = np.argmin(distances, axis=1)  # protein index
    nn_celltypes_prot = adata_prot.obs["cell_types"][nearest_indices]
    return nn_celltypes_prot


# F1
def f1_score_calc(adata_rna, adata_prot):
    return f1_score(adata_rna.obs["cell_types"], calc_dist(adata_rna, adata_prot), average="macro")


# ARI
def ari_score_calc(adata_rna, adata_prot):
    return adjusted_rand_score(adata_rna.obs["cell_types"], calc_dist(adata_rna, adata_prot))


# matching_accuracy 1-1
def matching_accuracy(adata_rna, adata_prot):
    correct_matches = 0
    nn_celltypes_prot = calc_dist(adata_rna, adata_prot)
    for index, cell_type in enumerate(adata_rna.obs["cell_types"]):
        if cell_type == nn_celltypes_prot[index]:
            correct_matches += 1
    accuracy = correct_matches / len(nn_celltypes_prot)
    return accuracy


if __name__ == "__main__":
    import os
    from datetime import datetime
    from pathlib import Path

    import scanpy as sc

    # Set working directory to project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Path to trained data directory
    data_dir = Path("CODEX_RNA_seq/data/trained_data").absolute()

    # Find latest RNA and protein files
    rna_file = bar_nick_utils.get_latest_file(data_dir, "rna_vae_trained")
    prot_file = bar_nick_utils.get_latest_file(data_dir, "protein_vae_trained")

    if not rna_file or not prot_file:
        print("Error: Could not find trained data files.")
        sys.exit(1)

    print(f"Using RNA file: {os.path.basename(rna_file)}")
    print(f"Using Protein file: {os.path.basename(prot_file)}")

    # Load data
    print("\nLoading data...")
    adata_rna = sc.read_h5ad(rna_file)
    adata_prot = sc.read_h5ad(prot_file)
    print("✓ Data loaded")

    # Calculate and print all metrics
    print("\nCalculating metrics...")
    silhouette = silhouette_score_calc(adata_rna, adata_prot)
    f1 = f1_score_calc(adata_rna, adata_prot)
    ari = ari_score_calc(adata_rna, adata_prot)
    accuracy = matching_accuracy(adata_rna, adata_prot)

    # Calculate advanced metrics if available
    try:
        silhouette_f1 = compute_silhouette_f1(adata_rna, adata_prot)
        ari_f1 = compute_ari_f1(adata_rna, adata_prot)
        has_advanced_metrics = True
    except Exception as e:
        print(f"Warning: Could not calculate advanced metrics: {e}")
        has_advanced_metrics = False

    # Print results
    print(f"\nMetrics Results:")
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"ARI Score: {ari:.3f}")
    print(f"Matching Accuracy: {accuracy:.3f}")

    if has_advanced_metrics:
        print(f"Silhouette F1 Score: {silhouette_f1.mean():.3f}")
        print(f"ARI F1 Score: {ari_f1:.3f}")

    # Save results to log file
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = Path("CODEX_RNA_seq/logs").absolute()
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir / f"metrics_log_{timestamp}.txt"

    with open(log_file, "w") as f:
        f.write(f"Metrics calculated on: {timestamp}\n")
        f.write(f"RNA file: {os.path.basename(rna_file)}\n")
        f.write(f"Protein file: {os.path.basename(prot_file)}\n\n")
        f.write(f"Silhouette Score: {silhouette:.3f}\n")
        f.write(f"F1 Score: {f1:.3f}\n")
        f.write(f"ARI Score: {ari:.3f}\n")
        f.write(f"Matching Accuracy: {accuracy:.3f}\n")

        if has_advanced_metrics:
            f.write(f"Silhouette F1 Score: {silhouette_f1.mean():.3f}\n")
            f.write(f"ARI F1 Score: {ari_f1:.3f}\n")

    print(f"\nResults saved to: {log_file}")
    print("\nMetrics calculation completed!")


def leiden_from_embeddings(embeddings, resolution=1.0, neighbors=15):
    """
    Run Leiden clustering on given embeddings.
    Parameters:
        embeddings: np.ndarray of shape (n_cells, n_features)
        resolution: float, resolution parameter for Leiden
        neighbors: int, number of neighbors for graph construction
    Returns:
        cluster_labels: np.ndarray of Leiden cluster assignments
    """
    # Create an AnnData object from embeddings
    adata = anndata.AnnData(X=embeddings)
    # Compute neighborhood graph
    sc.pp.neighbors(adata, n_neighbors=neighbors, use_rep="X")
    # Run Leiden clustering
    sc.tl.leiden(adata, resolution=resolution)
    # Extract cluster labels
    cluster_labels = adata.obs["leiden"].astype(int).values
    return cluster_labels


def normalize_silhouette(silhouette_vals):
    """Normalize silhouette scores from [-1, 1] to [0, 1]."""
    return (np.mean(silhouette_vals) + 1) / 2


def compute_silhouette_f1(adata_rna, adata_prot):
    """
    Compute the Silhouette F1 score.

    embeddings: np.ndarray, shape (n_samples, n_features)
    celltype_labels: list or array of ground-truth biological labels
    modality_labels: list or array of modality labels (e.g., RNA, ATAC)
    """

    # protein embeddings
    prot_embeddings = adata_prot.obsm["X_scVI"]
    # rna embeddings
    rna_embeddings = adata_rna.obsm["X_scVI"]
    embeddings = np.concatenate([rna_embeddings, prot_embeddings], axis=0)
    celltype_labels = np.concatenate(
        [adata_rna.obs["major_cell_types"], adata_prot.obs["major_cell_types"]], axis=0
    )
    modality_labels = np.concatenate(
        [adata_rna.obs["Data_Type"], adata_prot.obs["Data_Type"]], axis=0
    )

    le_ct = LabelEncoder()
    le_mod = LabelEncoder()
    ct = le_ct.fit_transform(celltype_labels)
    mod = le_mod.fit_transform(modality_labels)

    slt_clust = normalize_silhouette(silhouette_samples(embeddings, ct))
    slt_mod_raw = silhouette_samples(embeddings, mod)
    slt_mod = 1 - normalize_silhouette(slt_mod_raw)  # We want mixing, so invert

    slt_f1 = (
        2 * (slt_clust * slt_mod) / (slt_clust + slt_mod + 1e-8)
    )  # just so we don't divide by zero
    return slt_f1


def compute_ari_f1(adata_rna, adata_prot):
    """
    Compute the ARI F1 score.

    cluster_labels: clustering result (e.g. from k-means or Leiden)
    celltype_labels: ground-truth biological labels
    modality_labels: original modality labels
    """
    prot_embeddings = adata_prot.obsm["X_scVI"]
    # rna embeddings
    rna_embeddings = adata_rna.obsm["X_scVI"]
    embeddings = np.concatenate([rna_embeddings, prot_embeddings], axis=0)
    celltype_labels = np.concatenate(
        [adata_rna.obs["major_cell_types"], adata_prot.obs["major_cell_types"]], axis=0
    )
    modality_labels = np.concatenate(
        [adata_rna.obs["Data_Type"], adata_prot.obs["Data_Type"]], axis=0
    )

    cluster_labels = leiden_from_embeddings(embeddings)

    le_ct = LabelEncoder()
    le_mod = LabelEncoder()
    ct = le_ct.fit_transform(celltype_labels)
    mod = le_mod.fit_transform(modality_labels)
    clust = LabelEncoder().fit_transform(cluster_labels)

    ari_clust = adjusted_rand_score(ct, clust)
    ari_mod = 1 - adjusted_rand_score(mod, clust)  # invert for mixing

    ari_f1 = 2 * (ari_clust * ari_mod) / (ari_clust + ari_mod + 1e-8)

    return ari_f1

# %% Metrics Functions
# This module contains functions for calculating various metrics.


# %% Imports and Setup
import importlib
import os
import sys


import numpy as np
import scanpy as sc
from anndata import AnnData, concat
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, f1_score, silhouette_score
import numpy as np
from sklearn.metrics import silhouette_samples, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
import anndata

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
   from pathlib import Path
   import scanpy as sc

   # Set working directory to project root
   os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


   # Load trained data
   save_dir = Path("CODEX_RNA_seq/data/trained_data").absolute()
   adata_rna = sc.read_h5ad(save_dir / "rna_vae_trained.h5ad")
   adata_prot = sc.read_h5ad(save_dir / "protein_vae_trained.h5ad")


   # Calculate and print all metrics
   print("\nCalculating metrics...")
   print(f"Silhouette Score: {silhouette_score_calc(adata_rna, adata_prot):.3f}")
   print(f"F1 Score: {f1_score_calc(adata_rna, adata_prot):.3f}")
   print(f"ARI Score: {ari_score_calc(adata_rna, adata_prot):.3f}")
   print(f"Matching Accuracy: {matching_accuracy(adata_rna, adata_prot):.3f}")


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
   sc.pp.neighbors(adata, n_neighbors=neighbors, use_rep='X')
   # Run Leiden clustering
   sc.tl.leiden(adata, resolution=resolution)
   # Extract cluster labels
   cluster_labels = adata.obs['leiden'].astype(int).values
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
   prot_embeddings = adata_prot.obsm['X_scVI']
   # rna embeddings
   rna_embeddings = adata_rna.obsm['X_scVI']
   embeddings = np.concatenate([rna_embeddings, prot_embeddings], axis=0)
   celltype_labels = np.concatenate([adata_rna.obs['major_cell_types'], adata_prot.obs['major_cell_types']], axis =0)
   modality_labels = np.concatenate([adata_rna.obs['Data_Type'], adata_prot.obs['Data_Type']], axis = 0)

   le_ct = LabelEncoder()
   le_mod = LabelEncoder()
   ct = le_ct.fit_transform(celltype_labels)
   mod = le_mod.fit_transform(modality_labels)

   slt_clust = normalize_silhouette(silhouette_samples(embeddings, ct))
   slt_mod_raw = silhouette_samples(embeddings, mod)
   slt_mod = 1 - normalize_silhouette(slt_mod_raw)  # We want mixing, so invert

   slt_f1 = 2 * (slt_clust * slt_mod) / (slt_clust + slt_mod + 1e-8) # just so we don't divide by zero
   return slt_f1, slt_clust, slt_mod


def compute_ari_f1(adata_rna, adata_prot):
   """
   Compute the ARI F1 score.
  
   cluster_labels: clustering result (e.g. from k-means or Leiden)
   celltype_labels: ground-truth biological labels
   modality_labels: original modality labels
   """
   prot_embeddings = adata_prot.obsm['X_scVI']
   # rna embeddings
   rna_embeddings = adata_rna.obsm['X_scVI']
   embeddings = np.concatenate([rna_embeddings, prot_embeddings], axis=0)
   celltype_labels = np.concatenate([adata_rna.obs['major_cell_types'], adata_prot.obs['major_cell_types']], axis =0)
   modality_labels = np.concatenate([adata_rna.obs['Data_Type'], adata_prot.obs['Data_Type']], axis = 0)
  
   cluster_labels = leiden_from_embeddings(embeddings)


   le_ct = LabelEncoder()
   le_mod = LabelEncoder()
   ct = le_ct.fit_transform(celltype_labels)
   mod = le_mod.fit_transform(modality_labels)
   clust = LabelEncoder().fit_transform(cluster_labels)


   ari_clust = adjusted_rand_score(ct, clust)
   ari_mod = 1 - adjusted_rand_score(mod, clust)  # invert for mixing


   ari_f1 = 2 * (ari_clust * ari_mod) / (ari_clust + ari_mod + 1e-8)
   return ari_f1, ari_clust, ari_mod

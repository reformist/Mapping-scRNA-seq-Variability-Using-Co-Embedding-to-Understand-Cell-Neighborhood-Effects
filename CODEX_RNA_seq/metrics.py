import scanpy as sc
import pandas as pd
from anndata import AnnData
import anndata as ad
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist 
import numpy as np
from sklearn.metrics import f1_score 
from sklearn.metrics import adjusted_rand_score

def silhouette_score_calc(adata_rna, adata_prot):
    embedding_key = "X_scVI" 
    assert embedding_key in adata_rna.obsm, f"No embeddings found in adata_rna.obsm['{embedding_key}']."
    assert embedding_key in adata_prot.obsm, f"No embeddings found in adata_prot.obsm['{embedding_key}']."

    rna_latent = AnnData(adata_rna.obsm[embedding_key].copy())
    prot_latent = AnnData(adata_prot.obsm[embedding_key].copy())
    rna_latent.obs = adata_rna.obs.copy()
    prot_latent.obs = adata_prot.obs.copy()

    combined_latent = ad.concat([rna_latent, prot_latent], join='outer', label='modality', keys=['RNA', 'Protein'])

    silhouette_avg = silhouette_score(combined_latent.X, combined_latent.obs["cell_types"])
    # print(f"Silhouette Score (RNA+Protein): {silhouette_avg}")
    return silhouette_avg


# returns list of indices of proteins that are most aligned with adata_rna. 
# for example, first item in return array (adata_prot) is closest match to adata_rna 
def calc_dist(adata_rna, adata_prot):
    embedding_key = "X_scVI" 
    assert embedding_key in adata_rna.obsm, f"No embeddings found in adata_rna.obsm['{embedding_key}']."
    assert embedding_key in adata_prot.obsm, f"No embeddings found in adata_prot.obsm['{embedding_key}']."

    distances = cdist(adata_rna.obsm['X_scVI'], adata_prot.obsm['X_scVI'], metric='euclidean')
    nearest_indices = np.argmin(distances, axis=1) # protein index
    nn_celltypes_prot = adata_prot.obs['cell_types'][nearest_indices]
    return nn_celltypes_prot

#F1
def f1_score_calc(adata_rna, adata_prot):
    return f1_score(adata_rna.obs['cell_types'], calc_dist(adata_rna, adata_prot),average="macro")

#ARI
def ari_score_calc(adata_rna, adata_prot):
    return adjusted_rand_score(adata_rna.obs['cell_types'], calc_dist(adata_rna, adata_prot))

# matching_accuracy 1-1
def matching_accuracy(adata_rna, adata_prot):
    correct_matches = 0
    nn_celltypes_prot = calc_dist(adata_rna, adata_prot)
    for index,cell_type in enumerate(adata_rna.obs['cell_types']):
        if cell_type == nn_celltypes_prot[index]:
            correct_matches+=1
    accuracy = correct_matches / len(nn_celltypes_prot)
    return accuracy
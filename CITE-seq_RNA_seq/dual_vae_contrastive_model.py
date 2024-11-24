from functools import partial
import tempfile
from collections import Counter

import anndata as ad
import matplotlib.pyplot as plt
import mudata as md
from anndata import AnnData
# import muon
import scanpy as sc
import scvi
import seaborn as sns
import torch
# ignore warnings
import warnings
import pandas as pd
import numpy    as np
import os
import tempfile
import torch.functional as F
from orbax.checkpoint.metadata.tree_test import Custom
from scipy.sparse import issparse
from scvi.model import SCVI
from scvi.train import TrainingPlan
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
import scvi
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# set pandas display options
pd.set_option("display.max_columns", 10)
pd.set_option("display.max_rows", 10)

warnings.filterwarnings("ignore")
# limit show df size
pd.options.display.max_rows = 10
pd.options.display.max_columns = 10
np.set_printoptions(threshold=100)
np.random.seed(0)
torch.random.manual_seed(0)
plot_flag = False
device = "cuda:0" if torch.cuda.is_available() else "cpu"
adata_rna_subset = sc.read('/home/barroz/projects/Mapping-scRNA-seq-Variability-Using-Co-Embedding-to-Understand-Cell-Neighborhood-Effects/CITE-seq_RNA_seq/adata_rna_subset.hd5ad.h5ad')
adata_prot_subset = sc.read('/home/barroz/projects/Mapping-scRNA-seq-Variability-Using-Co-Embedding-to-Understand-Cell-Neighborhood-Effects/CITE-seq_RNA_seq/adata_prot_subset.hd5ad.h5ad')
# take subset of data
adata_rna_subset = adata_rna_subset[adata_rna_subset.obs['major_cell_types'].isin([ 'B cells'])].copy()
adata_prot_subset = adata_prot_subset[adata_prot_subset.obs['major_cell_types'].isin([ 'B cells'])].copy()


sc.pp.neighbors(adata_prot_subset,key_added='original_neighbors')
sc.pp.neighbors(adata_rna_subset,key_added='original_neighbors')
sc.tl.umap(adata_prot_subset,neighbors_key='original_neighbors')
sc.tl.umap(adata_rna_subset,neighbors_key='original_neighbors')

sc.pl.umap(adata_prot_subset, color="cell_types",neighbors_key='original_neighbors',title='Original protein data minor cell types')
sc.pl.umap(adata_rna_subset, color="cell_types",neighbors_key='original_neighbors',title='Original RNA data minor cell types')
# sc.pl.umap(adata_prot_subset, color="major_cell_types",neighbors_key='original_neighbors',title='Original protein data major cell types')
# sc.pl.umap(adata_rna_subset, color="major_cell_types",neighbors_key='original_neighbors',title='Original RNA data major cell types')
# sc.pl.umap(adata_prot_subset[adata_prot_subset.obs['major_cell_types'] =='B cells'], color="cell_types",neighbors_key='original_neighbors',title='Latent space MINOR cell types, B cells only')

class DualVAETrainingPlan(TrainingPlan):
    def __init__(self, rna_module, **kwargs):
        protein_vae = kwargs.pop('protein_vae')
        linkage_matrix = kwargs.pop('linkage_matrix')
        contrastive_weight = kwargs.pop('contrastive_weight', 1.0)

        # Call super().__init__() after removing custom kwargs
        super().__init__(rna_module, **kwargs)

        # Now assign the attributes
        self.protein_vae = protein_vae
        self.linkage_matrix = linkage_matrix
        self.contrastive_weight = contrastive_weight
        self.protein_vae.module = self.protein_vae.module.to(device)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.module.parameters()) + list(self.protein_vae.module.parameters()),
            lr=0.001,
            weight_decay=1e-5,
        )
        return optimizer
    def training_step(self, batch, batch_idx):
        # Call the base training step for standard loss computation for RNA VAE
        _, _, rna_loss_output = self.forward(batch)
        rna_loss = rna_loss_output.loss

        # Get protein batch data
        protein_batch = self._get_protein_batch(batch)

        # Compute protein loss using the protein_module's forward method
        _, _, protein_loss_output = self.protein_vae.module.forward(protein_batch)
        prot_loss = protein_loss_output.loss

        # Get latent embeddings from RNA VAE
        rna_inference_outputs = self.module.inference(
            batch["X"], batch_index=batch["batch"], n_samples=1
        )
        rna_latent_embeddings = rna_inference_outputs["z"].squeeze(0)

        # Get latent embeddings from Protein VAE
        protein_inference_outputs = self.protein_vae.module.inference(
            protein_batch["X"], batch_index=protein_batch["batch"], n_samples=1
        )
        protein_latent_embeddings = protein_inference_outputs["z"].squeeze(0)
        torch.cdist(rna_latent_embeddings, protein_latent_embeddings, p=1)
        matching_rna_protein_latent_distances = torch.norm(rna_latent_embeddings - protein_latent_embeddings, p=2, dim=1)
        # Compute pairwise distances for linked cells
        # distances = torch.cdist(rna_latent_embeddings, rna_latent_embeddings, p=2)
        prot_distances = torch.cdist(protein_latent_embeddings, protein_latent_embeddings, p=2)
        rna_distances = torch.cdist(rna_latent_embeddings, rna_latent_embeddings, p=2)
        if False: # plot    histogram of distances prot and rna on top of each other after each epoch
            plt.hist(prot_distances.detach().cpu().numpy().flatten(),bins=100,alpha=0.5)
            plt.hist(rna_distances.detach().cpu().numpy().flatten(),bins=100,alpha=0.5)
            plt.show()
        distances = rna_distances+prot_distances #+rna_distances
        linked_mask = self._get_linked_mask(batch_idx, distances.shape)

        # Contrastive loss
        cell_neighborhood_info = batch["labels"].squeeze()
        major_cell_type = batch["extra_categorical_covs"][:,0].squeeze()

        # positive_pairs = distances[linked_mask]
        # negative_pairs = distances[~linked_mask]
        num_cells = cell_neighborhood_info.shape[0]
        diagonal_mask = torch.eye(num_cells, dtype=torch.bool, device=cell_neighborhood_info.device)
        # this will give us each row represents a item in the array, and each col is whether it is the same as the items in that index of the col
        # this way we get for each cell(a row) which other cells (index of each item in the row, which is the col) are matching
        # so if we got 1,2,1, we will get [[1,0,1],[0,1,0],[1,0,1]]
        same_cn_mask = cell_neighborhood_info.unsqueeze(0) == cell_neighborhood_info.unsqueeze(1)
        same_major_cell_type = major_cell_type.unsqueeze(0) == major_cell_type.unsqueeze(1)

        if batch["batch"][
            0].item() == 0:  # show the mask only for the first batch to make sure it is working as expected
            plt.imshow(same_cn_mask.cpu().numpy())
        distances = distances.masked_fill(diagonal_mask, 0)

        same_major_type_same_cn_mask = (same_major_cell_type * same_cn_mask).type(torch.bool)
        same_major_type_different_cn_mask = (same_major_cell_type * ~same_cn_mask).type(torch.bool)
        different_major_type_same_cn_mask = (~same_major_cell_type * same_cn_mask).type(torch.bool)
        different_major_type_different_cn_mask = (~same_major_cell_type * ~same_cn_mask).type(torch.bool)

        same_major_type_same_cn_mask.masked_fill_(diagonal_mask, 0)
        same_major_type_different_cn_mask.masked_fill_(diagonal_mask, 0)
        different_major_type_same_cn_mask.masked_fill_(diagonal_mask, 0)
        different_major_type_different_cn_mask.masked_fill_(diagonal_mask, 0)

        same_major_type_same_cn_loss = (distances ** 2) * same_major_type_same_cn_mask
        same_major_type_different_cn_loss = ((10 - distances).clamp(min=0) ** 2) * same_major_type_different_cn_mask
        different_major_type_same_cn_loss = ((10 - distances).clamp(min=0) ** 2) * different_major_type_same_cn_mask
        different_major_type_different_cn_loss = ((10 - distances).clamp(min=0) ** 2) * different_major_type_different_cn_mask
        # for debugging only: # todo remove this same_cn_loss, it is not valid
        same_cn_loss = (distances ** 2) * same_cn_mask
        same_major_type_loss = (distances ** 2) * same_major_cell_type
        # end of debugging

        # positive_loss = (positive_pairs ** 2).mean()
        # negative_loss = ((10 - negative_pairs).clamp(min=0) ** 2).mean()
        # contrastive_loss = positive_loss + negative_loss
        positive_loss = same_major_type_same_cn_loss

        # TODO change to temperature, learnable,
        negative_loss = different_major_type_different_cn_loss + different_major_type_same_cn_loss + 2 * same_major_type_different_cn_loss
        cn_loss = (positive_loss.sum() + negative_loss.sum()) / (num_cells * (num_cells - 1))

        # Combine losses

        # Combine losses
        total_loss = (rna_loss*1+prot_loss*1 +
                      self.contrastive_weight * cn_loss+
                      100*matching_rna_protein_latent_distances.mean())
        # total_loss =   self.contrastive_weight * cn_loss

        # Log losses
        self.log("train_rna_loss", rna_loss, prog_bar=True)
        self.log("train_contrastive_loss", cn_loss, prog_bar=True)
        self.log("train_total_loss", total_loss, prog_bar=True)
        # print(sum(torch.sum(x) for x in protein_vae.module.parameters()))
        return total_loss

    def _get_protein_batch(self, batch):
        # Implement logic to fetch the corresponding protein batch
        indices = batch['extra_categorical_covs'][:,1].detach().cpu().numpy().flatten()  # Assuming batch contains indices
        protein_data = self.protein_vae.adata[indices]
        protein_batch = {
            'X': torch.tensor(protein_data.X.A if issparse(protein_data.X) else protein_data.X).to(device),
            'batch': torch.tensor(protein_data.obs['_scvi_batch'].values, dtype=torch.long).to(device),
        'extra_categorical_covs':batch['extra_categorical_covs'],
        'labels':batch['labels'].to(device)
        }
        return protein_batch

    def _get_linked_mask(self, batch_idx, shape):
        # Create a mask for linked cells (assuming one-to-one mapping)
        mask = torch.eye(shape[0], dtype=torch.bool, device=self.module.device)
        return mask


# Setup anndata for RNA and Protein datasets
adata_rna_subset.obs['index'] = np.arange(adata_rna_subset.shape[0])
adata_prot_subset.obs['index'] = np.arange(adata_prot_subset.shape[0])
SCVI.setup_anndata(
    adata_rna_subset,
    labels_key="CN",
    categorical_covariate_keys=["major_cell_types",'index'],
)

SCVI.setup_anndata(
    adata_prot_subset,
    labels_key="CN",
    categorical_covariate_keys=["major_cell_types",'index'],
)

# Initialize VAEs
rna_vae = scvi.model.SCVI(adata_rna_subset, gene_likelihood="nb", n_hidden=128)
protein_vae = scvi.model.SCVI(adata_prot_subset, gene_likelihood="poisson", n_hidden=50)
protein_vae

def custom_optimizer(params):
    # protein_vae.module.parameters() # 27 params
    # rna_vae.module.parameters() # 27 params
    # return torch.optim.AdamW(list(protein_vae.module.parameters()) + list(params), lr=0.001, weight_decay=1e-5)
    return torch.optim.AdamW(list(protein_vae.module.parameters()),lr=0.0001)


# Create linkage matrix (define your own linkage based on your data)
linkage_matrix = ...  # Should be defined based on your dataset

# Initialize the custom TrainingPlan
training_plan = partial(DualVAETrainingPlan,
                        # rna_module=rna_vae.module,
                        protein_vae=protein_vae,
                        linkage_matrix=linkage_matrix,
                        contrastive_weight=10.0,

                        )
# Assign the training plan to the SCVI model
rna_vae._training_plan_cls = training_plan
# Train the model
rna_vae.train(
    check_val_every_n_epoch=1,
    max_epochs=300,
    early_stopping=True,
    early_stopping_patience=70,
    early_stopping_monitor="elbo_validation",
    batch_size=200,

)

SCVI_LATENT_KEY = "X_scVI"
protein_vae.module.to('cpu')
protein_vae.is_trained = True
latent_rna = rna_vae.get_latent_representation()
latent_prot = protein_vae.get_latent_representation()
adata_rna_subset.obsm[SCVI_LATENT_KEY] = latent_rna
adata_prot_subset.obsm[SCVI_LATENT_KEY] = latent_prot

sc.pp.neighbors(adata_rna_subset, use_rep=SCVI_LATENT_KEY,key_added='latent_space_neighbors')
sc.pp.neighbors(adata_prot_subset, use_rep=SCVI_LATENT_KEY,key_added='latent_space_neighbors')
sc.tl.umap(adata_rna_subset,neighbors_key='latent_space_neighbors')
sc.tl.umap(adata_prot_subset,neighbors_key='latent_space_neighbors')

# sc.pl.umap(adata_rna_subset, color="major_cell_types",neighbors_key='latent_space_neighbors',title='RNA Latent space, major cell types')
# sc.pl.umap(adata_prot_subset, color="major_cell_types",neighbors_key='latent_space_neighbors',title='Protein Latent space, major cell types')
# minor cell types
# sc.pl.umap(adata_rna_subset[adata_rna_subset.obs['major_cell_types'] =='B cells'], color="CN",neighbors_key='latent_space_neighbors',title='Latent space, minor cell types (B-cells only) with observed CN')
# make sure that the latent space of the protein and rna are aligned by plotting them on top of each other
sc.pl.umap(adata_rna_subset, color="cell_types",neighbors_key='latent_space_neighbors',title='Latent space, minor cell types RNA',legend_loc='on data')
sc.pl.umap(adata_prot_subset, color="cell_types",neighbors_key='latent_space_neighbors',title='Latent space, minor cell types Protein',legend_loc='on data')
# Combine the latent embeddings into one AnnData object before plotting
combined_latent = ad.concat([AnnData(adata_rna_subset.obsm[SCVI_LATENT_KEY]), AnnData(adata_prot_subset.obsm[SCVI_LATENT_KEY])], join='outer', label='modality', keys=['RNA', 'Protein'])

# Plot the combined latent space
sc.pp.neighbors(combined_latent)
sc.tl.umap(combined_latent)
sc.pl.umap(combined_latent,  color='modality', title='Combined Latent space, minor cell types')
# plot hist of distances between cells of diffrent modalities in latent space
combined_latent.obs['modality'] = combined_latent.obs['modality'].astype('category')
rna_latent = combined_latent[combined_latent.obs['modality']=='RNA']
prot_latent = combined_latent[combined_latent.obs['modality']=='Protein']
# calc distances with numpy
dis = (np.linalg.norm(rna_latent.X-prot_latent.X,axis=1))
rand_rna_latent = rna_latent.copy()
#  randomize order of cells in rna latent space
shuffled_indices = np.random.permutation(rand_rna_latent.obs.index)

# Reindex the AnnData object
rand_rna_latent = rand_rna_latent[shuffled_indices].copy()

rand_dis = np.linalg.norm(rand_rna_latent.X-prot_latent.X,axis=1)
# plot the distances hist with seaborn
sns.histplot(dis, bins=100, color='blue', label='True distances')
sns.histplot(rand_dis, bins=100, color='red', label='Randomized distances')
plt.legend()
plt.show()


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
plot_flag = False
device = "cuda:0" if torch.cuda.is_available() else "cpu"
adata_rna_subset = sc.read('/home/barroz/projects/Mapping-scRNA-seq-Variability-Using-Co-Embedding-to-Understand-Cell-Neighborhood-Effects/CITE-seq_RNA_seq/adata_rna_subset.hd5ad.h5ad')


adata_prot_subset = sc.read('/home/barroz/projects/Mapping-scRNA-seq-Variability-Using-Co-Embedding-to-Understand-Cell-Neighborhood-Effects/CITE-seq_RNA_seq/adata_prot_subset.hd5ad.h5ad')


# def __init__(self, rna_module, protein_module, linkage_matrix, contrastive_weight=1.0, **kwargs):
#     super().__init__(rna_module, **kwargs)
#     self.protein_module = protein_module
#     self.linkage_matrix = linkage_matrix  # A matrix linking RNA and protein cells
#     self.contrastive_weight = contrastive_weight

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
    def training_step(self, batch, batch_idx):
        print('sadfdsfasdf')
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

        # Compute pairwise distances for linked cells
        distances = torch.cdist(rna_latent_embeddings, protein_latent_embeddings, p=2)
        linked_mask = self._get_linked_mask(batch_idx, distances.shape)

        # Contrastive loss
        positive_pairs = distances[linked_mask]
        negative_pairs = distances[~linked_mask]

        positive_loss = (positive_pairs ** 2).mean()
        negative_loss = ((10 - negative_pairs).clamp(min=0) ** 2).mean()
        contrastive_loss = positive_loss + negative_loss

        # Combine losses
        total_loss = rna_loss + self.contrastive_weight * contrastive_loss

        # Log losses
        self.log("train_rna_loss", rna_loss, prog_bar=True)
        self.log("train_contrastive_loss", contrastive_loss, prog_bar=True)
        self.log("train_total_loss", total_loss, prog_bar=True)

        return total_loss

    def _get_protein_batch(self, batch):
        # Implement logic to fetch the corresponding protein batch
        indices = batch['batch'].detach().cpu().numpy().flatten()  # Assuming batch contains indices
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
SCVI.setup_anndata(
    adata_rna_subset,
    labels_key="CN",
    categorical_covariate_keys=["major_cell_types"],
)

SCVI.setup_anndata(
    adata_prot_subset,
    labels_key="CN",
    categorical_covariate_keys=["major_cell_types"],
)

# Initialize VAEs
rna_vae = scvi.model.SCVI(adata_rna_subset, gene_likelihood="nb", n_hidden=10)
protein_vae = scvi.model.SCVI(adata_prot_subset, gene_likelihood="poisson", n_hidden=10)


def custom_optimizer(params):
    return torch.optim.AdamW(params, lr=0.001, weight_decay=1e-5)


# Create linkage matrix (define your own linkage based on your data)
linkage_matrix = ...  # Should be defined based on your dataset

# Initialize the custom TrainingPlan
training_plan = partial(DualVAETrainingPlan,
                        # rna_module=rna_vae.module,
                        protein_vae=protein_vae,
                        linkage_matrix=linkage_matrix,
                        contrastive_weight=100.0,
                        optimizer="Custom",
                        optimizer_creator=custom_optimizer

                        )
# Assign the training plan to the SCVI model
rna_vae._training_plan_cls = training_plan

# Train the model
rna_vae.train(
    check_val_every_n_epoch=1,
    max_epochs=50,
    early_stopping=True,
    early_stopping_patience=20,
    early_stopping_monitor="elbo_validation",
    batch_size=10,
)



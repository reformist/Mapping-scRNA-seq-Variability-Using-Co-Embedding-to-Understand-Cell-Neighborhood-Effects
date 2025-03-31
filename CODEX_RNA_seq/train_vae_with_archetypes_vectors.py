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

# %%

# %%

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
from datetime import datetime
import scipy
import anndata as ad
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.optimize import linear_sum_assignment
from anndata import AnnData
import warnings
import os
import sys

# Add repository root to Python path without changing working directory

from scipy.sparse import issparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from sklearn.decomposition import PCA
from pathlib import Path
import scvi
from sklearn.manifold import TSNE
from matplotlib.patches import Arc

importlib.reload(scvi)
import re
from scvi.model import SCVI
from scvi.train import TrainingPlan
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score

import torch.nn.functional as F



import bar_nick_utils
importlib.reload(bar_nick_utils)

from bar_nick_utils import plot_latent, \
    compute_pairwise_kl, select_gene_likelihood, \
    clean_uns_for_h5ad, plot_normalized_losses, compute_pairwise_kl_two_items, get_latest_file, \
    match_datasets, calculate_cLISI, calculate_iLISI,get_umap_filtered_fucntion,mixing_score,plot_cosine_distance,plot_latent_mean_std,\
    plot_rna_protein_matching_means_and_scale,archetype_vs_latent_distances_plot,plot_inference_outputs,verify_gradients,compare_distance_distributions,plot_similarity_loss_history

if not hasattr(sc.tl.umap, '_is_wrapped'):
    sc.tl.umap = get_umap_filtered_fucntion()
    sc.tl.umap._is_wrapped = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pd.set_option("display.max_columns", 10)
pd.set_option("display.max_rows", 10)
warnings.filterwarnings("ignore")
pd.options.display.max_rows = 10
pd.options.display.max_columns = 10
np.set_printoptions(threshold=100)
np.random.seed(0)
save_dir = 'data'

plot_flag = True


# %%
folder = 'data/'
file_prefixes = ['adata_rna_', 'adata_prot_', 'adata_archetype_rna_', 'adata_archetype_prot_']



# Load the latest files
latest_files = {prefix: get_latest_file(folder, prefix) for prefix in file_prefixes}
print(latest_files)
adata_rna = sc.read(latest_files['adata_rna_'])
adata_prot = sc.read(latest_files['adata_prot_'])
adata_archetype_rna = sc.read(latest_files['adata_archetype_rna_'])
adata_archetype_prot = sc.read(latest_files['adata_archetype_prot_'])

# %%

# %%
sample_size = min(len(adata_prot), len(adata_rna), 2000)
adata_rna_subset = sc.pp.subsample(adata_rna, n_obs=sample_size, copy=True)
adata_prot_subset = sc.pp.subsample(adata_prot, n_obs=int(sample_size)-1, copy=True)
del adata_prot, adata_rna
if plot_flag:
    # making sure that the archetypes make sense in original data context
    sc.pp.neighbors(adata_rna_subset)
    sc.pp.neighbors(adata_prot_subset)
    sc.tl.umap(adata_rna_subset)
    sc.tl.umap(adata_prot_subset)
    sc.pl.umap(adata_rna_subset, color='archetype_label', title='Origanal RNA cells assosiate Archetypes')
    sc.pl.umap(adata_prot_subset, color='archetype_label', title='Original  Protein cells assosiate Archetypes')

# %%
# order cells by major and minor cell type for easy visualization
new_order_rna = adata_rna_subset.obs.sort_values(by=['major_cell_types', 'cell_types']).index
new_order_prot = adata_prot_subset.obs.sort_values(by=['major_cell_types', 'cell_types']).index
adata_rna_subset = adata_rna_subset[new_order_rna]
adata_prot_subset = adata_prot_subset[new_order_prot]
archetype_distances = scipy.spatial.distance.cdist(adata_rna_subset.obsm['archetype_vec'].values,
                                               adata_prot_subset.obsm['archetype_vec'].values,metric='cosine')
matching_distance_before = np.diag(archetype_distances).mean()

if plot_flag:
    plt.figure(figsize=(10, 5))
    plt.suptitle('Heatmap of archetype coor before matching\nordred by cell types only')
    plt.subplot(1, 2, 1)
    sns.heatmap(np.log1p(adata_rna_subset.obsm['archetype_vec'].values), cbar=False)
    plt.title('RNA')
    plt.ylabel('RNA cell index')
    plt.xlabel('Archetype Betas')
    plt.subplot(1, 2, 2)
    sns.heatmap(np.log1p(adata_prot_subset.obsm['archetype_vec'].values), cbar=False)
    plt.xlabel('Archetype Betas')
    plt.ylabel('Protein cell index')
    plt.title('Protein')
    plt.show()

# %%

# %%
# use_matched = input('Do you want to use previous saved matched data? (y/n)')
# if use_matched == 'y':
#     adata_rna_subset, adata_prot_subset = adata_rna_subset_matched, adata_prot_subset_matched
# else:
#     adata_rna_subset_matched,adata_prot_subset_matched = match_datasets(adata_rna_subset,adata_prot_subset,0.05,plot_flag=plot_flag)
#     adata_rna_subset,adata_prot_subset = adata_rna_subset_matched,adata_prot_subset_matched
#     # save the matched data
#     adata_rna_subset_matched.write(f'{save_dir}/adata_rna_matched.h5ad')
#     adata_prot_subset_matched.write(f'{save_dir}/adata_prot_matched.h5ad')
    
adata_rna_subset_matched,adata_prot_subset_matched = match_datasets(adata_rna_subset,adata_prot_subset,threshold=0.1,plot_flag=plot_flag)
# ok threhols example  = 0.01

adata_rna_subset,adata_prot_subset = adata_rna_subset_matched,adata_prot_subset_matched
adata_rna_subset.obs['CN']= adata_prot_subset.obs['CN'].values# add the CN to the rna data
    


# %%
if plot_flag:
    plt.figure(figsize=(10, 5))
    plt.suptitle('Heatmap of archetype coor after matching')
    plt.subplot(1, 2, 1)
    sns.heatmap(np.log1p(adata_rna_subset.obsm['archetype_vec'].values), cbar=False)
    plt.ylabel('RNA cell index')
    plt.xlabel('Archetype Betas')
    plt.subplot(1, 2, 2)
    sns.heatmap(np.log1p(adata_prot_subset.obsm['archetype_vec'].values), cbar=False)
    plt.ylabel('Protein cell index')
    plt.xlabel('Archetype Betas')

# %%
if plot_flag:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)

    sns.heatmap(np.log1p(archetype_distances[::5, ::5].T))
    plt.xlabel('RNA cell index')
    plt.ylabel('Protein cell index')
    plt.gca().invert_yaxis()

    plt.subplot(1, 2, 2)
    plt.scatter(
        np.arange(len(archetype_distances.argmin(axis=1))),
        archetype_distances.argmin(axis=1),
        s=1,
        rasterized=True
    )
    plt.xlabel('RNA cell index')
    plt.ylabel('Protein cell index')
    # plt.title('If this looks like a line, then the matching ARE THE SAME AND NOT ACROSS MODALITIES')
    plt.show()


# %%
adata_rna_subset

# %%
sc.pp.pca(adata_rna_subset)
sc.pp.pca(adata_prot_subset)
sc.pp.neighbors(adata_rna_subset, key_added='original_neighbors')
sc.tl.umap(adata_rna_subset, neighbors_key='original_neighbors')
adata_rna_subset.obsm['X_original_umap'] = adata_rna_subset.obsm["X_umap"]
sc.pp.neighbors(adata_prot_subset, key_added='original_neighbors')
sc.tl.umap(adata_prot_subset, neighbors_key='original_neighbors')
adata_prot_subset.obsm['X_original_umap'] = adata_prot_subset.obsm["X_umap"]

if plot_flag:
    sc.pl.pca(adata_rna_subset, color=['cell_types', 'major_cell_types'])
    sc.pl.pca(adata_prot_subset, color=['cell_types', 'major_cell_types'])
    sc.pl.embedding(adata_rna_subset, basis='X_original_umap', color=['cell_types', 'major_cell_types'])
    sc.pl.embedding(adata_prot_subset, basis='X_original_umap', color=['cell_types', 'major_cell_types'])


# %%
adata_rna_subset.obs['major_cell_types'][0]

# %%
if plot_flag:
    adata_B_cells = adata_rna_subset[adata_rna_subset.obs['major_cell_types'] == adata_rna_subset.obs['major_cell_types'][0]]
    sc.pp.pca(adata_B_cells)
    sc.pp.neighbors(adata_B_cells, use_rep='X_pca')
    sc.tl.umap(adata_B_cells)
    if 'tissue' in adata_B_cells.obs:
        sc.pl.umap(adata_B_cells, color=['tissue'], title='verifying tissue does not give a major effect')
    else:
        sc.pl.umap(adata_B_cells, color=['cell_types'], title='verifying cell types are well separated')


# %%
adata_prot_subset.obs.columns

# %%

sc.pp.neighbors(adata_prot_subset, use_rep='X_pca', key_added='X_neighborhood')
sc.tl.umap(adata_prot_subset, neighbors_key='X_neighborhood')
adata_prot_subset.obsm['X_original_umap'] = adata_prot_subset.obsm["X_umap"]
sc.pl.umap(adata_prot_subset, color='CN', title='Protein UMAP of CN vectors colored by CN label',
           neighbors_key='original_neighbors' )
one_cell_type = adata_prot_subset.obs['major_cell_types'][0]
sc.pl.umap(adata_prot_subset[adata_prot_subset.obs['major_cell_types'] == one_cell_type], color='cell_types',
           title='Protein UMAP of CN vectors colored by minor cell type label')
adata_prot_subset

# %%
if plot_flag:
    sns.histplot(adata_prot_subset[adata_prot_subset.obs['major_cell_types'] == one_cell_type].obs, x='cell_types',
                 hue='CN', multiple='fill', stat='proportion')
    # sns.histplot(adata_prot_subset.obs, x='cell_types',hue='CN', multiple='fill', stat='proportion')
    plt.title('Showcasing the signature CN progile of each minor B cell type')

# %%
if plot_flag:
    # sc.pl.embedding(adata_rna_subset, color=["major_cell_types","cell_types"], basis='X_original_umap',title='Original data major minor cell types')
    # sc.pl.embedding(adata_prot_subset, color=["major_cell_types","cell_types"], basis='X_original_umap',title='Original data major and minor cell types')

    # sc.pl.umap(adata_rna_subset, color="CN",neighbors_key='original_neighbors',title='Original RNA data CN')
    sc.pl.embedding(adata_rna_subset, color=[ "cell_types"], basis='X_original_umap', title='Original rna data CN')
    sc.pl.embedding(adata_rna_subset, color=["CN","cell_types"],basis='X_original_umap',title='Original rna data CN')
    # sc.pl.umap(adata_prot_subset, color="CN",neighbors_key='original_neighbors',title='Original protein data CN')
    sc.pl.embedding(adata_prot_subset, color=["CN", "cell_types", 'archetype_label'], basis='X_original_umap')
    sc.pl.embedding(adata_prot_subset, color=['archetype_label', "cell_types", ], basis='X_original_umap')
    sc.pl.umap(adata_prot_subset[adata_prot_subset.obs['major_cell_types'] == one_cell_type], color="cell_types",
               neighbors_key='original_neighbors', title='Latent space MINOR cell types, B cells only')



# %%
# DO NOT DELETE - save the adata of external processing
cwd = os.getcwd()
# adata_rna_subset = sc.read(f'{cwd}/adata_rna_subset.hd5ad')
# adata_prot_subset = sc.read(f'{cwd}/data_prot_subset.hd5ad')
clean_uns_for_h5ad(adata_prot_subset)
clean_uns_for_h5ad(adata_rna_subset)
sc.write(
    Path(f'{cwd}/adata_rna_subset'),
    adata_rna_subset)
sc.write(
    Path(f'{cwd}/adata_prot_subset'),
    adata_prot_subset)

adata_rna_subset.X = adata_rna_subset.X.astype(np.float32)
adata_prot_subset.X = adata_prot_subset.X.astype(np.float32)
adata_prot_subset.obs['CN'] = adata_prot_subset.obs['CN'].astype(int)
adata_rna_subset.obs['CN'] = adata_rna_subset.obs['CN'].astype(int)
adata_prot_subset.obs['CN'] = pd.Categorical(adata_prot_subset.obs['CN'])
adata_rna_subset.obs['CN'] = pd.Categorical(adata_rna_subset.obs['CN'])


# %%



# %%


class DualVAETrainingPlan(TrainingPlan):
    def __init__(self, rna_module, **kwargs):
        protein_vae = kwargs.pop('protein_vae')
        rna_vae = kwargs.pop('rna_vae')
        self.plot_x_times = kwargs.pop('plot_x_times',5)
        contrastive_weight = kwargs.pop('contrastive_weight', 1.0)
        self.batch_size = kwargs.pop('batch_size', 128)
        # super().__init__(protein_vae.module, **kwargs)
        super().__init__(rna_module, **kwargs)
        self.rna_vae = rna_vae
        self.protein_vae = protein_vae
        num_batches = 2
        latent_dim = self.rna_vae.module.n_latent
        self.batch_classifier = torch.nn.Linear(latent_dim, num_batches)
        self.contrastive_weight = contrastive_weight
        self.protein_vae.module.to(device)
        self.rna_vae.module = self.rna_vae.module.to(device)
        self.first_step = True
        if self.protein_vae.adata.uns.get('ordered_matching_cells') is not True:
            raise ValueError('The cells are not aligned across modalities, make sure ')
        n_samples = len(self.rna_vae.adata)
        steps_per_epoch = int(np.ceil(n_samples / self.batch_size))
        self.total_steps = steps_per_epoch * n_epochs
        self.similarity_loss_history = []
        self.steady_state_window = 50  # Number of steps to check for steady state
        self.steady_state_tolerance = 0.5  # Tolerance for determining steady state
        self.similarity_weight = 100000  # Initial weight
        self.similarity_active = True  # Flag to track if similarity loss is active
        self.reactivation_threshold = 0.1  # Threshold to reactivate similarity loss
        self.active_similarity_loss_active_history = []
        self.similarity_loss_all_history = []


        

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.rna_vae.module.parameters()) + list(self.protein_vae.module.parameters()),
            lr=0.001,
            weight_decay=1e-5,
        )
        # d = { # maybe add this?
        # "optimizer": optimizer,
        # "gradient_clip_val": 1.0,  # Critical for stability
        # "gradient_clip_algorithm": "value"
        # }
        return optimizer

    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        rna_batch = self._get_rna_batch(batch)
        _, _, rna_loss_output = self.rna_vae.module(rna_batch, loss_kwargs=self.loss_kwargs)
        protein_batch = self._get_protein_batch(batch)
        _, _, protein_loss_output = self.protein_vae.module(protein_batch, loss_kwargs=self.loss_kwargs)

        # Inference outputs
        rna_inference_outputs = self.rna_vae.module.inference(
            rna_batch["X"], batch_index=rna_batch["batch"], n_samples=1
        )
        protein_inference_outputs = self.protein_vae.module.inference(
            protein_batch["X"], batch_index=protein_batch["batch"], n_samples=1
        )

        # Compute latent distances
        matching_rna_protein_latent_distances = torch.distributions.kl_divergence(
            rna_inference_outputs["qz"], protein_inference_outputs["qz"]
        )

        # Compute pairwise distances for RNA and protein
        rna_distances = compute_pairwise_kl(
            rna_inference_outputs["qz"].mean, rna_inference_outputs["qz"].scale
        )
        prot_distances = compute_pairwise_kl(
            protein_inference_outputs["qz"].mean, protein_inference_outputs["qz"].scale
        )

        # Retrieve cell neighborhood information
        index = rna_batch["labels"]
        cell_neighborhood_info = torch.tensor(
            self.protein_vae.adata[index].obs["CN"].values, device=device
        )
        rna_major_cell_type = torch.tensor(
            self.rna_vae.adata[index].obs["major_cell_types"].values.codes, device=device
        ).squeeze()
        protein_major_cell_type = torch.tensor(
            self.rna_vae.adata[index].obs["major_cell_types"].values.codes, device=device
        ).squeeze()

        num_cells = cell_neighborhood_info.shape[0]
        diagonal_mask = torch.eye(num_cells, dtype=torch.bool, device=device)

        same_cn_mask = cell_neighborhood_info.unsqueeze(0) == cell_neighborhood_info.unsqueeze(1)
        same_major_cell_type_mask = rna_major_cell_type.unsqueeze(0) == protein_major_cell_type.unsqueeze(1)

        # Use protein distances or a combination as needed
        distances = prot_distances + rna_distances
        distances = distances.masked_fill(diagonal_mask, 0)

        # Define loss masks
        same_major_type_same_cn_loss = (distances ** 2) * (same_major_cell_type_mask & same_cn_mask)
        same_major_type_different_cn_loss = ((10 - distances).clamp(min=0) ** 2) * (
                same_major_cell_type_mask & ~same_cn_mask
        )
        different_major_type_same_cn_loss = ((10 - distances).clamp(min=0) ** 2) * (
                ~same_major_cell_type_mask & same_cn_mask
        )
        different_major_type_different_cn_loss = ((10 - distances).clamp(min=0) ** 2) * (
                ~same_major_cell_type_mask & ~same_cn_mask
        )

        cn_loss = (
                same_major_type_same_cn_loss.mean()
                + same_major_type_different_cn_loss.mean()
                + different_major_type_same_cn_loss.mean()
                + different_major_type_different_cn_loss.mean()
        ) / (num_cells * (num_cells - 1))

        validation_total_loss = (
              1 * self.contrastive_weight * cn_loss
                + 0.1 * rna_loss_output.loss
                + 0.1 * protein_loss_output.loss
                + 10 * matching_rna_protein_latent_distances.mean()
        )

        # Log metrics
        self.log(
            "validation_rna_loss", rna_loss_output.loss, on_epoch=True, sync_dist=self.use_sync_dist
        )
        self.log(
            "validation_protein_loss", protein_loss_output.loss, on_epoch=True, sync_dist=self.use_sync_dist
        )
        self.log(
            "validation_contrastive_loss", cn_loss, on_epoch=True, sync_dist=self.use_sync_dist
        )
        self.log(
            "validation_total_loss", validation_total_loss, on_epoch=True, sync_dist=self.use_sync_dist
        )
        self.log(
            "validation_matching_latent_distances", matching_rna_protein_latent_distances.mean(),
            on_epoch=True, sync_dist=self.use_sync_dist
        )

        # Compute and log additional metrics
        self.compute_and_log_metrics(rna_loss_output, self.val_metrics, "validation")
        self.compute_and_log_metrics(protein_loss_output, self.val_metrics, "validation")

        return validation_total_loss

    def training_step(self, batch, batch_idx):
        rna_batch = self._get_rna_batch(batch)
        kl_weight = 2  # maybe make sure this is proper
        self.loss_kwargs.update({"kl_weight": kl_weight})
        _, _, rna_loss_output = self.rna_vae.module(rna_batch, loss_kwargs=self.loss_kwargs)
        protein_batch = self._get_protein_batch(batch)
        _, _, protein_loss_output = self.protein_vae.module(protein_batch, loss_kwargs=self.loss_kwargs)

        rna_inference_outputs = self.rna_vae.module.inference(
            rna_batch["X"], batch_index=rna_batch["batch"], n_samples=1
        )
        index = rna_batch["labels"]
        # assert len(set(self.protein_vae.adata[index].obs['CN'].values)) != 1# should this be commented out?

        protein_inference_outputs = self.protein_vae.module.inference(
            protein_batch["X"], batch_index=protein_batch["batch"], n_samples=1
        )
        # here we assume that the cells have been aligned in the same order to their best match across modalities (check adata_prot_subset.uns['ordered_matching_cells'])

        archetype_dis = scipy.spatial.distance.cdist(
            rna_batch['archetype_vec'],
            protein_batch['archetype_vec'],
            metric='cosine'
        )

        latent_distances =compute_pairwise_kl_two_items(rna_inference_outputs["qz"].mean,protein_inference_outputs["qz"].mean,
                                        rna_inference_outputs["qz"].scale,protein_inference_outputs["qz"].scale)
        latent_distances = torch.clamp(latent_distances, max=torch.quantile(latent_distances, 0.90))


        if self.global_step > -1 and self.global_step % (1+int(self.total_steps /(self.plot_x_times)))== 0:
            latent_distances_temp = torch.cdist(rna_inference_outputs["qz"].mean,
                                protein_inference_outputs["qz"].mean,
                                p=2)
            plot_latent_mean_std(rna_inference_outputs,protein_inference_outputs)
            plot_rna_protein_matching_means_and_scale(rna_inference_outputs, protein_inference_outputs)
            print(f'min laten distances is {round(latent_distances.min().item(),3)}')
            print(f'max laten distances is {round(latent_distances.max().item(),3)}')
            print(f'mean laten distances is {round(latent_distances.mean().item(),3)}\n\n')

        archetype_dis_tensor = torch.tensor(archetype_dis, dtype=torch.float, device=latent_distances.device)
        threshold = 0.0005
        # normlize distances to [0,1] since we are using the same threshold for both archetype and latent distances
        archetype_dis_tensor = (archetype_dis_tensor - archetype_dis_tensor.min()) / (archetype_dis_tensor.max() - archetype_dis_tensor.min())
        latent_distances = (latent_distances - latent_distances.min()) / (latent_distances.max() - latent_distances.min())

        squared_diff = (latent_distances - archetype_dis_tensor)**2
        # Identify pairs that are close in the original space and remain close in the latent space
        acceptable_range_mask = (archetype_dis_tensor < threshold) & (latent_distances < threshold)
        stress_loss = squared_diff.mean()
        num_pairs = squared_diff.numel() 
        num_acceptable = acceptable_range_mask.sum()
        exact_pairs = 10*torch.diag(latent_distances).mean()

        reward_strength = 0  # should be zero, if it is positive I think it cause all the sampel to be as close as possible into one central point which is not good
        # Apply the reward by subtracting from the loss based on how many acceptable pairs we have
        reward = reward_strength * (num_acceptable.float() / num_pairs)
        matching_loss = stress_loss - reward + exact_pairs
        rna_distances = compute_pairwise_kl(rna_inference_outputs["qz"].mean,
                                            rna_inference_outputs["qz"].scale)
        prot_distances = compute_pairwise_kl(protein_inference_outputs["qz"].mean,
                                             protein_inference_outputs["qz"].scale)
        distances = 5* prot_distances + rna_distances
        
        rna_size = prot_size = rna_batch['X'].shape[0]
        mixed_latent = torch.cat([rna_inference_outputs["qz"].mean, protein_inference_outputs["qz"].mean], dim=0)
        batch_labels = torch.cat([torch.zeros(rna_size), torch.ones(prot_size)]).to(device)
        batch_pred = self.batch_classifier(mixed_latent)
        adv_loss = -F.cross_entropy(batch_pred, batch_labels.long())

        if self.first_step and plot_flag and False:  # show the mask only for the first batch to make sure it is working as expected
            plot_inference_outputs(rna_inference_outputs, protein_inference_outputs,
                                   latent_distances, rna_distances, prot_distances)
            self.first_step = False
        if self.global_step > -1 and self.global_step %( 1+int(self.total_steps /(self.plot_x_times))) == 0:
            print('mean prot distances is ',round(prot_distances.mean().item(),3))
            print('mean rna distances is ',round(rna_distances.mean().item(),3))
            print('after I multiply the prot distances by 5')
            
            verify_gradients(self.rna_vae.module,self.protein_vae) # no funcitonal
            print('acceptable ratio',round(num_acceptable.float().item() / num_pairs,3))    
            print('stress_loss',round(stress_loss.item(),3))
            print('reward',round(reward.item(),3))
            print('exact_pairs_loss',round(exact_pairs.item(),3))
            print('matching_loss',round(matching_loss.item(),3),'\n\n')
            'sssssssssssssssssssssssssssssssssssssssssssss'
            if plot_flag:
                plot_latent(rna_inference_outputs["qz"].mean.clone().detach().cpu().numpy(),
                            protein_inference_outputs["qz"].mean.clone().detach().cpu().numpy(),
                            self.rna_vae.adata, self.protein_vae.adata, index=protein_batch["labels"])
            mixing_score_ = mixing_score(rna_inference_outputs["qz"].mean, protein_inference_outputs["qz"].mean, 
                                        adata_rna_subset, adata_prot_subset,index,plot_flag)
            print(f'mixing score is {mixing_score_}\n\n')

            self.log("extra_metric_acceptable_ratio", num_acceptable.float().item() / num_pairs, on_epoch=False, on_step=True)
            self.log("extra_metric_stress_loss", stress_loss.item(), on_epoch=False, on_step=True)
            self.log("extra_metric_reward", reward.item(), on_epoch=False, on_step=True)
            self.log("extra_metric_exact_pairs_loss", exact_pairs.item(), on_epoch=False, on_step=True)
            self.log("extra_metric_iLISI", mixing_score_['iLISI'], on_epoch=False, on_step=True)
            self.log("extra_metric_cLISI", mixing_score_['cLISI'], on_epoch=False, on_step=True)


            # price accuracy for diversity
            accuracy = (batch_pred.argmax(dim=1) == batch_labels).float().mean()
            print(f'accuracy is {accuracy}')
            self.log("extra_metric_accuracy", accuracy, on_epoch=False, on_step=True)

            # plt.figure()
            # archetype_dis_tensor_ = archetype_dis_tensor.detach().cpu().numpy()
            # plt.hist(np.sort(archetype_dis_tensor_.flatten()),bins=100)
            # plt.hist(np.sort(archetype_dis_tensor_)[latent_distances.detach().cpu().numpy() < threshold].flatten(),bins=100)
            # plt.title(f'num of below threshold {np.sum(latent_distances.detach().cpu().numpy() < threshold)}')
            # plt.show()

            if plot_flag:
                archetype_vs_latent_distances_plot(archetype_dis_tensor,latent_distances,threshold)
                plot_cosine_distance(rna_batch,protein_batch)
        cell_neighborhood_info = torch.tensor(self.protein_vae.adata[index].obs["CN"].values).to(device)
        rna_major_cell_type = torch.tensor(self.rna_vae.adata[index].obs["major_cell_types"].values.codes).to(
            device).squeeze()
        protein_major_cell_type = torch.tensor(self.protein_vae.adata[index].obs["major_cell_types"].values.codes).to(
            device).squeeze()

        num_cells = self.rna_vae.adata[index].shape[0]
        # this will give us each row represents a item in the array, and each col is whether it is the same as the items in that index of the col
        # this way we get for each cell(a row) which other cells (index of each item in the row, which is the col) are matching
        # so if we got 1,2,1, we will get [[1,0,1],[0,1,0],[1,0,1]]
        same_cn_mask = cell_neighborhood_info.unsqueeze(0) == cell_neighborhood_info.unsqueeze(1)
        same_major_cell_type = rna_major_cell_type.unsqueeze(0) == protein_major_cell_type.unsqueeze(1)
        diagonal_mask = torch.eye(num_cells, dtype=torch.bool, device=cell_neighborhood_info.device)

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
        # for debugging only: #
        same_cn_loss = (distances ** 2) * same_cn_mask
        same_major_type_loss = (distances ** 2) * same_major_cell_type
        # end of debugging
        positive_loss = same_major_type_same_cn_loss

        # negative_loss = different_major_type_different_cn_loss + different_major_type_same_cn_loss + 10* same_major_type_different_cn_loss
        negative_loss = same_major_type_different_cn_loss # try to simplify loss
        cn_loss = (positive_loss.mean() + negative_loss.mean()) / (num_cells * (num_cells - 1))

        matching_loss = 1000 * matching_loss.mean()
        reconstruction_loss = rna_loss_output.loss * 000.1 + protein_loss_output.loss * 000.1
        # Calculate silhouette score for the CN clusters
        # if self.global_step % 50 ==0:
        #     cn_labels = cell_neighborhood_info.cpu().numpy()
        #     silhouette_avg_rna = silhouette_score(rna_inference_outputs["qz"].mean.detach().cpu().numpy(), cn_labels)
        #     silhouette_avg_prot = silhouette_score(protein_inference_outputs["qz"].mean.detach().cpu().numpy(), cn_labels)
        #     silhouette_avg = (silhouette_avg_rna + silhouette_avg_prot) / 2

        #     if silhouette_avg < 0.1:
        #         self.contrastive_weight = self.contrastive_weight * 10
        #     else :
        #         self.contrastive_weight =  self.contrastive_weight / 10
        contrastive_loss = self.contrastive_weight * cn_loss * 1000000

        probs = F.softmax(batch_pred, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        adv_loss = -entropy.mean()  # Negative entropy to maximize uncertainty

        adv_loss = 1 * adv_loss
        
        # In training_step:
        all_latent = torch.cat([rna_inference_outputs["qz"].mean,
                            protein_inference_outputs["qz"].mean], dim=0)
        rna_dis =torch.cdist(rna_inference_outputs["qz"].mean, rna_inference_outputs["qz"].mean)
        prot_dis =torch.cdist(protein_inference_outputs["qz"].mean, protein_inference_outputs["qz"].mean)
        rna_prot_dis =torch.cdist(rna_inference_outputs["qz"].mean, protein_inference_outputs["qz"].mean)
        # 
        # apply loss on similarity between rna and prot vs internal similarity 
        # Calculate the similarity loss as the absolute difference between the average of the mean absolute clamped RNA and protein distances
        # and the mean absolute clamped RNA-protein distances. This helps in measuring the similarity between RNA and protein distances
        # while avoiding the influence of outliers by using clamped distances.
        similarity_loss_raw = torch.abs(((rna_dis.abs().mean() + prot_dis.abs().mean())/2) - 
                                        rna_prot_dis.abs().mean())
        # Store the current loss value
        
        self.similarity_loss_history.append(similarity_loss_raw.item())
        
        # Only keep the most recent window of values
        if len(self.similarity_loss_history) > self.steady_state_window:
            self.similarity_loss_history.pop(0)
        
        # Determine if we're in steady state
        in_steady_state = False
        coeff_of_variation  = 0 # default value
        if len(self.similarity_loss_history) == self.steady_state_window:
            # Calculate mean and standard deviation over the window
            mean_loss = sum(self.similarity_loss_history) / self.steady_state_window
            std_loss = (sum((x - mean_loss) ** 2 for x in self.similarity_loss_history) / self.steady_state_window) ** 0.5
            
            # Check if variation is small enough to be considered steady state
            coeff_of_variation = std_loss / mean_loss
            if coeff_of_variation < self.steady_state_tolerance:
                in_steady_state = True
        
        # Determine if loss has increased significantly from steady state
        loss_increased = False
        if not self.similarity_active and len(self.similarity_loss_history) > 0:
            recent_loss = similarity_loss_raw.item()
            min_steady_loss = min(self.similarity_loss_history)
            if recent_loss > min_steady_loss * (1 + self.reactivation_threshold):
                loss_increased = True
        # Update the weight based on steady state detection
        if in_steady_state and self.similarity_active:
            current_similarity_weight = self.similarity_weight/1000  # Zero out weight when in steady state
            self.similarity_active = False
        elif loss_increased and not self.similarity_active:
            current_similarity_weight = self.similarity_weight  # Reactivate with full weight
            self.similarity_active = True
        else:
            current_similarity_weight = self.similarity_weight if self.similarity_active else 0
        # if the mixing score is lower than 1.8 activaet the similarity loss
        # Apply weight to loss
        
        combined_latent = ad.concat([
        AnnData(rna_inference_outputs["qz"].mean.detach().cpu().numpy()), 
        AnnData(protein_inference_outputs["qz"].mean.detach().cpu().numpy())
        ], join='outer', label='modality', keys=['RNA', 'Protein'])


        if self.global_step % 50 == 0:
            sc.pp.pca(combined_latent, n_comps=5)
            sc.pp.neighbors(combined_latent,use_rep='X_pca',n_neighbors=10)
            iLISI_score =calculate_iLISI(combined_latent, 'modality',plot_flag=False)
            if iLISI_score < 1.9 and self.similarity_weight > 1e8:
                print()
                self.similarity_weight = self.similarity_weight * 10
            elif self.similarity_weight > 100: # make it smaller only if it is not too small
                self.similarity_weight = self.similarity_weight / 10
        similarity_loss = current_similarity_weight * similarity_loss_raw
        self.active_similarity_loss_active_history.append( self.similarity_active)
        self.similarity_loss_all_history.append(similarity_loss.item())
        if self.global_step > -1 and self.global_step %( 1+int(self.total_steps /(self.plot_x_times))) == 0:
            plot_similarity_loss_history(self.similarity_loss_all_history, self.active_similarity_loss_active_history)


        dis=torch.cdist(all_latent, all_latent)
        dis1 = dis[:rna_size, rna_size:]
        dis2 = dis[rna_size:, rna_size:]
        diversity_loss = torch.abs(dis1.mean() - dis2.mean())
        # print('diversity_loss',diversity_loss)
        diversity_loss = diversity_loss * 1000000
        total_loss = (
                reconstruction_loss
                + contrastive_loss
                + matching_loss
                +similarity_loss
                # + adv_loss
                # + diversity_loss    
        )
        # Log losses
        
        self.log("train_similarity_loss_raw", similarity_loss_raw.item(), on_epoch=False, on_step=True)
        # self.log("train_silhouette_score", silhouette_avg, on_epoch=False, on_step=True)
        self.log("train_similarity_weight", current_similarity_weight, on_epoch=False, on_step=True)
        self.log("train_similarity_weighted", similarity_loss.item(), on_epoch=False, on_step=True)

        # Log ratios of loss components to total loss
        similarity_ratio = similarity_loss.item() / (total_loss.item() + 1e-8)
        self.log("train_similarity_ratio", similarity_ratio, on_epoch=False, on_step=True)          
        
          
        self.log("train_rna_reconstruction_loss", rna_loss_output.loss, on_epoch=False, on_step=True)
        self.log("train_protein_reconstruction_loss", protein_loss_output.loss, on_epoch=False, on_step=True)
        self.log("train_contrastive_loss", contrastive_loss, on_epoch=False, on_step=True)
        self.log("train_matching_rna_protein_loss", matching_loss, on_epoch=False, on_step=True)
        self.log("train_total_loss", total_loss, on_epoch=False, on_step=True)
        self.log("train_adv_loss", adv_loss, on_epoch=False, on_step=True)
        self.log("train_diversity_loss", diversity_loss, on_epoch=False, on_step=True)
        
        
        if (self.global_step > -1 and self.global_step % (1+max(int(self.total_steps / self.plot_x_times)), 100) == 0):
            print(
                f'losses are:\n reconstruction_loss:{reconstruction_loss}, contrastive_loss:{contrastive_loss},\n', 
                f'matching_loss:{matching_loss}, similarity_loss:{similarity_loss},total_loss:{total_loss}\n'
                f'coeff_of_variation: {coeff_of_variation}')
                # , silhouette_avg: {silhouette_avg}\n\n')
                # adv_loss:{adv_loss},divesity loss {diversity_loss} 
            
        # self.saved_model = False if (self.current_epoch % 49 == 0) else True
        # if self.current_epoch % 50 == 0 and self.saved_model:
        #     print('sved model')
        #     rna_vae.save(save_dir, prefix=f'batch_{self.current_epoch}_', save_anndata=False, overwrite=True)
        #     self.saved_model =True
        return total_loss

    def _get_protein_batch(self, batch):
        indices = batch['labels'].detach().cpu().numpy().flatten()  # Assuming batch contains indices
        indices = np.sort(indices)

        protein_data = self.protein_vae.adata[indices]
        protein_batch = {
            'X': torch.tensor(protein_data.X.toarray() if issparse(protein_data.X) else protein_data.X).to(device),
            'batch': torch.tensor(protein_data.obs['_scvi_batch'].values, dtype=torch.long).to(device),
            'labels': indices,
            'archetype_vec': protein_data.obsm['archetype_vec']
        }
        return protein_batch

    def _get_rna_batch(self, batch):
        indices = batch['labels'].detach().cpu().numpy().flatten()
        indices = np.sort(indices)
        rna_data = self.rna_vae.adata[indices]
        rna_batch = {
            'X': torch.tensor(rna_data.X.toarray() if issparse(rna_data.X) else rna_data.X).to(device),
            'batch': torch.tensor(rna_data.obs['_scvi_batch'].values, dtype=torch.long).to(device),
            'labels': indices,
            'archetype_vec': rna_data.obsm['archetype_vec']

        }
        return rna_batch


SCVI.setup_anndata(
    adata_rna_subset,
    labels_key="index_col",
)
if adata_prot_subset.X.min() < 0:
    adata_prot_subset.X = adata_prot_subset.X - adata_prot_subset.X.min()

SCVI.setup_anndata(
    adata_prot_subset,
    labels_key="index_col",
)

# Initialize VAEs
rna_vae = None
protein_vae = None
rna_vae = scvi.model.SCVI(adata_rna_subset, gene_likelihood=select_gene_likelihood(adata_rna_subset), n_hidden=128,
                          n_layers=3)
protein_vae = scvi.model.SCVI(adata_prot_subset, gene_likelihood="normal", n_hidden=50, n_layers=3)
initial_weights = {name: param.clone() for name, param in rna_vae.module.named_parameters()}

rna_vae._training_plan_cls = DualVAETrainingPlan
protein_vae._training_plan_cls = DualVAETrainingPlan
protein_vae.module.to('cpu')
rna_vae.module.to('cpu')
rna_vae.is_trained = protein_vae.is_trained = True
# latent_rna_before = rna_vae.get_latent_representation().copy()
# latent_prot_before = protein_vae.get_latent_representation().copy()
rna_vae.is_trained = protein_vae.is_trained = False

# Create a TensorBoard logger
# It will create a folder nam   ed "my_logs" with subfolders for each run.
logger = TensorBoardLogger(save_dir="my_logs", name=f"experiment_name_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
n_epochs = 2
plot_flag = False
# rna_vae.train(  # for debug only1
#  check_val_every_n_epoch=1,
#  max_epochs=n_epochs,
#  early_stopping=False,
#  early_stopping_patience=70,
#  early_stopping_monitor="train_total_loss",
#  batch_size=9999,# does not have any effect
#  shuffle_set_split=True,
#  plan_kwargs={'protein_vae': protein_vae,
#               'rna_vae': rna_vae,
#               'contrastive_weight': 10.0,
#                 'plot_x_times': 1
#               },
# )

# %%
plot_flag = True
rna_vae.is_trained = protein_vae.is_trained = False
n_epochs =2
batch_size = 1000
# scvi.settings.batch_size = 1

rna_vae.train(
    check_val_every_n_epoch=20,
    max_epochs=n_epochs,
    early_stopping=False,
    accelerator='cuda',
    

    # early_stopping_patience=70,
    # early_stopping_monitor="train_total_loss",
 batch_size=batch_size,# does not have any effect, just for gobal step calculation
    plan_kwargs={'protein_vae': protein_vae,
                 'rna_vae': rna_vae,
                 'batch_size':batch_size,
                 'contrastive_weight': 10.0,
                 'plot_x_times' : 10
                 },
    # logger=logger  # Pass lddogger directly, not within another dictionary

)

# %%

# %%
protein_vae.is_trained = rna_vae.is_trained = True
if 'train_total_loss' not in rna_vae.history.keys():
    raise Exception('make sure you did not run the training twice (in the same cell as the custom training plan)')

# %%
plot_normalized_losses(rna_vae.history)
# plot_aligned_normalized_losses(rna_vae.history)

# %%
'ddddddddddddddddddddddd'

# %%
SCVI_LATENT_KEY = "X_scVI"
rna_vae.module.to(device)
protein_vae.module.to(device)
rna_vae.module.eval()
protein_vae.module.eval()

protein_vae.is_trained = True
with torch.no_grad():
    latent_rna = rna_vae.get_latent_representation()
    latent_prot = protein_vae.get_latent_representation()
    
    # latent_rna = rna_vae.module.inference(
    # torch.tensor(adata_rna_subset.X.todense()), batch_index=1, n_samples=1
    # )["qz"].mean.clone().detach().cpu().numpy()
    # latent_prot = protein_vae.module.inference(
    # torch.tensor(adata_prot_subset.X), batch_index=1, n_samples=1
    # )["qz"].mean.clone().detach().cpu().numpy()
    plot_latent(latent_rna, latent_prot, adata_rna_subset, adata_prot_subset,index=range(len(adata_prot_subset.obs.index)))

adata_rna_subset.obs['CN'] = adata_prot_subset.obs['CN'].values
adata_rna_subset.obsm[SCVI_LATENT_KEY] = latent_rna
adata_prot_subset.obsm[SCVI_LATENT_KEY] = latent_prot
# Set up neighbors and UMAP for RNA and protein subsets
sc.pp.neighbors(adata_rna_subset, key_added='latent_space_neighbors', use_rep=SCVI_LATENT_KEY)
adata_rna_subset.obsm['X_umap_scVI'] = adata_rna_subset.obsm['X_umap']
sc.tl.umap(adata_rna_subset, neighbors_key='latent_space_neighbors')

sc.pp.neighbors(adata_prot_subset, key_added='latent_space_neighbors', use_rep=SCVI_LATENT_KEY)
sc.tl.umap(adata_prot_subset, neighbors_key='latent_space_neighbors')
adata_prot_subset.obsm['X_umap_scVI'] = adata_prot_subset.obsm['X_umap']

# PCA and UMAP for archetype vectors
rna_latent = AnnData(adata_rna_subset.obsm[SCVI_LATENT_KEY].copy())
prot_latent = AnnData(adata_prot_subset.obsm[SCVI_LATENT_KEY].copy())
rna_latent.obs = adata_rna_subset.obs.copy()
prot_latent.obs = adata_prot_subset.obs.copy()


combined_latent = ad.concat([rna_latent, prot_latent], join='outer', label='modality', keys=['RNA', 'Protein'])
combined_major_cell_types=pd.concat((adata_rna_subset.obs['major_cell_types']
,adata_prot_subset.obs['major_cell_types']),join='outer')
combined_latent.obs['major_cell_types']=combined_major_cell_types
# sc.pp.pca(combined_latent)
sc.pp.pca(combined_latent)
sc.pp.neighbors(combined_latent)
sc.tl.umap(combined_latent)

rna_archtype = AnnData(adata_rna_subset.obsm['archetype_vec'])
rna_archtype.obs = adata_rna_subset.obs
sc.pp.neighbors(rna_archtype)
sc.tl.umap(rna_archtype)

prot_archtype = AnnData(adata_prot_subset.obsm['archetype_vec'])
prot_archtype.obs = adata_prot_subset.obs
sc.pp.neighbors(prot_archtype)
sc.tl.umap(prot_archtype)


# %%

# Combine RNA and protein UMAP plots into two side-by-side plots
sc.pl.embedding(adata_rna_subset, color=["CN","cell_types"], basis='X_umap_scVI',
                title=['Latent space, CN RNA', 'Latent space, minor cell types RNA'])
sc.pl.embedding(adata_prot_subset, color=["CN", "cell_types"], basis='X_umap_scVI',
                title=['Latent space, CN Protein', 'Latent space, minor cell types Protein'])

# sc.pl.umap(prot_archtype, color="CN", title='Latent space, CN Protein (Archetype)')
# sc.pl.umap(rna_archtype, color="CN", title='Latent space, CN RNA (Archetype)')

# Combine RNA and Protein latent spaces
sc.tl.umap(combined_latent,min_dist=0.1)
sc.pl.umap(combined_latent, color=['CN', 'modality'],
           title=['UMAP Combined Latent space CN', 'UMAP Combined Latent space modality'],alpha=0.5)
sc.pl.umap(combined_latent, color=['CN', 'modality', 'cell_types'],
           title=['UMAP Combined Latent space CN', 'UMAP Combined Latent space modality',
                  'UMAP Combined Latent space cell types'],alpha=0.5)
sc.pl.pca(combined_latent, color=['CN', 'modality'],
          title=['PCA Combined Latent space CN', 'PCA Combined Latent space modality'],alpha=0.5)

# Analyze distances between modalities in the combined latent space
rna_latent = combined_latent[combined_latent.obs['modality'] == 'RNA']
rna_latent.obs['major_cell_types'] = adata_rna_subset.obs['major_cell_types'].values

prot_latent = combined_latent[combined_latent.obs['modality'] == 'Protein']
distances = np.linalg.norm(rna_latent.X - prot_latent.X, axis=1)

# Randomize RNA latent space to compare distances
rand_rna_latent = rna_latent.copy()
shuffled_indices = np.random.permutation(rand_rna_latent.obs.index)
rand_rna_latent = rand_rna_latent[shuffled_indices].copy()
rand_distances = np.linalg.norm(rand_rna_latent.X - prot_latent.X, axis=1)
# Plot randomized latent space distances
rand_rna_latent.obs['latent_dis'] = np.log(distances)

sc.pl.umap(rand_rna_latent, cmap='coolwarm', color='latent_dis',
           title='Latent space distances between RNA and Protein cells')


compare_distance_distributions(rand_distances, rna_latent,prot_latent,distances)
mixing_score(latent_rna, latent_prot, adata_rna_subset, adata_prot_subset,index = range(len(adata_rna_subset)),plot_flag=True)



# %%
# Identify the top 3 most common cell types
top_3_cell_types = combined_latent.obs['cell_types'].value_counts().index[:3]

# Plot UMAP for each of the top 3 most common cell types separately
for cell_type in top_3_cell_types:
    cell_type_data = combined_latent[combined_latent.obs['cell_types'] == cell_type]
    sc.pl.umap(cell_type_data, color=['CN', 'modality', 'cell_types'],
               title=[f'UMAP {cell_type} CN', f'UMAP {cell_type} modality', f'UMAP {cell_type} cell types'], alpha=0.5)

# %%
adata_prot_subset

# %%
plt.figure(figsize=(12, 6))

# Plot with cell type as color
plt.subplot(1, 3, 1)
sns.scatterplot(x=adata_prot_subset.obs['x_um'], y=adata_prot_subset.obs['y_um'], hue=adata_prot_subset.obs['cell_types'], palette='tab10', s=10)
plt.title('Protein cells colored by cell type')
plt.legend(loc='upper right', fontsize='small', title_fontsize='small')
plt.xlabel('x_um')
plt.ylabel('y_um')


# Add x and y coordinates to RNA data
adata_rna_subset.obs['x_um'] = adata_prot_subset.obs['x_um'].values
adata_rna_subset.obs['y_um'] = adata_prot_subset.obs['y_um'].values


# Plot RNA cells with CN as color
plt.subplot(1, 3, 2)
sns.scatterplot(x=adata_rna_subset.obs['x_um'], y=adata_rna_subset.obs['y_um'],  hue=adata_rna_subset.obs['cell_types'], s=10)
plt.title('RNA cells colored by CN')
plt.xlabel('x_um')
plt.ylabel('y_um')
plt.legend([],[], frameon=False)


# Plot with CN as color
plt.subplot(1, 3, 3)
sns.scatterplot(x=adata_prot_subset.obs['x_um'], y=adata_prot_subset.obs['y_um'], hue=adata_prot_subset.obs['CN'], s=10)
plt.title('Protein cells colored by CN')
plt.xlabel('x_um')
plt.ylabel('y_um')



plt.tight_layout()
plt.show()


# %%
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score

# Normalized Mutual Information between cell types and CN labels
nmi_cell_types_cn = adjusted_mutual_info_score(adata_rna_subset.obs['cell_types'], adata_rna_subset.obs['CN'])
print(f"Normalized Mutual Information between cell types and CN labels RNA : {nmi_cell_types_cn:.3f}")
nmi_cell_types_cn = adjusted_mutual_info_score(adata_prot_subset.obs['cell_types'], adata_prot_subset.obs['CN'])
print(f"Normalized Mutual Information between cell types and CN labels protein: {nmi_cell_types_cn:.3f}")
# Normalized Mutual Information between cell types across modalities
nmi_cell_types_modalities = adjusted_mutual_info_score(adata_rna_subset.obs['cell_types'], adata_prot_subset.obs['cell_types'])
print(f"Normalized Mutual Information between cell types across modalities: {nmi_cell_types_modalities:.3f}")

# adata_rna_subset.obs['cell_types']== adata_prot_subset.obs['cell_types']
mathces = adata_rna_subset.obs['cell_types'].values==adata_prot_subset.obs['cell_types'].values
accuray = mathces.sum()/len(mathces)
accuray

# %%
sc.pp.pca(combined_latent)
# sc.pp.neighbors(combined_latent,n_neighbors=5)
# sc.tl.umap(combined_latent)
# sc.pl.umap(combined_latent, color=['CN', 'modality'])
# sc.pl.pca(combined_latent, color=['CN', 'modality'],)
# sns.heatmap(combined_latent.X)
# plot 3d pca of the combined latent space
sc.pl.pca(combined_latent, color='modality', projection='3d')
# plot pca 3 4 5 of the combined latent space in 3d
sc.pl.pca(combined_latent, color='modality', components=['3,4,5'], projection='3d')
sc.pl.pca(combined_latent, color='modality', components=['1,2,3'], projection='3d')
sc.pl.pca(combined_latent, color='modality', components=['6,7,8'], projection='3d')

# %%
# Save the adata subsets
cwd = os.getcwd()
# Save RNA and protein subsets
clean_uns_for_h5ad(adata_rna_subset)
clean_uns_for_h5ad(adata_prot_subset)
sc.write(
    Path(f'{cwd}/adata_rna_subset_vae.h5ad'),
    adata_rna_subset)
sc.write(
    Path(f'{cwd}/adata_prot_subset_vae.h5ad'),
    adata_prot_subset)


# a=calculate_iLISI(combined_latent,batch_key='modality')
# b =calculate_cLISI(combined_latent,label_key='major_cell_types')
# a,b



# %%

# %%
# plot the CN vector componenst as hue:
# color = adata_prot_subset.var
# cn_cols = adata_prot_subset.var['feature_type']=='CN'
# cn_vec = adata_prot_subset[:,cn_cols.values].X.copy()
# set(adata_prot_subset.var['feature_type'])
# adata_prot_subset.obsm['CN']

# %%
# sc.pp.pca(adata_prot_subset)
# # sc.pl.pca(adata_prot_subset, color=["CN", "cell_types"],title=['Latent space, CN Protein', 'Latent space, minor cell types Protein'])
# basis = 'X_umap_scVI'
# sc.pp.neighbors(adata_prot_subset, use_rep=basis)
# sc.tl.umap(adata_prot_subset)
# sc.pl.umap(adata_prot_subset, color=["CN", "cell_types"],
#            title=['Latent space, CN Protein', 'Latent space, minor cell types Protein'])
# aa = adata_prot_subset.obsm[basis]
# # pca on aa and plot
# pca = PCA(n_components=2)
# pca.fit(aa)
# aa_pca = pca.transform(aa)
# plt.scatter(aa_pca[:, 0], aa_pca[:, 1], c=adata_prot_subset.obs['CN'])
# plt.show()

# %%
# SCVI_LATENT_KEY = "X_scVI"
# latent = model.get_latent_representation()
# adata_rna_subset.obsm[SCVI_LATENT_KEY] = latent
# 
# sc.pp.neighbors(adata_rna_subset, use_rep=SCVI_LATENT_KEY, key_added='latent_space_neighbors')
# sc.tl.umap(adata_rna_subset, neighbors_key='latent_space_neighbors')
# 
# sc.pl.umap(adata_rna_subset, color="major_cell_types", neighbors_key='latent_space_neighbors',
#            title='Latent space, major cell types')
# sc.pl.umap(adata_rna_subset[adata_rna_subset.obs['major_cell_types'] == 'B cells'], color="CN",
#            neighbors_key='latent_space_neighbors',
#            title='Latent space, minor cell types (B-cells only) with observed CN')
# # B cells only adta
# adata_rna_subset_B_cells = adata_rna_subset[adata_rna_subset.obs['major_cell_types'] == 'B cells']
# adata_prot_subset_B_cells = adata_prot_subset[adata_prot_subset.obs['major_cell_types'] == 'B cells']
# sc.pp.neighbors(adata_rna_subset_B_cells, use_rep=SCVI_LATENT_KEY, key_added='latent_space_neighbors')
# sc.pp.neighbors(adata_prot_subset_B_cells, use_rep='X_pca', key_added='original_neighbors')
# sc.tl.umap(adata_rna_subset_B_cells, neighbors_key='latent_space_neighbors')
# sc.tl.umap(adata_prot_subset_B_cells, neighbors_key='original_neighbors')
# 
# sc.pl.umap(adata_rna_subset_B_cells, color="cell_types", neighbors_key='latent_space_neighbors',
#            title='Latent space minor cell types (B-cells only)')
# sc.pl.umap(adata_prot_subset_B_cells, color="cell_types",
#            neighbors_key='original_neighbors', title='Original Latent space MINOR cell types, B cells only')
# 


# %%
# adata_2 = adata_prot_subset[adata_prot_subset.obs['major_cell_types'] == 'B cells']
# sc.pp.pca(adata_2)
# sc.pp.neighbors(adata_2, use_rep='X_pca')
# sc.tl.umap(adata_2)
# sc.pl.umap(adata_2, color='cell_types')


# %%
# silhouette_score_per_cell_type_original = {}
# silhouette_score_per_cell_type_latent = {}
# silhouette_score_per_cell_type = {}
# cell_type_indexes = adata_rna_subset.obs['major_cell_types'] == 'B cells'
# cell_type_data = adata_rna_subset[cell_type_indexes].X
# minor_cell_type_lables = adata_rna_subset[cell_type_indexes].obs['cell_types']
# curr_latent = adata_rna_subset.obsm[SCVI_LATENT_KEY][cell_type_indexes]
# 
# silhouette_score_per_cell_type['original_B cells'] = silhouette_score(cell_type_data, minor_cell_type_lables)
# silhouette_score_per_cell_type['Ours B cells'] = silhouette_score(curr_latent, minor_cell_type_lables)
# adata_rna_subset[cell_type_indexes].obs['cell_types']

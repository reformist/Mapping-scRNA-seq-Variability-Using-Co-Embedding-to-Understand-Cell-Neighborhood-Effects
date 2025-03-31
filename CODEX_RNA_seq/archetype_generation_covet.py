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

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import importlib
import os
import re
import sys
import os
from sklearn.manifold import TSNE

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.metrics import silhouette_score
import torch
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp
import scvi
import seaborn as sns
from anndata import AnnData
from matplotlib import pyplot as plt
from py_pcha import PCHA
from scipy.sparse import issparse
from scipy.spatial.distance import cdist
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sympy.physics.units import current
from tqdm import tqdm
from kneed import KneeLocator
import torch
import scvi
import numpy as np
from scvi.train import TrainingPlan
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import bar_nick_utils
import covet_utils

importlib.reload(bar_nick_utils)
importlib.reload(covet_utils)
from covet_utils import compute_covet

from bar_nick_utils import preprocess_rna, preprocess_protein, plot_archetypes, \
    get_cell_representations_as_archetypes_cvxpy, reorder_rows_to_maximize_diagonal, evaluate_distance_metrics, \
    plot_archetypes_matching, compare_matchings, find_best_pair_by_row_matching, add_spatial_data_to_prot, \
    clean_uns_for_h5ad, get_latest_file,get_latest_file,plot_latent_single

plot_flag = True
# computationally figure out which ones are best
np.random.seed(8)


# %%
# ### reading in data


# %%
file_prefixes = ['preprocessed_adata_rna_', 'preprocessed_adata_prot_']
folder = 'CODEX_RNA_seq/data/'

latest_files = {prefix: get_latest_file(folder, prefix) for prefix in file_prefixes}
adata_1_rna = sc.read(latest_files['preprocessed_adata_rna_'])
adata_2_prot = sc.read(latest_files['preprocessed_adata_prot_'])

num_rna_cells = 6000
num_protein_cells = 200000
# num_rna_cells = num_protein_cells= 2000
subsample_n_obs_rna = min(adata_1_rna.shape[0], num_rna_cells)
subsample_n_obs_protein = min(adata_2_prot.shape[0], num_protein_cells)
sc.pp.subsample(adata_1_rna, n_obs=subsample_n_obs_rna)
sc.pp.subsample(adata_2_prot, n_obs=subsample_n_obs_protein)


original_protein_num = adata_2_prot.X.shape[1]
sc.pp.neighbors(adata_2_prot, use_rep='spatial_location', key_added='spatial_neighbors', n_neighbors=15)

distances = adata_2_prot.obsp['spatial_neighbors_distances'].data
log_transformed_distances = (distances + 1)  # since the distances are not normally distributed, log-transform them

sns.histplot(log_transformed_distances[log_transformed_distances > 0])
plt.title('Distribution of distances between spatial neighbors before applying cutoff')
plt.show()
distances_mean = log_transformed_distances.mean()
distances_std = log_transformed_distances.std()

one_std_dev = distances_mean + distances_std
# Zero out neighbors in the connectivity matrix and remove them from the distances matrix if their distance is more than 2 standard deviations above the mean
two_std_dev = distances_mean + 2 * distances_std

# Get the indices of the neighbors to be zeroed out
indices_to_zero_out = np.where(adata_2_prot.obsp['spatial_neighbors_distances'].data > two_std_dev)[0]

# Zero out the corresponding entries in the connectivity matrix
adata_2_prot.obsp['spatial_neighbors_connectivities'].data[indices_to_zero_out] = 0
adata_2_prot.obsp['spatial_neighbors_distances'].data[indices_to_zero_out] = 0
# Recompute the connectivity matrix
non_zero_distances = adata_2_prot.obsp['spatial_neighbors_distances'].data[
    adata_2_prot.obsp['spatial_neighbors_distances'].data > 0
]
sns.histplot(non_zero_distances)
plt.title('Distribution of distances between spatial neighbors after applying cutoff')
plt.show()
# todo  I am using the data as if its notmarl and its not, so I need to see how to remove the tail in a different way



# %%
adata_2_prot

# %%

# %%
exp_adata_2_prot= adata_2_prot.copy()
exp_adata_2_prot.X = np.exp(adata_2_prot.X)
adata_2_prot.obsm['COVET'], adata_2_prot.obsm['COVET_SQRT'], adata_2_prot.uns['CovGenes'] = compute_covet(exp_adata_2_prot, k=21, spatial_key='spatial')
adata_2_prot.obsm['COVET_SQRT'] = adata_2_prot.obsm['COVET_SQRT'].reshape(adata_2_prot.obsm['COVET_SQRT'].shape[0], -1)
conv_sqrt = AnnData(adata_2_prot.obsm['COVET_SQRT'])
conv_sqrt.obs = adata_2_prot.obs
sc.pp.pca(conv_sqrt, n_comps=10)
sc.pp.neighbors(conv_sqrt, n_neighbors=15, use_rep='X_pca', key_added='pca_neighbors')
sc.tl.leiden(conv_sqrt, resolution=0.15, key_added='CN', neighbors_key='pca_neighbors')
sc.tl.umap(conv_sqrt,neighbors_key='pca_neighbors')

adata_2_prot.obs['CN'] = conv_sqrt.obs['CN']

sc.tl.pca(adata_2_prot, n_comps=10)
sc.pp.neighbors(adata_2_prot, n_neighbors=15, use_rep='X_pca', key_added='pca_neighbors')
sc.tl.umap(adata_2_prot, min_dist=0.5, spread=1, neighbors_key='pca_neighbors')


sc.pl.umap(conv_sqrt, color='CN',title='COVET_SQRT - CN as color')
sc.pl.umap(adata_2_prot, color=['cell_type', 'CN'])
sc.pl.scatter(adata_2_prot, x='X', y='Y', color=['CN', 'cell_type'],title=['spatial and CN as color', 'spatial and cell type as color'])
# del conv_sqrt


sil_score = silhouette_score(adata_2_prot.obsm['X_pca'], adata_2_prot.obs['CN'].astype(int))
print(f"Silhouette Score: {sil_score}")

assert sil_score < 0.5, "Silhouette score is too high the CN labels should not depend on the spatial location"



# %%
# Perform PCA on the combined data
sc.pp.normalize_total(conv_sqrt)
conv_sqrt.X = conv_sqrt.X - conv_sqrt.X.min()

sc.pp.pca(conv_sqrt, n_comps=50)

# Compute the neighborhood graph using PCA representation
sc.pp.neighbors(conv_sqrt, use_rep='X_pca')

# Run UMAP for visualization
sc.tl.umap(conv_sqrt)
sc.pl.pca(conv_sqrt, color=['cell_types'], title=['PCA - Cell Types', 'PCA - Original Protein Clusters'])

# Plot the UMAP embedding
sc.pl.umap(conv_sqrt, color=['cell_types'], title=['UMAP - Cell Types', 'UMAP - Original Protein Clusters'])


# %%
assert sil_score < 0.5

# %%
# normalized_data = zscore(neighbor_means, axis=0) # each cell we get the mean of the features of its neighbors
# inertias = []
# silhouette_scores = []
# min_cn_lbls = 4
# k_range = range(min_cn_lbls, 11)  # You can adjust this range as needed
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     kmeans.fit(normalized_data)
#     inertias.append(kmeans.inertia_)sns.heatmap(adata_2_prot.obsp kmeans.labels_)
#     silhouette_scores.append(score)
# # num_clusters = min_cn_lbls + np.diff(inertias, 2).argmin() 
# num_clusters = k_range[np.argmax(silhouette_scores)] 
# inertias =np.array(inertias)
# silhouette_scores = np.array(silhouette_scores)
# # norm in the range 0-1
# silhouette_scores = (silhouette_scores - silhouette_scores.min()) / (silhouette_scores.max() - silhouette_scores.min())
# inertias = (inertias - inertias.min()) / (inertias.max() - inertias.min())
# # Plot the elbow curve
# plt.figure(figsize=(10, 6))
# plt.plot(k_range,inertias, 'bo-', label='norm Inertia')
# plt.plot(k_range, silhouette_scores, 'bo-', color='red', label='norm Silhouette Score')
# plt.legend()
# plt.xlabel('Number of Clusters (k)')
# plt.xticks(k_range)
# plt.ylabel('Inertia')
# plt.title('Elbow Method for Optimal k')
# plt.grid(True)
# plt.axvline(x=num_clusters, color='red', linestyle='dashed')
# plt.show()
# kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(normalized_data)
# final_labels = np.array(kmeans.labels_)
# if 'CN' in adata_2_prot.obs:
#     adata_2_prot.obs.drop(columns=['CN'], inplace=True, errors='ignore')
#     # adata_1_rna.obs.drop(columns=['CN'], inplace=True, errors='ignore')
# adata_2_prot.obs['CN'] = pd.Categorical(final_labels)
# # adata_1_rna.obs['CN'] = pd.Categorical(final_labels)
# num_clusters = len(adata_2_prot.obs['CN'].unique())
# palette = sns.color_palette("tab10", num_clusters)  # "tab10" is a good color map, you can choose others too
# adata_2_prot.uns['spatial_clusters_colors'] = palette.as_hex()  # Save as hex color codes


# %%
if plot_flag:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(adata_2_prot.obsm[
                    'COVET_SQRT'])  # less noisy because we mean, makes sense as we made sure each cell is of same cell type
    plt.title('convet sqrt')
    plt.subplot(1, 2, 2)
    sns.heatmap(adata_2_prot.X.todense() if issparse(adata_2_prot.X) else adata_2_prot.X)
    plt.title('Proteins expressions of each cell')
    plt.show()


# %%
# (different adata technique)
if plot_flag:
    # merge the CN to the protein data
    fig, ax = plt.subplots()

    sc.pl.scatter(
        adata_2_prot,
        x='X', y='Y',
        color='CN',  # got this CN from kmeans
        title='spatial location with CN as color',
        ax=ax,  # Use the ax created above
        show=False  # Prevent scanpy from showing the plot immediately
    )
    # for x in horizontal_splits[1:-1]:  # Exclude the edges to avoid border doubling
    #     ax.axvline(x=x, color='black', linestyle='--')
    # for y in vertical_splits[1:-1]:  # Exclude the edges to avoid border doubling
    #     ax.axhline(y=y, color='black', linestyle='--')
    # plt.show()\
    # Run PCA on adata_2_prot.obsm['COVET_SQRT'] and keep 0.8 variance
covet_adata = anndata.AnnData(adata_2_prot.obsm['COVET_SQRT'])

sc.pp.pca(covet_adata, n_comps=50)
cumulative_variance_ratio = np.cumsum(covet_adata.uns['pca']['variance_ratio'])
n_comps_thresh = np.argmax(cumulative_variance_ratio >= 0.5) + 1
covet_adata = AnnData(covet_adata.obsm['X_pca'][:, :n_comps_thresh])
# covet_adata.obsm['X_pca'] = covet_adata.obsm['X_pca'][:, :n_comps_thresh]

print(f"Number of components explaining 0.5 variance: {n_comps_thresh}")

if plot_flag:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(covet_adata.X)  # less noisy because we mean, makes sense as we made sure each cell is of same cell type
    plt.title('convet sqrt')
    plt.subplot(1, 2, 2)
    sns.heatmap(adata_2_prot.X.todense() if issparse(adata_2_prot.X) else adata_2_prot.X)
    plt.title('Proteins expressions of each cell')
    plt.show()

# rna no longer has the CN, using the protein
covet_adata.obs['CN'] = pd.Categorical(adata_2_prot.obs['CN'])
sc.pp.pca(covet_adata)
sc.pp.neighbors(covet_adata)
sc.tl.umap(covet_adata)
sc.pl.umap(covet_adata, color='CN', title='UMAP of CN embedding')
#  pop obsm from both
# adata_2_prot.obsm.pop('COVET')
# adata_2_prot.obsm.pop('COVET_SQRT')
# covet_adata.obsm.pop('X_pca')
# making sure the CN and the protein are distinct
# adata_prot_cn_concat = anndata.concat([adata_2_prot, covet_adata], join='outer', label='modality',
#                                       keys=['Protein', 'CN'])
# X = adata_prot_cn_concat.X.toarray() if issparse(adata_prot_cn_concat.X) else adata_prot_cn_concat.X
# X = np.nan_to_num(X)
# adata_prot_cn_concat.X = X
# sc.pp.pca(adata_prot_cn_concat)
# sc.pp.neighbors(adata_prot_cn_concat)
# if plot_flag:
#     sc.tl.umap(adata_prot_cn_concat)
#     sc.pl.umap(adata_prot_cn_concat, color=['CN', 'modality'],
#                title=['UMAP of CN embedding to make sure they are not mixed',
#                       'UMAP of CN embedding to make sure they are not mixed'])
#     sc.pl.pca(adata_prot_cn_concat, color=['CN', 'modality'],
#               title=['PCA of CN embedding to make sure they are not mixed',
#                      'PCA of CN embedding to make sure they are not mixed'])


# %%
import scanpy as sc

# Compute neighborhood graph
sc.pp.neighbors(adata_2_prot, n_neighbors=15, n_pcs=40)

# Run Leiden clustering
sc.tl.leiden(adata_2_prot, resolution=0.5, key_added='orig_prot_clusters')

# Get cluster labels
cluster_labels = adata_2_prot.obs['orig_prot_clusters'].cat.codes.values
# Create a DataFrame for plotting
adata_2_prot.obs['orig_prot_clusters'] = pd.Categorical(cluster_labels)
sc.tl.umap(adata_2_prot)
sc.pl.umap(adata_2_prot, color='orig_prot_clusters', title='Leiden clustering of protein data')


# %%
a = PCA(n_components=100).fit(adata_2_prot.obsm['COVET_SQRT'])
# pring variance explained
print(np.cumsum(a.explained_variance_ratio_))
# sns.heatmap(adata_2_prot.obsm['COVET_SQRT'])
# norm cols
adata_2_prot.obsm['COVET_SQRT'] = zscore(adata_2_prot.obsm['COVET_SQRT'], axis=0)
sns.heatmap(adata_2_prot.obsm['COVET_SQRT'])


# %%

class ClusterPreservingTrainingPlan(TrainingPlan):
    def __init__(self, module, **kwargs):
        self.protein_dim = kwargs.pop('protein_dim', 30)
        self.protein_vae = kwargs.pop('protein_vae', None)
        self.cluster_weight = kwargs.pop('cluster_weight', 1.0)
        self.initial_cluster_weight = self.cluster_weight
        self.cluster_labels = kwargs.pop('cluster_labels', None)
        self.batch_size = kwargs.pop('batch_size', 1000)
        self.plot_x_times = kwargs.pop('plot_x_times', 10)
        self.combined_adata = kwargs.pop('combined_adata', None)
        
        # Parameters for silhouette-based weight adjustment
        self.target_silhouette = kwargs.pop('target_silhouette', 0.5)
        self.silhouette_tolerance = kwargs.pop('silhouette_tolerance', 0.1)
        self.min_cluster_weight = kwargs.pop('min_cluster_weight', 0.1)
        self.max_cluster_weight = kwargs.pop('max_cluster_weight', 10.0)
        
        super().__init__(module, **kwargs)
        
        # Initialize tracking variables for silhouette calculation
        self.initial_silhouette = None
        self.all_latent = None
        self.all_labels = None
        self.last_epoch_checked = -1

        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.protein_vae.module.parameters(),
            lr=0.001,
            weight_decay=1e-5,
        )
        d = { # maybe add this?
        "optimizer": optimizer,
        "gradient_clip_val": 1.0,  # Critical for stability
        "gradient_clip_algorithm": "value"
        }
        return d
    def training_step(self, batch, batch_idx):
        # Initialize steps if needed
        if not hasattr(self, 'total_steps'):
            n_samples = len(self.combined_adata)
            steps_per_epoch = int(np.ceil(n_samples / self.batch_size))
            self.total_steps = steps_per_epoch * self.trainer.max_epochs
            self.plot_interval = max(1, (steps_per_epoch * self.trainer.max_epochs) // self.plot_x_times)
        self.loss_kwargs['kl_weight'] = 0.01
        # Standard ELBO loss calculation
        _, _, loss_output = self.module(batch, loss_kwargs=self.loss_kwargs)

        elbo_loss = loss_output.loss
        
        # Get latent representation
        x = batch["X"]
        inference_outputs = self.module.inference(
            x, batch_index=batch["batch"], n_samples=1
        )
        z = inference_outputs["qz"].loc
        
        # Calculate cluster preservation loss
        self.cluster_labels = self.cluster_labels.to(z.device)
        cluster_loss = triplet_cluster_loss(z, self.cluster_labels[batch["labels"]])

        # Adjust cluster weight based on silhouette score (periodically)
        if self.trainer.current_epoch >= 5 and batch_idx % 2 == 0:
            # Calculate silhouette score on current batch only
            current_silhouette = self.calculate_batch_silhouette(z, batch["labels"])
            self.log("batch_silhouette", current_silhouette)
            
            # Adjust weight based on comparison with target
            if current_silhouette < self.target_silhouette - self.silhouette_tolerance:
                # Silhouette below target - increase weight (more gradually than before)
                new_weight = min(self.cluster_weight * 1.2, self.max_cluster_weight)
                self.cluster_weight = new_weight
                
            elif current_silhouette > self.target_silhouette + self.silhouette_tolerance:
                # Silhouette above target - decrease weight
                new_weight = max(self.cluster_weight / 1.2, self.min_cluster_weight)
                self.cluster_weight = new_weight
            
            self.log("cluster_weight", self.cluster_weight)

        # Use the dynamic cluster weight
        cluster_loss = self.cluster_weight * cluster_loss
        dynamic_range = torch.max(z,dim=0)[0] -torch.min(z,dim=0)[0]
        if torch.mean(dynamic_range) < 0.5:
            dynamic_range_loss = -10000 *torch.log(torch.mean(dynamic_range))  # Encourage larger dynamic range
        else:
            dynamic_range_loss = 0
        total_loss = elbo_loss + cluster_loss +dynamic_range_loss
        
        # Log losses
        self.log("train_elbo_loss", elbo_loss, on_epoch=False, on_step=True)
        self.log("train_cluster_loss", cluster_loss, on_epoch=False, on_step=True)
        self.log("train_total_loss", total_loss, on_epoch=False, on_step=True)
        self.log("cluster_weight", self.cluster_weight, on_epoch=False, on_step=True)
        self.log("dynamic_range_loss", dynamic_range_loss, on_epoch=False, on_step=True)
        
        # Plot periodically
        if self.global_step > -1 and self.global_step % self.plot_interval == 0:
            # print all losses vals:
            print(f"total_loss: {total_loss}, elbo_loss: {elbo_loss}, \ncluster_loss: {cluster_loss}, dynamic_range_loss: {dynamic_range_loss}")
            print(f'dynamic_range: {torch.mean(dynamic_range)}')
            plot_latent_single(z, self.combined_adata, batch["labels"], 
                            color_label='orig_prot_clusters', 
                            title=f"precentage {int(100*(self.global_step/self.total_steps))}")
        
        return total_loss


    def on_validation_epoch_end(self):
        # This runs exactly once per epoch after all validation batches
        # Concatenate all stored latent representations
        if self.trainer.current_epoch % max(1, self.trainer.max_epochs // self.plot_x_times) == 0:

            if hasattr(self, 'validation_z') and len(self.validation_z) > 0:
                all_z = torch.cat(self.validation_z, dim=0)
                all_labels = torch.cat(self.validation_labels, dim=0)
                
                # Plot
                plot_latent_single(all_z, self.combined_adata, all_labels, 
                                color_label='orig_prot_clusters',
                                title=f"_val_epoch_{self.trainer.current_epoch}")
                
                # Clear stored data
                self.validation_z = []
                self.validation_labels = []

    def calculate_batch_silhouette(self, z, batch_labels):
        """Calculate silhouette score on the current batch"""
        
        # Get latent representation and labels
        latent = z.detach().cpu().numpy()
        labels = self.cluster_labels[batch_labels].cpu().numpy()
        
        # Need at least 2 clusters and enough samples per cluster
        unique_labels = np.unique(labels)
        
        if len(unique_labels) < 2:
            return 0
        
        # Check if we have at least 2 samples per cluster
        valid_clusters = True
        for label in unique_labels:
            if np.sum(labels == label) < 2:
                valid_clusters = False
                break
        
        if not valid_clusters:
            return 0
            
        try:
            # norm between 0 and 1 to avoid the latent space from just using large distances instead of distinct clusters

            latent = (latent - latent.min(axis=0)) / (latent.max(axis=0) - latent.min(axis=0) + 1e-8)
            score = silhouette_score(latent, labels.ravel())
            # Add the size of the dynamic range as a loss
            
            return score 
        except Exception as e:
            print(f"Error calculating batch silhouette: {e}")
            return 0

    def validation_step(self, batch, batch_idx):
        # Standard validation logic (calculate losses, etc.)
        _, _, loss_output = self.module(batch, loss_kwargs=self.loss_kwargs)
        elbo_loss = loss_output.loss
        
        # Get latent representation
        x = batch["X"]
        inference_outputs = self.module.inference(
            x, batch_index=batch["batch"], n_samples=1
        )
        z = inference_outputs["qz"].loc
        
        # Calculate cluster loss
        self.cluster_labels = self.cluster_labels.to(z.device)
        cluster_loss = improved_triplet_cluster_loss(z, self.cluster_labels[batch["labels"]])        


        total_loss = elbo_loss + self.cluster_weight * cluster_loss
        
        # Log metrics
        self.log("val_elbo_loss", elbo_loss, on_epoch=True, on_step=False)
        self.log("val_cluster_loss", cluster_loss, on_epoch=True, on_step=False)
        self.log("val_total_loss", total_loss, on_epoch=True, on_step=False)
        
        # Store the latent representation for later use in on_validation_epoch_end
        if not hasattr(self, 'validation_z'):
            self.validation_z = []
            self.validation_labels = []
        
        self.validation_z.append(z.detach().cpu())
        self.validation_labels.append(batch["labels"].detach().cpu())
        
        return total_loss
    
    def plot_latent_space(self, z, labels, cell_types, step):
        plt.figure(figsize=(12, 6))
        
        # TSNE for visualization
        tsne = TSNE(n_components=2, random_state=42)
        z_tsne = tsne.fit_transform(z.detach().cpu().numpy())
        
        # Plot cluster labels
        plt.subplot(121)
        scatter = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=labels.cpu().numpy(), cmap='tab20', s=5)
        plt.colorbar(scatter)
        plt.title(f'Latent Space - Clusters (Step {step})')
        
        # Plot cell types
        plt.subplot(122)
        scatter = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=cell_types, cmap='tab20', s=5)
        plt.colorbar(scatter)
        plt.title(f'Latent Space - Cell Types (Step {step})')
        
        plt.tight_layout()
        plt.savefig(f'latent_space_step_{step}.png')
        plt.close()
def improved_triplet_cluster_loss(z, labels, margin=1.0, var_weight=0.1, use_semi_hard=True):
    """
    Improved triplet loss with semi-hard negative mining and variance regularization
    """
    # Compute pairwise distances
    pairwise_dist = torch.cdist(z, z, p=2)
    labels = labels.squeeze()
    # Create same-cluster and different-cluster masks
    labels_expanded_i = labels.unsqueeze(1)
    labels_expanded_j = labels.unsqueeze(0)
    same_cluster_mask = (labels_expanded_i == labels_expanded_j)
    identity_mask = torch.eye(z.size(0), device=z.device, dtype=torch.bool)
    
    # Find valid positives (same cluster, not self)
    positive_mask = same_cluster_mask & ~identity_mask
    
    # Find valid negatives (different cluster)
    negative_mask = ~same_cluster_mask
    
    # Compute mean positive distance (average distance to same cluster)
    mean_pos_dist = torch.sum(pairwise_dist * positive_mask.float(), dim=1) / (positive_mask.sum(dim=1) + 1e-8)
    
    # Calculate loss with semi-hard negative mining or standard hard mining
    triplet_losses = torch.zeros_like(mean_pos_dist)
    valid_anchors = torch.zeros_like(mean_pos_dist, dtype=torch.bool)
    
    for i in range(z.size(0)):
        pos_dist = mean_pos_dist[i]
        
        # Skip if no positives for this anchor
        if positive_mask[i].sum() == 0:
            continue
            
        neg_dists = pairwise_dist[i][negative_mask[i]]
        
        # Skip if no negatives for this anchor
        if neg_dists.shape[0] == 0:
            continue
            
        valid_anchors[i] = True
        
        if use_semi_hard:
            # Semi-hard negative: negatives that are further than the positive
            # but still within margin (pos_dist < neg_dist < pos_dist + margin)
            semi_hard_mask = (neg_dists > pos_dist) & (neg_dists < pos_dist + margin)
            
            if semi_hard_mask.any():
                # Use closest semi-hard negative
                semi_hard_negs = neg_dists[semi_hard_mask]
                closest_semi_hard = semi_hard_negs.min()
                triplet_losses[i] = torch.clamp(pos_dist - closest_semi_hard + margin, min=0.0)
            else:
                # If no semi-hard negatives, use closest negative
                triplet_losses[i] = torch.clamp(pos_dist - neg_dists.min() + margin, min=0.0)
        else:
            # Hard negative mining (original approach)
            triplet_losses[i] = torch.clamp(pos_dist - neg_dists.min() + margin, min=0.0)
    
    # Add variance regularization to encourage spread
    # Calculate variance of each dimension of z
    z_var = torch.var(z, dim=0).mean()
    # We want to maximize variance (spread), so we minimize negative variance
    var_loss = -var_weight * z_var
    
    # Only use valid triplets in final loss
    if valid_anchors.any():
        triplet_loss = triplet_losses[valid_anchors].mean()
        return triplet_loss + var_loss
    else:
        return var_loss  # At least encourage spread if no valid triplets

def triplet_cluster_loss(z, labels, margin=1.0):
    """
    Triplet loss for cluster preservation that works with backpropagation
    """
    labels = labels.squeeze()
    # Compute pairwise distances
    pairwise_dist = torch.cdist(z, z, p=2)
    
    # Create same-cluster mask (should be shape [batch_size, batch_size])
    labels_expanded_i = labels.unsqueeze(1)  # [batch_size, 1]
    labels_expanded_j = labels.unsqueeze(0)  # [1, batch_size]
    same_cluster_mask = (labels_expanded_i == labels_expanded_j)  # [batch_size, batch_size]
    
    # Create identity mask to exclude self-pairs
    batch_size = z.size(0)
    identity_mask = torch.eye(batch_size, device=z.device, dtype=torch.bool)
    
    # Find valid positives (same cluster, not self)
    positive_mask = same_cluster_mask & ~identity_mask
    
    # Find valid negatives (different cluster)
    negative_mask = ~same_cluster_mask
    
    # Compute valid triplets mask - anchors that have at least one positive and one negative
    valid_triplets = (positive_mask.sum(dim=1) > 0) & (negative_mask.sum(dim=1) > 0)
    
    if not valid_triplets.any():
        return torch.tensor(0.0, device=z.device)
    
    # Set masked values for efficient max/min calculations
    # For positives: replace non-positives with -inf so they don't affect max
    masked_dist_pos = pairwise_dist.clone()
    masked_dist_pos[~positive_mask] = torch.nan
    
    # For negatives: replace non-negatives with inf so they don't affect min
    masked_dist_neg = pairwise_dist.clone()
    masked_dist_neg[~negative_mask] = torch.nan
    
    # Compute hardest positives and negatives efficiently
    # remoevc nan and inf etc from the tensor
     
    hardest_positives = torch.nanquantile(masked_dist_pos, 0.9, dim=1)  # 90th percentile distance to positive
    hardest_negatives = torch.nanquantile(masked_dist_neg, 0.1, dim=1)  # 10th percentile distance to negative
    
    # Compute triplet loss with margin
    triplet_losses = torch.clamp(hardest_positives - hardest_negatives + margin, min=0.0)
    
    # Only use valid triplets in final loss
    return triplet_losses[valid_triplets].mean()
# def compute_cluster_loss(z, labels, k=15):
#     # Compute k-nearest neighbors graph in latent space
#     knn_graph = kneighbors_graph(z.detach().cpu().numpy(), k, mode='connectivity', include_self=False)
#     knn_graph = torch.from_numpy(knn_graph.toarray()).float().to(z.device)
    
#     # Compute cluster assignment matrix
#     unique_labels = torch.unique(labels)
#     cluster_assignment = torch.zeros(len(labels), len(unique_labels), device=z.device)
#     for i, label in enumerate(unique_labels):
#         cluster_assignment[:, i] = (labels == label).float().squeeze()
#     # Compute within-cluster connectivity
#     within_cluster_conn = torch.sum(knn_graph * torch.mm(cluster_assignment, cluster_assignment.t()))
#     # Compute between-cluster connectivity
#     between_cluster_conn = torch.sum(knn_graph * (1 - torch.mm(cluster_assignment, cluster_assignment.t())))
#     # Compute loss (maximize within-cluster connectivity, minimize between-cluster connectivity)
#     loss = -within_cluster_conn + between_cluster_conn
    
# 1. Set up your AnnData object with combined data
COVET_SQRT = PCA(n_components=100).fit_transform(adata_2_prot.obsm['COVET_SQRT'])
combined_data = np.concatenate([adata_2_prot.X.todense() if issparse(adata_2_prot.X) else adata_2_prot.X,
                                 COVET_SQRT], axis=1)

# combined_data = adata_2_prot
combined_adata = AnnData(X=combined_data, obs=adata_2_prot.obs)
combined_adata.obs['index_col'] = range(combined_adata.shape[0])  
# scale all cols
# sc.pp.scale(combined_adata)
combined_adata.X = combined_adata.X - combined_adata.X.min()
# sc.pp.normalize_total(combined_adata)
combined_adata.X = np.abs(combined_adata.X)
# 2. Set up scVI
scvi.model.SCVI.setup_anndata(
    combined_adata,
    labels_key="index_col",
)


# 3. Create a standard scVI model with smaller architecture
protein_vae = scvi.model.SCVI(
    combined_adata,
    n_hidden=50,  # Smaller hidden layers
    n_latent=10,  # Latent space dimensionality
    n_layers=1,   # Single layer for simplicity
    gene_likelihood="normal",  # Use normal distribution for non-count data
    use_layer_norm=True,  # Add layer normalization
    use_batch_norm=True,  # Add batch normalization

)

protein_vae._training_plan_cls = ClusterPreservingTrainingPlan
num_epochs = 200
batch_size = 1200

sc.pp.pca( adata_2_prot,n_comps=10)
target_silhouette = silhouette_avg = silhouette_score(adata_2_prot.obsm['X_pca'], adata_2_prot.obs['orig_prot_clusters'].astype(int))

print(f"Silhouette score of the original protein clustering: {target_silhouette}")

protein_vae.train(
    max_epochs=num_epochs,
    check_val_every_n_epoch=10,
    early_stopping=False,
    batch_size=batch_size,
    plan_kwargs={
        'protein_vae': protein_vae,
        'cluster_labels': torch.tensor(combined_adata.obs['orig_prot_clusters'].values).long(),
        'batch_size': batch_size,
        'plot_x_times': 20,
        'combined_adata': combined_adata,
        'target_silhouette': 2*target_silhouette,  # Target silhouette score
        'silhouette_tolerance': 0.05,  # Acceptable range around target
        'cluster_weight': 200.0,  # Start with a moderate weight
        'min_cluster_weight': 100,  # Minimum allowed weight
        'max_cluster_weight': 2000.0   # Maximum allowed weight
    }
)

# %%
# Perform PCA on the combined data
sc.pp.scale(combined_adata)
combined_adata.X = combined_adata.X - combined_adata.X.min()

sc.pp.pca(combined_adata, n_comps=50)

# Compute the neighborhood graph using PCA representation
sc.pp.neighbors(combined_adata, use_rep='X_pca')

# Run UMAP for visualization
sc.tl.umap(combined_adata)

# Plot the UMAP embedding
sc.pl.umap(combined_adata, color=['cell_types', 'orig_prot_clusters'], title=['UMAP - Cell Types', 'UMAP - Original Protein Clusters'])

# %% vscode={"languageId": "r"}
# Perform PCA on the combined data
sc.pp.scale(combined_adata)
combined_adata.X = combined_adata.X - combined_adata.X.min()

sc.pp.pca(combined_adata, n_comps=50)

# Compute the neighborhood graph using PCA representation
sc.pp.neighbors(combined_adata, use_rep='X_pca')

# Run UMAP for visualization
sc.tl.umap(combined_adata)

# Plot the UMAP embedding
sc.pl.umap(combined_adata, color=['cell_types', 'orig_prot_clusters'], title=['UMAP - Cell Types', 'UMAP - Original Protein Clusters'])

# %%
from bar_nick_utils import plot_normalized_losses


plot_normalized_losses(protein_vae.history)


# %%
combined_adata,adata_2_prot

# %%


# Assuming the model is trained and latent representation is extracted
latent_prot = protein_vae.get_latent_representation()
# Add latent representation to AnnData object
combined_adata.obsm['X_scVI'] = latent_prot
# Compute neighborhood graph
sc.pp.neighbors(combined_adata, use_rep='X_scVI',key_added='scVI_neighbors')
sc.tl.umap(combined_adata,neighbors_key='scVI_neighbors')
sc.pl.umap(combined_adata, color=['cell_types','orig_prot_clusters'], title=['latent - Cell Types', 'latent - original prot Clustering'])
# sc.pl.umap(combined_adata, color=combined_adata.var_names[:5], ncols=3, title='UMAP of Protein Latent Space - Top 5 Features')

plt.tight_layout()
plt.show()
sc.pp.pca(adata_2_prot, n_comps=10)
sc.pp.neighbors(adata_2_prot, use_rep='X_pca')
sc.pl.umap(adata_2_prot, color=['cell_types', 'orig_prot_clusters'], title=['UMAP of Original Protein Data - Cell Types', 
                                                                            'UMAP of Original Protein Data - Original Protein Clustering'],
                                                                            neighbors_key='neighbors')
# Identify the largest cluster in 'orig_prot_clusters'
largest_cluster = adata_2_prot.obs['orig_prot_clusters'].value_counts().idxmax()

# Subset the data to include only the largest cluster
adata_largest_cluster = adata_2_prot[adata_2_prot.obs['orig_prot_clusters'] == largest_cluster]

# Plot UMAP for the largest cluster, colored by 'cell_types'
sc.pl.umap(adata_largest_cluster, color='cell_types', title=f'Largest Cluster ({largest_cluster}) - Colored by Cell Types')


# %%


silhouette_avg = silhouette_score(combined_adata.obsm['X_scVI'],
                                  combined_adata.obs['orig_prot_clusters'].astype(int))
print(f"Silhouette Score: {silhouette_avg}")
'a'+1

# %%
adata_2_prot.layers["original_data"] = adata_2_prot.X.copy()
adata_2_prot = AnnData(combined_adata.obsm['X_scVI'], obs=combined_adata.obs)


# %%
#this is the old way of combineing the protein expressin and the COVET, no longer needed
# Assuming `adata_prot` is the original AnnData object
# And `neighbor_means` is the new matrix to be concatenated
# different adata
# new_feature_names = [f"CN_{i}" for i in range(covet_adata.shape[1])]
# covet_adata_dense = covet_adata.X.toarray() if issparse(covet_adata.X) else covet_adata.X
# # sc.pp.pca(covet_adata, n_comps=20)
# adata_2_prot.X = adata_2_prot.X.toarray() if issparse(adata_2_prot.X) else adata_2_prot.X
# new_X = np.hstack([adata_2_prot.obsm['X_pca']
#                    , covet_adata_dense])
# additional_var = pd.DataFrame(index=new_feature_names)
# prot_vars_pca =   [f"_{i}" for i in range(adata_2_prot.obsm['X_pca'].shape[1])]
# prot_vars_pca = pd.DataFrame(index=prot_vars_pca)
# new_vars = pd.concat([prot_vars_pca, additional_var])
# new_vars['higly_variable'] = True
# new_vars['n_cells'] = covet_adata.shape[0]
# adata_2_prot_new = anndata.AnnData(
#     X=new_X,
#     obs=adata_2_prot.obs.copy(),  # Keep the same observation metadata
#     var=new_vars,  # Keep the same variable metadata
#     uns=adata_2_prot.uns.copy(),  # Keep the same unstructured data #todo brin back?
#     obsm=adata_2_prot.obsm.copy(),  # Keep the same observation matrices ? #todo bring back?
#     # varm=adata_prot.varm.copy(), # Keep the same variable matrices
#     # layers=adata_2_prot.layers.copy()  # Keep the same layers
# )
# prot_feat_num = prot_vars_pca.shape[0]
# adata_2_prot_new.var['feature_type'] = ['protein'] * prot_feat_num + ['CN'] * covet_adata.shape[1]
# # plot pca variance using scanpy
# # remove pca
# if 'X_pca' in adata_2_prot_new.obsm:
#     adata_2_prot_new.obsm.pop('X_pca')
# #  set new highly variable genes
# adata_2_prot_new.var['highly_variable'] = True
# # adata_2_prot_new.obs['CN'] = adata_2_prot.obs['CN']
# adata_2_prot_new.X = (zscore(adata_2_prot_new.X, axis=0))
# sc.pp.highly_variable_genes(adata_2_prot_new)
# sc.pp.pca(adata_2_prot_new)
# sc.pp.neighbors(adata_2_prot_new, n_neighbors=15)
# sc.tl.umap(adata_2_prot_new)
# sc.pl.umap(adata_2_prot_new, color='CN', title='UMAP of CN embedding to make sure they are not mixed')
# # todo make sure no more things to remove
# new_pca = adata_2_prot_new.obsm['X_pca']
# sns.heatmap(adata_2_prot_new.X)
# sc.pl.pca_variance_ratio(adata_2_prot_new, log=True, n_pcs=50, save='.pdf')
# sc.pp.pca(adata_2_prot_new)
# if 'adata_2_prot_old' not in globals():
#     adata_2_prot_old = adata_2_prot.copy()
# print(f"New adata shape (protein features + cell neighborhood vector): {adata_2_prot_new.shape}")
# # make sure adata2 prot is float32 and dense mat
# adata_2_prot_new.X = adata_2_prot_new.X.astype('float32')
# adata_2_prot_new.X = adata_2_prot_new.X.toarray() if issparse(adata_2_prot_new.X) else adata_2_prot_new.X
# adata_2_prot = adata_2_prot_new  # todo uncomment this
# todo why this crash the num of dim make not sence 
# adata_2_prot_new = AnnData(adata_2_prot_new.X, obs=adata_2_prot_new.obs, var=adata_2_prot_new.var)
# max_possible_pca_dim_prot = min(adata_2_prot_new.X.shape[1], adata_2_prot_new.X.shape[0])
# # sc.pp.pca(adata_2_prot_new, n_comps=max_possible_pca_dim_prot - 1)
# cumulative_variance_ratio = np.cumsum(adata_2_prot_new.uns['pca']['variance_ratio'])



# %%
# # End of CN concatenation to protein features


# %%
# different rna, protein data analysis
sc.pp.pca(adata_1_rna)
sc.pp.pca(adata_2_prot)
sc.pp.neighbors(adata_1_rna, key_added='original_neighbors', use_rep='X_pca')
sc.tl.umap(adata_1_rna, neighbors_key='original_neighbors')
adata_1_rna.obsm['X_original_umap'] = adata_1_rna.obsm["X_umap"]
sc.pp.neighbors(adata_2_prot, key_added='original_neighbors', use_rep='X_pca')
sc.tl.umap(adata_2_prot, neighbors_key='original_neighbors')
adata_2_prot.obsm['X_original_umap'] = adata_2_prot.obsm["X_umap"]

if plot_flag:
    sc.tl.umap(adata_2_prot, neighbors_key='original_neighbors')
    sc.pl.pca(adata_1_rna, color=['cell_types', 'major_cell_types'],
              title=['RNA pca minor cell types', 'RNA pca major cell types'])
    sc.pl.pca(adata_2_prot, color=['cell_types', 'major_cell_types'],
              title=['Protein pca minor cell types', 'Protein pca major cell types'])
    sc.pl.embedding(adata_1_rna, basis='X_umap', color=['major_cell_types', 'cell_types'],
                    title=['RNA UMAP major cell types', 'RNA UMAP major cell types'])
    sc.pl.embedding(adata_2_prot, basis='X_original_umap', color=['major_cell_types', 'cell_types'],
                    title=['Protein UMAp major cell types', 'Protein UMAP major cell types'])


# %%
adata_2_prot.varm['PCs'].shape
adata_2_prot.shape
# Plot the explained variance ratio using scanpy's built-in function
sc.pl.pca_variance_ratio(adata_2_prot, log=True, n_pcs=10)
# pcs variance
pcs_variance = adata_2_prot.uns['pca']['variance_ratio']
print(f"Explained variance ratio of the first 10 PCs: {pcs_variance}")


# %%
# import numpy as np
# from sklearn.neighbors import NearestNeighbors
# import networkx as nx
# from node2vec import Node2Vec
# # Load your RNA expression data
# # Assuming 'adata' is your AnnData object
# X = adata.X
# # Compute kNN graph
# nn = NearestNeighbors(n_neighbors=10)
# nn.fit(X)
# distances, indices = nn.kneighbors(X)
# # Create adjacency matrix
# adj = np.zeros((X.shape[0], X.shape[0]))
# for i in range(X.shape[0]):
#     adj[i, indices[i]] = 1
# # Create a NetworkX graph
# G = nx.Graph()
# G.add_nodes_from(range(X.shape[0]))
# for i in range(X.shape[0]):
#     for j in indices[i]:
#         G.add_edge(i, j)
# # Apply Node2Vec
# node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
# model = node2vec.fit(window=10, min_count=1, batch_words=4)
# # Get node embeddings
# node_embeddings = model.wv.vectors
# # Use the embeddings for further analysis
# print(node_embeddings.shape)


# %%
# computing a gene module score 
terminal_exhaustion = [
    "CD3G", "FASLG", "ID2", "LAG3", "RGS1",
    "CCL3", "CCL3L1", "KIAA1671", "SH2D2A", "DUSP2",
    "PDCD1", "CD7", "NR4A2", "CD160", "PTPN22",
    "ABI3", "PTGER4", "GZMK", "GZMA", "MBNL1",
    "VMP1", "PLAC8", "RGS3", "EFHD2", "GLRX",
    "CXCR6", "ARL6IP1", "CCL4", "ISG15", "LAX1",
    "CD8A", "SERPINA3", "GZMB", "TOX"
]
precursor_exhaustion = [
    "TCF7", "MS4A4A", "TNFSF8", "CXCL10", "EEF1B2",
    "ID3", "IL7R", "JUN", "LTB", "XCL1",
    "SOCS3", "TRAF1", "EMB", "CRTAM", "EEF1G",
    "CD9", "ITGB1", "GPR183", "ZFP36L1", "SLAMF6",
    "LY6E"
]
cd8_t_cell_activation = [
    "CD69", "CCR7", "CD27", "BTLA", "CD40LG",
    "IL2RA", "CD3E", "CD47", "EOMES", "GNLY",
    "GZMA", "GZMB", "PRF1", "IFNG", "CD8A",
    "CD8B", "CD95L", "LAMP1", "LAG3", "CTLA4",
    "HLA-DRA", "TNFRSF4", "ICOS", "TNFRSF9", "TNFRSF18"
]


# %%
adata_1_rna.var_names = adata_1_rna.var_names.str.upper()
adata_2_prot.var_names = adata_2_prot.var_names.str.upper()


# %%
sc.tl.score_genes(adata_1_rna, gene_list=terminal_exhaustion, score_name="terminal_exhaustion_score")


# %%
plot_flag = True
if plot_flag:
    sc.pl.umap(adata_1_rna, color="terminal_exhaustion_score", cmap="viridis")


# %%
adata_1_rna


# %%
# ### analysis to get to scatter plot


# %%
# # different adata analysis
# adata_2_prot_first_110_vars = adata_2_prot[:, adata_2_prot.var_names[:110]].copy()
# adata_2_prot_second_110_vars = adata_2_prot[:, adata_2_prot.var_names[110:]].copy()
# sc.pp.pca(adata_2_prot_first_110_vars)
# sc.pp.pca(adata_2_prot_second_110_vars)
# # plot umap each separately
# sc.pp.neighbors(adata_2_prot_first_110_vars)
# sc.tl.umap(adata_2_prot_first_110_vars)
# sc.pp.neighbors(adata_2_prot_first_110_vars)
# sc.tl.umap(adata_2_prot_first_110_vars)
# sc.pp.neighbors(adata_2_prot_second_110_vars)
# sc.tl.umap(adata_2_prot_second_110_vars)

# if plot_flag:
#     sc.pl.embedding(adata_2_prot_first_110_vars, basis='X_umap', color=[ 'major_cell_types','cell_types'], title=['Protein UMAP first 110 vars major cell types','Protein UMAP first 110 vars major cell types'])
#     sc.pl.embedding(adata_2_prot_second_110_vars, basis='X_umap', color=[ 'major_cell_types','cell_types'], title=['Protein UMAP second 110 vars major cell types','Protein UMAP second 110 vars major cell types'])


# %%

adata_2_prot.uns['pca']['variance_ratio']

# %%

# different rna, protein adata analysis
max_possible_pca_dim_rna = min(adata_1_rna.X.shape[1], adata_1_rna.X.shape[0])
max_possible_pca_dim_prot = min(adata_2_prot.X.shape[1], adata_2_prot.X.shape[0])
sc.pp.pca(adata_1_rna, n_comps=max_possible_pca_dim_rna - 1)
# sc.pp.pca(adata_2_prot, n_comps=max_possible_pca_dim_prot - 1)
# make PCA explain X% of variance

# going to make pca 25 here just so they have the same number of pca
max_dim = 50
variance_ration_selected = 0.75

cumulative_variance_ratio = np.cumsum(adata_1_rna.uns['pca']['variance_ratio'])
n_comps_thresh = np.argmax(cumulative_variance_ratio >= variance_ration_selected) + 1
n_comps_thresh = min(n_comps_thresh, max_dim)
if n_comps_thresh == 1:
    raise ValueError('n_comps_thresh is 1, this is not good, try to lower the variance_ration_selected')
real_ratio = np.cumsum(adata_1_rna.uns['pca']['variance_ratio'])[n_comps_thresh]
# sc.pp.pca(adata_1_rna, n_comps=n_comps_thresh)
sc.pp.pca(adata_1_rna, n_comps=n_comps_thresh)
print(f"\nNumber of components explaining {real_ratio} of rna variance: {n_comps_thresh}\n")
sc.pp.pca(adata_2_prot)
variance_ration_selected = 0.999

cumulative_variance_ratio = np.cumsum(adata_2_prot.uns['pca']['variance_ratio'])
n_comps_thresh = np.argmax(cumulative_variance_ratio >= variance_ration_selected) + 1
n_comps_thresh = min(n_comps_thresh, max_dim)
real_ratio = np.cumsum(adata_2_prot.uns['pca']['variance_ratio'])[n_comps_thresh]
# sc.pp.pca(adata_2_prot, n_comps=n_comps_thresh)
sc.pp.pca(adata_1_rna, n_comps=n_comps_thresh)
print(f"\nNumber of components explaining {real_ratio} of protein variance: {n_comps_thresh}")
if n_comps_thresh == 1:
    raise ValueError('n_comps_thresh is 1, this is not good, try to lower the variance_ration_selected')


# %%
# # Find Archetypes


# %%
# Different adata analysis
archetype_list_protein = []
archetype_list_rna = []
converge = 1e-5
min_k = 9
max_k = 10
step_size = 1

# Store explained variances for plotting the elbow method
evs_protein = []
evs_rna = []

# Protein archetype detection
X_protein = adata_2_prot.obsm['X_pca'].T
total = (max_k - min_k) / step_size
for i, k in tqdm(enumerate(range(min_k, max_k, step_size)), total=total, desc='Protein Archetypes Detection'):
    archetype, _, _, _, ev = PCHA(X_protein, noc=k)
    evs_protein.append(ev)
    archetype_list_protein.append(np.array(archetype).T)
    if i > 0 and evs_protein[i] - evs_protein[i - 1] < converge:
        print('Early stopping for Protein')
        break

# RNA archetype detection
X_rna = adata_1_rna.obsm['X_pca'].T
for j, k in tqdm(enumerate(range(min_k, max_k, step_size)), total=total, desc='RNA Archetypes Detection'):
    if j > i:
        break
    archetype, _, _, _, ev = PCHA(X_rna, noc=k)
    evs_rna.append(ev)
    archetype_list_rna.append(np.array(archetype).T)
    if j > 0 and evs_rna[j] - evs_rna[j - 1] < converge:
        print('Early stopping for RNA')
        break

# Ensure both lists have the same length
min_len = min(len(archetype_list_protein), len(archetype_list_rna))
archetype_list_protein = archetype_list_protein[:min_len]
archetype_list_rna = archetype_list_rna[:min_len]


# %%
# Plot the Elbow Method for both protein and RNA
if plot_flag:
    plt.figure(figsize=(8, 6))
    ks = list(range(min_k, min_k + len(evs_protein)))  # Adjust for early stopping

    plt.plot(ks, evs_protein, marker='o', label='Protein EV')
    plt.plot(ks[:len(evs_rna)], evs_rna, marker='s', label='RNA EV')
    plt.xlabel("Number of Archetypes (k)")
    plt.ylabel("Explained Variance")
    plt.title("Elbow Method for Archetype Selection")
    plt.legend()
    plt.grid()
    plt.show()


# %%

# %%
# different sample analysis
minor_cell_types_list_prot = sorted(list(set(adata_2_prot.obs['cell_types'])))
major_cell_types_list_prot = sorted(list(set(adata_2_prot.obs['major_cell_types'])))

# have to do this above two lines for rna too
minor_cell_types_list_rna = sorted(list(set(adata_1_rna.obs['cell_types'])))
major_cell_types_list_rna = sorted(list(set(adata_1_rna.obs['major_cell_types'])))

major_cell_types_amount_prot = [adata_2_prot.obs['major_cell_types'].value_counts()[cell_type] for cell_type in
                                major_cell_types_list_prot]
major_cell_types_amount_rna = [adata_1_rna.obs['major_cell_types'].value_counts()[cell_type] for cell_type in
                               major_cell_types_list_rna]
assert set(adata_1_rna.obs['major_cell_types']) == set(adata_2_prot.obs['major_cell_types'])
archetype_proportion_list_rna, archetype_proportion_list_protein = [], []

# i made sure that archetype prot and archetype rna have the same dimensions but still not working

for archetypes_prot, archetypes_rna in tqdm(zip(archetype_list_protein, archetype_list_rna),
                                            total=len(archetype_list_protein),
                                            desc='Archetypes generating archetypes major cell types proportion vector '):
    weights_prot = get_cell_representations_as_archetypes_cvxpy(adata_2_prot.obsm['X_pca'], archetypes_prot)
    weights_rna = get_cell_representations_as_archetypes_cvxpy(adata_1_rna.obsm['X_pca'], archetypes_rna)

    archetypes_dim_prot = archetypes_prot.shape[1]  # these dimensions are not 25 anymore, made them 50
    archetype_num_prot = archetypes_prot.shape[0]
    # need to do the above two lines for rna too

    # archetype num prot and rna should be the same?
    archetypes_dim_rna = archetypes_rna.shape[1]  # these dimensions are 50
    archetype_num_rna = archetypes_rna.shape[0]

    # could it be because the minor and major cell types have different lengths for protein and rna?
    prot_arch_prop = pd.DataFrame(np.zeros((archetype_num_prot, len(major_cell_types_list_prot))),
                                  columns=major_cell_types_list_prot)
    rna_arch_prop = pd.DataFrame(np.zeros((archetype_num_rna, len(major_cell_types_list_rna))),
                                 columns=major_cell_types_list_rna)
    archetype_cell_proportions = np.zeros((archetype_num_prot, len(major_cell_types_list_rna)))
    for curr_archetype in range(archetype_num_prot):
        df_rna = pd.DataFrame([weights_prot[:, curr_archetype], adata_2_prot.obs['major_cell_types'].values],
                              index=['weight', 'major_cell_types']).T
        df_prot = pd.DataFrame([weights_rna[:, curr_archetype], adata_1_rna.obs['major_cell_types'].values],
                               index=['weight', 'major_cell_types']).T
        df_rna = df_rna.groupby('major_cell_types')['weight'].sum()[major_cell_types_list_rna]
        df_prot = df_prot.groupby('major_cell_types')['weight'].sum()[major_cell_types_list_prot]
        # normalize by the amount of major cell types
        rna_arch_prop.loc[curr_archetype, :] = df_rna.values / major_cell_types_amount_rna
        prot_arch_prop.loc[curr_archetype, :] = df_prot.values / major_cell_types_amount_prot

    prot_arch_prop = (prot_arch_prop.T / prot_arch_prop.sum(1)).T
    prot_arch_prop = prot_arch_prop / prot_arch_prop.sum(0)
    rna_arch_prop = (rna_arch_prop.T / rna_arch_prop.sum(1)).T
    rna_arch_prop = rna_arch_prop / rna_arch_prop.sum(0)
    archetype_proportion_list_rna.append(rna_arch_prop.copy())
    archetype_proportion_list_protein.append(prot_arch_prop.copy())


# %%
# lengths of major cell type amount rna and protein are the same
print(major_cell_types_amount_rna)
print(major_cell_types_amount_prot)


# %%
# plotting the results of the lowest num of archetypes
if plot_flag:
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    # sns.heatmap(reorder_rows_to_maximize_diagonal(archetype_proportion_list_rna[0])[0])
    sns.heatmap((archetype_proportion_list_rna[0]), cbar=False)
    plt.xticks()
    plt.title('RNA Archetypes')
    plt.yticks([])
    plt.ylabel('Archetypes')
    plt.subplot(1, 2, 2)
    plt.title('Protein Archetypes')
    # sns.heatmap(reorder_rows_to_maximize_diagonal(archetype_proportion_list_protein[0])[0])
    sns.heatmap((archetype_proportion_list_protein[0]), cbar=False)
    plt.suptitle('showcase the relationship between archetypes and cell types')
    plt.yticks([])
    plt.suptitle('Non-Aligned Archetypes Profiles')
    plt.ylabel('Archetypes')
    plt.show()

    new_order_1 = reorder_rows_to_maximize_diagonal(archetype_proportion_list_rna[0])[1]
    new_order_2 = reorder_rows_to_maximize_diagonal(archetype_proportion_list_protein[0])[1]
    data1 = archetype_proportion_list_rna[0].iloc[new_order_1, :]
    data2 = archetype_proportion_list_protein[0].iloc[new_order_2, :]
    # this just uses simple diagonal optimization for each one separatly, this is not final matching
    plot_archetypes_matching(data1, data2)


# %%
# todo find_best_pair_by_row_matching() find a mathing which has too many archetypes, should check it later
best_num_or_archetypes_index, best_total_cost, best_rna_archetype_order, best_protein_archetype_order = find_best_pair_by_row_matching(
    copy.deepcopy(archetype_proportion_list_rna), copy.deepcopy(archetype_proportion_list_protein), metric='correlation'
)

print("\nBest pair found:")
print(f"Best index: {best_num_or_archetypes_index}")
print(f"Best total matching cost: {best_total_cost}")
print(f"Row indices (RNA): {best_rna_archetype_order}")
print(f"Matched row indices (Protein): {best_protein_archetype_order}")


# %%
# reorder the archetypes based on the best matching so the archtypes across modalities are aligned
best_archetype_rna_prop = archetype_proportion_list_rna[best_num_or_archetypes_index].iloc[
                          best_rna_archetype_order, :].reset_index(drop=True)
# best_archetype_rna_prop = pd.DataFrame(best_archetype_rna_prop)
best_archetype_prot_prop = archetype_proportion_list_protein[best_num_or_archetypes_index].iloc[
                           best_protein_archetype_order, :].reset_index(drop=True)
# best_archetype_prot_prop = pd.DataFrame(best_archetype_prot_prop)
if plot_flag:
    plot_archetypes_matching(best_archetype_rna_prop, best_archetype_prot_prop, 8)


# %%
best_archetype_prot_prop.idxmax(axis=0)


# %%
if plot_flag:
    # show the proportion of each cell type in the archetypes, for both modalities, to see the overlap higer overlap is better
    best_archetype_prot_prop.idxmax(axis=0).plot(kind='bar', color='red', hatch='\\', label='Protein')
    best_archetype_rna_prop.idxmax(axis=0).plot(kind='bar', alpha=0.5, hatch='/', label='RNA')
    plt.title('show overlap of cell types proportions in archetypes')
    # add legend
    plt.legend()
    plt.xlabel('Major Cell Types')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    plt.show()
    compare_matchings(archetype_proportion_list_rna, archetype_proportion_list_protein, metric='cosine',
                      num_trials=100)


# %%
best_protein_archetype_order
adata_2_prot.obsm['X_pca'].shape


# %%
# get all cells archetype vec and see how they match
ordered_best_rna_archetype = archetype_list_rna[best_num_or_archetypes_index][best_protein_archetype_order, :]
ordered_best_protein_archetype = archetype_list_protein[best_num_or_archetypes_index][best_rna_archetype_order, :]
cells_archetype_vec_rna = get_cell_representations_as_archetypes_cvxpy(adata_1_rna.obsm['X_pca'],
                                                                       ordered_best_rna_archetype)
cells_archetype_vec_prot = get_cell_representations_as_archetypes_cvxpy(adata_2_prot.obsm['X_pca'],
                                                                        ordered_best_protein_archetype)

adata_1_rna.obsm['archetype_vec'] = pd.DataFrame(cells_archetype_vec_rna, index=adata_1_rna.obs.index,
                                                 columns=range(cells_archetype_vec_rna.shape[1]))
adata_2_prot.obsm['archetype_vec'] = pd.DataFrame(cells_archetype_vec_prot, index=adata_2_prot.obs.index,
                                                  columns=range(cells_archetype_vec_prot.shape[1]))
adata_1_rna.obsm['archetype_vec'].columns = adata_1_rna.obsm['archetype_vec'].columns.astype(str)
adata_2_prot.obsm['archetype_vec'].columns = adata_2_prot.obsm['archetype_vec'].columns.astype(str)

adata_1_rna.obs['archetype_label'] = pd.Categorical(np.argmax(cells_archetype_vec_rna, axis=1))
adata_2_prot.obs['archetype_label'] = pd.Categorical(np.argmax(cells_archetype_vec_prot, axis=1))
adata_1_rna.uns['archetypes'] = ordered_best_rna_archetype
adata_2_prot.uns['archetypes'] = ordered_best_protein_archetype
metrics = ['euclidean', 'cityblock', 'cosine', 'correlation', 'chebyshev']
evaluate_distance_metrics(cells_archetype_vec_rna, cells_archetype_vec_prot, metrics)


# %%
# add the best matching archetype to the metadata
adata_1_rna.obs['archetype_label'] = pd.Categorical(np.argmax(cells_archetype_vec_rna, axis=1))
adata_2_prot.obs['archetype_label'] = pd.Categorical(np.argmax(cells_archetype_vec_prot, axis=1))


# %%
adata_archetype_rna = AnnData(adata_1_rna.obsm['archetype_vec'])
adata_archetype_prot = AnnData(adata_2_prot.obsm['archetype_vec'])
adata_archetype_rna.obs = adata_1_rna.obs
adata_archetype_prot.obs = adata_2_prot.obs
adata_archetype_rna.index = adata_1_rna.obs.index
adata_archetype_prot.index = adata_2_prot.obs.index
# save all adata objects with time stamp
clean_uns_for_h5ad(adata_2_prot)
clean_uns_for_h5ad(adata_1_rna)
time_stamp = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
adata_1_rna.write(f'data/adata_rna_{time_stamp}.h5ad')
adata_2_prot.write(f'data/adata_prot_{time_stamp}.h5ad')
adata_archetype_rna.write(f'data/adata_archetype_rna_{time_stamp}.h5ad')
adata_archetype_prot.write(f'data/adata_archetype_prot_{time_stamp}.h5ad')
# load the latest of each sort by time as if I dont have the time stamp, read all files in the right name prefix and sort by time in a folder
folder = 'CODEX_RNA_seq/data/'
file_prefixes = ['adata_rna_', 'adata_prot_', 'adata_archetype_rna_', 'adata_archetype_prot_']

# Load the latest files (example)
latest_files = {prefix: get_latest_file(folder, prefix) for prefix in file_prefixes}
adata_rna = sc.read(latest_files['adata_rna_'])
adata_prot = sc.read(latest_files['adata_prot_'])
adata_archetype_rna = sc.read(latest_files['adata_archetype_rna_'])
adata_archetype_prot = sc.read(latest_files['adata_archetype_prot_'])
print(latest_files)


# %%
time_stamp


# %%
if plot_flag:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.ylabel('Archetypes')
    plt.xlabel('PCA dimensiton of archetypes')
    plt.title('RNA Archetypes')
    sns.heatmap(ordered_best_rna_archetype)
    plt.xlabel('PCA dimensiton of archetypes')
    plt.ylabel('Archetypes')
    plt.subplot(1, 2, 2)
    plt.ylabel('Archetypes')
    plt.xlabel('PCA dimensiton of archetypes')
    plt.title('Protein Archetypes')
    sns.heatmap(ordered_best_protein_archetype)
    plt.xlabel('PCA dimensiton of archetypes')
    plt.ylabel('Archetypes')

    plt.show()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('RNA Archetypes')
    plt.ylabel('Archetypes')
    _, row_order = reorder_rows_to_maximize_diagonal(best_archetype_rna_prop)
    sns.heatmap(pd.DataFrame(best_archetype_rna_prop).iloc[row_order], cbar=False)
    plt.yticks([])
    plt.ylabel('Archetypes')
    plt.subplot(1, 2, 2)

    plt.ylabel('Archetypes')
    plt.title('Protein Archetypes')
    sns.heatmap(pd.DataFrame(best_archetype_prot_prop).iloc[row_order], cbar=False)
    plt.ylabel('Archetypes')
    # plt.suptitle('The more similar the better, means that the archetypes are aligned in explaining different cell types')
    plt.suptitle('Aligned Archetypes Profiles')
    plt.yticks([])
    plt.xticks(rotation=45)

    plt.show()


# %%
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('RNA Archetypes')
plt.ylabel('Archetypes')
_, row_order = reorder_rows_to_maximize_diagonal(best_archetype_rna_prop)
sns.heatmap(pd.DataFrame(best_archetype_rna_prop).iloc[row_order], cbar=False)
plt.yticks([])
plt.ylabel('Archetypes')
plt.subplot(1, 2, 2)

plt.ylabel('Archetypes')
plt.title('Protein Archetypes')
sns.heatmap(pd.DataFrame(best_archetype_prot_prop).iloc[row_order], cbar=False)
plt.ylabel('Archetypes')
# plt.suptitle('The more similar the better, means that the archetypes are aligned in explaining different cell types')
plt.suptitle('Aligned Archetypes Profiles')
plt.yticks([])

plt.show()


# %%

archetype_distances = cdist(cells_archetype_vec_rna,
                            cells_archetype_vec_prot, metric='correlation')
if plot_flag:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(np.log1p(archetype_distances[:1000, :1000]))
    plt.title(
        'if diagonal bias this mean that the \nmathing is somewhat correct (remember that \ncells are sorted by tyeps for better visualization ')
    plt.subplot(1, 2, 2)
    # plt.plot(archetype_distances.argmin(axis=0))
    plt.title('If this looks like line, matching \nARE THE SAME AND NOT ACROSS MODALITIES')
    min_values_highlight = archetype_distances.copy()
    min_values_highlight[archetype_distances.argmin(axis=0), range(len(archetype_distances.argmin(axis=0)))] = 100
    sns.heatmap(min_values_highlight[:5000, :5000])
    # sns.heatmap(np.log1p(archetype_distances[:100,:100]))
    plt.show()


# %%
if plot_flag:
    ax = sns.histplot(adata_1_rna.obs, x='archetype_label', hue='major_cell_types', multiple='fill', stat='proportion')
    plt.xticks(rotation=45)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title('Proportion of Archetypes in Major Cell Types  in RNA')
    plt.show()

    ax = sns.histplot(adata_2_prot.obs, x='archetype_label', hue='major_cell_types', multiple='fill', stat='proportion')
    plt.xticks(rotation=45)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title('Proportion of Archetypes in Major Cell Types in Protein')



# %%
def plot_scatter(mtx1, mtx2, n_samples):
    pca = PCA(n_components=2)
    embeddings_combined = np.vstack((mtx1, mtx2))
    tsne_results = pca.fit_transform(embeddings_combined)
    # tsne_results = pca.fit_transform(embeddings_combined)

    labels = ['Dataset 1'] * n_samples + ['Dataset 2'] * n_samples
    df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    df['Dataset'] = labels

    sns.scatterplot(x='TSNE1', y='TSNE2', hue='Dataset', data=df)
    plt.title('t-SNE of Aligned Embeddings')
    plt.show()



# %%
# ### weights


# %%
if plot_flag:
    # plot both heatmaps as subplots, and add titel the more similar the better:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('RNA Archetypes')
    plt.ylabel('Archetypes')
    _, row_order = reorder_rows_to_maximize_diagonal(best_archetype_rna_prop)
    sns.heatmap(pd.DataFrame(best_archetype_rna_prop).loc[row_order, :])
    # sns.heatmap(pd.DataFrame(best_archetype_rna_prop))
    plt.subplot(1, 2, 2)
    plt.ylabel('Archetypes')
    # sns.heatmap(best_archetype_prot_prop)
    sns.heatmap(pd.DataFrame(best_archetype_prot_prop).loc[row_order, :])
    # sns.heatmap(pd.DataFrame(best_archetype_prot_prop).iloc[row_order,:])
    plt.title('Protein Archetypes')
    plt.suptitle('The more similar the better, means that the archtypes are aligned in explaining different cell types')
    plt.show()
    # errors = np.abs(ordered_arch_prot - ordered_arch_rna)
    # random_error =np.abs(ordered_arch_prot - np.random.permutation(ordered_arch_rna))
    # plt.plot(errors.values.flatten())
    # plt.plot(random_error.values.flatten())
    # plt.legend(['Error', 'Random Error'])
    # plt.show()


# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# Plot for RNA
ax = sns.histplot(adata_1_rna.obs, x='archetype_label', hue='major_cell_types', multiple='fill', stat='proportion',
                  ax=axes[0])
axes[0].set_xticks(axes[0].get_xticks())  # Set the ticks first
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
axes[0].set_title('Proportion of Archetypes in Major Cell Types in RNA')
axes[0].set_xlabel('Archetype Label')
axes[0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
axes[0].get_legend().remove()  # Remove the legend from the left subplot

# Plot for Protein
ax = sns.histplot(adata_2_prot.obs, x='archetype_label', hue='major_cell_types', multiple='fill', stat='proportion',
                  ax=axes[1])
axes[1].set_xticks(axes[1].get_xticks())  # Set the ticks first
axes[1].set_yticklabels([])
axes[1].set_title('Proportion of Archetypes in Major Cell Types in Protein')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()
# TODO this plot does not add up for some reason, it seems that the archetypes are not aligned correctly


# %%
len(data_points_rna_plot),len(ordered_best_rna_archetype),len(samples_cell_types_rna_plot),len(data_point_archetype_indices_rna_plot),
data_points_prot_plot.shape

# %%
if plot_flag:
    data_points_rna = adata_1_rna.obsm['X_pca']
    data_points_prot = adata_2_prot.obsm['X_pca']
    data_point_archetype_indices_rna = list(np.argmax(cells_archetype_vec_rna, axis=1))
    data_point_archetype_indices_prot = list(np.argmax(cells_archetype_vec_prot, axis=1))

    # use ground truth cell types
    show_ground_truth = True
    if show_ground_truth:
        major_or_minor = 'cell_types'
        major_or_minor = 'major_cell_types'
        samples_cell_types_rna = list(adata_1_rna.obs[major_or_minor])
        samples_cell_types_prot = list(adata_2_prot.obs[major_or_minor])
    else:  #
        # Get the archetype indices for each data point
        major_cell_types_list = minor_cell_types_list_rna
        current_cell_types_list = major_cell_types_list
        # Map the archetype indices to cell type names
        samples_cell_types_rna = [current_cell_types_list[i] for i in data_point_archetype_indices_rna]
        samples_cell_types_prot = [current_cell_types_list[i] for i in data_point_archetype_indices_prot]

    # Optionally limit the number of samples
    num_samples = 50000  # or any number you prefer
    data_points_rna_plot = data_points_rna[:num_samples]
    data_points_prot_plot = data_points_prot[:num_samples]
    samples_cell_types_rna_plot = samples_cell_types_rna[:num_samples]
    samples_cell_types_prot_plot = samples_cell_types_prot[:num_samples]
    data_point_archetype_indices_rna_plot = data_point_archetype_indices_rna[:num_samples]
    data_point_archetype_indices_prot_plot = data_point_archetype_indices_prot[:num_samples]

    # Create a consistent color mapping
    all_cell_types = set(samples_cell_types_rna_plot + samples_cell_types_prot_plot)
    all_cell_types.discard('archetype')
    all_cell_types = [ct for ct in all_cell_types if ct is not np.nan]
    all_cell_types = sorted(all_cell_types)
    palette = sns.color_palette("tab20", len(all_cell_types))
    cell_type_colors = {cell_type: color for cell_type, color in zip(all_cell_types, palette)}
    cell_type_colors["archetype"] = "black"

    # Call the updated function with the color mapping
    plot_archetypes(
        data_points_rna_plot,
        ordered_best_rna_archetype,
        samples_cell_types_rna_plot,
        data_point_archetype_indices_rna_plot,
        modality='RNA',
        cell_type_colors=cell_type_colors
    )
    plot_archetypes(
        data_points_prot_plot,
        ordered_best_protein_archetype,
        samples_cell_types_prot_plot,
        data_point_archetype_indices_prot_plot,
        modality='Protein',
        cell_type_colors=cell_type_colors
    )


# %%

if plot_flag:
    sc.pp.pca(adata_archetype_rna)
    sc.pp.pca(adata_archetype_prot)
    sc.pl.pca(adata_archetype_rna, color=['major_cell_types', 'archetype_label', 'cell_types'])
    sc.pl.pca(adata_archetype_prot, color=['major_cell_types', 'archetype_label', 'cell_types'])
    sc.pp.neighbors(adata_archetype_rna)
    sc.pp.neighbors(adata_archetype_prot)
    sc.tl.umap(adata_archetype_rna)
    sc.tl.umap(adata_archetype_prot)
    sc.pl.umap(adata_archetype_rna, color=['major_cell_types', 'archetype_label', 'cell_types'])
    sc.pl.umap(adata_archetype_prot, color=['major_cell_types', 'archetype_label', 'cell_types'])


# %%
if plot_flag:
    # making sure that the archetypes make sense in original data context
    sc.pp.neighbors(adata_1_rna)
    sc.pp.neighbors(adata_2_prot)
    sc.tl.umap(adata_1_rna)
    sc.tl.umap(adata_2_prot)
    sc.pl.umap(adata_1_rna, color='archetype_label', title='RNA Archetypes')
    sc.pl.umap(adata_2_prot, color='archetype_label', title='Protein Archetypes')


# %%
adata_1_rna


# %%
t_cell_terminal_exhaustion = [
    "CD3G", "FASLG", "ID2", "LAG3", "RGS1", "CCL3", "CCL3L1", "KIAA1671",
    "SH2D2A", "DUSP2", "PDCD1", "CD7", "NR4A2", "CD160", "PTPN22", "ABI3",
    "PTGER4", "GZMK", "GZMA", "MBNL1", "VMP1", "PLAC8", "RGS3", "EFHD2",
    "GLRX", "CXCR6", "ARL6IP1", "CCL4", "ISG15", "LAX1", "CD8A", "SERPINA3",
    "GZMB", "TOX"
]

t_cell_precursor_exhaustion = [
    "TCF7", "MS4A4A", "TNFSF8", "CXCL10", "EEF1B2", "ID3", "IL7R", "JUN",
    "LTB", "XCL1", "SOCS3", "TRAF1", "EMB", "CRTAM", "EEF1G", "CD9",
    "ITGB1", "GPR183", "ZFP36L1", "SLAMF6", "LY6E"
]

t_cell_t_reg = [
    "NT5E", "CD3D", "CD3G", "CD3E", "CD4",
    "CD5", "ENTPD1", "CTLA4", "IZUMO1R", "TNFRSF18",
    "IL2RA", "ITGAE", "LAG3", "TGFB1", "LRRC32",
    "TNFRSF4", "SELL", "FOXP3", "STAT5A", "STAT5B",
    "LGALS1", "IL10", "IL12A", "EBI3", "TGFB1"
]


# %%







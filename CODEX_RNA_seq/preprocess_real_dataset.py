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
# preprocess the real data from Elham lab and peprform the archetype analysis

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import importlib
import os
import re
import sys
import os

import scipy.sparse as sp
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import scvi
import torch
from scvi.model import SCVI
from scvi.train import TrainingPlan
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.sparse import issparse
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import re
from datetime import datetime
import importlib

# Set up the working directory and Python path
from setup_paths import setup_working_directory
setup_working_directory()

import bar_nick_utils
importlib.reload(bar_nick_utils)

from bar_nick_utils import plot_latent, \
    compute_pairwise_kl, select_gene_likelihood, \
    clean_uns_for_h5ad, plot_normalized_losses, compute_pairwise_kl_two_items, get_latest_file, \
    match_datasets, calculate_cLISI, calculate_iLISI,get_umap_filtered_fucntion,mixing_score,plot_cosine_distance,plot_latent_mean_std,\
    plot_rna_protein_matching_means_and_scale,archetype_vs_latent_distances_plot,plot_inference_outputs,verify_gradients,compare_distance_distributions,plot_similarity_loss_history

importlib.reload(bar_nick_utils)
from covet_utils import compute_covet

from bar_nick_utils import preprocess_rna, preprocess_protein, plot_archetypes, \
    get_cell_representations_as_archetypes_cvxpy, reorder_rows_to_maximize_diagonal, evaluate_distance_metrics, \
    plot_archetypes_matching, compare_matchings, find_best_pair_by_row_matching, add_spatial_data_to_prot, \
    clean_uns_for_h5ad, get_latest_file

plot_flag = False
# computationally figure out which ones are best
np.random.seed(8)



# %%
### reading in data

# %%

adata_rna = sc.read("../rna_umap.h5ad") # 5546 × 13447 
adata_prot = sc.read("../codex_cn_tumor.h5ad") # 893987 × 30

# filter out all tumor type cells
adata_prot = adata_prot[adata_prot.obs['cell_type'] != 'tumor']
adata_prot = adata_prot[adata_prot.obs['cell_type'] != 'dead']

num_rna_cells = 6000
num_protein_cells = 20000
num_rna_cells = num_protein_cells= 2000
subsample_n_obs_rna = min(adata_rna.shape[0],num_rna_cells)
subsample_n_obs_protein = min(adata_prot.shape[0],num_protein_cells)
sc.pp.subsample(adata_rna, n_obs=subsample_n_obs_rna)
sc.pp.subsample(adata_prot, n_obs=subsample_n_obs_protein)

adata_rna.obs['cell_types'] = adata_rna.obs['new_annotation']
adata_prot.obs['cell_types'] = adata_prot.obs['cell_type']

# nk cells seems to mess up the archetype matching, for now remove them
adata_rna = adata_rna[adata_rna.obs['cell_types'] != 'nk cells'] 
adata_prot = adata_prot[adata_prot.obs['cell_types'] != 'nk cells']


adata_rna = adata_rna[adata_rna.obs['cell_types'].argsort(), :]  # sort by cell types for easier visualization
# adata_prot = adata_prot[adata_prot.obs['cell_types'].argsort(), :]  # sort by cell types for easier visualization
adata_prot = adata_prot[adata_prot.obs['cell_types'].argsort(), :]  # sort by cell types for easier visualization

# initial_adata_rna = adata[adata.obs['batch'] == f'SLN111-D1']
# initial_adata_protein = adata[adata.obs['batch'] == f'SLN208-D1'] 
# get rna from one patient, protein from the other then run the whole archetype analysis
# cell type protein, cell type rna, see which one's 
# plot data before preprocessing with a subsample of 1000 cells
if plot_flag:
    subsample_n_obs_rna_plot = min(adata_rna.shape[0], 1000)
    subsample_n_obs_protein_plot = min(adata_prot.shape[0], 1000)
    
    adata_rna_plot = adata_rna[np.random.choice(adata_rna.shape[0], subsample_n_obs_rna_plot, replace=False), :]
    adata_prot_plot = adata_prot[np.random.choice(adata_prot.shape[0], subsample_n_obs_protein_plot, replace=False), :]
    
    # same for RNA
    sc.pp.pca(adata_rna_plot, n_comps=10)
    sc.pp.neighbors(adata_rna_plot)  # Compute the neighbors needed for UMAP
    sc.tl.umap(adata_rna_plot)  # Calculate UMAP coordinates
    sc.pl.umap(adata_rna_plot, color='cell_types', title='RNA data (1000 subsample)')
    
    # same for Protein
    sc.pp.pca(adata_prot_plot, n_comps=10)
    sc.pp.neighbors(adata_prot_plot)  # Compute the neighbors needed for UMAP
    sc.tl.umap(adata_prot_plot)  # Calculate UMAP coordinates
    sc.pl.umap(adata_prot_plot, color='cell_types', title='Protein data (1000 subsample)')

    

# %%
# make sure we dont have gene column in var if it is equal to the index
if 'gene' in adata_rna.var.columns and np.array_equal(adata_rna.var['gene'].values, (adata_rna.var.index.values)):
    adata_rna.var.drop(columns='gene', inplace=True)
if 'gene' in adata_prot.var.columns and np.array_equal(adata_prot.var['gene'].values, (adata_prot.var.index.values)):
    adata_prot.var.drop(columns='gene', inplace=True)

# %%
set(adata_rna.obs['new_annotation']), set(adata_prot.obs['cell_type'])

# %%
# filer out the cell types that are not in the other dataset
mutual_cell_types = set(adata_rna.obs['new_annotation']).intersection(set(adata_prot.obs['cell_type']))
adata_rna = adata_rna[adata_rna.obs['new_annotation'].isin(mutual_cell_types)]
adata_prot = adata_prot[adata_prot.obs['cell_type'].isin(mutual_cell_types)]
adata_rna.obs['major_cell_types'] = adata_rna.obs['new_annotation'].values
adata_prot.obs['major_cell_types'] = adata_prot.obs['cell_type'].values

# %%

# %%

# Scatter plot of variance vs. mean expression
# common approach to inspect the variance of genes. It shows the relationship between mean expression and variance (or dispersion) and highlights the selected highly variable genes.
if plot_flag:
    sc.pp.highly_variable_genes(adata_rna, n_top_genes=2000)
    plt.figure(figsize=(8, 6))
    plt.scatter(adata_rna.var['means'], adata_rna.var['variances'], alpha=0.3, label='All genes')
    plt.scatter(adata_rna.var['means'][adata_rna.var['highly_variable']],
                adata_rna.var['variances'][adata_rna.var['highly_variable']],
                color='red', label='Highly variable genes')
    plt.xlabel('Mean expression')
    plt.ylabel('Variance')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title('Raw data - Variance vs. Mean Expression of Genes')
    plt.show()

# %%
# plt.figure(figsize=(8, 6))
# plt.hist(adata_rna.var['variances'], bins=75, alpha=0.7)
# plt.axvline(x=adata_rna.var['variances'][adata_rna.var['highly_variable']].min(), color='red', linestyle='dashed', label='Cutoff')
# plt.xlabel('Variance')
# plt.ylabel('Frequency')
# plt.yscale('log')
# plt.title('Distribution of Gene Variances')
# plt.legend()
# plt.show()

# %%
variances_sorted = np.sort(adata_rna.var['variances'])[::-1]

if plot_flag:
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(variances_sorted) + 1), variances_sorted)
    plt.xlabel('Gene rank')
    plt.ylabel('Variance')
    # plt.xscale('log')

    plt.yscale('log')
    plt.title('Elbow plot of Gene Variances')
    plt.axvline(x=1000, color='red', linestyle='dashed', label='n_top_genes=1000')
    plt.legend()
    plt.show()
    plt.figure()
kneedle = KneeLocator(range(1, len(variances_sorted) + 1), np.log(variances_sorted), S=4.0, curve="convex",
                      direction="decreasing")
if plot_flag:
    kneedle.plot_knee()


# %%
adata_rna = preprocess_rna(adata_rna,n_top_genes=kneedle.knee)
if plot_flag:
    plt.figure(figsize=(8, 6))
    plt.scatter(adata_rna.var['means'], adata_rna.var['variances'], alpha=0.3, label='All genes')
    plt.scatter(adata_rna.var['means'][adata_rna.var['highly_variable']],
                adata_rna.var['variances'][adata_rna.var['highly_variable']],
                color='red', label='Highly variable genes')
    plt.xlabel('Mean expression')
    plt.ylabel('Variance')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title('Processed data - Variance vs. Mean Expression of Genes')
    plt.show()

# %%
adata_rna = adata_rna[
    adata_rna.obs.sort_values(by=['cell_types']).index
]

# %%

adata_rna = adata_rna.copy()
sc.pp.pca(adata_rna)
print(f'variance explained by first 10 PCs {adata_rna.uns["pca"]["variance_ratio"][:10].sum()}')
adata_rna = preprocess_rna(adata_rna)
sc.pp.pca(adata_rna)

# %%
adata_rna

# %%
# adata_rna.obsm.pop('protein_expression')
# assert len(set(adata.obs['batch']))!=1
adata_prot = preprocess_protein(adata_prot)
# # add all obs and var to adata_prot
# adata_prot.obs = adata_rna.obs (this is a potential problem as rna and protein obs don't match currently)


# %%
if plot_flag:
    # same for rna
    sc.pp.pca(adata_rna)
    sc.pp.neighbors(adata_rna)  # Compute the neighbors needed for UMAP
    sc.tl.umap(adata_rna)  # Calculate UMAP coordinates
    sc.pl.umap(adata_rna, color='cell_types')
    # same for protein
    sc.pp.pca(adata_prot)
    sc.pp.neighbors(adata_prot)  # Compute the neighbors needed for UMAP
    sc.tl.umap(adata_prot)  # Calculate UMAP coordinates
    sc.pl.umap(adata_prot, color='cell_types')


# %%

# %%
adata_prot.obs = adata_prot.obs.drop(columns=['n_genes'])
adata_prot.obsm.pop('X_pca')
adata_prot.varm.pop('PCs')
original_protein_num = adata_prot.X.shape[1]

# %%
adata_prot.obs['major_cell_types'].unique()
adata_prot.obs['cell_types'].unique()

# %%

assert adata_prot.obs.index.is_unique
x_coor = adata_prot.obsm['spatial'][:, 0]
y_coor = adata_prot.obsm['spatial'][:, 1]
temp = pd.DataFrame([x_coor, y_coor], index=['x', 'y']).T
temp.index = adata_prot.obs.index
adata_prot.obsm['spatial_location'] = temp
adata_prot.obs['X'] = x_coor
adata_prot.obs['Y'] = y_coor
if plot_flag:
    sc.pl.scatter(adata_prot, x='X', y='Y', color='cell_types', title='T Cell subtypes locations')
    # sc.pl.scatter(adata_prot[adata_prot.obs['major_cell_types']=='CD8 T'], x='X', y='Y', color='cell_types', title='T Cell subtypes locations')

# %%
adata_prot = adata_prot[
    adata_prot.obs.sort_values(by=['cell_types']).index
]

# %%
if plot_flag:
    # Randomly select 100 cells
    num_cells = min(1000, adata_rna.n_obs, adata_prot.n_obs)
    random_indices_protein = np.random.choice(adata_prot.n_obs, num_cells, replace=False)
    random_indices_rna = np.random.choice(adata_rna.n_obs, num_cells, replace=False)

    # For protein data
    protein_data = adata_prot.X[random_indices_protein, :]
    sns.heatmap(protein_data, xticklabels=False, yticklabels=False)
    plt.title("Protein Expression Heatmap (Random 100 Cells)")
    plt.show()

    # For RNA data
    rna_data = adata_rna.X[random_indices_rna, :].todense() if issparse(adata_rna.X) else adata_rna.X[random_indices_rna, :]
    sns.heatmap(rna_data, xticklabels=False, yticklabels=False)
    plt.title("RNA Expression Heatmap (Random 100 Cells)")
    plt.show()

# %%
clean_uns_for_h5ad(adata_prot)
clean_uns_for_h5ad(adata_rna)
time_stamp = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
adata_rna.write(f'data/preprocessed_adata_rna_{time_stamp}.h5ad')
adata_prot.write(f'data/preprocessed_adata_prot_{time_stamp}.h5ad')


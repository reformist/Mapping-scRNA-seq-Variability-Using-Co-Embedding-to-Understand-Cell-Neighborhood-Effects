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

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.metrics import silhouette_score
import numpy as np
import ot
import warnings
from sklearn.metrics.pairwise import pairwise_distances

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
from tqdm import tqdm
from kneed import KneeLocator

import bar_nick_utils
import covet_utils

importlib.reload(bar_nick_utils)
importlib.reload(covet_utils)
from covet_utils import compute_covet

from bar_nick_utils import preprocess_rna, preprocess_protein, plot_archetypes, \
    get_cell_representations_as_archetypes_cvxpy, reorder_rows_to_maximize_diagonal, evaluate_distance_metrics, \
    plot_archetypes_matching, compare_matchings, find_best_pair_by_row_matching, add_spatial_data_to_prot, \
    clean_uns_for_h5ad, get_latest_file,get_latest_file

plot_flag = False
# computationally figure out which ones are best
np.random.seed(8)


# %%



file_prefixes = ['preprocessed_adata_rna_', 'preprocessed_adata_prot_']
folder = 'CODEX_RNA_seq/data/'

# Load the latest files (example)
latest_files = {prefix: get_latest_file(folder, prefix) for prefix in file_prefixes}
adata_1_rna = sc.read(latest_files['preprocessed_adata_rna_'])
adata_2_prot = sc.read(latest_files['preprocessed_adata_prot_'])

num_rna_cells = 6000
num_protein_cells = 20000
num_rna_cells = num_protein_cells= 2000
subsample_n_obs_rna = min(adata_1_rna.shape[0], num_rna_cells)
subsample_n_obs_protein = min(adata_2_prot.shape[0], num_protein_cells)
sc.pp.subsample(adata_1_rna, n_obs=subsample_n_obs_rna)
sc.pp.subsample(adata_2_prot, n_obs=subsample_n_obs_protein)


original_protein_num = adata_2_prot.X.shape[1]


sc.pp.neighbors(adata_2_prot, use_rep='spatial_location')

connectivities = adata_2_prot.obsp['connectivities']
connectivities[connectivities > 0] = 1
assert np.array_equal(np.array([0., 1.], dtype=np.float32), np.unique(np.array(connectivities.todense())))
if plot_flag:
    sns.heatmap(connectivities.todense()[:1000, :1000])



# %%
neighbor_sums = connectivities.todense().dot(adata_2_prot.X.todense() if issparse(adata_2_prot.X) else adata_2_prot.X) # sum the neighbors
neighbor_sums = np.asarray(neighbor_sums)
# neighbor_means = neighbor_sums/(0.00001+neighbor_sums.sum(1))[:,np.newaxis] # normalize
neighbor_means = np.asarray(neighbor_sums / connectivities.sum(1))
if plot_flag:
    # sns.heatmap(adata_2_prot.obsp['distances'].todense()[:1000,:1000])
    plt.show()
sc.pp.neighbors(adata_2_prot, use_rep='spatial_location', key_added='spatial_neighbors', n_neighbors=15)

distances = adata_2_prot.obsp['spatial_neighbors_distances'].data
log_transformed_distances = (distances + 1)  # since the distances are not normally distributed, log-transform them



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


if plot_flag:
    sns.histplot(log_transformed_distances)
    plt.title('Distribution of distances between spatial neighbors before applying cutoff')
    plt.show()
    # Recompute the connectivity matrix
    sns.histplot(adata_2_prot.obsp['spatial_neighbors_distances'].data)
    plt.title('Distribution of distances between spatial neighbors after applying cutoff')
    plt.show()
    # todo  I am using the data as if its notmarl and its not, so I need to see how to remove the tail in a different way





# %%
np.array(neighbor_means)

# %%
# (different samples)
# since we have different samples, we need different final labels after the clustering
# Standardize the data
normalized_data = zscore(neighbor_means.toarrray() if issparse(neighbor_means) else neighbor_means, axis=0)
                         # each cell we get the mean of the features of its neighbors


temp = AnnData(normalized_data)
temp.obs = adata_2_prot.obs
sc.pp.pca(temp)
sc.pp.neighbors(temp)
sc.tl.leiden(temp, resolution=0.3, key_added='CN')
num_clusters = len(adata_2_prot.obs['CN'].unique())
palette = sns.color_palette("tab10", num_clusters)  # "tab10" is a good color map, you can choose others too
adata_2_prot.uns['spatial_clusters_colors'] = palette.as_hex()  # Save as hex color codes





# %%



# %%

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.heatmap(neighbor_means)  # less noisy because we mean, makes sense as we made sure each cell is of same cell type
plt.title('Mean proteins expression of neighbors of each cell')
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
        title='Cluster cells by their CN, can see the different CN in different regions, \nthanks to the different B cell types in each region',
        ax=ax,  # Use the ax created above
        show=False  # Prevent scanpy from showing the plot immediately
    )
    # for x in horizontal_splits[1:-1]:  # Exclude the edges to avoid border doubling
    #     ax.axvline(x=x, color='black', linestyle='--')
    # for y in vertical_splits[1:-1]:  # Exclude the edges to avoid border doubling
    #     ax.axhline(y=y, color='black', linestyle='--')
    # plt.show()

    neighbor_adata = anndata.AnnData(neighbor_means)
if plot_flag:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(
        neighbor_adata.X)  # less noisy because we mean, makes sense as we made sure each cell is of same cell type
    plt.title('prot')
    plt.subplot(1, 2, 2)
    sns.heatmap(adata_2_prot.X.todense() if issparse(adata_2_prot.X) else adata_2_prot.X)
    plt.title('Proteins expressions of each cell')
    plt.show()
    # rna no longer has the CN, using the protein
    neighbor_adata.obs['CN'] = pd.Categorical(adata_2_prot.obs['CN'])

    # rna no longer has the CN, using the protein
    neighbor_adata.obs['CN'] = pd.Categorical(adata_2_prot.obs['CN'])
    sc.pp.pca(neighbor_adata)
    sc.pp.neighbors(neighbor_adata)
    sc.tl.umap(neighbor_adata)
    sc.pl.umap(neighbor_adata, color='CN', title='UMAP of CN embedding')

    # making sure the CN and the protein are distinct
    adata_prot_cn_concat = anndata.concat([adata_2_prot, neighbor_adata], join='outer', label='modality',
                                          keys=['Protein', 'CN'])
    X = adata_prot_cn_concat.X.toarray() if issparse(adata_prot_cn_concat.X) else adata_prot_cn_concat.X
    X = np.nan_to_num(X)
    adata_prot_cn_concat.X = X
    sc.pp.pca(adata_prot_cn_concat)
    sc.pp.neighbors(adata_prot_cn_concat)
if plot_flag:
    sc.tl.umap(adata_prot_cn_concat)
    sc.pl.umap(adata_prot_cn_concat, color=['CN', 'modality'],
               title=['UMAP of CN embedding to make sure they are not mixed',
                      'UMAP of CN embedding to make sure they are not mixed'])
    sc.pl.pca(adata_prot_cn_concat, color=['CN', 'modality'],
              title=['PCA of CN embedding to make sure they are not mixed',
                     'PCA of CN embedding to make sure they are not mixed'])



# %%




# %%
# Assuming `adata_prot` is the original AnnData object
# And `neighbor_means` is the new matrix to be concatenated
# different adata

new_feature_names = [f"CN_{i}" for i in adata_2_prot.var.index]
if adata_2_prot.X.shape[1] == neighbor_means.shape[1]:
    new_X = np.hstack([adata_2_prot.X.todense() if issparse(adata_2_prot.X) else adata_2_prot.X, neighbor_means])
    additional_var = pd.DataFrame(index=new_feature_names)
    new_vars = pd.concat([adata_2_prot.var, additional_var])
else:
    new_X = adata_2_prot.X
    new_vars = adata_2_prot.var

adata_2_prot = anndata.AnnData(
    X=new_X,
    obs=adata_2_prot.obs.copy(),  # Keep the same observation metadata
    var=new_vars,  # Keep the same variable metadata
    uns=adata_2_prot.uns.copy(),  # Keep the same unstructured data #todo brin back?
    obsm=adata_2_prot.obsm.copy(),  # Keep the same observation matrices ? #todo bring back?
    # varm=adata_prot.varm.copy(), # Keep the same variable matrices
    # layers=adata_2_prot.layers.copy()  # Keep the same layers
)
adata_2_prot.var['feature_type'] = ['protein'] * original_protein_num + ['CN'] * neighbor_means.shape[1]
sc.pp.pca(adata_2_prot)  # rerun PCA
print(f"New adata shape (protein features + cell neighborhood vector): {adata_2_prot.shape}")
# adata_2_prot_new.var['feature_type'] = ['protein'] * original_protein_num + ['CN'] * neighbor_adata.shape[1]
# plot pca variance using scanpy
# remove pca
# if 'X_pca' in adata_2_prot_new.obsm:
#    adata_2_prot_new.obsm.pop('X_pca')
#  set new highly variable genes
# adata_2_prot_new.var['highly_variable'] = True
# adata_2_prot_new.obs['CN'] = adata_2_prot.uns['CN']
# adata_2_prot_new.X = (zscore(adata_2_prot_new.X, axis=0)) 
# sc.pp.highly_variable_genes(adata_2_prot_new)
# sc.pp.pca(adata_2_prot_new)
# sc.pp.neighbors(adata_2_prot_new,n_neighbors=15)
# sc.tl.umap(adata_2_prot_new)
# sc.pl.umap(adata_2_prot_new, color='CN', title='UMAP of CN embedding to make sure they are not mixed')
# todo make sure no more things to remove
# new_pca = adata_2_prot_new.obsm['X_pca']
# sns.heatmap(adata_2_prot_new.X)
# sc.pl.pca_variance_ratio(adata_2_prot_new, log=True, n_pcs=50, save='.pdf')
# sc.pp.pca(adata_2_prot_new)
# if 'adata_2_prot_old' not in globals():
#    adata_2_prot_old = adata_2_prot.copy()
# print(f"New adata shape (protein features + cell neighborhood vector): {adata_2_prot_new.shape}")
# make sure adata2 prot is float32 and dense mat
# adata_2_prot_new.X = adata_2_prot_new.X.astype('float32')
# adata_2_prot_new.X = adata_2_prot_new.X.toarray() if issparse(adata_2_prot_new.X) else adata_2_prot_new.X
# adata_2_prot = adata_2_prot_new # todo uncomment this



# %%
# if plot_flag:
#     sc.pl.pca(adata_2_prot_new, color='cell_types')
#     sc.pp.neighbors(adata_2_prot_new, n_neighbors=5, use_rep='X_pca')
#     sc.tl.umap(adata_2_prot_new)
#     sc.pl.umap(adata_2_prot_new, color='cell_types')



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
# different rna, protein adata analysis
max_possible_pca_dim_rna = min(adata_1_rna.X.shape[1], adata_1_rna.X.shape[0])
max_possible_pca_dim_prot = min(adata_2_prot.X.shape[1], adata_2_prot.X.shape[0])
sc.pp.pca(adata_1_rna, n_comps=max_possible_pca_dim_rna - 1)
sc.pp.pca(adata_2_prot, n_comps=max_possible_pca_dim_prot - 1)
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
min_k = 6
max_k = 7
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
# # different adata analysis
# archetype_list_protein = []

# archetype_list_rna = []
# converge = 1e-5
# min_k = 7  # this is where we set number of archetypes
# max_k = 8
# step_size = 1

# evs = []
# X_protein = adata_2_prot.obsm['X_pca'].T
# total = (max_k - min_k) / step_size
# for i, k in tqdm(enumerate(range(min_k, max_k, step_size)), total=total, desc='Protein Archetypes Detection'):
#     archetype, _, _, _, ev = PCHA(X_protein, noc=k)
#     evs.append(ev)
#     archetype_list_protein.append(np.array(archetype).T)
#     if i > 0 and ev - evs[i - 1] < converge:
#         print('early stopping')
#         break
# evs = []
# X_rna = adata_1_rna.obsm['X_pca'].T

# for j, k in tqdm(enumerate(range(min_k, max_k, step_size)), total=total, desc='RNA Archetypes Detection'):
#     if j > i:
#         break
#     archetype, _, _, _, ev = PCHA(X_rna, noc=k)
#     evs.append(ev)
#     archetype_list_rna.append(np.array(archetype).T)
#     if j > 0 and ev - evs[j - 1] < converge:
#         print('early stopping')
#         break
# min_len = min([len(archetype_list_protein), len(archetype_list_rna)])
# archetype_list_protein = archetype_list_protein[:min_len]
# archetype_list_rna = archetype_list_rna[:min_len]



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

# %%
num_archetypes = weights_rna.shape[1]
rna_archetypes = []
print(num_archetypes)
# Find cells for each RNA archetype
arche_index_for_each_cell_rna = np.argmax(weights_rna, axis=1)
for i in range(num_archetypes):
    locs = arche_index_for_each_cell_rna == i
    archetype_cells = adata_1_rna[locs]
    rna_archetypes.append(archetype_cells.X.toarray() if issparse(archetype_cells.X) else archetype_cells.X)

# Create lists to store cells for each archetype (Protein)
prot_archetypes = []

# Find cells for each Protein archetype
arche_index_for_each_cell_prot = np.argmax(weights_prot, axis=1)
for i in range(num_archetypes):
    locs = arche_index_for_each_cell_prot == i
    archetype_cells = adata_2_prot[locs]
    prot_archetypes.append(archetype_cells.X.toarray() if issparse(archetype_cells.X) else archetype_cells.X)

# Example: Access the cells for the third archetype (index 2)
earchetyp_1_rna_cell = rna_archetypes[2]
example_prot_cell = prot_archetypes[2]

# print(rna_archetypes)
# Plot heatmap of some RNA and Protein archetypes
if plot_flag:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(rna_archetypes[0], cmap="viridis", cbar=True)
    plt.title("RNA Archetype 1")
    plt.xlabel("Features")
    plt.ylabel("Cells")

    plt.subplot(1, 2, 2)
    sns.heatmap(prot_archetypes[0], cmap="viridis", cbar=True)
    plt.title("Protein Archetype 1")
    plt.xlabel("Features")
    plt.ylabel("Cells")

    plt.tight_layout()
    plt.show()



# %%

def match_archetypes_with_gw(archetypes_space1, archetypes_space2, metric='euclidean', 
                           loss_type='square_loss', epsilon=0.01, max_iter=1000):
    """
    Match archetypes across different feature spaces using Gromov-Wasserstein Optimal Transport.
    """
    print(f'sizes of the archetypes {(archetypes_space1[0].shape)} {(archetypes_space2)[0].shape}')
    # Number of archetypes in each space
    num_archetypes1 = len(archetypes_space1)
    num_archetypes2 = len(archetypes_space2)
    # Create a cost matrix between archetypes across spaces
    cost_matrix = np.zeros((num_archetypes1, num_archetypes2))
    # Compute distances between each pair of archetypes across spaces
    for i, archetype1 in tqdm(enumerate(archetypes_space1),total=num_archetypes1, desc='Computing GW distances'):
        for j, archetype2 in enumerate(archetypes_space2):
            # Skip if either archetype has no cells
            if len(archetype1) == 0 or len(archetype2) == 0:
                cost_matrix[i, j] = np.inf
                continue
            try:
                # Compute distance matrices within each archetype
                C1 = pairwise_distances(archetype1, metric=metric)
                C2 = pairwise_distances(archetype2, metric=metric)
                # Normalize the distance matrices
                if C1.max() > 0:
                    C1 = C1 / C1.max()
                if C2.max() > 0:
                    C2 = C2 / C2.max()
                # Define weights for samples (uniform weights)
                p = np.ones(len(archetype1)) / len(archetype1)
                q = np.ones(len(archetype2)) / len(archetype2)
                
                # Compute Gromov-Wasserstein distance
                gw_dist = ot.gromov.entropic_gromov_wasserstein2(
                    C1, C2, p, q, loss_type, epsilon=epsilon, 
                    max_iter=max_iter, verbose=False
                )
                
                # Use the Gromov-Wasserstein distance as the cost
                cost_matrix[i, j] = gw_dist
                
            except Exception as e:
                print(f"Error computing GW distance between archetype {i} and {j}: {e}")
                cost_matrix[i, j] = np.inf
    
    # Handle potential numerical issues in the cost matrix
    cost_matrix = np.nan_to_num(cost_matrix, nan=np.inf, posinf=np.inf, neginf=0)
    
    # If all values are infinite, set them to 1 to avoid algorithm failure
    if np.all(~np.isfinite(cost_matrix)):
        warnings.warn("All values in cost matrix are invalid. Using uniform costs.")
        cost_matrix = np.ones((num_archetypes1, num_archetypes2))
    
    # Define weights for archetypes (uniform weights)
    weights_archetypes1 = np.ones(num_archetypes1) / num_archetypes1
    weights_archetypes2 = np.ones(num_archetypes2) / num_archetypes2
    
    try:
        # Solve the optimal transport problem to match archetypes
        matching = ot.emd(weights_archetypes1, weights_archetypes2, cost_matrix) # give one to one matching? am  I sure?
        # matching = ot.sinkhorn(weights_archetypes1, weights_archetypes2, 
        #               cost_matrix, reg=1.0)  # Try higher reg values
        # matching = ot.unbalanced.sinkhorn_unbalanced(
        # weights_archetypes1, weights_archetypes2, 
        # cost_matrix, reg=1.0, reg_m=1.0
        # )

    except Exception as e:
        warnings.warn(f"OT algorithm failed: {e}. Falling back to uniform matching.")
        matching = np.ones((num_archetypes1, num_archetypes2)) / (num_archetypes1 * num_archetypes2)
    cost_matrix = soften_matching(cost_matrix)

    return matching, cost_matrix
def soften_matching(M, temperature=0.1):
    M_exp = np.exp(-M/temperature)
    return M_exp / M_exp.sum(axis=1, keepdims=True)

# Example usage
# if __name__ == "__main__":
#     archetyp_rna_cells = rna_archetypes[:1]
#     archetyp_prot_cells = prot_archetypes[:1]
#     # Take a subset of each of the archetypes
#     archetyp_prot_cells = [archetype[:min(len(archetype), 30)] for archetype in archetyp_prot_cells]
#     archetyp_rna_cells = [archetype[:min(len(archetype), 30)] for archetype in archetyp_rna_cells]
    
#     earchetyp_1_rna_cell_dummy = copy.deepcopy(archetyp_rna_cells)
#     eaxmple_prot_cell_dummy = copy.deepcopy(archetyp_prot_cells)
#     # eaxmple_prot_cell_dummy[0] = eaxmple_prot_cell_dummy[0]+ np.random.normal(0, 0.1, eaxmple_prot_cell_dummy[0].shape)# PCA().fit_transform(earchetyp_1_rna_cell_dummy[0])
#     earchetyp_1_rna_cell_dummy[0] = eaxmple_prot_cell_dummy[0]+ np.random.normal(0, 0.1, eaxmple_prot_cell_dummy[0].shape)# PCA().fit_transform(earchetyp_1_rna_cell_dummy[0])
#     # Add some noise to the RNA archetypes
#     earchetyp_1_rna_cell_dummy[1] = eaxmple_prot_cell_dummy[1]+ np.random.normal(0, 0.1, eaxmple_prot_cell_dummy[1].shape)
    
#     # Match archetypes using Gromov-Wasserstein
#     matching, cost_matrix = match_archetypes_with_gw(earchetyp_1_rna_cell_dummy, eaxmple_prot_cell_dummy)
    
#     print("Cost matrix between archetypes:")
#     print(cost_matrix)
#     print("\nOptimal matching between archetypes:")
#     print(matching)
    
#     # Interpret the matching matrix
#     for i in range(matching.shape[0]):
#         matches = [(j, matching[i, j]) for j in range(matching.shape[1]) if matching[i, j] > 0.01]
#         for j, weight in matches:
#             print(f"Archetype {i+1} from space 1 matches with archetype {j+1} from space 2 with weight {weight:.4f}")



# %%
matching, cost_matrix = match_archetypes_with_gw(rna_archetypes, prot_archetypes)



# %%
# Plot the heatmap of matching and cost_matrix
if plot_flag:
    plt.figure(figsize=(12, 6))

    # Plot matching heatmap
    # This shows the transport plan between archetypes - higher values indicate stronger correspondences
    # Values close to 1/(num_archetypes*num_archetypes) suggest uniform matching
    plt.subplot(1, 2, 1)
    sns.heatmap(100*matching, annot=True, fmt=".2f", cmap="viridis", cbar=True)
    plt.title("Matching Heatmap")
    plt.xlabel("Archetypes in Space 2")
    plt.ylabel("Archetypes in Space 1")

    # This shows the Gromov-Wasserstein distances between archetypes
    # Lower values (darker in magma colormap) indicate more structural similarity
    plt.subplot(1, 2, 2)
    sns.heatmap(100*cost_matrix, annot=True, fmt=".2f", cmap="magma", cbar=True)
    plt.title("Cost Matrix Heatmap")
    plt.xlabel("Archetypes in Space 2")
    plt.ylabel("Archetypes in Space 1")

    plt.tight_layout()
    plt.show()

# %%
# Find the row indices (RNA) and matched column indices (Protein) using argmax
row_indices_rna_ot = np.arange(matching.shape[0])
matched_indices_protein_ot = np.argmax(matching, axis=0)

# Print the results
print(f"Row indices (RNA): {row_indices_rna_ot}")
print(f"Matched row indices (Protein): {matched_indices_protein_ot}")

# %%

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
matched_indices_protein_ot
matched_indices_protein_ot = np.argmax(matching, axis=0)
# # precentage of agreemet ot and best matching
print(f"Percentage of agreement between OT and Best Matching: {np.mean(matched_indices_protein_ot == best_protein_archetype_order) * 100:.2f}%")
print(matched_indices_protein_ot , best_protein_archetype_order)
# best_protein_archetype_order = matched_indices_protein_ot # todo remove this line

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
if plot_flag or True:
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
adata_archetype_rna = AnnData(adata_1_rna.obsm['archetype_vec'])
adata_archetype_prot = AnnData(adata_2_prot.obsm['archetype_vec'])
adata_archetype_rna.obs = adata_1_rna.obs
adata_archetype_prot.obs = adata_2_prot.obs
adata_archetype_rna.index = adata_1_rna.obs.index
adata_archetype_prot.index = adata_2_prot.obs.index
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
    "SH2D2A", "DUSP2", "PDCD1", "CD7", "NR4A2", "CD160", "PTPN22",
    "ABI3", "PTGER4", "GZMK", "GZMA", "MBNL1", "VMP1", "PLAC8", "RGS3",
    "EFHD2", "GLRX", "CXCR6", "ARL6IP1", "CCL4", "ISG15", "LAX1", "CD8A",
    "SERPINA3", "GZMB", "TOX"
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









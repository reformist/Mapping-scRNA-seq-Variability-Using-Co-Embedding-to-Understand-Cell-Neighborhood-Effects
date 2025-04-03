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

# %% Archetype Generation with Neighbors Means and MaxFuse
# This notebook generates archetypes for RNA and protein data using neighbor means and MaxFuse alignment.

# %% Imports and Setup
import copy
import importlib
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData, concat
from matplotlib import pyplot as plt
from py_pcha import PCHA
from scipy.sparse import issparse
from scipy.spatial.distance import cdist
from scipy.stats import zscore
from sklearn.decomposition import PCA
from tqdm import tqdm

# Add repository root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set working directory to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cell_lists import cd8_t_cell_activation, precursor_exhaustion, terminal_exhaustion
from plotting_functions import (
    plot_archetype_proportions,
    plot_archetype_visualizations,
    plot_archetype_weights,
    plot_elbow_method,
    plot_modality_embeddings,
    plot_neighbor_means,
    plot_spatial_clusters,
)

import bar_nick_utils
import covet_utils

importlib.reload(bar_nick_utils)
importlib.reload(covet_utils)
from bar_nick_utils import (
    clean_uns_for_h5ad,
    compare_matchings,
    evaluate_distance_metrics,
    find_best_pair_by_row_matching,
    get_cell_representations_as_archetypes_cvxpy,
    get_latest_file,
    plot_archetypes,
    plot_archetypes_matching,
    reorder_rows_to_maximize_diagonal,
)

# Set random seed for reproducibility
np.random.seed(8)

# Global variables
plot_flag = False

# %% Load and Preprocess Data
# Load data
file_prefixes = ["preprocessed_adata_rna_", "preprocessed_adata_prot_"]
folder = "CODEX_RNA_seq/data/processed_data"

# Load the latest files
latest_files = {prefix: get_latest_file(folder, prefix) for prefix in file_prefixes}
adata_1_rna = sc.read(latest_files["preprocessed_adata_rna_"])
adata_2_prot = sc.read(latest_files["preprocessed_adata_prot_"])

# Subsample data
num_rna_cells = 80000
num_protein_cells = 80000
num_rna_cells = num_protein_cells = 2000
subsample_n_obs_rna = min(adata_1_rna.shape[0], num_rna_cells)
subsample_n_obs_protein = min(adata_2_prot.shape[0], num_protein_cells)
sc.pp.subsample(adata_1_rna, n_obs=subsample_n_obs_rna)
sc.pp.subsample(adata_2_prot, n_obs=subsample_n_obs_protein)

original_protein_num = adata_2_prot.X.shape[1]
print(f"data shape: {adata_1_rna.shape}, {adata_2_prot.shape}")
# %% Compute Spatial Neighbors and Means
# Compute spatial neighbors
sc.pp.neighbors(adata_2_prot, use_rep="spatial_location")

connectivities = adata_2_prot.obsp["connectivities"]
connectivities[connectivities > 0] = 1
assert np.array_equal(
    np.array([0.0, 1.0], dtype=np.float32), np.unique(np.array(connectivities.todense()))
)
if plot_flag:
    sns.heatmap(connectivities.todense()[:1000, :1000])

# Compute neighbor means
neighbor_sums = connectivities.dot(adata_2_prot.X)  # get the sum of all neighbors
if issparse(neighbor_sums):
    my_array = neighbor_sums.toarray()
neighbor_means = np.asarray(neighbor_sums / connectivities.sum(1))
if plot_flag:
    plt.show()

# %% Spatial Neighbors with Cutoff
# Compute spatial neighbors with cutoff
sc.pp.neighbors(
    adata_2_prot, use_rep="spatial_location", key_added="spatial_neighbors", n_neighbors=15
)

distances = adata_2_prot.obsp["spatial_neighbors_distances"].data
log_transformed_distances = distances + 1  # log-transform distances

if plot_flag:
    sns.histplot(log_transformed_distances)
    plt.title("Distribution of spatial distances between protein neighbors before cutoff")
    plt.show()

# Apply distance cutoff
distances_mean = log_transformed_distances.mean()
distances_std = log_transformed_distances.std()
two_std_dev = distances_mean + 2 * distances_std

indices_to_zero_out = np.where(adata_2_prot.obsp["spatial_neighbors_distances"].data > two_std_dev)[
    0
]
indices_to_zero_out = indices_to_zero_out[
    indices_to_zero_out < adata_2_prot.obsp["spatial_neighbors_connectivities"].data.shape[0]
]

adata_2_prot.obsp["spatial_neighbors_connectivities"].data[indices_to_zero_out] = 0
adata_2_prot.obsp["spatial_neighbors_distances"].data[indices_to_zero_out] = 0

if plot_flag:
    sns.histplot(adata_2_prot.obsp["spatial_neighbors_distances"].data)
    plt.title("Distribution of spatial distances between protein neighbors after cutoff")
    plt.show()

print(
    indices_to_zero_out.max(), adata_2_prot.obsp["spatial_neighbors_connectivities"].data.shape[0]
)
print(
    len(adata_2_prot.obsp["spatial_neighbors_distances"].data),
    len(adata_2_prot.obsp["spatial_neighbors_connectivities"].data),
)

# %% Compute Cell Neighborhoods
# Compute cell neighborhoods
normalized_data = zscore(neighbor_means, axis=0)

temp = AnnData(normalized_data)
temp.obs = adata_2_prot.obs
sc.pp.pca(temp)
sc.pp.neighbors(temp)
sc.tl.leiden(temp, resolution=0.1, key_added="CN")
num_clusters = len(adata_2_prot.obs["CN"].unique())
palette = sns.color_palette("tab10", num_clusters)
adata_2_prot.uns["spatial_clusters_colors"] = palette.as_hex()

if issparse(adata_2_prot.X):
    adata_2_prot.X = adata_2_prot.X.toarray()

# Plot neighbor means
if plot_flag:
    plot_neighbor_means(adata_2_prot, neighbor_means)

# %% Plot Spatial Clusters
# Plot spatial clusters
if plot_flag:
    plot_spatial_clusters(adata_2_prot, neighbor_means)

# %% Add CN Features to Protein Data
# Add CN features to protein data
new_feature_names = [f"CN_{i}" for i in adata_2_prot.var.index]
if adata_2_prot.X.shape[1] == neighbor_means.shape[1]:
    new_X = np.hstack([adata_2_prot.X, neighbor_means])
    additional_var = pd.DataFrame(index=new_feature_names)
    new_vars = pd.concat([adata_2_prot.var, additional_var])
else:
    new_X = adata_2_prot.X
    new_vars = adata_2_prot.var

adata_2_prot = AnnData(
    X=new_X,
    obs=adata_2_prot.obs.copy(),
    var=new_vars,
    uns=adata_2_prot.uns.copy(),
    obsm=adata_2_prot.obsm.copy(),
)
adata_2_prot.var["feature_type"] = ["protein"] * original_protein_num + [
    "CN"
] * neighbor_means.shape[1]
sc.pp.pca(adata_2_prot)
print(f"New adata shape (protein features + cell neighborhood vector): {adata_2_prot.shape}")

# %% Compute PCA and UMAP for Both Modalities
# Compute PCA and UMAP for both modalities
sc.pp.pca(adata_1_rna)
sc.pp.pca(adata_2_prot)
sc.pp.neighbors(adata_1_rna, key_added="original_neighbors", use_rep="X_pca")
sc.tl.umap(adata_1_rna, neighbors_key="original_neighbors")
adata_1_rna.obsm["X_original_umap"] = adata_1_rna.obsm["X_umap"]
sc.pp.neighbors(adata_2_prot, key_added="original_neighbors", use_rep="X_pca")
sc.tl.umap(adata_2_prot, neighbors_key="original_neighbors")
adata_2_prot.obsm["X_original_umap"] = adata_2_prot.obsm["X_umap"]

if plot_flag:
    plot_modality_embeddings(adata_1_rna, adata_2_prot)

# %% Convert Gene Names and Compute Module Scores
# Convert gene names to uppercase
adata_1_rna.var_names = adata_1_rna.var_names.str.upper()
adata_2_prot.var_names = adata_2_prot.var_names.str.upper()

# Compute gene module scores
sc.tl.score_genes(
    adata_1_rna, gene_list=terminal_exhaustion, score_name="terminal_exhaustion_score"
)

if plot_flag:
    sc.pl.umap(adata_1_rna, color="terminal_exhaustion_score", cmap="viridis")

# %% Compute PCA Dimensions
# Compute PCA dimensions
max_possible_pca_dim_rna = min(adata_1_rna.X.shape[1], adata_1_rna.X.shape[0])
max_possible_pca_dim_prot = min(adata_2_prot.X.shape[1], adata_2_prot.X.shape[0])
sc.pp.pca(adata_1_rna, n_comps=max_possible_pca_dim_rna - 1)
sc.pp.pca(adata_2_prot, n_comps=max_possible_pca_dim_prot - 1)

# Select PCA components based on variance explained
max_dim = 50
variance_ration_selected = 0.75

cumulative_variance_ratio = np.cumsum(adata_1_rna.uns["pca"]["variance_ratio"])
n_comps_thresh = np.argmax(cumulative_variance_ratio >= variance_ration_selected) + 1
n_comps_thresh = min(n_comps_thresh, max_dim)
if n_comps_thresh == 1:
    raise ValueError(
        "n_comps_thresh is 1, this is not good, try to lower the variance_ration_selected"
    )
real_ratio = np.cumsum(adata_1_rna.uns["pca"]["variance_ratio"])[n_comps_thresh]
sc.pp.pca(adata_1_rna, n_comps=n_comps_thresh)
print(f"\nNumber of components explaining {real_ratio} of rna variance: {n_comps_thresh}\n")

sc.pp.pca(adata_2_prot)
cumulative_variance_ratio = np.cumsum(adata_2_prot.uns["pca"]["variance_ratio"])
n_comps_thresh = np.argmax(cumulative_variance_ratio >= variance_ration_selected) + 1
n_comps_thresh = min(n_comps_thresh, max_dim)
real_ratio = np.cumsum(adata_2_prot.uns["pca"]["variance_ratio"])[n_comps_thresh]
sc.pp.pca(adata_1_rna, n_comps=n_comps_thresh)
print(f"\nNumber of components explaining {real_ratio} of protein variance: {n_comps_thresh}")
if n_comps_thresh == 1:
    raise ValueError(
        "n_comps_thresh is 1, this is not good, try to lower the variance_ration_selected"
    )

# %% Find Archetypes
# Find archetypes
archetype_list_protein = []
archetype_list_rna = []
converge = 1e-5
min_k = 7
max_k = 8
step_size = 1

# Store explained variances for plotting the elbow method
evs_protein = []
evs_rna = []

# Protein archetype detection
X_protein = adata_2_prot.obsm["X_pca"].T
total = (max_k - min_k) / step_size
for i, k in tqdm(
    enumerate(range(min_k, max_k, step_size)), total=total, desc="Protein Archetypes Detection"
):
    archetype, _, _, _, ev = PCHA(X_protein, noc=k)
    evs_protein.append(ev)
    archetype_list_protein.append(np.array(archetype).T)
    if i > 0 and evs_protein[i] - evs_protein[i - 1] < converge:
        print("Early stopping for Protein")
        break

# RNA archetype detection
X_rna = adata_1_rna.obsm["X_pca"].T
for j, k in tqdm(
    enumerate(range(min_k, max_k, step_size)), total=total, desc="RNA Archetypes Detection"
):
    if j > i:
        break
    archetype, _, _, _, ev = PCHA(X_rna, noc=k)
    evs_rna.append(ev)
    archetype_list_rna.append(np.array(archetype).T)
    if j > 0 and evs_rna[j] - evs_rna[j - 1] < converge:
        print("Early stopping for RNA")
        break

# Ensure both lists have the same length
min_len = min(len(archetype_list_protein), len(archetype_list_rna))
archetype_list_protein = archetype_list_protein[:min_len]
archetype_list_rna = archetype_list_rna[:min_len]

# Plot elbow method results
if plot_flag:
    plot_elbow_method(evs_protein, evs_rna)

# %% Get Cell Type Lists and Compute Archetype Proportions
# Get cell type lists
minor_cell_types_list_prot = sorted(list(set(adata_2_prot.obs["cell_types"])))
if "major_cell_types" not in adata_2_prot.obs.columns:
    adata_2_prot.obs["major_cell_types"] = adata_2_prot.obs["cell_types"]
major_cell_types_list_prot = sorted(list(set(adata_2_prot.obs["major_cell_types"])))
minor_cell_types_list_rna = sorted(list(set(adata_1_rna.obs["cell_types"])))
if "major_cell_types" not in adata_1_rna.obs.columns:
    adata_1_rna.obs["major_cell_types"] = adata_1_rna.obs["cell_types"]
major_cell_types_list_rna = sorted(list(set(adata_1_rna.obs["major_cell_types"])))

major_cell_types_amount_prot = [
    adata_2_prot.obs["major_cell_types"].value_counts()[cell_type]
    for cell_type in major_cell_types_list_prot
]
major_cell_types_amount_rna = [
    adata_1_rna.obs["major_cell_types"].value_counts()[cell_type]
    for cell_type in major_cell_types_list_rna
]
assert set(adata_1_rna.obs["major_cell_types"]) == set(adata_2_prot.obs["major_cell_types"])
archetype_proportion_list_rna, archetype_proportion_list_protein = [], []

# Compute archetype proportions
for archetypes_prot, archetypes_rna in tqdm(
    zip(archetype_list_protein, archetype_list_rna),
    total=len(archetype_list_protein),
    desc="Archetypes generating archetypes major cell types proportion vector ",
):
    weights_prot = get_cell_representations_as_archetypes_cvxpy(
        adata_2_prot.obsm["X_pca"], archetypes_prot
    )
    weights_rna = get_cell_representations_as_archetypes_cvxpy(
        adata_1_rna.obsm["X_pca"], archetypes_rna
    )

    archetypes_dim_prot = archetypes_prot.shape[1]
    archetype_num_prot = archetypes_prot.shape[0]
    archetypes_dim_rna = archetypes_rna.shape[1]
    archetype_num_rna = archetypes_rna.shape[0]

    prot_arch_prop = pd.DataFrame(
        np.zeros((archetype_num_prot, len(major_cell_types_list_prot))),
        columns=major_cell_types_list_prot,
    )
    rna_arch_prop = pd.DataFrame(
        np.zeros((archetype_num_rna, len(major_cell_types_list_rna))),
        columns=major_cell_types_list_rna,
    )
    archetype_cell_proportions = np.zeros((archetype_num_prot, len(major_cell_types_list_rna)))
    for curr_archetype in range(archetype_num_prot):
        df_rna = pd.DataFrame(
            [weights_prot[:, curr_archetype], adata_2_prot.obs["major_cell_types"].values],
            index=["weight", "major_cell_types"],
        ).T
        df_prot = pd.DataFrame(
            [weights_rna[:, curr_archetype], adata_1_rna.obs["major_cell_types"].values],
            index=["weight", "major_cell_types"],
        ).T
        df_rna = df_rna.groupby("major_cell_types")["weight"].sum()[major_cell_types_list_rna]
        df_prot = df_prot.groupby("major_cell_types")["weight"].sum()[major_cell_types_list_prot]
        rna_arch_prop.loc[curr_archetype, :] = df_rna.values / major_cell_types_amount_rna
        prot_arch_prop.loc[curr_archetype, :] = df_prot.values / major_cell_types_amount_prot

    prot_arch_prop = (prot_arch_prop.T / prot_arch_prop.sum(1)).T
    prot_arch_prop = prot_arch_prop / prot_arch_prop.sum(0)
    rna_arch_prop = (rna_arch_prop.T / rna_arch_prop.sum(1)).T
    rna_arch_prop = rna_arch_prop / rna_arch_prop.sum(0)
    archetype_proportion_list_rna.append(rna_arch_prop.copy())
    archetype_proportion_list_protein.append(prot_arch_prop.copy())

print(major_cell_types_amount_rna)
print(major_cell_types_amount_prot)

# %% Plot Archetype Proportions
# Plot archetype proportions
if plot_flag:
    plot_archetype_proportions(archetype_proportion_list_rna, archetype_proportion_list_protein)

    new_order_1 = reorder_rows_to_maximize_diagonal(archetype_proportion_list_rna[0])[1]
    new_order_2 = reorder_rows_to_maximize_diagonal(archetype_proportion_list_protein[0])[1]
    data1 = archetype_proportion_list_rna[0].iloc[new_order_1, :]
    data2 = archetype_proportion_list_protein[0].iloc[new_order_2, :]
    plot_archetypes_matching(data1, data2)

# %% Find Best Matching Archetypes
# Find best matching archetypes
(
    best_num_or_archetypes_index,
    best_total_cost,
    best_rna_archetype_order,
    best_protein_archetype_order,
) = find_best_pair_by_row_matching(
    copy.deepcopy(archetype_proportion_list_rna),
    copy.deepcopy(archetype_proportion_list_protein),
    metric="correlation",
)

print("\nBest pair found:")
print(f"Best index: {best_num_or_archetypes_index}")
print(f"Best total matching cost: {best_total_cost}")
print(f"Row indices (RNA): {best_rna_archetype_order}")
print(f"Matched row indices (Protein): {best_protein_archetype_order}")

# %% Reorder and Plot Best Matching Archetypes
# Reorder archetypes based on best matching
best_archetype_rna_prop = (
    archetype_proportion_list_rna[best_num_or_archetypes_index]
    .iloc[best_rna_archetype_order, :]
    .reset_index(drop=True)
)
best_archetype_prot_prop = (
    archetype_proportion_list_protein[best_num_or_archetypes_index]
    .iloc[best_protein_archetype_order, :]
    .reset_index(drop=True)
)
if plot_flag:
    plot_archetypes_matching(best_archetype_rna_prop, best_archetype_prot_prop, 8)

best_archetype_prot_prop.idxmax(axis=0)

if plot_flag:
    best_archetype_prot_prop.idxmax(axis=0).plot(
        kind="bar", color="red", hatch="\\", label="Protein"
    )
    best_archetype_rna_prop.idxmax(axis=0).plot(kind="bar", alpha=0.5, hatch="/", label="RNA")
    plt.title("show overlap of cell types proportions in archetypes")
    plt.legend()
    plt.xlabel("Major Cell Types")
    plt.ylabel("Proportion")
    plt.xticks(rotation=45)
    plt.show()
    compare_matchings(
        archetype_proportion_list_rna,
        archetype_proportion_list_protein,
        metric="cosine",
        num_trials=100,
    )

best_protein_archetype_order

# %% Get Cell Archetype Vectors
# Get cell archetype vectors
ordered_best_rna_archetype = archetype_list_rna[best_num_or_archetypes_index][
    best_protein_archetype_order, :
]
ordered_best_protein_archetype = archetype_list_protein[best_num_or_archetypes_index][
    best_rna_archetype_order, :
]
cells_archetype_vec_rna = get_cell_representations_as_archetypes_cvxpy(
    adata_1_rna.obsm["X_pca"], ordered_best_rna_archetype
)
cells_archetype_vec_prot = get_cell_representations_as_archetypes_cvxpy(
    adata_2_prot.obsm["X_pca"], ordered_best_protein_archetype
)

# Add archetype vectors to AnnData objects
adata_1_rna.obsm["archetype_vec"] = pd.DataFrame(
    cells_archetype_vec_rna,
    index=adata_1_rna.obs.index,
    columns=range(cells_archetype_vec_rna.shape[1]),
)
adata_2_prot.obsm["archetype_vec"] = pd.DataFrame(
    cells_archetype_vec_prot,
    index=adata_2_prot.obs.index,
    columns=range(cells_archetype_vec_prot.shape[1]),
)
adata_1_rna.obsm["archetype_vec"].columns = adata_1_rna.obsm["archetype_vec"].columns.astype(str)
adata_2_prot.obsm["archetype_vec"].columns = adata_2_prot.obsm["archetype_vec"].columns.astype(str)

# Add archetype labels
adata_1_rna.obs["archetype_label"] = pd.Categorical(np.argmax(cells_archetype_vec_rna, axis=1))
adata_2_prot.obs["archetype_label"] = pd.Categorical(np.argmax(cells_archetype_vec_prot, axis=1))
adata_1_rna.uns["archetypes"] = ordered_best_rna_archetype
adata_2_prot.uns["archetypes"] = ordered_best_protein_archetype

# %% Evaluate Distance Metrics
# Evaluate distance metrics
metrics = ["euclidean", "cityblock", "cosine", "correlation", "chebyshev"]
evaluate_distance_metrics(cells_archetype_vec_rna, cells_archetype_vec_prot, metrics)

# %% Plot Archetype Weights
# Plot archetype weights
if plot_flag:
    _, row_order = reorder_rows_to_maximize_diagonal(best_archetype_rna_prop)
    plot_archetype_weights(best_archetype_rna_prop, best_archetype_prot_prop, row_order)

# %% Create and Plot Archetype AnnData Objects
# Create archetype AnnData objects
adata_archetype_rna = AnnData(adata_1_rna.obsm["archetype_vec"])
adata_archetype_prot = AnnData(adata_2_prot.obsm["archetype_vec"])
adata_archetype_rna.obs = adata_1_rna.obs
adata_archetype_prot.obs = adata_2_prot.obs
adata_archetype_rna.index = adata_1_rna.obs.index
adata_archetype_prot.index = adata_2_prot.obs.index

# Plot archetype visualizations
if plot_flag:
    plot_archetype_visualizations(
        adata_archetype_rna, adata_archetype_prot, adata_1_rna, adata_2_prot
    )

# %% Save Results
# Save results
clean_uns_for_h5ad(adata_2_prot)
clean_uns_for_h5ad(adata_1_rna)
time_stamp = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
save_dir = "CODEX_RNA_seq/data/processed_data"
os.makedirs(save_dir, exist_ok=True)
adata_1_rna.write(f"{save_dir}/adata_rna_{time_stamp}.h5ad")
adata_2_prot.write(f"{save_dir}/adata_prot_{time_stamp}.h5ad")
adata_archetype_rna.write(f"{save_dir}/adata_archetype_rna_{time_stamp}.h5ad")
adata_archetype_prot.write(f"{save_dir}/adata_archetype_prot_{time_stamp}.h5ad")

# Load latest files
file_prefixes = ["adata_rna_", "adata_prot_", "adata_archetype_rna_", "adata_archetype_prot_"]
latest_files = {prefix: get_latest_file(save_dir, prefix) for prefix in file_prefixes}

# Check if any files were found
if any(v is None for v in latest_files.values()):
    print("Warning: Some files were not found. Skipping file loading.")
    print("Missing files:", [k for k, v in latest_files.items() if v is None])
else:
    adata_rna = sc.read(latest_files["adata_rna_"])
    adata_prot = sc.read(latest_files["adata_prot_"])
    adata_archetype_rna = sc.read(latest_files["adata_archetype_rna_"])
    adata_archetype_prot = sc.read(latest_files["adata_archetype_prot_"])

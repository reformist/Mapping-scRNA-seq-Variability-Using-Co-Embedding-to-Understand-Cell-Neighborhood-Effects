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
import json
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from matplotlib import pyplot as plt
from py_pcha import PCHA
from scipy.sparse import issparse
from scipy.stats import zscore
from tqdm import tqdm

# Add repository root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set working directory to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load config if exists
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    num_rna_cells = config["subsample"]["num_rna_cells"]
    num_protein_cells = config["subsample"]["num_protein_cells"]
    plot_flag = config["plot_flag"]
else:
    num_rna_cells = num_protein_cells = 2000
    plot_flag = True

import cell_lists
import plotting_functions as pf

import bar_nick_utils
import covet_utils

importlib.reload(cell_lists)
importlib.reload(pf)
importlib.reload(bar_nick_utils)
importlib.reload(covet_utils)

from plotting_functions import (
    plot_archetype_proportions,
    plot_archetype_visualizations,
    plot_archetype_weights,
    plot_elbow_method,
    plot_modality_embeddings,
    plot_neighbor_means,
    plot_spatial_clusters,
)

from bar_nick_utils import (
    clean_uns_for_h5ad,
    compare_matchings,
    evaluate_distance_metrics,
    find_best_pair_by_row_matching,
    get_cell_representations_as_archetypes_cvxpy,
    get_latest_file,
    plot_archetypes_matching,
    reorder_rows_to_maximize_diagonal,
)

# Set random seed for reproducibility
np.random.seed(8)

# Global variables
# plot_flag = True  # Removed as it's now loaded from config

# %% Load and Preprocess Data
# Load data
file_prefixes = ["preprocessed_adata_rna_", "preprocessed_adata_prot_"]
folder = "CODEX_RNA_seq/data/processed_data"

# Load the latest files
latest_files = {prefix: get_latest_file(folder, prefix) for prefix in file_prefixes}
adata_1_rna = sc.read(latest_files["preprocessed_adata_rna_"])
adata_2_prot = sc.read(latest_files["preprocessed_adata_prot_"])

# Subsample data
subsample_n_obs_rna = min(adata_1_rna.shape[0], num_rna_cells)
subsample_n_obs_protein = min(adata_2_prot.shape[0], num_protein_cells)
sc.pp.subsample(adata_1_rna, n_obs=subsample_n_obs_rna)
sc.pp.subsample(adata_2_prot, n_obs=subsample_n_obs_protein)

original_protein_num = adata_2_prot.X.shape[1]
print(f"data shape: {adata_1_rna.shape}, {adata_2_prot.shape}")
# %% Compute Spatial Neighbors and Means
# remove far away neighbors before setting up the neighbors means
sc.pp.neighbors(
    adata_2_prot, use_rep="spatial_location", key_added="spatial_neighbors", n_neighbors=15
)
connectivities = adata_2_prot.obsp["spatial_neighbors_connectivities"]
spatial_distances = adata_2_prot.obsp["spatial_neighbors_distances"]
if plot_flag:
    sns.histplot(spatial_distances.data)
    plt.title("Distribution of spatial distances between protein neighbors before cutoff")
    plt.show()
    plt.close()

# %%
percentile_threshold = 95
percentile_value = np.percentile(spatial_distances.data, percentile_threshold)
connectivities[spatial_distances > percentile_value] = 0.0
spatial_distances[spatial_distances > percentile_value] = 0.0
connectivities[connectivities > 0] = 1
adata_2_prot.obsp["spatial_neighbors_connectivities"] = connectivities
adata_2_prot.obsp["spatial_neighbors_distances"] = spatial_distances
if plot_flag:
    sns.heatmap(connectivities[:1000, :1000].todense())
    plt.show()
    sns.heatmap(spatial_distances[:1000, :1000].todense())
    plt.show()
    plt.close()

# Compute neighbor means
neighbor_sums = connectivities.dot(adata_2_prot.X)  # get the sum of all neighbors
if issparse(neighbor_sums):
    my_array = neighbor_sums.toarray()
neighbor_means = np.asarray(neighbor_sums / connectivities.sum(1))


# %% save umap of original protein data, save the umap in adata_2_prot.obsm["X_original_umap"]


if plot_flag:
    sns.histplot(adata_2_prot.obsp["spatial_neighbors_distances"].data)
    plt.title("Distribution of spatial distances between protein neighbors after cutoff")
    plt.show()
    plt.close()


# %% Compute Cell Neighborhoods
# Compute cell neighborhoods
normalized_data = zscore(neighbor_means, axis=0)

temp = AnnData(normalized_data)
temp.obs = adata_2_prot.obs
sc.pp.pca(temp)
sc.pp.neighbors(temp)
resolution = 0.1
while True:
    sc.tl.leiden(temp, resolution=resolution, key_added="CN")
    print(f"Resolution: {resolution}, num_clusters: {len(temp.obs['CN'].unique())}")

    num_clusters = len(temp.obs["CN"].unique())
    if num_clusters < 12:
        break
    resolution = resolution / 1.5
adata_2_prot.obs["CN"] = temp.obs["CN"]
num_clusters = len(adata_2_prot.obs["CN"].unique())
# Make sure to create a color palette with enough colors for all clusters
palette = sns.color_palette("tab10", num_clusters)
adata_2_prot.uns["spatial_clusters_colors"] = palette.as_hex()

# Convert CN to categorical with CN_ prefix
adata_2_prot.obs["CN"] = pd.Categorical(
    [f"CN_{cn}" for cn in adata_2_prot.obs["CN"]],
    categories=sorted([f"CN_{i}" for i in range(num_clusters)]),
)

# Verify that the colors and categories match
if "spatial_clusters_colors" in adata_2_prot.uns:
    if len(adata_2_prot.uns["spatial_clusters_colors"]) < len(
        adata_2_prot.obs["CN"].cat.categories
    ):
        # Add more colors if needed
        new_palette = sns.color_palette("tab10", len(adata_2_prot.obs["CN"].cat.categories))
        adata_2_prot.uns["spatial_clusters_colors"] = new_palette.as_hex()
    print(f"Number of categories: {len(adata_2_prot.obs['CN'].cat.categories)}")
    print(f"Number of colors: {len(adata_2_prot.uns['spatial_clusters_colors'])}")

if issparse(adata_2_prot.X):
    adata_2_prot.X = adata_2_prot.X.toarray()

# Plot neighbor means
if plot_flag:
    plot_neighbor_means(adata_2_prot, neighbor_means)
    plt.close()

# %% Plot Spatial Clusters
# Plot spatial clusters
if plot_flag:
    plot_spatial_clusters(adata_2_prot, neighbor_means)
    plt.close()

# %% Add CN Features to Protein Data
if False:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    neighbor_means = scaler.fit_transform(neighbor_means)
    print("Applied standard scaling to both feature types.")
    new_feature_names = [
        f"CN_{i}" for i in adata_2_prot.var.index
    ]  # in case we run the cell multiple times
    if adata_2_prot.X.shape[1] == neighbor_means.shape[1]:
        norm_original_protein_data = scaler.fit_transform(adata_2_prot.X)
        new_X = np.hstack([norm_original_protein_data, neighbor_means])
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

else:
    # Import neighborhood_utils module
    import tree_model

    importlib.reload(tree_model)
    from tree_model import analyze_residual_variation

    # Extract protein data
    if issparse(adata_2_prot.X):
        protein_data = adata_2_prot.X.toarray()
    else:
        protein_data = adata_2_prot.X.copy()

    # Get neighborhood statistics
    print("\nExtracting neighborhood feature statistics...")
    adata_2_prot, neighborhood_stats = analyze_residual_variation(
        adata_obj=adata_2_prot, plot=plot_flag, verbose=True
    )


# %%
sc.pp.pca(adata_2_prot)
print(f"New adata shape (protein features + cell neighborhood vector): {adata_2_prot.shape}")
# %% make sure prot data and cn data features are similar same scale and variance
# Compare statistical properties between protein features and cell neighborhood features
protein_mask = adata_2_prot.var["feature_type"] == "protein"
cn_mask = adata_2_prot.var["feature_type"] == "CN_projection"

# Extract data for each feature type
if issparse(adata_2_prot.X):
    protein_data = adata_2_prot.X[:, protein_mask].toarray()
    cn_data = adata_2_prot.X[:, cn_mask].toarray()
else:
    protein_data = adata_2_prot.X[:, protein_mask]
    cn_data = adata_2_prot.X[:, cn_mask]

# Calculate basic statistics
protein_stats = {
    "mean": np.mean(protein_data),
    "std": np.std(protein_data),
    "min": np.min(protein_data),
    "max": np.max(protein_data),
    "median": np.median(protein_data),
}

cn_stats = {
    "mean": np.mean(cn_data),
    "std": np.std(cn_data),
    "min": np.min(cn_data),
    "max": np.max(cn_data),
    "median": np.median(cn_data),
}

print("\nComparing statistical properties of protein vs CN features:")
print(f"{'Statistic':10} {'Protein':15} {'CN':15} {'Ratio (Protein/CN)':20}")
print("-" * 60)
for stat in protein_stats:
    ratio = protein_stats[stat] / cn_stats[stat] if cn_stats[stat] != 0 else float("inf")
    print(f"{stat:10} {protein_stats[stat]:<15.4f} {cn_stats[stat]:<15.4f} {ratio:<20.4f}")

# Visual comparison of distributions
if plot_flag:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution of means per feature
    protein_feature_means = np.mean(protein_data, axis=0)
    cn_feature_means = np.mean(cn_data, axis=0)
    axes[0].hist(protein_feature_means, alpha=0.5, bins=30, label="Protein Features")
    axes[0].hist(cn_feature_means, alpha=0.5, bins=30, label="CN Features")
    axes[0].set_title("Distribution of Feature Means")
    axes[0].set_xlabel("Mean Value")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # Distribution of variances per feature
    protein_feature_vars = np.var(protein_data, axis=0)
    cn_feature_vars = np.var(cn_data, axis=0)
    axes[1].hist(protein_feature_vars, alpha=0.5, bins=30, label="Protein Features")
    axes[1].hist(cn_feature_vars, alpha=0.5, bins=30, label="CN Features")
    axes[1].set_title("Distribution of Feature Variances")
    axes[1].set_xlabel("Variance")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    plt.close()

    # Boxplot comparison
    plt.figure(figsize=(10, 6))
    data_to_plot = [protein_feature_means, cn_feature_means, protein_feature_vars, cn_feature_vars]
    labels = ["Protein Means", "CN Means", "Protein Variances", "CN Variances"]
    plt.boxplot(data_to_plot, labels=labels)
    plt.title("Comparison of Feature Statistics")
    plt.ylabel("Value")
    plt.yscale("log")  # Log scale to better visualize differences
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()

# If statistics are very different, consider scaling
scale_threshold = 10  # Define threshold for when scaling is needed
if (
    protein_stats["std"] / cn_stats["std"] > scale_threshold
    or cn_stats["std"] / protein_stats["std"] > scale_threshold
):
    print("\nWARNING: Large difference in variance between protein and CN features!")
    print("Consider scaling features before PCA to prevent bias.")

    # Optional: Perform scaling here if needed


# %% Compute PCA and UMAP for Both Modalities


minor_cell_types_list_prot = sorted(list(set(adata_2_prot.obs["cell_types"])))
if "major_cell_types" not in adata_2_prot.obs.columns:
    adata_2_prot.obs["major_cell_types"] = adata_2_prot.obs["cell_types"]
major_cell_types_list_prot = sorted(list(set(adata_2_prot.obs["major_cell_types"])))
if "major_cell_types" not in adata_1_rna.obs.columns:
    adata_1_rna.obs["major_cell_types"] = adata_1_rna.obs["cell_types"]
    major_cell_types_list_rna = sorted(list(set(adata_1_rna.obs["major_cell_types"])))
minor_cell_types_list_rna = sorted(list(set(adata_1_rna.obs["cell_types"])))

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
    plt.close()

# %% Convert Gene Names and Compute Module Scores
# Convert gene names to uppercase
adata_1_rna.var_names = adata_1_rna.var_names.str.upper()
adata_2_prot.var_names = adata_2_prot.var_names.str.upper()

# Compute gene module scores
# sc.tl.score_genes(
#     adata_1_rna, gene_list=terminal_exhaustion, score_name="terminal_exhaustion_score"
# )

# if plot_flag:
#     sc.pl.umap(adata_1_rna, color="terminal_exhaustion_score", cmap="viridis")
#     plt.close()

# %% Compute PCA Dimensions
# Compute PCA dimensions
print("\nComputing PCA dimensions...")
max_possible_pca_dim_rna = min(adata_1_rna.X.shape[1], adata_1_rna.X.shape[0])
max_possible_pca_dim_prot = min(adata_2_prot.X.shape[1], adata_2_prot.X.shape[0])
sc.pp.pca(adata_1_rna, n_comps=max_possible_pca_dim_rna - 1)
sc.pp.pca(adata_2_prot, n_comps=max_possible_pca_dim_prot - 1)
# %%
# Select PCA components based on variance explained
print("Selecting PCA components...")
max_dim = 50
variance_ratio_selected = 0.80  # was 0.75

cumulative_variance_ratio = np.cumsum(adata_1_rna.uns["pca"]["variance_ratio"])
n_comps_thresh = np.argmax(cumulative_variance_ratio >= variance_ratio_selected) + 1
n_comps_thresh = min(n_comps_thresh, max_dim)
if n_comps_thresh == 1:
    raise ValueError(
        "n_comps_thresh is 1, this is not good, try to lower the variance_ratio_selected"
    )
real_ratio = np.cumsum(adata_1_rna.uns["pca"]["variance_ratio"])[n_comps_thresh]
sc.pp.pca(adata_1_rna, n_comps=n_comps_thresh)
print(f"\nNumber of components explaining {real_ratio} of rna variance: {n_comps_thresh}\n")

cumulative_variance_ratio = np.cumsum(adata_2_prot.uns["pca"]["variance_ratio"])
n_comps_thresh = np.argmax(cumulative_variance_ratio >= variance_ratio_selected) + 1
n_comps_thresh = min(n_comps_thresh, max_dim)
real_ratio = np.cumsum(adata_2_prot.uns["pca"]["variance_ratio"])[n_comps_thresh]
sc.pp.pca(adata_2_prot, n_comps=n_comps_thresh)
print(f"\nNumber of components explaining {real_ratio} of protein variance: {n_comps_thresh}")
if n_comps_thresh == 1:
    raise ValueError(
        "n_comps_thresh is 1, this is not good, try to lower the variance_ratio_selected"
    )
# %% plot umap of original protein data and the umap to new protein data
sc.pp.neighbors(adata_2_prot, n_neighbors=15, use_rep="X_pca")

if plot_flag:
    sc.tl.umap(adata_2_prot)
    sc.pl.embedding(
        adata_2_prot, basis="X_original_umap", color="cell_types", title="Original Protein UMAP"
    )
    plt.close()
    sc.pl.embedding(adata_2_prot, basis="X_umap", color="cell_types", title="New Protein UMAP")
    plt.close()
    print(
        "if those two plots are similar, that means that the new CN features are not affecting the protein data"
    )

# %%
# Plot heatmap of PCA feature contributions
if plot_flag:
    # RNA PCA feature contributions
    rna_pca_components = adata_1_rna.varm["PCs"]
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        rna_pca_components,
        cmap="viridis",
        center=0,
        xticklabels=range(1, rna_pca_components.shape[1] + 1),
        yticklabels=False,
    )
    plt.title("RNA: Feature Contributions to PCA Dimensions")
    plt.xlabel("PCA Dimensions")
    plt.ylabel("Original Features")
    plt.savefig(
        "CODEX_RNA_seq/plots/rna_pca_feature_contributions.png", dpi=300, bbox_inches="tight"
    )
    plt.show()
    plt.close()

    # Check feature contribution balance
    feature_total_contribution = np.abs(rna_pca_components).sum(axis=1)
    half_point = len(feature_total_contribution) // 2
    first_half_contrib = feature_total_contribution[:half_point].sum()
    second_half_contrib = feature_total_contribution[half_point:].sum()
    print(f"RNA PCA feature contribution balance:")
    print(f"First half contribution: {first_half_contrib:.2f}")
    print(f"Second half contribution: {second_half_contrib:.2f}")
    print(f"Ratio (first:second): {first_half_contrib/second_half_contrib:.2f}")

    # Protein PCA feature contributions
    prot_pca_components = adata_2_prot.varm["PCs"]
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        prot_pca_components,
        cmap="viridis",
        center=0,
        xticklabels=range(1, prot_pca_components.shape[1] + 1),
        yticklabels=False,
    )
    plt.title("Protein: Feature Contributions to PCA Dimensions")
    plt.xlabel("PCA Dimensions")
    plt.ylabel("Original Features")
    plt.savefig(
        "CODEX_RNA_seq/plots/protein_pca_feature_contributions.png", dpi=300, bbox_inches="tight"
    )
    plt.show()
    plt.close()

    # Check feature contribution balance
    feature_total_contribution = np.abs(prot_pca_components).sum(axis=1)
    half_point = len(feature_total_contribution) // 2
    first_half_contrib = feature_total_contribution[:half_point].sum()
    second_half_contrib = feature_total_contribution[half_point:].sum()
    print(f"Protein PCA feature contribution balance:")
    print(f"First half contribution: {first_half_contrib:.2f}")
    print(f"Second half contribution: {second_half_contrib:.2f}")
    print(f"Ratio (first:second): {first_half_contrib/second_half_contrib:.2f}")

    # Analyze contributions per feature type for protein data
    if "feature_type" in adata_2_prot.var:
        feature_types = adata_2_prot.var["feature_type"].unique()
        for ft in feature_types:
            mask = adata_2_prot.var["feature_type"] == ft
            ft_contribution = np.abs(prot_pca_components[mask]).sum()
            print(
                f"Contribution from {ft} features: {ft_contribution:.2f} "
                + f"({ft_contribution/np.abs(prot_pca_components).sum()*100:.2f}%)"
            )


# %% Find Archetypes
print("\nFinding archetypes...")
archetype_list_protein = []
archetype_list_rna = []
converge = 1e-5
min_k = 8
max_k = 9
step_size = 1

# Store explained variances for plotting the elbow method
evs_protein = []
evs_rna = []

# Protein archetype detection
print("Computing protein archetypes...")
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
print("Computing RNA archetypes...")
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
    plt.close()

# %% Get Cell Type Lists and Compute Archetype Proportions
print("\nComputing archetype proportions...")
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
print("Generating archetype proportions...")
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
    plt.close()

# Calculate diagonal maximization for archetype matching
new_order_1 = reorder_rows_to_maximize_diagonal(archetype_proportion_list_rna[0])[1]
new_order_2 = reorder_rows_to_maximize_diagonal(archetype_proportion_list_protein[0])[1]
data1 = archetype_proportion_list_rna[0].iloc[new_order_1, :]
data2 = archetype_proportion_list_protein[0].iloc[new_order_2, :]

# Plot archetype matching if plot_flag is enabled
if plot_flag:
    plot_archetypes_matching(data1, data2)
    plt.close()

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
    plt.close()

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
    plt.close()
    compare_matchings(
        archetype_proportion_list_rna,
        archetype_proportion_list_protein,
        metric="cosine",
        num_trials=100,
    )
    plt.close()

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
    plt.close()

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
    plt.close()

# %% Save Results
# Save results
print("\nSaving results...")
clean_uns_for_h5ad(adata_2_prot)
clean_uns_for_h5ad(adata_1_rna)
time_stamp = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
save_dir = "CODEX_RNA_seq/data/processed_data"
os.makedirs(save_dir, exist_ok=True)
adata_1_rna.write(f"{save_dir}/adata_rna_archetype_generated_{time_stamp}.h5ad")
adata_2_prot.write(f"{save_dir}/adata_prot_archetype_generated_{time_stamp}.h5ad")
print(f"\nRNA data dimensions: {adata_1_rna.shape[0]} samples x {adata_1_rna.shape[1]} features")
print(
    f"Protein data dimensions: {adata_2_prot.shape[0]} samples x {adata_2_prot.shape[1]} features"
)
print(
    f"RNA archetype dimensions: {adata_archetype_rna.shape[0]} samples x {adata_archetype_rna.shape[1]} features"
)
print(
    f"Protein archetype dimensions: {adata_archetype_prot.shape[0]} samples x {adata_archetype_prot.shape[1]} features\n"
)

# Load latest files
file_prefixes = ["adata_rna_archetype_generated_", "adata_prot_archetype_generated_"]
latest_files = {prefix: get_latest_file(save_dir, prefix) for prefix in file_prefixes}

# Check if any files were found
if any(v is None for v in latest_files.values()):
    print("Warning: Some files were not found. Skipping file loading.")
    print("Missing files:", [k for k, v in latest_files.items() if v is None])


# %%

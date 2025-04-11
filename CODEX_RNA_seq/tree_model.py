#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %% Imports and setup
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Add repository root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set working directory to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# %% Load the preprocessed data with neighbor means
print("Loading preprocessed protein data with neighbor information...")
adata_prot = sc.read(
    "CODEX_RNA_seq/data/processed_data/preprocessed_adata_prot_neighbors_means.h5ad"
)
print(f"Loaded data with shape: {adata_prot.shape}")

# Check if we have spatial neighbors information
if "spatial_neighbors_connectivities" in adata_prot.obsp:
    print("Spatial neighbors information found.")
    connectivities = adata_prot.obsp["spatial_neighbors_connectivities"]
else:
    print("No spatial neighbors found. Computing spatial neighbors...")
    sc.pp.neighbors(
        adata_prot, use_rep="spatial_location", key_added="spatial_neighbors", n_neighbors=15
    )
    connectivities = adata_prot.obsp["spatial_neighbors_connectivities"]

# %% Identify protein and CN features
# Get the number of protein features
if "feature_type" in adata_prot.var.columns:
    protein_mask = adata_prot.var["feature_type"] == "protein"
    cn_mask = adata_prot.var["feature_type"] == "CN"

    print(f"Found {np.sum(protein_mask)} protein features and {np.sum(cn_mask)} CN features")

    # Extract protein and CN data
    if issparse(adata_prot.X):
        protein_data = adata_prot.X[:, protein_mask].toarray()
        cn_data = adata_prot.X[:, cn_mask].toarray()
    else:
        protein_data = adata_prot.X[:, protein_mask]
        cn_data = adata_prot.X[:, cn_mask]
else:
    # If feature_type not present, assume all original features are proteins
    original_protein_num = adata_prot.X.shape[1]
    print(
        f"No feature type annotation found. Assuming all {original_protein_num} features are proteins."
    )
    if issparse(adata_prot.X):
        protein_data = adata_prot.X.toarray()
    else:
        protein_data = adata_prot.X

    # Compute neighbor means
    print("Computing neighbor means...")
    neighbor_sums = connectivities.dot(protein_data)
    neighbor_means = np.asarray(neighbor_sums / connectivities.sum(1))
    cn_data = neighbor_means
# %% Plot the protein data
print("Plotting original protein data...")
toplot = sc.AnnData(protein_data)
toplot.obs = adata_prot.obs.copy()  # Copy the entire obs dataframe to maintain index and data types

# Make sure cell_types is categorical
if not pd.api.types.is_categorical_dtype(toplot.obs["cell_types"]):
    toplot.obs["cell_types"] = toplot.obs["cell_types"].astype("category")

# Check how many cells have cell types assigned
print(f"Number of cells with assigned cell types: {toplot.obs['cell_types'].value_counts().sum()}")
print(f"Total number of cells: {toplot.shape[0]}")
print(f"Cell type distribution: {toplot.obs['cell_types'].value_counts()}")

# Run PCA and plot
sc.pp.pca(toplot)
sc.pl.pca(toplot, color="cell_types")
plt.savefig("CODEX_RNA_seq/plots/original_protein_pca.png", dpi=300, bbox_inches="tight")

# Run UMAP and plot
sc.pp.neighbors(toplot)
sc.tl.umap(toplot)
sc.pl.umap(toplot, color="cell_types")
plt.savefig("CODEX_RNA_seq/plots/original_protein_umap.png", dpi=300, bbox_inches="tight")
plt.show()
# %% Standardize data for comparison
scaler = StandardScaler()
norm_protein_data = scaler.fit_transform(protein_data)
norm_cn_data = scaler.fit_transform(cn_data)

# %% Approach 1: Residual Modeling to explain intra-cluster variation
print("\n--- Approach 1: Residual Modeling ---")

# Get cell types
cell_types = adata_prot.obs["cell_types"].values
unique_cell_types = adata_prot.obs["cell_types"].unique()

# Create dummy variables for cell types
cell_type_dummies = pd.get_dummies(cell_types)

# Train a model to predict protein expression based on cell type
print("Training model to predict protein expression based on cell type...")
model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(cell_type_dummies, norm_protein_data)
predicted_baseline = model.predict(cell_type_dummies)

# Calculate residuals (variations not explained by cell type)
residuals = norm_protein_data - predicted_baseline

# Use canonical correlation analysis to find relationships between CN features and residual protein expression
print("Finding correlations between CN features and residual protein expression...")
min_dim = min(norm_cn_data.shape[1], residuals.shape[1])
n_components = min(min_dim, 10)  # Cap at 10 components

# Fit CCA
cca = CCA(n_components=n_components)
cca.fit(norm_cn_data, residuals)

# Project CN data onto directions that best explain residuals
cn_projection = cca.transform(norm_cn_data)

# Create enhanced feature space
enhanced_features = np.hstack([norm_protein_data, cn_projection])

# UMAP to visualize enhanced features
print("Generating UMAP visualization...")
temp_adata = sc.AnnData(enhanced_features)
temp_adata.obs = adata_prot.obs
sc.pp.neighbors(temp_adata)
sc.tl.umap(temp_adata)

# Evaluate if CN features explain intra-cluster variation
print("\nEvaluating how well CN features explain intra-cluster variation:")
r2_values = {}
for cell_type in unique_cell_types:
    mask = cell_types == cell_type
    if sum(mask) > 10:  # Only analyze clusters with enough cells
        cluster_residuals = residuals[mask]
        cluster_cn_proj = cn_projection[mask]

        # Calculate R² for this cluster
        reg = LinearRegression().fit(cluster_cn_proj, cluster_residuals)
        r2 = reg.score(cluster_cn_proj, cluster_residuals)
        r2_values[cell_type] = r2
        print(f"Cell type {cell_type}: CN explains {r2:.2f} of residual variation")

# %% Visualize the results of residual modeling

# Plot 1: UMAP of enhanced features colored by cell type
print("Plotting UMAP visualization...")
sc.pl.umap(temp_adata, color="cell_types", title="Enhanced Features (Protein + CN Projection)")
plt.savefig("CODEX_RNA_seq/plots/enhanced_features_umap.png", dpi=300, bbox_inches="tight")

# Plot 2: R² values for each cell type
plt.figure(figsize=(10, 6))
cell_types_plot = list(r2_values.keys())
r2_vals_plot = list(r2_values.values())
sorted_indices = np.argsort(r2_vals_plot)[::-1]
plt.bar(range(len(cell_types_plot)), [r2_vals_plot[i] for i in sorted_indices])
plt.xticks(range(len(cell_types_plot)), [cell_types_plot[i] for i in sorted_indices], rotation=90)
plt.title("Explained Residual Variation by Cell Type")
plt.ylabel("R² Value")
plt.tight_layout()
plt.savefig(
    "CODEX_RNA_seq/plots/explained_variation_by_cell_type.png", dpi=300, bbox_inches="tight"
)
plt.show()

# Plot 3: Detailed analysis for top cell types with highest R²
top_cell_types = [cell_types_plot[i] for i in sorted_indices[:2]]
for i, cell_type in enumerate(top_cell_types):
    plt.figure(figsize=(8, 6))
    mask = cell_types == cell_type

    # Project data to 2D using CCA components
    projection = cn_projection[mask][:, :2]

    # Get one protein feature to visualize (pick the one most correlated with CN projection)
    protein_idx = np.argmax(
        np.abs(
            np.corrcoef(residuals[mask].T, projection[:, 0])[
                residuals[mask].shape[1], : residuals[mask].shape[1]
            ]
        )
    )
    protein_values = residuals[mask][:, protein_idx]

    # Create scatter plot
    plt.scatter(
        projection[:, 0], projection[:, 1], c=protein_values, cmap="viridis", s=50, alpha=0.8
    )
    plt.colorbar(label=f"Residual Protein Expression")
    plt.title(f"Cell Type: {cell_type}")
    plt.xlabel("CN Projection 1")
    plt.ylabel("CN Projection 2")
    plt.tight_layout()
    plt.savefig(
        f"CODEX_RNA_seq/plots/cell_type_{cell_type}_projection.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

# %% Try to implement GNN approach if PyTorch Geometric is available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    has_torch_geometric = True
    print("\n--- Approach 2: Graph Neural Networks ---")
    print("PyTorch Geometric is available, implementing GNN approach...")

    # Prepare a graph from spatial neighbors
    edges = np.array(connectivities.nonzero())
    edge_index = torch.from_numpy(edges).long()

    # Convert features to PyTorch tensors
    protein_features = torch.from_numpy(norm_protein_data).float()
    cn_features = torch.from_numpy(norm_cn_data).float()

    # Define a GNN model
    class SpatialAwareGNN(nn.Module):
        def __init__(self, protein_dim, cn_dim, hidden_dim=64, latent_dim=32):
            super().__init__()
            # Protein feature processing
            self.protein_encoder = nn.Sequential(nn.Linear(protein_dim, hidden_dim), nn.ReLU())

            # CN feature processing
            self.cn_encoder = nn.Sequential(nn.Linear(cn_dim, hidden_dim // 2), nn.ReLU())

            # Graph convolution layers
            self.conv1 = GCNConv(hidden_dim + hidden_dim // 2, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, latent_dim)

            # Decoder for reconstruction
            self.decoder = nn.Linear(latent_dim, protein_dim)

        def forward(self, protein_data, cn_data, edge_index):
            # Encode features separately
            h_protein = self.protein_encoder(protein_data)
            h_cn = self.cn_encoder(cn_data)

            # Concatenate features
            h = torch.cat([h_protein, h_cn], dim=1)

            # Apply graph convolutions
            h = self.conv1(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=0.2, training=self.training)

            h = self.conv2(h, edge_index)
            h = F.relu(h)

            # Final embedding
            embedding = self.conv3(h, edge_index)

            # Reconstruction
            reconstruction = self.decoder(embedding)

            return embedding, reconstruction

    # Create model
    model = SpatialAwareGNN(protein_dim=norm_protein_data.shape[1], cn_dim=norm_cn_data.shape[1])

    # Just run a forward pass to get embeddings (no training in this simple demo)
    print("Running forward pass through GNN...")
    with torch.no_grad():
        embeddings, reconstructions = model(protein_features, cn_features, edge_index)
        embeddings_np = embeddings.numpy()

    # Visualize GNN embeddings
    temp_adata_gnn = sc.AnnData(embeddings_np)
    temp_adata_gnn.obs = adata_prot.obs
    sc.pp.neighbors(temp_adata_gnn)
    sc.tl.umap(temp_adata_gnn)

    # Plot GNN embeddings by cell type (separate plot)
    print("Plotting GNN embeddings by cell type...")
    sc.pl.umap(temp_adata_gnn, color="cell_types", title="GNN Embeddings by Cell Type")
    plt.savefig("CODEX_RNA_seq/plots/gnn_embeddings_by_cell_type.png", dpi=300, bbox_inches="tight")

    # Compare with original PCA
    print("Plotting original PCA...")
    sc.pp.pca(adata_prot)
    sc.pp.neighbors(adata_prot)
    sc.tl.umap(adata_prot)
    sc.pl.umap(adata_prot, color="cell_types", title="Original PCA")
    plt.savefig("CODEX_RNA_seq/plots/original_pca_umap.png", dpi=300, bbox_inches="tight")

    # Plot spatial embeddings
    print("Plotting spatial embeddings with GNN dimensions...")
    plt.figure(figsize=(10, 8))
    plt.scatter(
        adata_prot.obsm["spatial_location"][:, 0],
        adata_prot.obsm["spatial_location"][:, 1],
        c=embeddings_np[:, 0],
        cmap="viridis",
        s=10,
    )
    plt.title("Spatial Plot Colored by GNN Embedding (Dim 0)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.colorbar(label="GNN Embedding Value")
    plt.tight_layout()
    plt.savefig("CODEX_RNA_seq/plots/spatial_gnn_dim0.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.scatter(
        adata_prot.obsm["spatial_location"][:, 0],
        adata_prot.obsm["spatial_location"][:, 1],
        c=embeddings_np[:, 1],
        cmap="viridis",
        s=10,
    )
    plt.title("Spatial Plot Colored by GNN Embedding (Dim 1)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.colorbar(label="GNN Embedding Value")
    plt.tight_layout()
    plt.savefig("CODEX_RNA_seq/plots/spatial_gnn_dim1.png", dpi=300, bbox_inches="tight")
    plt.show()

except ImportError:
    print("\nPyTorch Geometric not available. Skipping GNN approach.")
    print("To implement GNN approach, install with: pip install torch torch-geometric")
    has_torch_geometric = False

# %% Save enhanced data for future use
output_dir = "CODEX_RNA_seq/data/processed_data"
os.makedirs(output_dir, exist_ok=True)

# Save the enhanced data with residual modeling results
adata_enhanced = sc.AnnData(enhanced_features)
adata_enhanced.obs = adata_prot.obs.copy()
adata_enhanced.var_names = [f"feature_{i}" for i in range(enhanced_features.shape[1])]
adata_enhanced.var["feature_type"] = ["protein"] * norm_protein_data.shape[1] + [
    "cn_projection"
] * cn_projection.shape[1]
adata_enhanced.obsm["X_umap"] = temp_adata.obsm["X_umap"]
adata_enhanced.obsm["spatial_location"] = adata_prot.obsm["spatial_location"]
adata_enhanced.uns["residual_modeling_r2"] = r2_values

# Also save GNN embeddings if available
if has_torch_geometric:
    adata_enhanced.obsm["X_gnn"] = embeddings_np
    adata_enhanced.obsm["X_umap_gnn"] = temp_adata_gnn.obsm["X_umap"]

# Save the enhanced AnnData object
output_file = os.path.join(output_dir, "enhanced_protein_data_with_cn_analysis.h5ad")
adata_enhanced.write_h5ad(output_file)
print(f"\nEnhanced data saved to: {output_file}")

print("\nAnalysis complete!")

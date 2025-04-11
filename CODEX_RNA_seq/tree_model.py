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


def analyze_residual_variation(
    adata_obj=None, adata_path=None, output_dir="CODEX_RNA_seq/plots", plot=True, verbose=True
):
    """
    Analyze residual variation in protein data using cell neighborhood (CN) information.

    Parameters
    ----------
    adata_obj : AnnData, optional
        AnnData object with protein and cell neighborhood (CN) data. This should be a pre-concatenated
        object with both protein and CN data as columns in the .X matrix and identified by the
        'feature_type' column in .var.
    adata_path : str, optional
        Path to the preprocessed h5ad file with protein and CN data (used only if adata_obj is None)
    output_dir : str, default="CODEX_RNA_seq/plots"
        Directory to save output plots
    plot : bool, default=True
        Whether to generate and save plots
    verbose : bool, default=True
        Whether to print progress information

    Returns
    -------
    AnnData
        New AnnData object with the original data plus added CN projection features (CN_0, CN_1, ...)
        that capture residual variation explained by cell neighborhood effects.
    dict
        Dictionary containing additional results of the analysis:
        - r2_values: R² values for each cell type
        - cca: Trained CCA model
        - cn_projection: Raw CN data projected onto CCA components
        - residuals: Residual protein expression not explained by cell type
    """
    # Ensure output directory exists
    if plot and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the preprocessed data
    if verbose:
        print("Processing input data...")

    if adata_obj is not None:
        adata_prot = adata_obj
    elif adata_path is not None:
        adata_prot = sc.read(adata_path)
    else:
        adata_prot = sc.read(
            "CODEX_RNA_seq/data/processed_data/preprocessed_adata_prot_neighbors_means.h5ad"
        )

    if verbose:
        print(f"Working with data of shape: {adata_prot.shape}")

    # Check if we have spatial neighbors information
    if "spatial_neighbors_connectivities" in adata_prot.obsp:
        if verbose:
            print("Spatial neighbors information found.")
        connectivities = adata_prot.obsp["spatial_neighbors_connectivities"]
    else:
        if verbose:
            print("No spatial neighbors found. Computing spatial neighbors...")
        sc.pp.neighbors(
            adata_prot, use_rep="spatial_location", key_added="spatial_neighbors", n_neighbors=15
        )
        connectivities = adata_prot.obsp["spatial_neighbors_connectivities"]

    # Identify protein and CN features
    if "feature_type" in adata_prot.var.columns:
        protein_mask = adata_prot.var["feature_type"] == "protein"
        cn_mask = adata_prot.var["feature_type"] == "CN"

        if verbose:
            print(
                f"Found {np.sum(protein_mask)} protein features and {np.sum(cn_mask)} CN features"
            )

        # Extract protein and CN data
        if issparse(adata_prot.X):
            protein_data = adata_prot.X[:, protein_mask].toarray()
            cn_data = adata_prot.X[:, cn_mask].toarray()
        else:
            protein_data = adata_prot.X[:, protein_mask]
            cn_data = adata_prot.X[:, cn_mask]
    else:
        raise ValueError(
            "Input AnnData object must have a 'feature_type' column in .var identifying 'protein' and 'CN' features"
        )

    # Plot the protein data
    if plot:
        if verbose:
            print("Plotting original protein data...")
        toplot = sc.AnnData(protein_data)
        toplot.obs = (
            adata_prot.obs.copy()
        )  # Copy the entire obs dataframe to maintain index and data types

        # Make sure cell_types is categorical
        if not pd.api.types.is_categorical_dtype(toplot.obs["cell_types"]):
            toplot.obs["cell_types"] = toplot.obs["cell_types"].astype("category")

        if verbose:
            print(
                f"Number of cells with assigned cell types: {toplot.obs['cell_types'].value_counts().sum()}"
            )
            print(f"Total number of cells: {toplot.shape[0]}")
            print(f"Cell type distribution: {toplot.obs['cell_types'].value_counts()}")

        # Run PCA and plot
        sc.pp.pca(toplot)
        sc.pl.pca(toplot, color="cell_types")
        plt.savefig(f"{output_dir}/original_protein_pca.png", dpi=300, bbox_inches="tight")

        # Run UMAP and plot
        sc.pp.neighbors(toplot)
        sc.tl.umap(toplot)
        sc.pl.umap(toplot, color="cell_types")
        plt.savefig(f"{output_dir}/original_protein_umap.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Standardize data for comparison
    scaler = StandardScaler()
    norm_protein_data = scaler.fit_transform(protein_data)
    norm_cn_data = scaler.fit_transform(cn_data)

    # Residual Modeling to explain intra-cluster variation
    if verbose:
        print("\n--- Approach 1: Residual Modeling ---")

    # Get cell types
    cell_types = adata_prot.obs["cell_types"].values
    unique_cell_types = adata_prot.obs["cell_types"].unique()

    # Create dummy variables for cell types
    cell_type_dummies = pd.get_dummies(cell_types)

    # Train a model to predict protein expression based on cell type
    if verbose:
        print("Training model to predict protein expression based on cell type...")
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(cell_type_dummies, norm_protein_data)
    predicted_baseline = model.predict(cell_type_dummies)

    # Calculate residuals (variations not explained by cell type)
    residuals = norm_protein_data - predicted_baseline

    # Use canonical correlation analysis to find relationships between CN features and residual protein expression
    if verbose:
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

    # Create new AnnData object with original proteins and CN projections
    if verbose:
        print("Creating new AnnData object with CN projections...")

    # Get protein feature names
    protein_var_names = adata_prot.var_names[protein_mask]

    # Create new variable names for CN projections
    cn_proj_names = [f"CN_{i}" for i in range(cn_projection.shape[1])]

    # Create a new var dataframe
    new_var = pd.DataFrame(index=list(protein_var_names) + cn_proj_names)

    # Add feature type annotations
    new_var["feature_type"] = "protein"
    new_var.loc[cn_proj_names, "feature_type"] = "CN_projection"

    # For any other var columns in the original data that apply to proteins, copy them
    for col in adata_prot.var.columns:
        if col != "feature_type":
            new_var[col] = np.nan
            new_var.loc[protein_var_names, col] = adata_prot.var.loc[protein_var_names, col].values

    # Create the new AnnData object
    if issparse(adata_prot.X):
        # If original was sparse, we'll keep the new one sparse too
        from scipy.sparse import csr_matrix, hstack

        protein_mat = adata_prot.X[:, protein_mask]
        cn_proj_mat = csr_matrix(cn_projection)
        new_X = hstack([protein_mat, cn_proj_mat])
        new_adata = sc.AnnData(new_X, obs=adata_prot.obs.copy(), var=new_var)
    else:
        # Otherwise use numpy arrays
        protein_mat = protein_data
        combined_data = np.hstack([protein_mat, cn_projection])
        new_adata = sc.AnnData(combined_data, obs=adata_prot.obs.copy(), var=new_var)

    # Copy over other important annotations from original AnnData
    if hasattr(adata_prot, "obsm") and adata_prot.obsm:
        for key in adata_prot.obsm.keys():
            new_adata.obsm[key] = (
                adata_prot.obsm[key].copy()
                if hasattr(adata_prot.obsm[key], "copy")
                else adata_prot.obsm[key]
            )

    if hasattr(adata_prot, "obsp") and adata_prot.obsp:
        for key in adata_prot.obsp.keys():
            new_adata.obsp[key] = (
                adata_prot.obsp[key].copy()
                if hasattr(adata_prot.obsp[key], "copy")
                else adata_prot.obsp[key]
            )

    if hasattr(adata_prot, "uns") and adata_prot.uns:
        for key in adata_prot.uns.keys():
            new_adata.uns[key] = (
                adata_prot.uns[key].copy()
                if hasattr(adata_prot.uns[key], "copy")
                else adata_prot.uns[key]
            )

    # UMAP to visualize enhanced features
    if verbose:
        print("Generating UMAP visualization...")
    temp_adata = sc.AnnData(enhanced_features)
    temp_adata.obs = adata_prot.obs
    sc.pp.neighbors(temp_adata)
    sc.tl.umap(temp_adata)

    # Add UMAP coordinates to the new AnnData
    new_adata.obsm["X_umap"] = temp_adata.obsm["X_umap"]

    # Evaluate if CN features explain intra-cluster variation
    if verbose:
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
            if verbose:
                print(f"Cell type {cell_type}: CN explains {r2:.2f} of residual variation")

    # Visualize the results of residual modeling
    if plot:
        # Plot 1: UMAP of enhanced features colored by cell type
        if verbose:
            print("Plotting UMAP visualization...")
        sc.pl.umap(
            temp_adata, color="cell_types", title="Enhanced Features (Protein + CN Projection)"
        )
        plt.savefig(f"{output_dir}/enhanced_features_umap.png", dpi=300, bbox_inches="tight")

        # Plot 2: R² values for each cell type
        plt.figure(figsize=(10, 6))
        cell_types_plot = list(r2_values.keys())
        r2_vals_plot = list(r2_values.values())
        sorted_indices = np.argsort(r2_vals_plot)[::-1]
        plt.bar(range(len(cell_types_plot)), [r2_vals_plot[i] for i in sorted_indices])
        plt.xticks(
            range(len(cell_types_plot)), [cell_types_plot[i] for i in sorted_indices], rotation=90
        )
        plt.title("Explained Residual Variation by Cell Type")
        plt.ylabel("R² Value")
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/explained_variation_by_cell_type.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

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
                projection[:, 0],
                projection[:, 1],
                c=protein_values,
                cmap="viridis",
                s=50,
                alpha=0.8,
            )
            plt.colorbar(label=f"Residual Protein Expression")
            plt.title(f"Cell Type: {cell_type}")
            plt.xlabel("CN Projection 1")
            plt.ylabel("CN Projection 2")
            plt.tight_layout()
            plt.savefig(
                f"{output_dir}/cell_type_{cell_type}_projection.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

    # Additional results information
    results_dict = {
        "r2_values": r2_values,
        "cca": cca,
        "cn_projection": cn_projection,
        "residuals": residuals,
    }

    # Store results in the AnnData object's uns
    new_adata.uns["cn_analysis"] = {"r2_values": r2_values, "n_components": n_components}

    return new_adata, results_dict


if __name__ == "__main__":
    # Add repository root to Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Example usage
    adata = sc.read(
        "CODEX_RNA_seq/data/processed_data/preprocessed_adata_prot_neighbors_means.h5ad"
    )
    new_adata, results = analyze_residual_variation(adata_obj=adata, plot=True, verbose=True)

    # Save the new AnnData object with CN projections
    new_adata.write_h5ad("CODEX_RNA_seq/data/processed_data/adata_with_cn_projections.h5ad")

# %%

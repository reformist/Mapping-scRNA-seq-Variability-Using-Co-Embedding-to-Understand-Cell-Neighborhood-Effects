#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %% Imports and setup
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.sparse import issparse
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# from remote_plot import plt


# Add repository root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set working directory to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def robust_clamp_outliers(data, lower_quantile=0.001, upper_quantile=0.999):
    lower_bound = np.quantile(data, lower_quantile, axis=0)
    upper_bound = np.quantile(data, upper_quantile, axis=0)
    return np.clip(data, lower_bound, upper_bound)


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
    if plot:
        subset_size = 2000
        if adata_prot.shape[0] > subset_size:
            subset_idx = np.random.choice(adata_prot.shape[0], subset_size, replace=False)
        else:
            subset_idx = np.arange(adata_prot.shape[0])
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

    # Original protein data plots
    if plot:
        if verbose:
            print("Plotting original protein data...")
        toplot = sc.AnnData(protein_data[subset_idx])
        toplot.obs = adata_prot.obs.iloc[subset_idx].copy()

        if not pd.api.types.is_categorical_dtype(toplot.obs["cell_types"]):
            toplot.obs["cell_types"] = toplot.obs["cell_types"].astype("category")

        if verbose:
            print(
                f"Number of cells with assigned cell types: {toplot.obs['cell_types'].value_counts().sum()}"
            )
            print(f"Total number of cells: {toplot.shape[0]}")
            print(f"Cell type distribution: {toplot.obs['cell_types'].value_counts()}")

        # PCA plot
        sc.pp.pca(toplot)
        plt.figure()

        ax = sc.pl.pca(toplot, color="cell_types", show=False)
        plt.savefig(f"{output_dir}/original_protein_pca.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

        # UMAP plot
        sc.pp.neighbors(toplot)
        sc.tl.umap(toplot)
        plt.figure()

        sc.pl.umap(toplot, color="cell_types", show=False)
        plt.savefig(f"{output_dir}/original_protein_umap.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    # Standardize data for comparison
    scaler = StandardScaler()
    norm_protein_data = scaler.fit_transform(robust_clamp_outliers(protein_data))
    norm_cn_data = scaler.fit_transform(robust_clamp_outliers(cn_data))

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
    # sort predicted_baseline by cell_types
    predicted_baseline_temp = predicted_baseline[np.argsort(cell_types)]
    # Calculate residuals (variations not explained by cell type)
    residuals = norm_protein_data - predicted_baseline
    # Residuals heatmap
    if plot:
        plt.figure(figsize=(10, 10))
        sns.heatmap(residuals[subset_idx], cmap="viridis")
        plt.title("Residual Protein Expression Heatmap (Subset)")
        plt.savefig(f"{output_dir}/residuals_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

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
    # CN projection heatmap
    if plot:
        plt.figure(figsize=(10, 10))
        sns.heatmap(cn_projection[subset_idx], cmap="viridis")
        plt.title("CN Projection Heatmap (Subset)")
        plt.savefig(f"{output_dir}/cn_projection_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

        cn_projection_adata = sc.AnnData(cn_projection[subset_idx])
        cn_projection_adata.obs = adata_prot.obs.iloc[subset_idx].copy()
        sc.pp.pca(cn_projection_adata)
        sc.pp.neighbors(cn_projection_adata)
        sc.tl.umap(cn_projection_adata)
        sc.pl.umap(
            cn_projection_adata,
            color="cell_types",
            title="CN Projection UMAP, colored by cell type",
        )
        sc.pl.umap(cn_projection_adata, color="CN", title="CN Projection UMAP, colored by CN")
        plt.savefig(f"{output_dir}/cn_projection_umap.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"temp.png", dpi=300, bbox_inches="tight")
        plt.close()

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

    # Copy over other important annotations from original AnnData - simplified version
    if hasattr(adata_prot, "obsm") and adata_prot.obsm:
        for key in adata_prot.obsm.keys():
            new_adata.obsm[key] = adata_prot.obsm[key]

    if hasattr(adata_prot, "obsp") and adata_prot.obsp:
        for key in adata_prot.obsp.keys():
            new_adata.obsp[key] = adata_prot.obsp[key]

    if hasattr(adata_prot, "uns") and adata_prot.uns:
        for key in adata_prot.uns.keys():
            new_adata.uns[key] = adata_prot.uns[key]

    # Enhanced features UMAP
    if plot:
        if verbose:
            print("Generating UMAP visualization...")
        temp_adata = sc.AnnData(enhanced_features[subset_idx])
        temp_adata.obs = adata_prot.obs.iloc[subset_idx]
        sc.pp.neighbors(temp_adata)
        sc.tl.umap(temp_adata)
        sc.pl.umap(temp_adata, color="cell_types")
        plt.savefig(f"{output_dir}/enhanced_features_umap.png", dpi=300, bbox_inches="tight")
        plt.close()
        # plot the following, original only protein feateurs umap,
        # origanl protein + CN vector  umap,
        #   CN only umap
        # projeccted CN only umap
        # enhance features (protein + projected CN vector) umap
    if plot:
        if verbose:
            print("Generating UMAP visualizations...")

        # 1. Original protein features UMAP
        protein_adata = sc.AnnData(protein_data[subset_idx])
        protein_adata.obs = adata_prot.obs.iloc[subset_idx]
        sc.pp.neighbors(protein_adata)
        sc.tl.umap(protein_adata)
        sc.pl.umap(
            protein_adata, color="cell_types", title="Original Protein Features - Cell Types"
        )
        plt.savefig(f"{output_dir}/protein_only_umap_celltypes.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"temp.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Original protein + CN vectors UMAP
        orig_combined = np.hstack([protein_data, cn_data])
        orig_combined_adata = sc.AnnData(orig_combined[subset_idx])
        orig_combined_adata.obs = adata_prot.obs.iloc[subset_idx]
        sc.pp.neighbors(orig_combined_adata)
        sc.tl.umap(orig_combined_adata)
        sc.pl.umap(
            orig_combined_adata, color="cell_types", title="Original Protein + CN - Cell Types"
        )
        plt.savefig(f"{output_dir}/protein_cn_umap_celltypes.png", dpi=300, bbox_inches="tight")
        sc.pl.umap(orig_combined_adata, color="CN", title="Original Protein + CN - CN")
        plt.savefig(f"{output_dir}/protein_cn_umap_cn.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"temp_1.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 3. CN only UMAP
        cn_adata = sc.AnnData(cn_data[subset_idx])
        cn_adata.obs = adata_prot.obs.iloc[subset_idx]
        sc.pp.neighbors(cn_adata)
        sc.tl.umap(cn_adata)
        sc.pl.umap(cn_adata, color="cell_types", title="CN Only - Cell Types")
        plt.savefig(f"{output_dir}/cn_only_umap_celltypes.png", dpi=300, bbox_inches="tight")
        sc.pl.umap(cn_adata, color="CN", title="CN Only - CN")
        plt.savefig(f"{output_dir}/cn_only_umap_cn.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 4. Projected CN only UMAP
        cn_proj_adata = sc.AnnData(cn_projection[subset_idx])
        cn_proj_adata.obs = adata_prot.obs.iloc[subset_idx]
        sc.pp.neighbors(cn_proj_adata)
        sc.tl.umap(cn_proj_adata)
        sc.pl.umap(cn_proj_adata, color="cell_types", title="Projected CN - Cell Types")
        plt.savefig(f"{output_dir}/cn_projected_umap_celltypes.png", dpi=300, bbox_inches="tight")
        sc.pl.umap(cn_proj_adata, color="CN", title="Projected CN - CN")
        plt.savefig(f"{output_dir}/cn_projected_umap_cn.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 5. Enhanced features (protein + projected CN) UMAP
        enhanced_adata = sc.AnnData(enhanced_features[subset_idx])
        enhanced_adata.obs = adata_prot.obs.iloc[subset_idx]
        sc.pp.neighbors(enhanced_adata)
        sc.tl.umap(enhanced_adata)
        sc.pl.umap(enhanced_adata, color="cell_types", title="Enhanced Features - Cell Types")
        plt.savefig(
            f"{output_dir}/enhanced_features_umap_celltypes.png", dpi=300, bbox_inches="tight"
        )
        sc.pl.umap(enhanced_adata, color="CN", title="Enhanced Features - CN")
        plt.savefig(f"{output_dir}/enhanced_features_umap_cn.png", dpi=300, bbox_inches="tight")
        plt.close()

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
    # Enhanced R² Plot with Variance Explanation
    if plot:
        plt.figure(figsize=(12, 6))
        cell_types_plot = list(r2_values.keys())
        r2_vals_plot = list(r2_values.values())

        # Sort by R² value descending
        sorted_indices = np.argsort(r2_vals_plot)[::-1]
        sorted_cell_types = [cell_types_plot[i] for i in sorted_indices]
        sorted_r2 = [r2_vals_plot[i] for i in sorted_indices]

        # Create bar plot with annotations
        bars = plt.bar(sorted_cell_types, sorted_r2, color="skyblue")
        plt.axhline(
            np.mean(sorted_r2),
            color="red",
            linestyle="--",
            label=f"Mean R²: {np.mean(sorted_r2):.2f}",
        )

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )

        plt.title("Variance Explained by CN Projections per Cell Type", pad=20)
        plt.ylabel("Proportion of Variance Explained (R²)")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()

        # Save to output directory
        plt.savefig(
            f"{output_dir}/cn_variance_explained_per_celltype.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

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
    print("done, returning new_adata and results_dict")
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
    # new_adata.write_h5ad("CODEX_RNA_seq/data/processed_data/adata_with_cn_projections.h5ad")

# %%

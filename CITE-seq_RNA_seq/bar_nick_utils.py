import copy
import os
import re
from typing import List, Dict, Any

from scipy.optimize import linear_sum_assignment

import torch
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from itertools import product, zip_longest
import matplotlib.pyplot as plt

from scipy.optimize import nnls
import cvxpy as cp
from sklearn.linear_model import OrthogonalMatchingPursuit
# import scib
import scipy
# Function to get the latest file based on the timestamp

def calculate_cLISI(adata, label_key='cell_type', neighbors_key='neighbors'):
    """
    Calculate cell-type Local Inverse Simpson's Index (LISI) using precomputed neighbors.
    
    The cLISI score measures how well cell types are separated in the embedding space.
    Higher scores indicate better cell type separation, with a minimum value of 1 
    (all neighbors same cell type) and maximum of k+1 (all neighbors different cell types),
    where k is the number of neighbors used.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with precomputed neighbors
    label_key : str, default='cell_type'
        Column in adata.obs containing cell type labels
    neighbors_key : str, default='neighbors'
        Key where neighbor information is stored in adata.uns
        
    Returns
    -------
    float
        Median cLISI score across all cells. Higher values indicate better
        cell type separation in the embedding space.
    """

    if neighbors_key not in adata.uns:
        raise ValueError(f"Run sc.pp.neighbors with key='{neighbors_key}' first")
    
    connectivities = adata.obsp[f'connectivities']
    n_cells = adata.n_obs
    
    lisi_scores = []
    for i in range(n_cells):
        neighbors = connectivities[i].indices
        neighbors = np.append(neighbors, i)
        
        labels = adata.obs[label_key].iloc[neighbors].values
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        proportions = counts / len(neighbors)
        simpson = np.sum(proportions**2)
        lisi = 1 / simpson if simpson > 0 else 0
        lisi_scores.append(lisi)
    
    return np.median(lisi_scores)

def calculate_iLISI(adata, batch_key='batch', neighbors_key='neighbors'):
    """
    Calculate integration Local Inverse Simpson's Index (LISI) using precomputed neighbors.
    
    The iLISI score measures how well different batches are mixed in the embedding space.
    Higher scores indicate better batch mixing, with a minimum value of 1 
    (all neighbors same batch) and maximum of k+1 (all neighbors different batches),
    where k is the number of neighbors used.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with precomputed neighbors
    batch_key : str, default='batch'
        Column in adata.obs containing batch labels
    neighbors_key : str, default='neighbors'
        Key where neighbor information is stored in adata.uns
        
    Returns
    -------
    float
        Median iLISI score across all cells. Higher values indicate better
        batch mixing in the embedding space.
    """
    
    if neighbors_key not in adata.uns:
        raise ValueError(f"Run sc.pp.neighbors with key='{neighbors_key}' first")
    
    connectivities = adata.obsp[f'connectivities']
    n_cells = adata.n_obs
    plt.figure()
    plt.title('neighbors, first half are RNA cells \nthe second half, protein cells')
    sns.heatmap(connectivities.todense())
    plt.show()
    lisi_scores = []
    for i in range(n_cells):
        neighbors = connectivities[i].indices
        neighbors = np.append(neighbors, i)
        
        batches = adata.obs[batch_key].iloc[neighbors].values
        unique_batches, counts = np.unique(batches, return_counts=True)
        
        proportions = counts / len(neighbors)
        simpson = np.sum(proportions**2)
        lisi = 1 / simpson if simpson > 0 else 0
        lisi_scores.append(lisi)
    
    return np.median(lisi_scores)



def plot_merged_pca_tsne(
    adata1,
    adata2,
    unmatched_prot_indices,
    unmatched_rna_indices,
    pca_components=5
):
    """
    1) Combines Protein + RNA 'archetype_vec' data.
    2) Dynamically adjusts PCA components if needed.
    3) Plots the first two principal components in one figure.
    4) Applies TSNE to the PCA output, and generates TWO separate figures:
       - Figure A: Colors by modality (Protein vs. RNA).
       - Figure B: Colors by matched vs. unmatched status.

    Parameters
    ----------
    adata1 : anndata.AnnData
        Protein subset with 'archetype_vec' in obsm.
    adata2 : anndata.AnnData
        RNA subset with 'archetype_vec' in obsm.
    unmatched_prot_indices : list or np.ndarray
        Indices of unmatched protein cells.
    unmatched_rna_indices : list or np.ndarray
        Indices of unmatched RNA cells.
    pca_components : int
        Requested number of principal components before TSNE.
    """

    # -------------------- MERGE DATA --------------------
    prot_data = adata1.obsm['archetype_vec']
    rna_data = adata2.obsm['archetype_vec']
    merged_data = np.vstack([prot_data, rna_data])

    max_valid_components = min(merged_data.shape[0], merged_data.shape[1])
    final_pca_components = min(pca_components, max_valid_components)

    # -------------------- PCA --------------------
    pca_model = PCA(n_components=final_pca_components)
    pca_result = pca_model.fit_transform(merged_data)

    # Split PCA results back
    prot_pca = pca_result[:len(prot_data)]
    rna_pca = pca_result[len(prot_data):]

    # -------------------- PCA PLOT --------------------
    plt.figure(figsize=(6, 5))

    # Matched vs. unmatched (Protein)
    plt.scatter(
        prot_pca[:, 0],
        prot_pca[:, 1],
        c='blue',
        s=5,
        label='Matched Protein'
    )
    plt.scatter(
        prot_pca[unmatched_prot_indices, 0],
        prot_pca[unmatched_prot_indices, 1],
        c='black',
        marker='x',
        s=10,
        label='Unmatched Protein',
        alpha=0.5
    )

    # Matched vs. unmatched (RNA)
    plt.scatter(
        rna_pca[:, 0],
        rna_pca[:, 1],
        c='red',
        s=5,
        label='Matched RNA',
        alpha=0.5

    )
    plt.scatter(
        rna_pca[unmatched_rna_indices, 0],
        rna_pca[unmatched_rna_indices, 1],
        c='green',
        marker='D',
        s=10,
        label='Unmatched RNA',
        alpha=0.5

    )

    plt.title("PCA (First Two PCs)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.show()

    # -------------------- TSNE --------------------
    tsne_model = TSNE(n_components=2)
    tsne_result = tsne_model.fit_transform(pca_result)

    prot_tsne = tsne_result[:len(prot_data)]
    rna_tsne = tsne_result[len(prot_data):]

    # -------------------- FIGURE A (Modality Colors) --------------------
    plt.figure(figsize=(6, 5))

    plt.scatter(
        prot_tsne[:, 0],
        prot_tsne[:, 1],
        c='blue',
        s=5,
        label='Protein'
    )
    plt.scatter(
        rna_tsne[:, 0],
        rna_tsne[:, 1],
        c='red',
        s=5,
        label='RNA',
                alpha=0.5

    )

    plt.title("TSNE by Modality (Protein vs. RNA)")
    plt.xlabel("TSNE 1")
    plt.ylabel("TSNE 2")
    plt.legend()
    plt.show()

    # -------------------- FIGURE B (Matched vs. Unmatched) --------------------
    plt.figure(figsize=(6, 5))

    # Matched Protein
    plt.scatter(
        prot_tsne[:, 0],
        prot_tsne[:, 1],
        c='red',
        s=5,
        label='Matched Protein'
    )
    # Unmatched Protein
    plt.scatter(
        prot_tsne[unmatched_prot_indices, 0],
        prot_tsne[unmatched_prot_indices, 1],
        c='green',
        marker='x',
        s=10,
        label='Unmatched Protein',
                alpha=0.5

    )

    # Matched RNA
    plt.scatter(
        rna_tsne[:, 0],
        rna_tsne[:, 1],
        c='red',
        s=5,
        label='Matched RNA',
                alpha=0.5

    )
    # Unmatched RNA
    plt.scatter(
        rna_tsne[unmatched_rna_indices, 0],
        rna_tsne[unmatched_rna_indices, 1],
        c='black',
        marker='x',
        s=10,
        label='Unmatched RNA',
                alpha=0.5

    )

    plt.title("TSNE by Match Status")
    plt.xlabel("TSNE 1")
    plt.ylabel("TSNE 2")
    plt.legend()
    plt.show()

def match_datasets(adata1, adata2, threshold=0.3,
                   obs_key1='archetype_vec', obs_key2='archetype_vec',plot_flag=False):
    # Compute pairwise distance matrix
    dist_matrix = scipy.spatial.distance.cdist(
        adata1.obsm[obs_key1],
        adata2.obsm[obs_key2],
        metric='cosine'
    )
    matching_distance_before = np.diag(dist_matrix).mean()


    n1, n2 = len(adata1), len(adata2)

    # Optimal initial matching using Hungarian algorithm
    rows, cols = linear_sum_assignment(dist_matrix)

    # Collect all potential matches
    all_matches = []

    # 1. Primary matching (one-to-one)
    primary_matches = []
    for r, c in zip(rows, cols):
        if dist_matrix[r, c] <= threshold:
            primary_matches.append((r, c))

    # 2. Secondary matching for remaining adata1 cells
    matched_adata1 = set(r for r, _ in primary_matches)
    remaining_adata1 = [i for i in range(n1) if i not in matched_adata1]

    # Find best remaining matches for unpaired adata1 cells
    secondary_matches = []
    for r in remaining_adata1:
        c = np.argmin(dist_matrix[r])
        if dist_matrix[r, c] <= threshold:
            secondary_matches.append((r, c))

    # Combine matches and remove duplicates
    combined = primary_matches + secondary_matches
    unique_adata2 = set(c for _, c in combined)

    # Create final index arrays
    adata1_indices = np.array([r for r, _ in combined])
    adata2_indices = np.array([c for _, c in combined])
    remaining_adata1 = [i for i in range(n1) if i not in adata1_indices]
    remaining_adata2 = [i for i in range(n2) if i not in adata2_indices]

    adata1.uns['ordered_matching_cells'] = True
    adata2.uns['ordered_matching_cells'] = True
    adata1.obs['index_col'] = np.arange(adata1.shape[0])
    adata2.obs['index_col'] = np.arange(adata2.shape[0])


    # Calculate statistics
    stats = {
        'total_adata1': n1,
        'total_adata2': n2,
        'matched_adata1': len(adata1_indices),
        'unique_adata2_used': len(unique_adata2),
        'adata1_unmatched': n1 - len(adata1_indices),
        'adata2_unmatched': n2 - len(unique_adata2),
        'adata2_reuses': len(adata2_indices) - len(unique_adata2),
        'mean_distance': dist_matrix[adata1_indices, adata2_indices].mean()
    }

    # Print comprehensive report
    print(f"Matching Report:\n"
          f"- Matched {stats['matched_adata1']}/{n1} adata1 cells "
          f"({stats['adata1_unmatched']} unmatched)\n"
          f"- Used {stats['unique_adata2_used']}/{n2} adata2 cells "
          f"({stats['adata2_unmatched']} never matched)\n"
          f"- adata2 reuses: {stats['adata2_reuses']}\n"
          f'- Average match distance before matching:{matching_distance_before:.3f}\n'
          f"- Average match distance after matching: {stats['mean_distance']:.3f}")
    if plot_flag:
        plt.figure()
        sns.histplot(dist_matrix[adata1_indices,adata2_indices],bins=int(100*np.max(dist_matrix[adata1_indices,adata2_indices])),color='blue',label='final matches')
        sns.histplot(dist_matrix[rows, cols],bins=int(100*np.max(dist_matrix[rows, cols])),color='red',label='raw matches of Hungarian algo')
        # plt.title('distances used')
        plt.legend()
        plt.xlabel('cosine distance')

        plt.show()

        plot_merged_pca_tsne(
            adata1,
            adata2,
            unmatched_prot_indices=remaining_adata2,
            unmatched_rna_indices=remaining_adata1,
            pca_components=5)
    return adata1[adata1_indices], adata2[adata2_indices]


def get_latest_file(prefix,folder):
    files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith('.h5ad')]
    if not files:
        return None
    files.sort(key=lambda x: re.search(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', x).group(), reverse=True)
    return os.path.join(folder, files[0])

def nnls_omp(basis_matrix, target_vector, tol=1e-4):
    omp = OrthogonalMatchingPursuit(tol=tol, fit_intercept=False)
    omp.fit(basis_matrix.T, target_vector)
    weights = omp.coef_
    weights = np.maximum(0, weights)  # Enforce non-negativity
    return weights


def get_cell_representations_as_archetypes_ols(count_matrix, archetype_matrix):
    """
    Compute archetype weights for each cell using Ordinary Least Squares (OLS).

    Parameters:
    -----------
    count_matrix : np.ndarray
        Matrix of cells in reduced-dimensional space (e.g., PCA),
        shape (n_cells, n_features).
    archetype_matrix : np.ndarray
        Matrix of archetypes,
        shape (n_archetypes, n_features).

    Returns:
    --------
    weights : np.ndarray
        Matrix of archetype weights for each cell,
        shape (n_cells, n_archetypes).
    """
    n_cells = count_matrix.shape[0]
    n_archetypes = archetype_matrix.shape[0]
    weights = np.zeros((n_cells, n_archetypes))

    # Transpose the archetype matrix
    A_T = archetype_matrix.T  # Shape: (n_features, n_archetypes)

    # For each cell, solve the least squares problem
    for i in range(n_cells):
        x = count_matrix[i]
        # Solve for w in A_T w = x
        w, residuals, rank, s = np.linalg.lstsq(A_T, x, rcond=None)
        weights[i] = w

    return weights


def get_cell_representations_as_archetypes_omp(count_matrix, archetype_matrix, tol=1e-4):
    # Preprocess archetype matrix

    n_cells = count_matrix.shape[0]
    n_archetypes = archetype_matrix.shape[0]
    weights = np.zeros((n_cells, n_archetypes))

    for i in range(n_cells):
        weights[i] = nnls_omp(archetype_matrix, count_matrix[i], tol=tol)

    row_sums = weights.sum(axis=1, keepdims=True)
    weights[row_sums == 0] = 1.0 / n_archetypes  # Assign uniform weights to zero rows

    weights /= weights.sum(axis=1, keepdims=True)

    return weights




def nnls_cvxpy(A, b):
    """
    Solve the NNLS problem using cvxpy.
    """
    n_features = A.shape[1]
    x = cp.Variable(n_features, nonneg=True)
    objective = cp.Minimize(cp.sum_squares(A @ x - b))
    problem = cp.Problem(objective)
    problem.solve()
    return x.value


def reorder_rows_to_maximize_diagonal(matrix):
    """
    Reorders rows of a matrix to maximize diagonal dominance by placing the highest values
    in the closest positions to the diagonal.

    Parameters:
    -----------
    matrix : np.ndarray
        An m x n matrix.

    Returns:
    --------
    reordered_matrix : np.ndarray
        The input matrix with reordered rows.
    row_order : list
        The indices of the rows in their new order.
    """
    # Track available rows and
    original = None
    if isinstance(matrix, pd.DataFrame):
        original = copy.deepcopy(matrix)
        matrix = matrix.values
    available_rows = list(range(matrix.shape[0]))
    available_cols = list(range(matrix.shape[1]))
    row_order = []

    # Reorder rows iteratively
    for col in range(matrix.shape[1]):
        if not available_rows:
            break

        # Find the row with the maximum value for the current column
        best_row = max(available_rows, key=lambda r: matrix[r, col])
        row_order.append(best_row)
        available_rows.remove(best_row)

    # Handle leftover rows if there are more rows than columns
    row_order += available_rows

    # Reorder the matrix
    reordered_matrix = matrix[row_order]
    if original is not None:
        reordered_matrix = pd.DataFrame(reordered_matrix, index=original.index, columns=original.columns)
    return reordered_matrix, row_order


def get_cell_representations_as_archetypes_cvxpy(count_matrix, archetype_matrix, solver=cp.ECOS):
    """
    Compute archetype weights for each cell using constrained optimization (non-negative least squares).

    Parameters:
    -----------
    count_matrix : np.ndarray
        Matrix of cells in reduced-dimensional space,
        shape (n_cells, n_features).
    archetype_matrix : np.ndarray
        Matrix of archetypes,
        shape (n_archetypes, n_features).
    solver : cvxpy solver, optional
        Solver to use for optimization. Default is cp.ECOS.

    Returns:
    --------
    weights : np.ndarray
        Non-negative weights for each cell,
        shape (n_cells, n_archetypes).
    """
    n_cells = count_matrix.shape[0]
    n_archetypes = archetype_matrix.shape[0]
    weights = np.zeros((n_cells, n_archetypes))

    A_T = archetype_matrix.T  # Shape: (n_features, n_archetypes)
    assert not np.isnan(count_matrix).any(), "count_matrix contains NaNs"
    assert not np.isinf(count_matrix).any(), "count_matrix contains infinities"
    assert not np.isnan(archetype_matrix).any(), "archetype_matrix contains NaNs"
    assert not np.isinf(archetype_matrix).any(), "archetype_matrix contains infinities"
    # make sure all values are positive
    # if not (count_matrix >= 0).all():
    #     count_matrix = count_matrix - count_matrix.min()

    for i in range(n_cells):
        x = count_matrix[i]
        w = cp.Variable(n_archetypes)
        objective = cp.Minimize(cp.sum_squares(A_T @ w - x))
        # constraints = [w >= 0] # legacy=
        constraints = [w >= 0, cp.sum(w) == 1] # this make sure that each data points is a convex combination of the archetypes

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=solver)
        except cp.SolverError:
            problem.solve(solver=cp.SCS)  # Try SCS if the primary solver fails

        weights[i] = w.value

    return weights


def get_cell_representations_as_archetypes(count_matrix, archetype_matrix):
    """
    Compute archetype weights for each cell using cvxpy.
    """
    n_cells = count_matrix.shape[0]
    n_archetypes = archetype_matrix.shape[0]
    weights = np.zeros((n_cells, n_archetypes))
    for i in range(n_cells):
        weights[i], _ = nnls(archetype_matrix.T, count_matrix[i])
    weights /= weights.sum(axis=1, keepdims=True)  # Normalize rows
    return weights


def preprocess_rna(adata_rna):
    """
    Preprocess RNA data for downstream analysis with PCA and variance tracking.
    """

    # Annotate mitochondrial, ribosomal, and hemoglobin genes
    adata_rna.var["mt"] = adata_rna.var_names.str.startswith("Mt-")  # Mouse data
    adata_rna.var["ribo"] = adata_rna.var_names.str.startswith(("RPS", "RPL"))
    adata_rna.var["hb"] = adata_rna.var_names.str.contains("^HB[^(P)]", regex=True)

    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(adata_rna, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)

    # Add raw counts to layers for future reference
    adata_rna.layers["counts"] = adata_rna.X.copy()

    # Log-transform the data
    sc.pp.log1p(adata_rna)
    sc.pp.pca(adata_rna)
    print(
        f"Variance ratio after log transformation PCA (10 PCs): {adata_rna.uns['pca']['variance_ratio'][:10].sum():.4f}")
    # Normalize total counts
    sc.pp.normalize_total(adata_rna, target_sum=5e3)
    sc.pp.pca(adata_rna)
    print(f"Variance ratio after normalization PCA (10 PCs): {adata_rna.uns['pca']['variance_ratio'][:10].sum():.4f}")

    # Scale the data
    # sc.pp.scale(adata_rna, max_value=10)
    # sc.pp.pca(adata_rna)
    # print(f"Variance ratio after scaling PCA: {adata_rna.uns['pca']['variance_ratio'][:10].sum():.4f}")

    return adata_rna


def preprocess_protein(adata_prot):
    sc.pp.pca(adata_prot)
    print(f"Variance ratio after PCA (10 PCs): {adata_prot.uns['pca']['variance_ratio'][:10].sum():.4f}")
    print()
    sc.pp.normalize_total(adata_prot)
    sc.pp.pca(adata_prot)
    print(f"Variance ratio after normalization PCA (10 PCs): {adata_prot.uns['pca']['variance_ratio'][:10].sum():.4f}")
    print()
    sc.pp.log1p(adata_prot)
    sc.pp.pca(adata_prot)
    print(
        f"Variance ratio after log transformation PCA (10 PCs): {adata_prot.uns['pca']['variance_ratio'][:10].sum():.4f}")
    # matrix = adata_prot.X
    # np.log1p(matrix / np.exp(np.mean(np.log1p(matrix + 1), axis=1, keepdims=True)))
    # adata_prot.X = matrix
    # sc.pp.scale(adata_prot, max_value=10)

    return adata_prot


def select_gene_likelihood(adata):
    """
    Determines the appropriate gene likelihood distribution for the SCVI model
    based on the properties of the input AnnData object.

    Parameters:
    - adata: AnnData object containing single-cell RNA-seq data.

    Returns:
    - str: Selected gene likelihood distribution ("nb", "zinb", "poisson").
    """

    # Check for zero-inflation by counting the proportion of zero values in the data
    zero_proportion = (adata.X == 0).sum() / adata.X.size

    # Select likelihood based on zero inflation and count properties
    if zero_proportion > 0.4:
        gene_likelihood = "zinb"  # Zero-Inflated Negative Binomial for high zero-inflation
    elif adata.X.mean() < 5:
        gene_likelihood = "poisson"  # Poisson for low-count data
    else:
        gene_likelihood = "nb"  # Negative Binomial for typical gene expression

    print(f"Selected gene likelihood: {gene_likelihood}")
    return gene_likelihood


def add_spatial_data_to_prot(adata_prot_subset, major_to_minor_dict, plot_flag=False) -> (AnnData, list, list):


    horizontal_splits = [0, 500, 1000]
    vertical_splits = [0, 333, 666, 1000]

    # Create regions as a list of coordinate grids
    regions = [
        list(product(range(horizontal_splits[i], horizontal_splits[i + 1]),
                     range(vertical_splits[j], vertical_splits[j + 1])))
        for i in range(len(horizontal_splits) - 1)
        for j in range(len(vertical_splits) - 1)
    ]

    # Create a board for visualization (optional)
    board = np.zeros((1000, 1000))
    for idx, region in enumerate(regions):
        coords = np.array(region)
        board[coords[:, 0], coords[:, 1]] = idx + 1

    if plot_flag:
        plt.imshow(board)
        plt.title('CNs')
        plt.colorbar()
        plt.show()

    # Assign random initial coordinates to all cells
    adata_prot_subset.obs['X'] = np.random.randint(0, 1000, adata_prot_subset.n_obs)
    adata_prot_subset.obs['Y'] = np.random.randint(0, 1000, adata_prot_subset.n_obs)

    # Create a dictionary mapping tuples of (cell_type_1, cell_type_2, cell_type_3) to a region index
    minor_to_region_dict = {}
    major_B = major_to_minor_dict.get('B cells', [])
    major_CD4 = major_to_minor_dict.get('CD4 T', [])
    major_CD8 = major_to_minor_dict.get('CD8 T', [])

    for i, (cell_type_1, cell_type_2, cell_type_3) in enumerate(zip_longest(major_B, major_CD4, major_CD8)):
        # If any of these are None, they won't match any cell, but we can still store them
        minor_to_region_dict[(cell_type_1, cell_type_2, cell_type_3)] = i

    # Now place the cells of each subgroup into their assigned region
    for (cell_type_1, cell_type_2, cell_type_3), region_index in minor_to_region_dict.items():
        coords = np.array(regions[region_index])

        # Update positions for each cell type if not None
        if cell_type_1 is not None:
            cell_indices_1 = (adata_prot_subset.obs['cell_types'] == cell_type_1)
            if cell_indices_1.sum() > 0:
                adata_prot_subset.obs.loc[cell_indices_1, 'X'] = np.random.choice(coords[:, 0], cell_indices_1.sum())
                adata_prot_subset.obs.loc[cell_indices_1, 'Y'] = np.random.choice(coords[:, 1], cell_indices_1.sum())

        if cell_type_2 is not None:
            cell_indices_2 = (adata_prot_subset.obs['cell_types'] == cell_type_2)
            if cell_indices_2.sum() > 0:
                adata_prot_subset.obs.loc[cell_indices_2, 'X'] = np.random.choice(coords[:, 0], cell_indices_2.sum())
                adata_prot_subset.obs.loc[cell_indices_2, 'Y'] = np.random.choice(coords[:, 1], cell_indices_2.sum())

        if cell_type_3 is not None:
            cell_indices_3 = (adata_prot_subset.obs['cell_types'] == cell_type_3)
            if cell_indices_3.sum() > 0:
                adata_prot_subset.obs.loc[cell_indices_3, 'X'] = np.random.choice(coords[:, 0], cell_indices_3.sum())
                adata_prot_subset.obs.loc[cell_indices_3, 'Y'] = np.random.choice(coords[:, 1], cell_indices_3.sum())

    # Store the spatial coordinates in obsm
    adata_prot_subset.obsm['X_spatial'] = adata_prot_subset.obs[['X', 'Y']].to_numpy()

    return adata_prot_subset, horizontal_splits, vertical_splits


def verify_gradients(*models):
    return
    for model in models:
        if all(param.grad is None for param in model.module.parameters()):
            print("No gradients found for any parameter in the model.")
            # raise ValueError("No gradients found for any parameter in the model.")


def compute_pairwise_kl(loc, scale):
    # Expand for broadcasting
    loc1 = loc.unsqueeze(1)
    loc2 = loc.unsqueeze(0)
    scale1 = scale.unsqueeze(1)
    scale2 = scale.unsqueeze(0)
    # Compute KL divergence for each pair
    kl_matrix = (
            torch.log(scale2 / scale1) +
            (scale1 ** 2 + (loc1 - loc2) ** 2) / (2 * scale2 ** 2) - 0.5
    ).sum(dim=-1)  # Sum over latent dimensions
    return kl_matrix

def compute_pairwise_kl_two_items(loc1,loc2, scale1, scale2):
    # Expand for broadcasting
    loc1 = loc1.unsqueeze(1)
    loc2 = loc2.unsqueeze(0)
    scale1 = scale1.unsqueeze(1)
    scale2 = scale2.unsqueeze(0)
    # Compute KL divergence for each pair
    kl_matrix = (
            torch.log(scale2 / scale1) +
            (scale1 ** 2 + (loc1 - loc2) ** 2) / (2 * scale2 ** 2) - 0.5
    ).sum(dim=-1)  # Sum over latent dimensions
    return kl_matrix
def plot_torch_normal(mean, std_dev, num_points=1000):
    """
    Plots a Normal distribution given the mean and standard deviation.

    Parameters:
        mean (float): The mean of the distribution.
        std_dev (float): The standard deviation of the distribution.
        num_points (int): Number of points to plot (default: 1000).
    """
    # Create the Normal distribution
    normal_dist = torch.distributions.Normal(mean, std_dev)
    x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, num_points)
    y = torch.exp(normal_dist.log_prob(torch.tensor(x))).numpy()
    plt.plot(x, y, label=f"Mean={mean:.2f}, Variance={std_dev ** 2:.2f}")
    plt.title("Normal Distribution (Torch)")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()


def plot_archetypes(
        data_points,
        archetype,
        samples_cell_types: List[str],
        data_point_archetype_indices: List[int],
        modality='',
        cell_type_colors: Dict[str, Any] = None
):
    if not isinstance(samples_cell_types, List):
        raise TypeError("samples_cell_types should be a list of strings.")
    if not isinstance(data_point_archetype_indices, List):
        raise TypeError("data_point_archetype_indices should be a list of integers.")
    if len(data_points) != len(samples_cell_types) or len(data_points) != len(data_point_archetype_indices):
        raise ValueError("Length of data_points, samples_cell_types, and data_point_archetype_indices must be equal.")

    # Check the shapes of data_points and archetype
    print("Shape of data_points:", data_points.shape)
    print("Shape of archetype before any adjustment:", archetype.shape)

    # Ensure archetype has the same number of features as data_points
    if archetype.shape[1] != data_points.shape[1]:
        # Check if transposing helps
        if archetype.T.shape[1] == data_points.shape[1]:
            print("Transposing archetype array to match dimensions.")
            archetype = archetype.T
        else:
            raise ValueError("archetype array cannot be reshaped to match data_points dimensions.")

    print("Shape of archetype after adjustment:", archetype.shape)

    # Combine data points and archetypes
    num_archetypes = archetype.shape[0]
    data = np.concatenate((data_points, archetype), axis=0)
    labels = ["data"] * len(data_points) + ["archetype"] * num_archetypes
    cell_types = samples_cell_types + ['archetype'] * num_archetypes

    # Perform PCA and t-SNE
    data_pca = data[:, :50]
    data_tsne = TSNE(n_components=2).fit_transform(data_pca)

    # Create a numbering for archetypes
    archetype_numbers = [np.nan] * len(data_points) + list(range(num_archetypes))

    # Create DataFrames for plotting
    df_pca = pd.DataFrame({
        "PCA1": data_pca[:, 0],
        "PCA2": data_pca[:, 1],
        "type": labels,
        "cell_type": cell_types,
        "archetype_number": archetype_numbers,
        "data_point_archetype_index": data_point_archetype_indices + [np.nan] * num_archetypes
    })

    df_tsne = pd.DataFrame({
        "TSNE1": data_tsne[:, 0],
        "TSNE2": data_tsne[:, 1],
        "type": labels,
        "cell_type": cell_types,
        "archetype_number": archetype_numbers,
        "data_point_archetype_index": data_point_archetype_indices + [np.nan] * num_archetypes
    })

    # Use the provided color mapping or generate a new one
    if cell_type_colors is not None:
        palette_dict = cell_type_colors
    else:
        # Define color palette based on unique cell types
        unique_cell_types = list(pd.unique(samples_cell_types))
        palette = sns.color_palette("tab20", len(unique_cell_types))
        palette_dict = {cell_type: color for cell_type, color in zip(unique_cell_types, palette)}
        palette_dict["archetype"] = "black"  # Assign black to archetype

    # Ensure 'archetype' color is set
    if "archetype" not in palette_dict:
        palette_dict["archetype"] = "black"

    # Plot PCA
    plt.figure(figsize=(10, 6))
    df_pca = df_pca.sort_values(by='cell_type')
    sns.scatterplot(
        data=df_pca,
        x="PCA1",
        y="PCA2",
        hue="cell_type",
        style="type",
        palette=palette_dict,
        size="type",
        sizes={"data": 40, "archetype": 500},
        legend="brief",
        alpha=1
    )

    # Remove 'type' from the legend
    handles, labels_ = plt.gca().get_legend_handles_labels()
    cell_type_legend = [(h, l) for h, l in zip(handles, labels_) if l in palette_dict.keys() and l != "archetype"]
    if cell_type_legend:
        handles, labels_ = zip(*cell_type_legend)
    plt.legend(handles, labels_, title="Cell Types", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Annotate archetype points with numbers
    archetype_points = df_pca[df_pca['type'] == 'archetype']
    for _, row in archetype_points.iterrows():
        plt.text(
            row['PCA1'],
            row['PCA2'],
            str(int(row['archetype_number'])),
            fontsize=12,
            fontweight='bold',
            color='red'
        )

    # Add lines from each data point to its matching archetype
    df_pca_data = df_pca[df_pca['type'] == 'data'].copy()
    df_pca_archetypes = df_pca[df_pca['type'] == 'archetype'].copy()

    # 'archetype_number' is already assigned to archetypes
    df_pca_archetypes['archetype_number'] = df_pca_archetypes['archetype_number'].astype(int)

    # Create a mapping from archetype_number to its PCA coordinates
    archetype_coords = df_pca_archetypes.set_index('archetype_number')[['PCA1', 'PCA2']]

    # Now for each data point, draw a line to its corresponding archetype
    for idx, row in df_pca_data.iterrows():
        archetype_index = int(row['data_point_archetype_index'])
        data_point_coords = (row['PCA1'], row['PCA2'])
        try:
            archetype_point_coords = archetype_coords.loc[archetype_index]
            plt.plot(
                [data_point_coords[0], archetype_point_coords['PCA1']],
                [data_point_coords[1], archetype_point_coords['PCA2']],
                color='gray', linewidth=0.5, alpha=0.3
            )
        except KeyError:
            # If archetype_index does not match any archetype_number, skip
            pass

    plt.title(f"{modality} PCA Scatter Plot with Archetypes Numbered")
    plt.tight_layout()
    plt.show()

    # Plot t-SNE
    plt.figure(figsize=(10, 6))
    df_tsne = df_tsne.sort_values(by='cell_type')
    sns.scatterplot(
        data=df_tsne,
        x="TSNE1",
        y="TSNE2",
        hue="cell_type",
        style="type",
        palette=palette_dict,
        size="type",
        sizes={"data": 20, "archetype": 500},
        legend="brief",
        alpha=1
    )

    # Remove 'type' from the legend
    handles, labels_ = plt.gca().get_legend_handles_labels()
    cell_type_legend = [(h, l) for h, l in zip(handles, labels_) if l in palette_dict.keys() and l != "archetype"]

    if cell_type_legend:
        handles, labels_ = zip(*cell_type_legend)
    plt.legend(handles, labels_, title="Cell Types", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Annotate archetype points with numbers
    archetype_points_tsne = df_tsne[df_tsne['type'] == 'archetype']
    for _, row in archetype_points_tsne.iterrows():
        plt.text(
            row['TSNE1'],
            row['TSNE2'],
            str(int(row['archetype_number'])),
            fontsize=12,
            fontweight='bold',
            color='red'
        )

    # Add lines from each data point to its matching archetype in t-SNE plot
    df_tsne_data = df_tsne[df_tsne['type'] == 'data'].copy()
    df_tsne_archetypes = df_tsne[df_tsne['type'] == 'archetype'].copy()

    # Create a mapping from archetype_number to its t-SNE coordinates
    archetype_coords_tsne = df_tsne_archetypes.set_index('archetype_number')[['TSNE1', 'TSNE2']]

    # Now for each data point, draw a line to its corresponding archetype
    for idx, row in df_tsne_data.iterrows():
        archetype_index = int(row['data_point_archetype_index'])
        data_point_coords = (row['TSNE1'], row['TSNE2'])
        try:
            archetype_point_coords = archetype_coords_tsne.loc[archetype_index]
            plt.plot(
                [data_point_coords[0], archetype_point_coords['TSNE1']],
                [data_point_coords[1], archetype_point_coords['TSNE2']],
                color='gray', linewidth=0.2, alpha=0.3
            )
        except KeyError:
            # If archetype_index does not match any archetype_number, skip
            pass

    plt.title(f"{modality} t-SNE Scatter Plot with Archetypes Numbered")
    plt.tight_layout()
    plt.show()


def plot_aligned_normalized_losses(history):
    """
    Plot aligned and normalized losses for train and validation data.

    Parameters:
        history (dict): Dictionary containing training and validation loss data as Pandas DataFrames.
    """
    # Extract all loss keys that contain 'loss'
    all_loss_keys = [k for k in history.keys() if 'loss' in k]

    # Identify unique base loss names
    unique_losses = list(set(k.replace('train_', '').replace('validation_', '') for k in all_loss_keys))

    # Filter to keep only those with both train and validation keys
    filtered_losses = [
        loss_name
        for loss_name in unique_losses
        if f"train_{loss_name}" in history.keys() and f"validation_{loss_name}" in history.keys()
    ]

    # Create figure and subplots
    fig, axes = plt.subplots(len(filtered_losses), 1, figsize=(8, 4 * len(filtered_losses)), sharex=True)

    # Handle single loss case
    if len(filtered_losses) == 1:
        axes = [axes]

    for ax, loss_name in zip(axes, filtered_losses):
        # Get train data
        train_key = f"train_{loss_name}"
        train_df = history[train_key]
        train_epochs = train_df.index.astype(float)  # Ensure numeric indices
        train_data = train_df.values.flatten().astype(float)  # Ensure numeric values

        train_min, train_max = train_data.min(), train_data.max()
        norm_train_data = (train_data - train_min) / (train_max - train_min) if train_min != train_max else train_data

        # Get validation data
        val_key = f"validation_{loss_name}"
        val_df = history[val_key]
        val_epochs = val_df.index.astype(float)  # Ensure numeric indices
        val_data = val_df.values.flatten().astype(float)  # Ensure numeric values

        val_min, val_max = val_data.min(), val_data.max()
        norm_val_data = (val_data - val_min) / (val_max - val_min) if val_min != val_max else val_data

        # Interpolate validation data to align with training epochs
        interpolated_val_data = np.interp(train_epochs, val_epochs, norm_val_data)

        # Plot both on the same subplot
        ax.plot(train_epochs, norm_train_data, label=f"Train {loss_name} (min: {train_min:.2f}, max: {train_max:.2f})")
        ax.plot(train_epochs, interpolated_val_data, label=f"Val {loss_name} (min: {val_min:.2f}, max: {val_max:.2f})")

        ax.set_title(loss_name)
        ax.set_ylabel('Normalized Loss')
        ax.legend()

    axes[-1].set_xlabel('Epoch')
    plt.tight_layout()
    plt.show()


def plot_normalized_losses(history, figsize=(6, 8)):
    """
    Plot normalized loss values from a training history dictionary.

    Parameters:
        history (dict): Dictionary containing loss values for training and validation.
        figsize (tuple): Tuple specifying the figure size (width, height).
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)  # Two subplots: one for training and one for validation
    train_ax, val_ax = axes

    for key in history.keys():
        if 'loss' in key:
            # Extract the data and ensure it's numeric
            loss_data = history[key].to_numpy()

            # Calculate min, max, and range
            min_val = loss_data.min()
            max_val = loss_data.max()
            range_val = max_val - min_val

            # Safeguard against division by zero
            if range_val == 0:
                norm_loss = loss_data * 0  # Normalize to zero if no range
            else:
                norm_loss = (loss_data - min_val) / range_val

            label = f'{key} min: {min_val:.0f} max: {max_val:.0f}'

            # Plot on the respective subplot
            if 'train' in key:
                train_ax.plot(norm_loss, label=label)
            elif 'val' in key:
                val_ax.plot(norm_loss, label=label)

    # Formatting subplots
    train_ax.set_title('Training Losses')
    train_ax.set_xlabel('Epoch')
    train_ax.set_ylabel('Normalized Loss')
    train_ax.legend()

    val_ax.set_title('Validation Losses')
    val_ax.set_xlabel('Epoch')
    val_ax.set_ylabel('Normalized Loss')
    val_ax.legend()

    plt.tight_layout()
    plt.show()


def evaluate_distance_metrics_old(A: np.ndarray, B: np.ndarray, metrics: List[str]) -> Dict:
    """
    Evaluates multiple distance metrics to determine which one best captures the similarity
    between matching rows in matrices A and B.

    Parameters:
    - A: np.ndarray of shape (n_samples, n_features)
    - B: np.ndarray of shape (n_samples, n_features)
    - metrics: List of distance metrics to evaluate

    Returns:
    - results: Dictionary containing evaluation metrics for each distance metric
    """
    results = {}

    for metric in metrics:
        print(f"Evaluating distance metric: {metric}")

        # Compute the distance matrix between rows of A and rows of B
        distances = cdist(A, B, metric=metric)
        # For each row i, get the distances between A[i] and all rows in B
        # Then compute the rank of the matching distance
        ranks = []
        for i in range(len(A)):
            row_distances = distances[i, :]
            # Get the rank of the matching distance
            # Rank 1 means the smallest distance
            rank = np.argsort(row_distances).tolist().index(i) + 1
            ranks.append(rank)
        ranks = np.array(ranks)
        total_samples = len(A)
        # Compute evaluation metrics
        num_correct_matches = np.sum(ranks == 1)
        percentage_correct = num_correct_matches / total_samples * 100
        mean_rank = np.mean(ranks)
        mrr = np.mean(1 / ranks)
        print(f"Percentage of correct matches (rank 1): {percentage_correct:.2f}%")
        print(f"Mean rank of matching rows: {mean_rank:.2f}")
        print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
        print("")
        results[metric] = {
            'percentage_correct': percentage_correct,
            'mean_rank': mean_rank,
            'mrr': mrr,
            'ranks': ranks
        }
    return results


def plot_archetypes_matching(data1, data2, rows=5):
    offset = 1

    for i in range(rows):
        y1 = data1.iloc[i] + i * offset
        y2 = data2.iloc[i] + i * offset
        plt.plot(y1, label=f'modality 1 archetype {i + 1}')
        plt.plot(y2, linestyle='--', label=f'modality 2 archetype {i + 1}')
    plt.xlabel('Columns')
    plt.ylabel('proportion of cell types accounted for an archetype')
    plt.title('Show that the archetypes are aligned by using')
    plt.legend()
    # rotate x labels
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


def evaluate_distance_metrics(A: np.ndarray, B: np.ndarray, metrics: List[str]) -> Dict:
    """
    Evaluates multiple distance metrics to determine how much better they are compared
    to random assignment in matching rows in matrices A and B.

    Parameters:
    - A: np.ndarray of shape (n_samples, n_features)
    - B: np.ndarray of shape (n_samples, n_features)
    - metrics: List of distance metrics to evaluate

    Returns:
    - results: Dictionary containing evaluation metrics for each distance metric
    """
    results = {}
    n_samples =min( A.shape[0],B.shape[0])

    # Expected mean rank and MRR under random assignment
    expected_mean_rank = (n_samples + 1) / 2
    expected_mrr = np.mean(1 / np.arange(1, n_samples + 1))

    for metric in metrics:
        print(f"Evaluating distance metric: {metric}")

        # Compute the distance matrix between rows of A and rows of B
        distances = cdist(A, B, metric=metric)

        # For each row i, get the distances between A[i] and all rows in B
        # Then compute the rank of the matching distance
        ranks = []
        for i in range(n_samples):
            row_distances = distances[i, :]
            # Get the rank of the matching distance
            # Rank 1 means the smallest distance
            rank = np.argsort(row_distances).tolist().index(i) + 1
            ranks.append(rank)
        ranks = np.array(ranks)

        # Compute evaluation metrics
        mean_rank = np.mean(ranks)
        mrr = np.mean(1 / ranks)

        # Compare against expected values under random assignment
        rank_improvement = (expected_mean_rank - mean_rank) / (expected_mean_rank - 1)
        mrr_improvement = (mrr - expected_mrr) / (1 - expected_mrr)

        print(f"Mean Rank: {mean_rank:.2f} (Random: {expected_mean_rank:.2f})")
        print(f"MRR: {mrr:.4f} (Random: {expected_mrr:.4f})")
        print(f"Improvement over random (Rank): {rank_improvement * 100:.2f}%")
        print(f"Improvement over random (MRR): {mrr_improvement * 100:.2f}%\n")

        results[metric] = {
            'mean_rank': mean_rank,
            'expected_mean_rank': expected_mean_rank,
            'mrr': mrr,
            'expected_mrr': expected_mrr,
            'rank_improvement': rank_improvement,
            'mrr_improvement': mrr_improvement,
            'ranks': ranks
        }
    return results


def compute_random_matching_cost(rna, protein, metric='correlation'):
    """Compute normalized cost and distances for a random row assignment."""
    n_samples = rna.shape[0]
    random_indices = np.random.permutation(n_samples)
    protein_random = protein[random_indices]

    if metric == 'euclidean':
        distances = np.linalg.norm(rna - protein_random, axis=1)
    elif metric == 'cosine':
        # Normalize rows to compute cosine similarity
        rna_norm = rna / np.linalg.norm(rna, axis=1, keepdims=True)
        protein_random_norm = protein_random / np.linalg.norm(protein_random, axis=1, keepdims=True)
        cosine_similarity = np.sum(rna_norm * protein_random_norm, axis=1)
        distances = 1 - cosine_similarity  # Cosine distance
        for i in range(100):
            random_indices = np.random.permutation(n_samples)
            protein_random = protein[random_indices]
            protein_random_norm = protein_random / np.linalg.norm(protein_random, axis=1, keepdims=True)
            cosine_similarity = np.sum(rna_norm * protein_random_norm, axis=1)
            distances = np.vstack((distances, 1 - cosine_similarity))
        distances = np.mean(distances, axis=0)
    elif metric == 'correlation':
        # Compute Pearson correlation distance
        rna_mean = np.mean(rna, axis=1, keepdims=True)
        protein_random_mean = np.mean(protein_random, axis=1, keepdims=True)
        rna_centered = rna - rna_mean
        protein_random_centered = protein_random - protein_random_mean

        numerator = np.sum(rna_centered * protein_random_centered, axis=1)
        denominator = (
                np.sqrt(np.sum(rna_centered ** 2, axis=1)) *
                np.sqrt(np.sum(protein_random_centered ** 2, axis=1))
        )
        pearson_correlation = numerator / denominator
        distances = 1 - pearson_correlation  # Correlation distance

    else:
        raise ValueError("Unsupported metric. Use 'euclidean' or 'cosine'.")

    normalized_cost = np.sum(distances) / n_samples
    return normalized_cost, distances


def compare_matchings(archetype_proportion_list_rna, archetype_proportion_list_protein, metric='correlation',
                      num_trials=100):
    """Compare optimal matching cost with average random matching cost and plot norms."""
    # Extract the best pair based on optimal matching
    best_cost = float('inf')
    for i, (rna, protein) in enumerate(zip(archetype_proportion_list_rna, archetype_proportion_list_protein)):
        rna = rna.values if hasattr(rna, 'values') else rna
        protein = protein.values if hasattr(protein, 'values') else protein
        row_ind, col_ind, cost, cost_matrix = match_rows(rna, protein, metric)
        if cost < best_cost:
            best_cost = cost
            best_rna, best_protein = rna, protein
            best_rna_archetype_order, best_protein_archetype_order = row_ind, col_ind
            best_cost_matrix = cost_matrix
    print(f"Optimal normalized matching cost: {best_cost:.4f}")

    # Compute distances for the optimal matching
    optimal_distances = best_cost_matrix[best_rna_archetype_order, best_protein_archetype_order]

    # Compute distances for a single random matching
    random_cost, random_distances = compute_random_matching_cost(best_rna, best_protein, metric)

    # Visualization of distances
    n_samples = best_rna.shape[0]
    indices = np.arange(n_samples)

    plt.figure(figsize=(10, 6))
    plt.plot(indices, np.sort(optimal_distances), label='Optimal Matching', marker='o')
    plt.plot(indices, np.sort(random_distances), label='Random Matching', marker='x')
    plt.xlabel('Sample Index (sorted by distance)')
    plt.ylabel('Distance')
    plt.title('Comparison of Distances between Matched Rows')
    plt.legend()
    plt.show()

    # Compute average random matching cost over multiple trials
    random_costs = []
    for _ in range(num_trials):
        cost, _ = compute_random_matching_cost(best_rna, best_protein, metric)
        random_costs.append(cost)
    avg_random_cost = np.mean(random_costs)
    std_random_cost = np.std(random_costs)
    print(f"Average random matching cost over {num_trials} trials: {avg_random_cost:.4f}")
    print(f"Standard deviation: {std_random_cost:.4f}")

    # Bar plot of normalized matching costs
    labels = ['Optimal Matching', 'Random Matching']
    costs = [best_cost, avg_random_cost]
    errors = [0, std_random_cost]
    plt.figure(figsize=(8, 6))
    plt.bar(labels, costs, yerr=errors, capsize=5, color=['skyblue', 'lightgreen'])
    plt.ylabel('Normalized Matching Cost')
    plt.title('Optimal vs. Random Row Matching Costs')
    plt.show()


def match_rows(rna, protein, metric='correlation'):
    cost_matrix = cdist(rna, protein, metric=metric)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()
    n_archetypes = rna.shape[0]
    normalized_cost = total_cost / n_archetypes
    return row_ind, col_ind, normalized_cost, cost_matrix


def plot_latent(rna_mean, protein_mean, adata_rna_subset, adata_prot_subset, index):
    plt.figure(figsize=(10, 5))
    pca = PCA(n_components=2)
    pca.fit(rna_mean)
    rna_pca = pca.transform(rna_mean)
    plt.subplot(1, 3, 1)
    # plt.scatter(rna_pca[:, 0], rna_pca[:, 1], c=adata_rna_subset[index].obs['CN'], cmap='jet')
    plt.scatter(rna_pca[:, 0], rna_pca[:, 1], cmap='jet')
    plt.title('during training, RNA')

    pca.fit(protein_mean)
    protein_pca = pca.transform(protein_mean)
    plt.subplot(1, 3, 2)
    plt.scatter(protein_pca[:, 0], protein_pca[:, 1], c=adata_prot_subset[index].obs['CN'], cmap='jet')
    plt.title('during training, protein')

    # merge the two datasets

    plt.subplot(1, 3, 3)
    # plot merged RNA and protein
    plt.scatter(rna_pca[:, 0], rna_pca[:, 1], c='red', label='RNA')
    plt.scatter(protein_pca[:, 0], protein_pca[:, 1], c='blue', label='protein', alpha=0.5)
    plt.title('merged RNA and protein')
    plt.show()


def find_best_pair_by_row_matching(archetype_proportion_list_rna, archetype_proportion_list_protein,
                                   metric='correlation'):
    """
    Find the best index in the list by matching rows using linear assignment.

    Parameters:
    -----------
    archetype_proportion_list : list of tuples
        List where each tuple contains (rna, protein) matrices.
    metric : str, optional
        Distance metric to use ('euclidean' or 'cosine').

    Returns:
    --------
    best_num_or_archetypes_index : int
        Index of the best matching pair in the list.
    best_total_cost : float
        Total cost of the best matching.
    best_rna_archetype_order : np.ndarray
        Indices of RNA rows.
    best_protein_archetype_order : np.ndarray
        Indices of Protein rows matched to RNA rows.
    """
    best_num_or_archetypes_index = None
    best_total_cost = float('inf')
    best_rna_archetype_order = None
    best_protein_archetype_order = None

    for i, (rna, protein) in enumerate(zip(archetype_proportion_list_rna, archetype_proportion_list_protein)):
        rna = rna.values if hasattr(rna, 'values') else rna
        protein = protein.values if hasattr(protein, 'values') else protein

        assert rna.shape[1] == protein.shape[1], f"Mismatch in dimensions at index {i}."

        row_ind, col_ind, total_cost, _ = match_rows(rna, protein, metric=metric)
        print(f"Pair {i}: Total matching cost = {total_cost}")

        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_num_or_archetypes_index = i
            best_rna_archetype_order = row_ind
            best_protein_archetype_order = col_ind

    return best_num_or_archetypes_index, best_total_cost, best_rna_archetype_order, best_protein_archetype_order


def get_latest_file(folder, prefix):
    files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith('.h5ad')]
    if not files:
        return None
    files.sort(key=lambda x: re.search(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', x).group(), reverse=True)
    return os.path.join(folder, files[0])


def clean_uns_for_h5ad(adata: AnnData):
    """
    Remove or convert non-serializable objects from the `uns` attribute of an AnnData object.
    """
    keys_to_remove = []
    for key, value in adata.uns.items():
        if isinstance(value, sns.palettes._ColorPalette):
            # Convert seaborn ColorPalette to a list of colors
            adata.uns[key] = list(value)
        elif not isinstance(value, (str, int, float, list, dict, np.ndarray)):
            # Mark non-serializable keys for removal
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del adata.uns[key]
    if hasattr(adata, 'obsm'):
        adata.obsm = {str(key): value for key, value in adata.obsm.items()}


def plot_inference_outputs(rna_inference_outputs, protein_inference_outputs,
                           matching_rna_protein_latent_distances, rna_distances, prot_distances):
    """
    Plots latent distributions for RNA and protein, KL divergence scores,
    and histograms of distances for visual analysis.

    Parameters:
    - rna_inference_outputs: Dictionary with RNA latent space outputs.
    - protein_inference_outputs: Dictionary with protein latent space outputs.
    - matching_rna_protein_latent_distances: Torch tensor with KL divergence scores.
    - rna_distances: Torch tensor with RNA pairwise distances.
    - prot_distances: Torch tensor with protein pairwise distances.
    """

    # Plot for the first item in the batch
    plt.subplot(2, 1, 1)
    plot_torch_normal(rna_inference_outputs["qz"].mean[0][0].item(), rna_inference_outputs["qz"].scale[0][0].item())
    plot_torch_normal(protein_inference_outputs["qz"].mean[0][0].item(),
                      protein_inference_outputs["qz"].scale[0][0].item())
    plt.title(f'KL Divergence Score (Item 1): {matching_rna_protein_latent_distances[0][0].item()}')

    # Plot for the second item in the batch
    plt.subplot(2, 1, 2)
    plot_torch_normal(rna_inference_outputs["qz"].mean[1][0].item(), rna_inference_outputs["qz"].scale[1][0].item())
    plot_torch_normal(protein_inference_outputs["qz"].mean[1][0].item(),
                      protein_inference_outputs["qz"].scale[1][0].item())
    plt.title(f'KL Divergence Score (Item 2): {matching_rna_protein_latent_distances[1][0].item()}')
    plt.tight_layout()
    plt.show()

    # Histogram of distances for RNA, protein, and matching latent distances
    plt.figure(figsize=(10, 6))
    plt.hist(rna_inference_outputs["qz"].loc.detach().cpu().numpy().flatten(), bins=100, alpha=0.5,
             label='RNA Latent Distances')
    plt.hist(protein_inference_outputs["qz"].loc.detach().cpu().numpy().flatten(), bins=100, alpha=0.5,
             label='Protein Latent Distances')
    plt.hist(matching_rna_protein_latent_distances.detach().cpu().numpy().flatten(), bins=100, alpha=0.5,
             label='Matching Distances')
    plt.legend()
    plt.title("Histogram of Latent Distances")
    plt.show()

    # Histogram of RNA and protein distances
    plt.figure(figsize=(10, 6))
    plt.hist(prot_distances.detach().cpu().numpy().flatten(), bins=100, alpha=0.5, label='Protein Distances')
    plt.hist(rna_distances.detach().cpu().numpy().flatten(), bins=100, alpha=0.5, label='RNA Distances')
    plt.legend()
    plt.title("Histogram of Pairwise Distances")
    plt.show()

# funciton to avoid too much text in the notebook
from itertools import product, zip_longest
from typing import List

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy as np
# import starfysh
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
# import starfysh
from matplotlib import pyplot as plt
from py_pcha import PCHA
# !pip install starfysh
# !pip install pandas
# !pip install scanpy
# !pip install histomicstk
# !pip install --upgrade pip setuptools wheel
# !pip install pyvips --use-pep517
# !pip install histomicstk --find-links https://girder.github.io/large_image_wheels
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.backends.mkl import verbose

# computationally figure out which ones are best
np.random.seed(8)
plot_flag = False

from scipy.optimize import nnls
import cvxpy as cp
from sklearn.linear_model import OrthogonalMatchingPursuit




def nnls_omp(basis_matrix, target_vector, tol=1e-4):
    omp = OrthogonalMatchingPursuit(tol=tol, fit_intercept=False)
    omp.fit(basis_matrix.T, target_vector)
    weights = omp.coef_
    weights = np.maximum(0, weights)  # Enforce non-negativity
    return weights

import numpy as np
# from sklearn.linear_model import OrthogonalMatchingPursuit
#
#
# def nnls_omp(basis_matrix, target_vector, tol=1e-6):
#     """
#     Solve the non-negative least squares problem approximately using OMP.
#
#     Parameters:
#     -----------
#     basis_matrix : np.ndarray
#         Matrix of basis vectors, shape (n_basis_vectors, n_features).
#     target_vector : np.ndarray
#         Target vector to approximate, shape (n_features,).
#     tol : float, optional
#         Tolerance for reconstruction error (default is 1e-6).
#
#     Returns:
#     --------
#     weights : np.ndarray
#         Weights for the linear combination of basis vectors, shape (n_basis_vectors,).
#     """
#     # Initialize Orthogonal Matching Pursuit model
#     omp = OrthogonalMatchingPursuit(tol=tol, fit_intercept=False)
#     omp.fit(basis_matrix.T, target_vector)
#     weights = omp.coef_
#
#     # Enforce non-negativity (optional, depending on your requirement)
#     weights = np.maximum(0, weights)
#
#     return weights

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

    # Handle zero rows
    row_sums = weights.sum(axis=1, keepdims=True)
    weights[row_sums == 0] = 1.0 / n_archetypes  # Assign uniform weights to zero rows

    # Normalize weights
    weights /= weights.sum(axis=1, keepdims=True)

    return weights


# def get_cell_representations_as_archetypes_omp(count_matrix, archetype_matrix):
#     """
#     Compute archetype weights for each cell using OMP.
#     """
#     n_cells = count_matrix.shape[0]
#     n_archetypes = archetype_matrix.shape[0]
#     weights = np.zeros((n_cells, n_archetypes))
#     for i in range(n_cells):
#         weights[i] = nnls_omp(archetype_matrix.T, count_matrix[i])
#     weights /= weights.sum(axis=1, keepdims=True)  # Normalize rows
#     return weights

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
    # Track available rows and columns
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

    for i in range(n_cells):
        x = count_matrix[i]
        w = cp.Variable(n_archetypes)
        objective = cp.Minimize(cp.sum_squares(A_T @ w - x))
        constraints = [w >= 0]
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
        weights[i],_ = nnls(archetype_matrix.T, count_matrix[i])
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
    print(f"Variance ratio after log transformation PCA: {adata_rna.uns['pca']['variance_ratio'][:10].sum():.4f}")

    # Normalize total counts
    sc.pp.normalize_total(adata_rna, target_sum=5e3)
    sc.pp.pca(adata_rna)
    print(f"Variance ratio after normalization PCA: {adata_rna.uns['pca']['variance_ratio'][:10].sum():.4f}")


    # Scale the data
    # sc.pp.scale(adata_rna, max_value=10)
    # sc.pp.pca(adata_rna)
    # print(f"Variance ratio after scaling PCA: {adata_rna.uns['pca']['variance_ratio'][:10].sum():.4f}")

    return adata_rna


def preprocess_protein(adata_prot):
    sc.pp.pca(adata_prot)
    print(f"Variance ratio after PCA: {adata_prot.uns['pca']['variance_ratio'][:10].sum():.4f}")
    sc.pp.normalize_total(adata_prot)
    sc.pp.pca(adata_prot)
    print(f"Variance ratio after normalization PCA: {adata_prot.uns['pca']['variance_ratio'][:10].sum():.4f}")
    sc.pp.log1p(adata_prot)
    sc.pp.pca(adata_prot)
    print(f"Variance ratio after log transformation PCA: {adata_prot.uns['pca']['variance_ratio'][:10].sum():.4f}")
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
def add_spatial_data_to_prot(adata_prot_subset, major_to_minor_dict):
    horizontal_splits = [0, 500, 1000]
    vertical_splits = [0, 333, 666, 1000]
    regions = [
        list(product(range(horizontal_splits[i], horizontal_splits[i + 1]),
                     range(vertical_splits[j], vertical_splits[j + 1])))
        for i in range(len(horizontal_splits) - 1)
        for j in range(len(vertical_splits) - 1)
    ]
    board = np.zeros((1000, 1000))
    for idx, region in enumerate(regions):
        # Convert each region's list of tuples to an array to index properly
        coords = np.array(region)
        board[coords[:, 0], coords[:, 1]] = idx + 1  # Assign different values for each region
    if plot_flag:
        plt.imshow(board)
        plt.title('CNs')
        plt.colorbar()
        plt.show()
    # set x and y coor for each cell in dataset and place it in the adata_prot_subset
    adata_prot_subset.obs['X'] = np.random.randint(0, 1000, adata_prot_subset.n_obs)
    adata_prot_subset.obs['Y'] = np.random.randint(0, 1000, adata_prot_subset.n_obs)
    minor_to_region_dict = {}
    for i, (cell_type_1, cell_type_2, cell_type_3) in enumerate(
            zip_longest(major_to_minor_dict['B cells'], major_to_minor_dict['CD4 T'],
                        major_to_minor_dict['CD8 T'])):
        minor_to_region_dict[(cell_type_1, cell_type_2, cell_type_3)] = i

    # Place the cells in the regions
    for (cell_type_1, cell_type_2, cell_type_3), region in minor_to_region_dict.items():
        # Get the indices of the cells of the current cell type
        cell_indices_1 = adata_prot_subset.obs['cell_types'] == cell_type_1
        cell_indices_2 = adata_prot_subset.obs['cell_types'] == cell_type_2
        cell_indices_3 = adata_prot_subset.obs['cell_types'] == cell_type_3
        # Get the coordinates of the cells
        coords = np.array(regions[region])
        # Place the cells in the regions
        adata_prot_subset.obs['X'][cell_indices_1] = np.random.choice(coords[:, 0], sum(cell_indices_1))
        adata_prot_subset.obs['Y'][cell_indices_1] = np.random.choice(coords[:, 1], sum(cell_indices_1))
        adata_prot_subset.obs['X'][cell_indices_2] = np.random.choice(coords[:, 0], sum(cell_indices_2))
        adata_prot_subset.obs['Y'][cell_indices_2] = np.random.choice(coords[:, 1], sum(cell_indices_2))
        adata_prot_subset.obs['X'][cell_indices_3] = np.random.choice(coords[:, 0], sum(cell_indices_3))
        adata_prot_subset.obs['Y'][cell_indices_3] = np.random.choice(coords[:, 1], sum(cell_indices_3))
    adata_prot_subset.obsm['X_spatial'] = np.array(adata_prot_subset.obs[['X', 'Y']])
    return adata_prot_subset,horizontal_splits,vertical_splits

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



def plot_archetypes(data_points,archetype,samples_cell_types:List[str]):
    if not isinstance(samples_cell_types, List):
        raise TypeError("samples_cell_types should be a list of strings.")
    categories = np.unique(samples_cell_types)
    num_categories = len(categories) + 1
    colormap = plt.cm.get_cmap("tab20", num_categories)  # Use a categorical colormap like 'tab20'
    labels = ["data"] * len(data_points) + ["archetype"] * len(archetype)
    cell_types = samples_cell_types + ['archetype'] * len(archetype)  # -1 for archetypes
    data = np.concatenate((data_points, archetype), axis=0)
    data = np.asarray(data)
    data_pca = PCA(n_components=10).fit_transform(data)
    data_tsne = TSNE(n_components=2).fit_transform(data_pca)
    # Create a DataFrame
    df_pca = pd.DataFrame({
        "PCA1": data_pca[:, 0],
        "PCA2": data_pca[:, 1],
        "type": labels,
        "cell_type": cell_types
    })

    unique_cell_types = df_pca["cell_type"].unique()
    palette = sns.color_palette("tab20", len(unique_cell_types))  # Exclude archetype
    palette_dict = {cell_type: color for cell_type, color in zip(unique_cell_types, palette)}
    palette_dict["archetype"] = "black"  # Assign black to archetype

    df_tsne = pd.DataFrame({
        "TSNE1": data_tsne[:, 0],
        "TSNE2": data_tsne[:, 1],
        "type": labels,
        "cell_type": cell_types
    })

    plt.figure(figsize=(10, 6))
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
    handles, labels = plt.gca().get_legend_handles_labels()
    # Keep only the unique cell types (skip "data" and "archetype")
    cell_type_legend = [(h, l) for h, l in zip(handles, labels) if l in palette_dict.keys()]
    if cell_type_legend:
        handles, labels = zip(*cell_type_legend)
    plt.legend(handles, labels, title="Cell Types", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title("PCA Scatter Plot with Archetypes Highlighted in Black")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
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

    handles, labels = plt.gca().get_legend_handles_labels()
    cell_type_legend = [(h, l) for h, l in zip(handles, labels) if l in palette_dict.keys()]
    if cell_type_legend:
        handles, labels = zip(*cell_type_legend)
    plt.legend(handles, labels, title="Cell Types", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title("t-SNE Scatter Plot with Archetypes Highlighted in Black")
    plt.tight_layout()
    plt.show()


# funciton to avoid too much text in the notebook
import copy
from itertools import product, zip_longest
from typing import List, Dict
from scipy.optimize import linear_sum_assignment

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
from scipy.spatial.distance import cdist
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
    # Track available rows and
    original= None
    if isinstance(matrix,pd.DataFrame):
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
        reordered_matrix = pd.DataFrame(reordered_matrix,index=original.index,columns=original.columns)
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
    print(f"Variance ratio after log transformation PCA (10 PCs): {adata_rna.uns['pca']['variance_ratio'][:10].sum():.4f}")
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
    print(f"Variance ratio after log transformation PCA (10 PCs): {adata_prot.uns['pca']['variance_ratio'][:10].sum():.4f}")
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



def plot_archetypes(data_points, archetype, samples_cell_types: List[str],modality=''):
    if not isinstance(samples_cell_types, List):
        raise TypeError("samples_cell_types should be a list of strings.")

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
    data_pca = data[:,:50]
    data_tsne = TSNE(n_components=2).fit_transform(data_pca)

    # Create a numbering for archetypes
    archetype_numbers = [np.nan] * len(data_points) + list(range(0, num_archetypes))

    # Create DataFrames for plotting
    df_pca = pd.DataFrame({
        "PCA1": data_pca[:, 0],
        "PCA2": data_pca[:, 1],
        "type": labels,
        "cell_type": cell_types,
        "archetype_number": archetype_numbers
    })

    df_tsne = pd.DataFrame({
        "TSNE1": data_tsne[:, 0],
        "TSNE2": data_tsne[:, 1],
        "type": labels,
        "cell_type": cell_types,
        "archetype_number": archetype_numbers
    })
    df_tsne = df_tsne.sort_values(by='cell_type')
    # Define color palette
    # Ensure unique_cell_types are sorted consistently
    unique_cell_types = sorted(
        [cell_type for cell_type in df_pca["cell_type"].unique() if cell_type != 'archetype'],
        key=int
    )
    palette = sns.color_palette("tab20", len(unique_cell_types))
    palette_dict = {cell_type: color for cell_type, color in zip(unique_cell_types, palette)}
    palette_dict["archetype"] = "black"  # Assign black to archetype

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
    handles, labels = plt.gca().get_legend_handles_labels()
    cell_type_legend = [(h, l) for h, l in zip(handles, labels) if l in palette_dict.keys()]
    if cell_type_legend:
        handles, labels = zip(*cell_type_legend)
    plt.legend(handles, labels, title="Cell Types", bbox_to_anchor=(1.05, 1), loc="upper left")

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

    # Convert cell_type labels to integers for matching
    df_pca_data['cell_type_int'] = df_pca_data['cell_type'].astype(int)
    df_pca_archetypes['archetype_number'] = df_pca_archetypes['archetype_number'].astype(int)

    # Create a mapping from archetype_number to its PCA coordinates
    archetype_coords = df_pca_archetypes.set_index('archetype_number')[['PCA1', 'PCA2']]

    # Now for each data point, draw a line to its corresponding archetype
    for idx, row in df_pca_data.iterrows():
        cell_type_int = row['cell_type_int']
        data_point_coords = (row['PCA1'], row['PCA2'])
        try:
            archetype_point_coords = archetype_coords.loc[cell_type_int]
            plt.plot(
                [data_point_coords[0], archetype_point_coords['PCA1']],
                [data_point_coords[1], archetype_point_coords['PCA2']],
                color='gray', linewidth=0.5, alpha=0.3
            )
        except KeyError:
            # If cell_type_int does not match any archetype_number, skip
            pass

    plt.title(f"{modality} PCA Scatter Plot with Archetypes Numbered")
    plt.tight_layout()
    plt.show()

    # Plot t-SNE
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

    # Remove 'type' from the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    cell_type_legend = [(h, l) for h, l in zip(handles, labels) if l in palette_dict.keys()]
    if cell_type_legend:
        handles, labels = zip(*cell_type_legend)
    plt.legend(handles, labels, title="Cell Types", bbox_to_anchor=(1.05, 1), loc="upper left")

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

    # Convert cell_type labels to integers for matching
    df_tsne_data['cell_type_int'] = df_tsne_data['cell_type'].astype(int)
    df_tsne_archetypes['archetype_number'] = df_tsne_archetypes['archetype_number'].astype(int)

    # Create a mapping from archetype_number to its t-SNE coordinates
    archetype_coords_tsne = df_tsne_archetypes.set_index('archetype_number')[['TSNE1', 'TSNE2']]

    # Now for each data point, draw a line to its corresponding archetype
    for idx, row in df_tsne_data.iterrows():
        cell_type_int = row['cell_type_int']
        data_point_coords = (row['TSNE1'], row['TSNE2'])
        try:
            archetype_point_coords = archetype_coords_tsne.loc[cell_type_int]
            plt.plot(
                [data_point_coords[0], archetype_point_coords['TSNE1']],
                [data_point_coords[1], archetype_point_coords['TSNE2']],
                color='gray', linewidth=0.2, alpha=0.3
            )
        except KeyError:
            # If cell_type_int does not match any archetype_number, skip
            pass

    plt.title(f"{modality} t-SNE Scatter Plot with Archetypes Numbered")
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


def plot_archetypes_matching(data1,data2,rows = 5):
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
    n_samples = A.shape[0]

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
    elif metric == 'correlation':
        # Compute Pearson correlation distance
        rna_mean = np.mean(rna, axis=1, keepdims=True)
        protein_random_mean = np.mean(protein_random, axis=1, keepdims=True)
        rna_centered = rna - rna_mean
        protein_random_centered = protein_random - protein_random_mean

        numerator = np.sum(rna_centered * protein_random_centered, axis=1)
        denominator = (
            np.sqrt(np.sum(rna_centered**2, axis=1)) *
            np.sqrt(np.sum(protein_random_centered**2, axis=1))
        )
        pearson_correlation = numerator / denominator
        distances = 1 - pearson_correlation  # Correlation distance

    else:
        raise ValueError("Unsupported metric. Use 'euclidean' or 'cosine'.")

    normalized_cost = np.sum(distances) / n_samples
    return normalized_cost, distances


def compare_matchings(archetype_proportion_list_rna,archetype_proportion_list_protein, metric='cosine', num_trials=100):
    """Compare optimal matching cost with average random matching cost and plot norms."""
    # Extract the best pair based on optimal matching
    best_cost = float('inf')
    for i, (rna, protein) in enumerate(zip(archetype_proportion_list_rna,archetype_proportion_list_protein)):
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

def match_rows(rna, protein, metric='cosine'):
    cost_matrix = cdist(rna, protein, metric=metric)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()
    n_archetypes = rna.shape[0]
    normalized_cost = total_cost / n_archetypes
    return row_ind, col_ind, normalized_cost, cost_matrix


def find_best_pair_by_row_matching(archetype_proportion_list_rna,archetype_proportion_list_protein, metric='cosine'):
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

    for i, (rna, protein) in enumerate(zip(archetype_proportion_list_rna,archetype_proportion_list_protein)):
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
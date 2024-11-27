# funciton to avoid too much text in the notebook
from itertools import product, zip_longest
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import torch
plot_flag = False

from scipy.optimize import nnls
def get_cell_representations_as_archetypes(count_matrix, archetype_matrix):
    """
    Compute the linear combination weights of archetypes for each cell.
    Parameters:
    -----------
    count_matrix : np.ndarray
        Matrix of cells in reduced-dimensional space (e.g., PCA) [n_cells, n_features].
    archetype_matrix : np.ndarray
        Matrix of archetypes [n_archetypes, n_features].
    Returns:
    --------
    weights : np.ndarray
        Matrix of archetype weights for each cell [n_cells, n_archetypes].
        Rows sum to 1.
    """
    n_cells = count_matrix.shape[0]
    n_archetypes = archetype_matrix.shape[0]
    # Initialize weight matrix
    weights = np.zeros((n_cells, n_archetypes))
    # For each cell, solve the NNLS problem
    for i in range(n_cells):
        weights[i], _ = nnls(archetype_matrix.T, count_matrix[i])
    # Normalize rows to sum to 1
    weights /= weights.sum(axis=1, keepdims=True)
    return weights

def preprocess_rna(adata, adata_rna):
    # now need to do normalization
    # mitochondrial genes, "MT-" for human, "Mt-" for mouse
    # this CITE-seq data is mouse data
    adata_rna.var["mt"] = adata.var_names.str.startswith("Mt-")
    # ribosomal genes
    adata_rna.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes
    adata_rna.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

    sc.pp.calculate_qc_metrics(adata_rna, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)

    # sc.pl.scatter(adata_rna, "total_counts", "n_genes_by_counts", color="pct_counts_mt")

    # sc.pl.violin(
    #     adata_rna,
    #     ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    #     jitter=0.4,
    #     multi_panel=True,
    # )
    # adata_qc.concatenate(adata_slice)
    sc.pp.filter_cells(adata_rna, min_genes=100)
    sc.pp.filter_genes(adata_rna, min_cells=3)

    # finding doublets
    # adata.layers["counts"] = adata.X.copy()
    adata_rna.layers["counts"] = adata_rna.X.copy()
    sc.pp.normalize_total(adata_rna)
    sc.pp.log1p(adata_rna)

    sc.pp.highly_variable_genes(adata_rna, n_top_genes=2000, batch_key="batch")
    # sc.pl.highly_variable_genes(adata_rna)

    # sc.pl.pca_variance_ratio(adata_rna, n_pcs=50, log=True)

    #     sc.pl.pca(
    #     adata_rna,
    #     color=["sample", "sample", "pct_counts_mt", "pct_counts_mt"],
    #     dimensions=[(0, 1), (2, 3), (0, 1), (2, 3)],
    #     ncols=2,
    #     size=2,
    # )
    # sc.pp.neighbors(adata_rna)
    # sc.tl.umap(adata_rna)
    return adata_rna


def preprocess_protein(adata_prot):
    sc.pp.normalize_total(adata_prot)
    # might need to adjust these parameters for protein, not sure what the filtering should be
    sc.pp.filter_cells(adata_prot, min_genes=20)
    sc.pp.filter_genes(adata_prot, min_cells=3)
    return  adata_prot

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
            zip_longest(major_to_minor_dict['B cells'], major_to_minor_dict['T cells-2'],
                        major_to_minor_dict['T cells-1'])):
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

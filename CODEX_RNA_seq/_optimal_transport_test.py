# %%
import copy
import os
import sys
import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import ot
import scanpy as sc
import seaborn as sns
import torch
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def soften_matching(M, temperature=0.1):
    M_exp = np.exp(-M / temperature)
    return M_exp / M_exp.sum(axis=1, keepdims=True)


def compute_nn_distance_matrix(data, k=5):
    # Build k-NN graph
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(data)
    adj_matrix = nn.kneighbors_graph(mode="distance").toarray()

    # Compute shortest path distances (number of hops)
    _, predecessors = shortest_path(
        csgraph=adj_matrix,
        directed=False,
        return_predecessors=True,
        unweighted=True,  # Treat as unweighted graph for hop count
    )

    # Convert predecessors to hop counts
    n_samples = data.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            path = []
            k = j
            while predecessors[i, k] != -9999:
                path.append(k)
                k = predecessors[i, k]
                if k == i:
                    break
            dist_matrix[i, j] = len(path)

    return dist_matrix


def plot_archetypes_comparison(archetype1, archetype2, i, j):
    """Plot the source and target archetypes side by side."""
    if archetype1.shape[1] == 2:  # Only for 2D data
        # Limit to first 1000 cells
        n_cells_to_plot = min(1000, len(archetype1), len(archetype2))
        archetype1_np = archetype1[:n_cells_to_plot].cpu().numpy()
        archetype2_np = archetype2[:n_cells_to_plot].cpu().numpy()

        fig = plt.figure(figsize=(10, 5))

        # Source archetype
        ax1 = fig.add_subplot(121)
        ax1.scatter(
            archetype1_np[:, 0],
            archetype1_np[:, 1],
            c="b",
            label="Source",
        )
        ax1.set_title(f"Archetype {i} (Source) - First {n_cells_to_plot} cells")
        # Target archetype
        ax2 = fig.add_subplot(122)
        ax2.scatter(
            archetype2_np[:, 0],
            archetype2_np[:, 1],
            c="r",
            label="Target",
        )
        ax2.set_title(f"Archetype {j} (Target) - First {n_cells_to_plot} cells")
        plt.show()
        plt.close()


def plot_distance_matrices(C1, C2, i, j):
    """Plot the distance matrices of the source and target archetypes."""
    # Select first 1000 cells to plot
    n_cells_to_plot = min(1000, C1.shape[0], C2.shape[0])
    C1_np = C1[:n_cells_to_plot, :n_cells_to_plot].cpu().numpy()
    C2_np = C2[:n_cells_to_plot, :n_cells_to_plot].cpu().numpy()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(C1_np, cmap="viridis")
    plt.title(f"Source Archetype {i} Distance Matrix - First {n_cells_to_plot} cells")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(C2_np, cmap="viridis")
    plt.title(f"Target Archetype {j} Distance Matrix - First {n_cells_to_plot} cells")
    plt.colorbar()

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_transport_plan(transport_plan, i, j, gw_dist):
    """Plot the transport plan as a heatmap."""
    transport_plan_np = (
        transport_plan.cpu().numpy() if isinstance(transport_plan, torch.Tensor) else transport_plan
    )
    # Limit to first 1000 cells
    n_cells_to_plot = min(400, transport_plan_np.shape[0], transport_plan_np.shape[1])
    transport_plan_np = transport_plan_np[:n_cells_to_plot, :n_cells_to_plot]

    plt.figure(figsize=(10, 8))
    plt.imshow(np.log1p(transport_plan_np), cmap="viridis", aspect="auto")
    plt.colorbar(label="Transport mass")
    plt.title(
        f"GW Logarithm of Transport Plan values: Archetype {i} ↔ {j}\nDistance: {gw_dist:.4f}, First {n_cells_to_plot} cells"
    )
    plt.xlabel("Target Archetype Cells")
    plt.ylabel("Source Archetype Cells")
    plt.show()
    plt.close()


def plot_transformed_points(archetype1_np, archetype2_np, transformed_source_scaled, i, j):
    """Plot original source, transformed source, and target points."""
    # Limit to first 1000 cells
    n_cells_to_plot = min(
        1000, len(archetype1_np), len(archetype2_np), len(transformed_source_scaled)
    )
    archetype1_np = archetype1_np[:n_cells_to_plot]
    archetype2_np = archetype2_np[:n_cells_to_plot]
    transformed_source_scaled = transformed_source_scaled[:n_cells_to_plot]

    # Plot original source, transformed source, and target
    plt.figure(figsize=(15, 5))

    # Original source
    plt.subplot(1, 3, 1)
    plt.scatter(archetype1_np[:, 0], archetype1_np[:, 1], c="blue", label="Source", alpha=0.7)
    plt.title(f"Original Source (Archetype {i}) - First {n_cells_to_plot} cells")
    plt.legend()

    # Transformed source (with proper scaling)
    plt.subplot(1, 3, 2)
    plt.scatter(
        transformed_source_scaled[:, 0],
        transformed_source_scaled[:, 1],
        c="green",
        label="Transformed Source",
        alpha=0.7,
    )
    plt.title(f"Transformed Source (Scaled) - First {n_cells_to_plot} cells")
    plt.legend()

    # Target and scaled transformed source overlay
    plt.subplot(1, 3, 3)
    plt.scatter(archetype2_np[:, 0], archetype2_np[:, 1], c="red", label="Target", alpha=0.5)
    plt.scatter(
        transformed_source_scaled[:, 0],
        transformed_source_scaled[:, 1],
        c="green",
        label="Transformed Source",
        alpha=0.5,
    )

    plt.title(f"Overlay: Target (Arch {j}) & Transformed - First {n_cells_to_plot} cells")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


def scale_transformed_source(transport_plan_np, archetype2_np):
    """Transform source points using transport plan and scale them to match target range."""
    # Limit to first 1000 cells for consistency with other plots
    n_cells_to_plot = min(1000, transport_plan_np.shape[0], len(archetype2_np))
    transport_plan_np_subset = transport_plan_np[:n_cells_to_plot, :n_cells_to_plot]
    archetype2_np_subset = archetype2_np[:n_cells_to_plot]

    # Use the transport plan to transform source points
    # Each source point is mapped as a weighted combination of target points
    transformed_source = transport_plan_np_subset @ archetype2_np_subset

    # Rescale transformed source to match target scale for better visualization
    # Get min/max of target data
    target_min = archetype2_np_subset.min(axis=0)
    target_max = archetype2_np_subset.max(axis=0)
    target_range = target_max - target_min

    # Get min/max of transformed source
    trans_min = transformed_source.min(axis=0)
    trans_max = transformed_source.max(axis=0)
    trans_range = trans_max - trans_min

    # Apply scaling to match target range
    transformed_source_scaled = np.zeros_like(transformed_source)
    for dim in range(transformed_source.shape[1]):
        if trans_range[dim] > 1e-10:  # Avoid division by zero
            transformed_source_scaled[:, dim] = (
                transformed_source[:, dim] - trans_min[dim]
            ) / trans_range[dim] * target_range[dim] + target_min[dim]
        else:
            transformed_source_scaled[:, dim] = target_min[dim] + target_range[dim] / 2
    return transformed_source_scaled


def plot_connections(
    archetype2_np, transformed_source_scaled, transport_plan_np, i, j, num_connections=50
):
    """Plot connections between transformed source and target points.

    Note: assumes transformed_source_scaled is already limited to 1000 cells
    and we need to use the same subset of transport_plan_np.
    """
    # Limit transport_plan to match the size of transformed_source_scaled
    n_source = min(len(transformed_source_scaled), transport_plan_np.shape[0])
    n_target = min(len(archetype2_np), transport_plan_np.shape[1])

    # Use the same subset for calculation
    transport_plan_subset = transport_plan_np[:n_source, :n_target]

    # Find the best match in space 2 for each cell in space 1
    matches_1_to_2 = np.argmax(transport_plan_subset, axis=1)

    # Get the matching weights for each source cell to its best target match
    match_weights = np.array(
        [transport_plan_subset[idx, matches_1_to_2[idx]] for idx in range(len(matches_1_to_2))]
    )

    # Get indices of cells sorted by match weight (highest to lowest)
    sorted_indices = np.argsort(match_weights)[::-1]  # Reverse to get highest first

    # Take only the top connections for arrows (based on match strength)
    source_indices = sorted_indices[:num_connections]
    target_indices = matches_1_to_2[source_indices]

    # Define a list of colors to cycle through
    line_colors = ["black", "orange", "purple", "brown"]

    plt.figure(figsize=(8, 8))

    # Plot all cells in the current subset
    plt.scatter(
        archetype2_np[:n_target, 0], archetype2_np[:n_target, 1], c="red", label="Target", alpha=0.5
    )
    plt.scatter(
        transformed_source_scaled[:, 0],
        transformed_source_scaled[:, 1],
        c="green",
        label="Transformed Source",
        alpha=0.5,
    )

    # Draw arrows for the top connections
    for idx, (src_idx, tgt_idx) in enumerate(zip(source_indices, target_indices)):
        weight = transport_plan_subset[src_idx, tgt_idx]
        # Use smaller arrow width parameters
        arrow_width = 0.01 + 0.05 * weight / np.max(transport_plan_subset)
        color_idx = idx % len(line_colors)
        plt.arrow(
            transformed_source_scaled[src_idx, 0],
            transformed_source_scaled[src_idx, 1],
            archetype2_np[tgt_idx, 0] - transformed_source_scaled[src_idx, 0],
            archetype2_np[tgt_idx, 1] - transformed_source_scaled[src_idx, 1],
            color=line_colors[color_idx],
            alpha=0.6,
            width=arrow_width,
            head_width=arrow_width * 2,  # Smaller multiplier
            head_length=arrow_width * 3,  # Smaller multiplier
            length_includes_head=True,
        )

    plt.title(
        f"Best Matches for Archetype {i} → {j} (Showing {len(source_indices)} Strongest Connections)"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_distribution_comparison(archetype1_np, archetype2_np, transformed_source_scaled, i, j):
    """Plot histograms comparing distributions before and after transport."""
    # Limit to first 1000 cells
    n_cells_to_plot = min(
        1000, len(archetype1_np), len(archetype2_np), len(transformed_source_scaled)
    )
    archetype1_np = archetype1_np[:n_cells_to_plot]
    archetype2_np = archetype2_np[:n_cells_to_plot]
    transformed_source_scaled = transformed_source_scaled[:n_cells_to_plot]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Define bins consistently across all histograms
    x_min = min(
        archetype1_np[:, 0].min(),
        archetype2_np[:, 0].min(),
        transformed_source_scaled[:, 0].min(),
    )
    x_max = max(
        archetype1_np[:, 0].max(),
        archetype2_np[:, 0].max(),
        transformed_source_scaled[:, 0].max(),
    )
    y_min = min(
        archetype1_np[:, 1].min(),
        archetype2_np[:, 1].min(),
        transformed_source_scaled[:, 1].min(),
    )
    y_max = max(
        archetype1_np[:, 1].max(),
        archetype2_np[:, 1].max(),
        transformed_source_scaled[:, 1].max(),
    )

    x_bins = np.linspace(x_min, x_max, 100)
    y_bins = np.linspace(y_min, y_max, 100)

    # Top-left: Source X distribution
    axs[0, 0].hist(
        archetype1_np[:, 0],
        bins=x_bins,
        alpha=0.5,
        color="blue",
        label="Source",
    )
    axs[0, 0].hist(
        transformed_source_scaled[:, 0],
        bins=x_bins,
        alpha=0.5,
        color="green",
        label="Transformed",
    )
    axs[0, 0].hist(
        archetype2_np[:, 0],
        bins=x_bins,
        alpha=0.5,
        color="red",
        label="Target",
    )
    axs[0, 0].set_title("X Dimension Distribution")
    axs[0, 0].legend()

    # Top-right: Source Y distribution
    axs[0, 1].hist(
        archetype1_np[:, 1],
        bins=y_bins,
        alpha=0.5,
        color="blue",
        label="Source",
    )
    axs[0, 1].hist(
        transformed_source_scaled[:, 1],
        bins=y_bins,
        alpha=0.5,
        color="green",
        label="Transformed",
    )
    axs[0, 1].hist(
        archetype2_np[:, 1],
        bins=y_bins,
        alpha=0.5,
        color="red",
        label="Target",
    )
    axs[0, 1].set_title("Y Dimension Distribution")
    axs[0, 1].legend()

    # Bottom-left: 2D density of source and transformed
    axs[1, 0].scatter(archetype1_np[:, 0], archetype1_np[:, 1], c="blue", alpha=0.5, label="Source")
    axs[1, 0].scatter(
        transformed_source_scaled[:, 0],
        transformed_source_scaled[:, 1],
        c="green",
        alpha=0.5,
        label="Transformed",
    )
    axs[1, 0].set_title("Source vs Transformed")
    axs[1, 0].legend()

    # Bottom-right: 2D density of transformed and target
    axs[1, 1].scatter(
        transformed_source_scaled[:, 0],
        transformed_source_scaled[:, 1],
        c="green",
        alpha=0.5,
        label="Transformed",
    )
    axs[1, 1].scatter(archetype2_np[:, 0], archetype2_np[:, 1], c="red", alpha=0.5, label="Target")
    axs[1, 1].set_title("Transformed vs Target")
    axs[1, 1].legend()

    plt.suptitle(f"Distribution Comparison: Archetype {i} → {j} - First {n_cells_to_plot} cells")
    plt.tight_layout()
    plt.show()
    plt.close()


def get_2D_data(archetype1, archetype2):
    """Convert high-dimensional data to 2D using PCA."""
    # Limit to first 1000 cells
    n_cells_to_plot = min(1000, len(archetype1), len(archetype2))
    archetype1_subset = archetype1[:n_cells_to_plot]
    archetype2_subset = archetype2[:n_cells_to_plot]

    # Visualize the transported source points to see overlap
    archetype1_np = archetype1_subset.cpu().numpy()
    archetype2_np = archetype2_subset.cpu().numpy()

    # Merge data and apply PCA
    len_archetype1 = len(archetype1_np)
    len_archetype2 = len(archetype2_np)
    merge_data = np.concatenate((archetype1_np, archetype2_np), axis=0)
    merge_data = PCA(n_components=2).fit_transform(merge_data)
    archetype1_np = merge_data[:len_archetype1, :]
    archetype2_np = merge_data[len_archetype1:, :]

    return archetype1_np, archetype2_np


def print_transport_plan_validation(transport_plan, p, q, i, j):
    print(f"\nTransport Plan Validation ({i}→{j}):")
    print(f"Mass conservation error: {torch.abs(p.sum() - q.sum()):.2e}")
    print(f"Source marginal L1 error: {torch.abs(transport_plan.sum(1) - p).sum().item():.2e}")
    print(f"Target marginal L1 error: {torch.abs(transport_plan.sum(0) - q).sum().item():.2e}")
    print(f"Percentage of mass transported: {100*transport_plan.sum().item()/p.sum().item():.1f}%")


def plot_convergence(log, i, j):
    """Plot the convergence of the Gromov-Wasserstein algorithm."""
    if "errs" in log:
        plt.figure()
        plt.plot(log["errs"])
        plt.yscale("log")
        plt.title(f"GW Convergence: Archetype {i} ↔ {j}")
        plt.xlabel("Iteration")
        plt.ylabel("Marginal Violation (log scale)")
        plt.show()
        plt.close()


def compute_distance_matrix(archetype1, archetype2, metric: Literal["euclidean", "nn"] = "nn"):
    if metric == "euclidean":
        C1 = torch.cdist(archetype1, archetype1)
        C2 = torch.cdist(archetype2, archetype2)
    elif metric == "nn":
        C1 = compute_nn_distance_matrix(archetype1, k=5)
        C2 = compute_nn_distance_matrix(archetype2, k=5)

    # Normalize the distance matrices
    if C1.max() > 0:
        C1 = C1 / C1.max()
    if C2.max() > 0:
        C2 = C2 / C2.max()
    return C1, C2


# read those adata_1_rna.write(f"CODEX_RNA_seq/data/processed_data/adata_rna_archetype_generated_ot_test.h5ad")
# adata_2_prot.write(f"CODEX_RNA_seq/data/processed_data/adata_prot_archetype_generated_ot_test.h5ad")
adata_1_rna = sc.read_h5ad(
    "CODEX_RNA_seq/data/processed_data/adata_rna_archetype_generated_ot_test.h5ad"
)
adata_2_prot = sc.read_h5ad(
    "CODEX_RNA_seq/data/processed_data/adata_prot_archetype_generated_ot_test.h5ad"
)
# subsample the adata_1_rna and adata_2_prot using scanpy subsample function
sc.pp.subsample(adata_1_rna, n_obs=2000)
sc.pp.subsample(adata_2_prot, n_obs=2000)
# %%
# %%
# start OT
########################################################
archetypes_rna = adata_1_rna.uns["archetypes"]
archetypes_prot = adata_2_prot.uns["archetypes"]
# weights_prot = get_cell_representations_as_archetypes_cvxpy(
#     adata_2_prot.obsm["X_pca"], archetypes_prot
# )
# weights_rna = get_cell_representations_as_archetypes_cvxpy(
#     adata_1_rna.obsm["X_pca"], archetypes_rna
# )
weights_rna = adata_1_rna.obsm["archetype_vec"]
weights_prot = adata_2_prot.obsm["archetype_vec"]

# %%
op_subsample_size = 100

num_archetypes = weights_rna.shape[1]
# each items in the list is an array of cells that are assigned to the archetype
rna_archetypes = []
print(num_archetypes)
# Find cells for each RNA archetype
arche_index_for_each_cell_rna = np.argmax(weights_rna, axis=1)
for i in range(num_archetypes):
    locs = arche_index_for_each_cell_rna == i
    archetype_cells = adata_1_rna.obsm["X_pca"][locs, :2]
    # Ensure exactly op_subsample_size cells
    if len(archetype_cells) > op_subsample_size:
        archetype_cells = archetype_cells[:op_subsample_size]
    elif len(archetype_cells) < op_subsample_size:
        # If we don't have enough cells, pad with random samples from existing cells
        n_pad = op_subsample_size - len(archetype_cells)
        padding_indices = np.random.choice(len(archetype_cells), n_pad)
        padding = archetype_cells[padding_indices]
        archetype_cells = np.vstack([archetype_cells, padding])
    rna_archetypes.append(archetype_cells)

# Create lists to store cells for each archetype (Protein)
# each items in the list is an array of cells that are assigned to the archetype
prot_archetypes = []
# Find cells for each Protein archetype
arche_index_for_each_cell_prot = np.argmax(weights_prot, axis=1)
for i in range(num_archetypes):
    locs = arche_index_for_each_cell_prot == i
    archetype_cells = adata_2_prot.obsm["X_pca"][locs, :2]
    # Ensure exactly     cells
    if len(archetype_cells) > op_subsample_size:
        archetype_cells = archetype_cells[:op_subsample_size]
    elif len(archetype_cells) < op_subsample_size:
        # If we don't have enough cells, pad with random samples from existing cells
        n_pad = op_subsample_size - len(archetype_cells)
        padding_indices = np.random.choice(len(archetype_cells), n_pad)
        padding = archetype_cells[padding_indices]
        archetype_cells = np.vstack([archetype_cells, padding])
    prot_archetypes.append(archetype_cells)


# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# make sure to move variables to GPU
archetyp_rna_cells = rna_archetypes[:4]
archetyp_prot_cells = prot_archetypes[:4]
# Take a subset of each of the archetypes
# archetyp_prot_cells = [archetype[:min(len(archetype), 30)] for archetype in archetyp_prot_cells]
# archetyp_rna_cells = [archetype[:min(len(archetype), 30)] for archetype in archetyp_rna_cells]

earchetyp_1_rna_cell_dummy = copy.deepcopy(archetyp_rna_cells)
eaxmple_prot_cell_dummy = copy.deepcopy(archetyp_prot_cells)
# eaxmple_prot_cell_dummy[0] = eaxmple_prot_cell_dummy[0]+ np.random.normal(0, 0.1, eaxmple_prot_cell_dummy[0].shape)# PCA().fit_transform(earchetyp_1_rna_cell_dummy[0])
# shuffle the dimension of the archetypes
earchetyp_1_rna_cell_dummy[0] = eaxmple_prot_cell_dummy[0][:, ::] + 0.01 * np.random.normal(
    0, 0.1, eaxmple_prot_cell_dummy[0].shape
)
earchetyp_1_rna_cell_dummy[0] = earchetyp_1_rna_cell_dummy[0][
    :, :
]  #  try unbalance the number of cells in one of the distribution
# PCA().fit_transform(earchetyp_1_rna_cell_dummy[0])
# Add some noise to the RNA archetypes
earchetyp_1_rna_cell_dummy[1] = eaxmple_prot_cell_dummy[1][:, ::-1] + 0.01 * np.random.normal(
    0, 0.1, eaxmple_prot_cell_dummy[1].shape
)
earchetyp_1_rna_cell_dummy[2] = eaxmple_prot_cell_dummy[2][:, ::-1] + 0.01 * np.random.normal(
    0, 0.1, eaxmple_prot_cell_dummy[2].shape
)
earchetyp_1_rna_cell_dummy[3] = eaxmple_prot_cell_dummy[3][:, ::-1] + 0.01 * np.random.normal(
    0, 0.1, eaxmple_prot_cell_dummy[3].shape
)
# make not a function
archetypes_space1 = earchetyp_1_rna_cell_dummy
archetypes_space2 = eaxmple_prot_cell_dummy
metric = "euclidean"  #  change to neighbor graph distance
loss_type = "kl_loss"
# loss_type = "square_loss"
epsilon = 0.01
max_iter = 100  # should be higher
# %%

"""
Match archetypes across different feature spaces using Gromov-Wasserstein Optimal Transport.
"""
print(f"sizes of the archetypes {(archetypes_space1[0].shape)} {(archetypes_space2)[0].shape}")
# Number of archetypes in each space
num_archetypes1 = len(archetypes_space1)
num_archetypes2 = len(archetypes_space2)
# Create a cost matrix between archetypes across spaces
cost_matrix = np.zeros((num_archetypes1, num_archetypes2))
# Compute distances between each pair of archetypes across spaces
# Before GW computation
archetypes_space1 = [
    torch.tensor(archetype, device=device, dtype=torch.float32) for archetype in archetypes_space1
]
archetypes_space2 = [
    torch.tensor(archetype, device=device, dtype=torch.float32) for archetype in archetypes_space2
]
for i, archetype1 in tqdm(
    enumerate(archetypes_space1), total=num_archetypes1, desc="Computing GW distances"
):
    for j, archetype2 in enumerate(archetypes_space2):
        # Visualize the archetypes
        plot_archetypes_comparison(archetype1, archetype2, i, j)

        # Skip if either archetype has no cells
        if len(archetype1) == 0 or len(archetype2) == 0:
            cost_matrix[i, j] = np.inf
            continue

        # Compute distance matrices within each archetype

        C1, C2 = compute_distance_matrix(archetype1, archetype2, metric=metric)
        # Plot the distance matrices
        plot_distance_matrices(C1, C2, i, j)

        # Define weights for samples (uniform weights)
        p = torch.ones(len(archetype1), device=device, dtype=torch.float32) / len(archetype1)
        q = torch.ones(len(archetype2), device=device, dtype=torch.float32) / len(archetype2)

        # Compute Gromov-Wasserstein distance
        gw_dist, log = ot.gromov.entropic_gromov_wasserstein2(
            C1,
            C2,
            p,
            q,
            loss_type,
            epsilon=epsilon,
            max_iter=max_iter,
            tol=1e-6,
            verbose=True,
            log=True,
        )
        transport_plan = log["T"]  # Get transport plan from log dict
        print_transport_plan_validation(transport_plan, p, q, i, j)

        # Plot the transport plan
        plot_transport_plan(transport_plan, i, j, gw_dist)
        transport_plan_np = transport_plan.cpu().numpy()

        archetype1_np, archetype2_np = get_2D_data(archetype1, archetype2)

        transformed_source_scaled = scale_transformed_source(transport_plan_np, archetype2_np)
        # Plot connections between points
        plot_transformed_points(archetype1_np, archetype2_np, transformed_source_scaled, i, j)
        plot_connections(archetype2_np, transformed_source_scaled, transport_plan_np, i, j)

        # Plot distribution comparison
        plot_distribution_comparison(archetype1_np, archetype2_np, transformed_source_scaled, i, j)

        # Plot convergence
        plot_convergence(log, i, j)

        # Use the Gromov-Wasserstein distance as the cost
        cost_matrix[i, j] = gw_dist

# %%
# Handle potential numerical issues in the cost matrix
cost_matrix = np.nan_to_num(cost_matrix, nan=np.inf, posinf=np.inf, neginf=0)

# If all values are infinite, set them to 1 to avoid algorithm failure
if np.all(~np.isfinite(cost_matrix)):
    warnings.warn("All values in cost matrix are invalid. Using uniform costs.")
    raise ValueError("All values in cost matrix are invalid. Using uniform costs.")
# Define weights for archetypes (uniform weights)
weights_archetypes1 = np.ones(num_archetypes1) / num_archetypes1
weights_archetypes2 = np.ones(num_archetypes2) / num_archetypes2

# Solve the optimal transport problem to match archetypes
matching = ot.emd(
    weights_archetypes1, weights_archetypes2, cost_matrix
)  # give one to one matching? am  I sure?
# matching = ot.sinkhorn(weights_archetypes1, weights_archetypes2,
#               cost_matrix, reg=1.0)  # Try higher reg values
# matching = ot.unbalanced.sinkhorn_unbalanced(
# weights_archetypes1, weights_archetypes2,
# cost_matrix, reg=1.0, reg_m=1.0
# )

# cost_matrix = soften_matching (cost_matrix)


# # Example usage
# if __name__ == "__main__":
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
# archetype_proportion_list_rna
# matching, cost_matrix = match_archetypes_with_gw(rna_archetypes, prot_archetypes)


# %%
# Plot the heatmap of matching and cost_matrix
plt.figure(figsize=(12, 6))

# Plot matching heatmap
# This shows the transport plan between archetypes - higher values indicate stronger correspondences
# Values close to 1/(num_archetypes*num_archetypes) suggest uniform matching
plt.subplot(1, 2, 1)
sns.heatmap(100 * matching, annot=True, fmt=".2f", cmap="viridis", cbar=True)
plt.title("Matching Heatmap")
plt.xlabel("Archetypes in Space 2")
plt.ylabel("Archetypes in Space 1")

# This shows the Gromov-Wasserstein distances between archetypes
# Lower values (darker in magma colormap) indicate more structural similarity
plt.subplot(1, 2, 2)
sns.heatmap(100 * cost_matrix, annot=True, fmt=".2f", cmap="magma", cbar=True)
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


# %%
# plotting the results of the lowest num of archetypes
fig = plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
# sns.heatmap(reorder_rows_to_maximize_diagonal(archetype_proportion_list_rna[0])[0])
sns.heatmap((archetype_proportion_list_rna[0]), cbar=False)
plt.xticks()
plt.title("RNA Archetypes")
plt.yticks([])
plt.ylabel("Archetypes")
plt.subplot(1, 2, 2)
plt.title("Protein Archetypes")
# sns.heatmap(reorder_rows_to_maximize_diagonal(archetype_proportion_list_protein[0])[0])
sns.heatmap((archetype_proportion_list_protein[0]), cbar=False)
plt.suptitle("showcase the relationship between archetypes and cell types")
plt.yticks([])
plt.suptitle("Non-Aligned Archetypes Profiles")
plt.ylabel("Archetypes")
plt.show()

# end OT

# %%

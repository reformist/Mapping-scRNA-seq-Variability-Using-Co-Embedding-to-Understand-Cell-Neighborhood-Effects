# %%
import copy
import functools
import os
import sys
import time
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
from tabulate import tabulate
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch as th

"""
for the KNN implementation
Author: Josue N Rivera (github.com/wzjoriv)
Date: 7/3/2021
Description: Snippet of various clustering implementations only using PyTorch
Full project repository: https://github.com/wzjoriv/Lign (A graph deep learning framework that works alongside PyTorch)
"""


def timeit(func):
    """
    Decorator to measure execution time of functions.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result

    return wrapper


@timeit
def plot_transport_plan_diagonal_vs_all(transport_plan_np, i, j, bins=50):
    """
    Create a histogram comparing the transport plan's diagonal elements vs all elements.

    Parameters:
    -----------
    transport_plan_np : numpy.ndarray
        The transport plan matrix
    i, j : int
        Source and target archetype indices (for title)
    bins : int
        Number of bins for the histogram
    """
    # Get diagonal and all elements
    n_cells = min(transport_plan_np.shape[0], transport_plan_np.shape[1])
    diag_elements = np.diag(transport_plan_np[:n_cells, :n_cells])
    all_elements = transport_plan_np.flatten()

    # Remove zeros from all_elements to focus on nonzero transport weights
    nonzero_elements = all_elements[all_elements > 0]

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot normalized histograms with log scale
    plt.hist(
        nonzero_elements,
        bins=bins,
        density=True,
        alpha=0.5,
        label="All nonzero elements",
        color="blue",
        log=True,
    )
    plt.hist(
        diag_elements,
        bins=bins,
        density=True,
        alpha=0.7,
        label="Diagonal elements",
        color="red",
        log=True,
    )

    # Add vertical lines for means
    diag_mean = diag_elements.mean()
    all_mean = nonzero_elements.mean()

    plt.axvline(x=diag_mean, color="red", linestyle="--", label=f"Diagonal mean: {diag_mean:.4f}")
    plt.axvline(x=all_mean, color="blue", linestyle="--", label=f"All nonzero mean: {all_mean:.4f}")

    # Add statistics as text annotation
    diag_sum = diag_elements.sum()
    all_sum = transport_plan_np.sum()
    diag_percentage = (diag_sum / all_sum) * 100

    plt.annotate(
        f"Diagonal sum: {diag_sum:.4f}\n"
        f"Total transport: {all_sum:.4f}\n"
        f"Diagonal %: {diag_percentage:.2f}%\n"
        f"Nonzero count: {len(nonzero_elements)}\n"
        f"Sparsity: {100 - 100*len(nonzero_elements)/transport_plan_np.size:.1f}%",
        xy=(0.02, 0.96),
        xycoords="axes fraction",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    # Formatting
    plt.title(f"Transport Plan Weight Distribution: Archetype {i} ↔ {j}")
    plt.xlabel("Transport weight")
    plt.ylabel("Normalized frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"temp1.png")
    # Add a second plot showing cumulative distribution
    plt.figure(figsize=(10, 6))

    # Sort transport weights by value (descending)
    sorted_weights = np.sort(nonzero_elements)[::-1]
    cumsum = np.cumsum(sorted_weights)
    cumsum_normalized = cumsum / cumsum[-1]

    # Calculate percentiles
    percentiles = np.linspace(0, 100, len(cumsum))

    # Find points where cumulative sum reaches certain thresholds
    thresholds = [50, 75, 90, 95, 99]
    threshold_indices = []
    for threshold in thresholds:
        idx = np.argmax(cumsum_normalized >= threshold / 100)
        threshold_indices.append(idx)

    plt.plot(percentiles, cumsum_normalized, "g-")

    # Add markers for thresholds
    for idx, threshold in zip(threshold_indices, thresholds):
        plt.scatter(percentiles[idx], cumsum_normalized[idx], color="red", s=50, zorder=5)
        plt.annotate(
            f"{threshold}% mass: top {idx/(len(sorted_weights))*100:.1f}% elements",
            xy=(percentiles[idx], cumsum_normalized[idx]),
            xytext=(10, -10),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )

    plt.title(f"Cumulative Transport Weight Distribution: Archetype {i} ↔ {j}")
    plt.xlabel("Percentile of transport weights (sorted desc)")
    plt.ylabel("Cumulative mass fraction")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"temp.png")
    plt.show()
    # plot heatmap of the x first cells of the transport plan with 1 for the top (size of the diagonal) number of cells and 0 for the rest
    # this is to see the most significant matches in the first x cells which should be the diagonal if we are using the same values just with noise
    subsample_size = 300
    binary_transport_plan = np.zeros_like(transport_plan_np[:subsample_size, :subsample_size])
    binary_transport_plan_shape = binary_transport_plan.shape
    size_of_diagonal = min(binary_transport_plan_shape[0], binary_transport_plan_shape[1])
    top_indices = np.argsort(transport_plan_np[:subsample_size, :subsample_size].flatten())[
        -size_of_diagonal:
    ]
    binary_transport_plan = binary_transport_plan.flatten()
    binary_transport_plan[top_indices] = 1
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        binary_transport_plan.reshape(
            binary_transport_plan_shape[0], binary_transport_plan_shape[1]
        )
    )
    plt.title(
        f"Transport Plan , most sigificant matches in the first {subsample_size} cells: Archetype {i} ↔ {j}"
    )
    plt.savefig(f"temp2.png")
    plt.show()


def distance_matrix(x, y=None, p=2):  # pairwise distance of vectors
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = (
        th.linalg.vector_norm(x - y, p, 2)
        if th.__version__ >= "1.7.0"
        else th.pow(x - y, p).sum(2) ** (1 / p)
    )

    return dist


class NN:
    def __init__(self, X=None, Y=None, p=2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p)
        labels = th.argmin(dist, dim=1)
        return self.train_label[labels]


def knn_distances(x, k=1, p=2):
    """Compute pairwise distances to k-nearest neighbors."""
    dist = torch.cdist(x, x, p=p)
    dist.fill_diagonal_(torch.inf)  # Ignore self-distance
    knn_dists, _ = dist.topk(k, largest=False, sorted=True)
    return knn_dists[:, 0]  # Return only nearest neighbor distances


def knn_graph(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Builds a k-NN graph adjacency matrix."""
    dist = torch.cdist(x, x)
    dist.fill_diagonal_(torch.inf)  # Ignore self-edges
    _, indices = dist.topk(k, dim=1, largest=False)
    return indices


def compute_hop_distance(adj: torch.Tensor, max_hops: int = 10) -> torch.Tensor:
    """Computes pairwise hop distances using BFS."""
    n = adj.size(0)
    hop_dists = torch.full((n, n), float("inf"), device=adj.device)

    for src in tqdm(range(n), desc="Computing hop distances", total=n):
        queue = torch.tensor([src], device=adj.device)
        visited = torch.tensor([src], device=adj.device)
        hop_dists[src, src] = 0
        current_hop = 1

        while queue.numel() > 0 and current_hop <= max_hops:
            neighbors = adj[queue].flatten().unique()
            new_nodes = neighbors[~torch.isin(neighbors, visited)]

            if new_nodes.numel() == 0:
                break

            hop_dists[src, new_nodes] = current_hop
            visited = torch.cat([visited, new_nodes])
            queue = new_nodes
            current_hop += 1

    return hop_dists


class KNN(NN):
    def __init__(self, X=None, Y=None, k=3, p=2):
        self.k = k
        super().__init__(X, Y, p)

    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p)

        knn = dist.topk(self.k, largest=False)
        votes = self.train_label[knn.indices]

        winner = th.zeros(votes.size(0), dtype=votes.dtype, device=votes.device)
        count = th.zeros(votes.size(0), dtype=votes.dtype, device=votes.device) - 1

        for lab in self.unique_labels:
            vote_count = (votes == lab).sum(1)
            who = vote_count >= count
            winner[who] = lab
            count[who] = vote_count[who]

        return winner


@timeit
def plot_distance_heatmap(C1, i, j, max_cells=100):
    """Plot enhanced hop distance matrix with histogram."""
    # Subset for visualization
    C1_np = C1[:max_cells, :max_cells].cpu().numpy()

    # Create figure with gridspec for layout
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4])

    # Top histogram of distances
    ax_histx = fig.add_subplot(gs[0, 0])
    flat_dists = C1_np.flatten()
    flat_dists = flat_dists[~np.isinf(flat_dists)]  # Remove inf values
    bins = np.sqrt(len(flat_dists))  # Square root rule for bin count
    ax_histx.hist(flat_dists, bins=int(bins), color="darkblue", alpha=0.7)
    ax_histx.set_title(f"Hop Distance Distribution (Archetype {i})")
    ax_histx.set_ylabel("Frequency")
    ax_histx.set_xticks([])

    # Right histogram for row-wise max distances
    ax_histy = fig.add_subplot(gs[1, 1])
    row_max = np.max(C1_np, axis=1)
    ax_histy.hist(
        row_max,
        bins=int(np.sqrt(len(row_max))),
        orientation="horizontal",
        color="darkred",
        alpha=0.7,
    )
    ax_histy.set_title("Max Distances")
    ax_histy.set_xlabel("Count")
    ax_histy.set_yticks(np.linspace(0, row_max.max(), 5))
    ax_histy.set_yticklabels([f"{x:.1f}" for x in np.linspace(0, row_max.max(), 5)])

    # Main heatmap
    ax_heat = fig.add_subplot(gs[1, 0])
    im = ax_heat.imshow(C1_np, cmap="viridis")
    ax_heat.set_title(f"Source Archetype {i} Hop Distance Matrix")
    ax_heat.set_xlabel("Cell Index")
    ax_heat.set_ylabel("Cell Index")

    fig.colorbar(im, ax=ax_heat, label="Hop Distance")
    plt.tight_layout()
    plt.show()


@timeit
def plot_k_distance(archetype, k=5, i=None, j=None):
    """Plot k-distance graph to determine optimal eps value."""
    # Calculate pairwise distances
    dist_matrix = torch.cdist(archetype, archetype)
    dist_matrix.fill_diagonal_(torch.inf)  # Exclude self-distances

    # Sort distances for each point and get k-th neighbor distance
    k_dists, _ = dist_matrix.topk(k + 1, largest=False)
    k_dists = k_dists[:, k].cpu().numpy()  # Get k-th distance

    # Sort for the plot
    k_dists = np.sort(k_dists)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(k_dists)), k_dists, "b-")
    plt.axhline(
        y=np.mean(k_dists), color="r", linestyle="--", label=f"Mean: {np.mean(k_dists):.3f}"
    )

    # Find potential "knee" points
    if len(k_dists) > 1:
        from kneed import KneeLocator

        x = np.arange(len(k_dists))
        kneedle = KneeLocator(x, k_dists, S=1.0, curve="convex", direction="increasing")
        if kneedle.knee is not None:
            plt.axvline(
                x=kneedle.knee,
                color="g",
                linestyle=":",
                label=f"Chosen eps: {k_dists[kneedle.knee]:.3f}",
            )

    plt.xlabel("Points sorted by k-distance")
    plt.ylabel(f"Distance to {k}-th nearest neighbor")

    # Include archetype indices in title if provided
    plt.title(f"K-Distance Plot for Archetype {i} → {j} (k={k})")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


@timeit
def plot_knn_graph(archetype1, archetype2, k=5, max_cells=100, i=None, j=None):
    """Visualize the k-NN graph structure for an archetype."""
    # Limit to a manageable number of cells
    # subsample randomly
    max_index_values = min(archetype1.shape[0], archetype2.shape[0])
    rand_indeces = np.random.permutation(max_index_values)[:max_cells]
    subset_1 = archetype1[rand_indeces]
    subset_2 = archetype2[rand_indeces]

    # Create k-NN graph
    knn_idx_1 = knn_graph(subset_1, k=k)
    knn_idx_2 = knn_graph(subset_2, k=k)

    subset_1 = subset_1.cpu().numpy()
    subset_2 = subset_2.cpu().numpy()
    subset_1_2d, subset_2_2d = get_2D_data(subset_1, subset_2)

    # Convert to networkx graph
    G_1 = nx.DiGraph()
    G_2 = nx.DiGraph()

    # Add nodes
    for idx in range(len(subset_1)):
        G_1.add_node(idx, pos=subset_1_2d[idx, :2])
        G_2.add_node(idx, pos=subset_2_2d[idx, :2])
    # Add edges
    for src in range(len(subset_1)):
        for dst in knn_idx_1[src].cpu().numpy():
            G_1.add_edge(src, dst)
        for dst in knn_idx_2[src].cpu().numpy():
            G_2.add_edge(src, dst)
    # Extract positions
    pos_1 = nx.get_node_attributes(G_1, "pos")
    pos_2 = nx.get_node_attributes(G_2, "pos")

    # Plot
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    nx.draw(
        G_1,
        pos_1,
        with_labels=False,
        node_size=50,
        node_color="blue",
        alpha=0.7,
        edge_color="green",
        arrows=False,
        width=0.5,
    )
    plt.legend(["Source"])
    plt.subplot(1, 2, 2)
    nx.draw(
        G_2,
        pos_2,
        with_labels=False,
        node_size=50,
        node_color="red",
        alpha=0.7,
        edge_color="orange",
        arrows=False,
        width=0.5,
    )

    # Include archetype indices in title if provided
    plt.suptitle(f"k-NN Graph for Archetype {i} → {j} (First {len(subset_1)} cells, k={k})")
    plt.legend(["Target"])
    plt.show()


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


@timeit
def plot_raw_source_vs_target(archetype1_np, archetype2_np, i, j):
    """Plot the source and target archetypes side by side."""
    # Limit to first 1000 cells
    archetype1_np_2d, archetype2_np_2d = get_2D_data(archetype1_np, archetype2_np)

    n_cells_to_plot = min(1000, len(archetype1_np_2d), len(archetype2_np_2d))
    archetype1_np_2d = archetype1_np_2d[:n_cells_to_plot]
    archetype2_np_2d = archetype2_np_2d[:n_cells_to_plot]

    fig = plt.figure(figsize=(10, 5))

    # Source archetype
    ax1 = fig.add_subplot(121)
    ax1.scatter(
        archetype1_np_2d[:, 0],
        archetype1_np_2d[:, 1],
        c="b",
        label="Source",
    )
    ax1.set_title(f"Archetype {i} (Source) - First {n_cells_to_plot} cells")
    # Target archetype
    ax2 = fig.add_subplot(122)
    ax2.scatter(
        archetype2_np_2d[:, 0],
        archetype2_np_2d[:, 1],
        c="r",
        label="Target",
    )
    ax2.set_title(f"Archetype {j} (Target) - First {n_cells_to_plot} cells")
    plt.show()


@timeit
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
    # print the min and max and mean values of the distance matrix
    print(f"Min value of distance matrix: {C1_np.min()}")
    print(f"Max value of distance matrix: {C1_np.max()}")
    print(f"Mean value of distance matrix: {C1_np.mean()}")
    # print the counter of the distance matrix percentage
    unique, counts = np.unique(C1_np, return_counts=True)
    print("Distance matrix value distribution:")
    for val, count in zip(unique, counts / C1_np.size):
        print(f"Value {val:.4f}: {count*100:.2f}%")
    plt.close()


@timeit
def plot_transport_plan_heatmap(transport_plan_np, i, j, gw_dist):
    """Plot the transport plan as a heatmap."""

    # Limit to first 1000 cells
    n_cells_to_plot = min(400, transport_plan_np.shape[0], transport_plan_np.shape[1])
    transport_plan_np = transport_plan_np[:n_cells_to_plot, :n_cells_to_plot]

    plt.figure(figsize=(10, 8))
    sns.heatmap(np.log1p(transport_plan_np))
    # add colorbfar
    # plt.colorbar(label="Transport mass")
    plt.title(
        f"GW Logarithm of Transport Plan values: Archetype {i} ↔ {j}\nDistance: {gw_dist:.4f}, First {n_cells_to_plot} cells"
    )
    plt.xlabel("Target Archetype Cells")
    plt.ylabel("Source Archetype Cells")
    plt.show()
    plt.close()


@timeit
def plot_distribution_comparison(archetype1_np, archetype2_np, transformed_source_scaled_np, i, j):
    """Plot histograms comparing distributions before and after transport."""
    # Compute transported target using the same method as in plot_target_transport_effect

    # Calculate transformed target using same method as plot_target_transport_effect
    # Limit to first 1000 cells
    n_cells_to_plot = min(
        1000, len(archetype1_np), len(archetype2_np), len(transformed_source_scaled_np)
    )
    archetype1_np = archetype1_np[:n_cells_to_plot]
    archetype2_np = archetype2_np[:n_cells_to_plot]
    transformed_source_scaled_np_subset = transformed_source_scaled_np[:n_cells_to_plot]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Define bins consistently across all histograms
    x_min = min(
        archetype1_np[:, 0].min(),
        archetype2_np[:, 0].min(),
        transformed_source_scaled_np_subset[:, 0].min(),
    )
    x_max = max(
        archetype1_np[:, 0].max(),
        archetype2_np[:, 0].max(),
        transformed_source_scaled_np_subset[:, 0].max(),
    )
    y_min = min(
        archetype1_np[:, 1].min(),
        archetype2_np[:, 1].min(),
        transformed_source_scaled_np_subset[:, 1].min(),
    )
    y_max = max(
        archetype1_np[:, 1].max(),
        archetype2_np[:, 1].max(),
        transformed_source_scaled_np_subset[:, 1].max(),
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
        transformed_source_scaled_np_subset[:, 0],
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
        transformed_source_scaled_np_subset[:, 1],
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
        transformed_source_scaled_np_subset[:, 0],
        transformed_source_scaled_np_subset[:, 1],
        c="green",
        alpha=0.5,
        label="Transformed",
    )
    axs[1, 0].set_title("Source vs Transformed")
    axs[1, 0].legend()

    # Bottom-right: 2D density of transformed and target
    axs[1, 1].scatter(
        transformed_source_scaled_np_subset[:, 0],
        transformed_source_scaled_np_subset[:, 1],
        c="green",
        alpha=0.5,
        label="Transformed",
    )
    axs[1, 1].scatter(archetype2_np[:, 0], archetype2_np[:, 1], c="red", alpha=0.5, label="Target")
    axs[1, 1].set_title("Transformed vs Target")
    axs[1, 1].legend()

    plt.suptitle(f"Distribution Comparison: Archetype {i} → {j} - First {n_cells_to_plot} cells")
    plt.tight_layout()
    plt.savefig(f"temp.png")
    plt.show()
    plt.close()


def get_2D_data(archetype1_np, archetype2_np):
    """Convert high-dimensional data to 2D using PCA."""
    # Visualize the transported source points to see overlap
    # pca each one separately
    archetype1_np = PCA(n_components=2).fit_transform(archetype1_np)
    archetype2_np = PCA(n_components=2).fit_transform(archetype2_np)

    return archetype1_np, archetype2_np


def print_transport_plan_validation(transport_plan, p, q, i, j):
    print(f"\nTransport Plan Validation ({i}→{j}):")
    print(f"Mass conservation error: {torch.abs(p.sum() - q.sum()):.2e}")
    print(f"Source marginal L1 error: {torch.abs(transport_plan.sum(1) - p).sum().item():.2e}")
    print(f"Target marginal L1 error: {torch.abs(transport_plan.sum(0) - q).sum().item():.2e}")
    print(f"Percentage of mass transported: {100*transport_plan.sum().item()/p.sum().item():.1f}%")


@timeit
def plot_convergence(log, i, j):
    """Plot the convergence of the Gromov-Wasserstein algorithm."""
    if "errs" in log:
        plt.figure()
        plt.plot(log["errs"])
        plt.yscale("log")
        plt.title(f"GW Convergence: Archetype {i} ↔ {j}")
        plt.xlabel("Iteration")
        plt.ylabel("Marginal Violation (log scale)")
        plt.savefig(f"temp.png")
        plt.show()
        plt.close()


@timeit
def plot_pre_post_transport_plan(
    archetype1_np, archetype2_np, transformed_target_scaled_np, i=None, j=None
):
    subsample_size = 1000
    skip_size = max(int(archetype1_np.shape[0] / subsample_size), 1)
    archetype1_subset = archetype1_np[::skip_size]
    archetype2_subset = archetype2_np[::skip_size]
    transformed_target_scaled_np_subset = transformed_target_scaled_np[::skip_size]
    plt.figure(figsize=(10, 5))
    # add gloabl axis names
    plt.xlabel("cells")
    plt.ylabel("dimensions")

    # Include archetype indices in title if provided
    if i is not None and j is not None:
        plt.suptitle(f"Pre and Post Transport Plan: Archetype {i} → {j}")
    else:
        plt.suptitle("Pre and Post Transport Plan")

    plt.subplot(1, 3, 1)
    # remove colorbar
    sns.heatmap(archetype1_subset, cbar=False)
    plt.title("archetype1")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 3, 2)
    plt.title("archetype2")
    sns.heatmap(archetype2_subset, cbar=False)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 3, 3)
    plt.title("transported archetype2")
    sns.heatmap(transformed_target_scaled_np_subset, cbar=False)
    # remove ticks to all subplots
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"temp.png")
    plt.show()


def visualize_transport_impact(archetype2, transport_plan, title="Transport Plan Impact"):
    """
    Visualize original vs transported target when source ≈ target (with noise).

    Parameters:
    source (np.ndarray): Original data (n, d)
    target (np.ndarray): Noisy version of source (n, d)
    transport_plan (np.ndarray): OT plan matrix (n, n)
    """
    # Compute transported target via barycentric projection
    transported_archetype2 = transport_plan @ archetype2 / transport_plan.sum(axis=1, keepdims=True)

    # PCA for visualization
    pca = PCA(n_components=2)
    combined = np.vstack([archetype2, transported_archetype2])
    pca.fit(combined)

    archetype2_2d = pca.transform(archetype2)
    transported_archetype2_2d = pca.transform(transported_archetype2)

    # Create plot
    plt.figure(figsize=(15, 5))

    # Plot 1: Original vs Transported Target
    plt.subplot(1, 2, 1)
    plt.scatter(
        archetype2_2d[:, 0], archetype2_2d[:, 1], c="blue", alpha=0.4, label="Original Target"
    )
    plt.scatter(
        transported_archetype2_2d[:, 0],
        transported_archetype2_2d[:, 1],
        c="red",
        alpha=0.4,
        label="Transported Target",
    )
    plt.title("Target Space Alignment")
    plt.legend()

    # Plot 2: Transport Plan Matrix
    plt.subplot(1, 2, 2)
    plt.imshow(transport_plan, cmap="viridis", aspect="auto")
    plt.plot([0, len(archetype1) - 1], [0, len(archetype1) - 1], "r--", alpha=0.5)
    plt.colorbar(label="Transport Weight")
    plt.title("Transport Plan Matrix\n(Diagonal Should Dominate)")

    plt.suptitle(title)
    plt.tight_layout()

    # Calculate diagonal dominance
    diag_strength = np.diag(transport_plan).sum() / transport_plan.sum()
    print(f"Diagonal mass: {diag_strength:.2%}")
    print(f"Mean off-diagonal: {transport_plan.sum() - np.diag(transport_plan).sum():.2e}")


def stabilize_matrices(C1, C2):
    # Replace inf with max finite value
    C1 = torch.where(C1 == float("inf"), C1[C1 != float("inf")].max(), C1)
    C2 = torch.where(C2 == float("inf"), C2[C2 != float("inf")].max(), C2)

    # --- Symmetrize the matrices ---
    C1 = ((C1 + C1.T) / 2).to(torch.int32)
    C2 = ((C2 + C2.T) / 2).to(torch.int32)

    print_top_4_most_common_distance_values(C1, C2)

    # for numerical stability
    # Normalize to [0,1]

    C1 = (C1 - C1.min()) / (C1.max() - C1.min() + 1e-10)
    C2 = (C2 - C2.min()) / (C2.max() - C2.min() + 1e-10)
    assert not torch.isnan(C1).any(), "C1 contains NaNs"
    assert not torch.isnan(C2).any(), "C2 contains NaNs"
    assert not (C1 < 0).any(), "C1 has negative values"
    assert not (C2 < 0).any(), "C2 has negative values"

    return C1, C2


@timeit
def compute_distance_matrix(
    archetype1,
    archetype2,
    metric: Literal["euclidean", "nn"],
    kwargs: dict = {"k": 15, "max_hops": 15},
):
    print(f"using max_hops: {kwargs['max_hops']} and k: {kwargs['k']}")
    if metric == "euclidean":
        C1 = torch.cdist(archetype1, archetype1)
        C2 = torch.cdist(archetype2, archetype2)
    elif metric == "nn":
        knn_indices_1 = knn_graph(archetype1, kwargs["k"])
        knn_indices_2 = knn_graph(archetype2, kwargs["k"])
        C1 = compute_hop_distance(knn_indices_1, kwargs["max_hops"])
        C2 = compute_hop_distance(knn_indices_2, kwargs["max_hops"])
    # Normalize the distance matrices
    print_top_4_most_common_distance_values(C1, C2)
    C1, C2 = stabilize_matrices(C1, C2)
    return C1, C2


@timeit
def plot_target_transport_effect(
    archetype2_np, transformed_target_scaled_np, n_components=2, i=None, j=None
):
    """
    Visualize how much the transport plan changes the target.
    Parameters:
        archetype2_np (np.ndarray): Target data, shape (n_samples, n_features)
        transported_target_np (np.ndarray): Transported target data, shape (n_samples, n_features)
        n_components (int): Number of PCA components for visualization (default: 2)
        i, j (int, optional): Source and target archetype indices for title
        title (str, optional): Custom plot title
    """
    # Compute transported target (barycentric projection)
    subset_size = 1000
    skip_size = max(int(archetype2_np.shape[0] / subset_size), 1)
    archetype2_np_subset = archetype2_np[::skip_size]
    transformed_target_scaled_np_subset = transformed_target_scaled_np[::skip_size]
    # First heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(archetype2_np_subset, cmap="viridis")
    plt.title("Original Target Data")
    plt.show()

    # Second heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(transformed_target_scaled_np_subset, cmap="viridis")
    plt.title("Transported Target Data")
    plt.show()

    # PCA visualization
    pca = PCA(n_components=n_components)
    all_data = np.vstack([archetype2_np_subset, transformed_target_scaled_np_subset])
    pca.fit(all_data)
    archetype2_2d = pca.transform(archetype2_np_subset)
    transported_archetype2_2d = pca.transform(transformed_target_scaled_np_subset)

    plt.figure(figsize=(8, 8))
    plt.scatter(
        archetype2_2d[:, 0], archetype2_2d[:, 1], c="blue", alpha=0.5, label="Original Target"
    )
    plt.scatter(
        transported_archetype2_2d[:, 0],
        transported_archetype2_2d[:, 1],
        c="red",
        alpha=0.5,
        label="Transported Target",
        marker="x",
    )
    for idx in range(len(archetype2_2d)):
        plt.plot(
            [archetype2_2d[idx, 0], transported_archetype2_2d[idx, 0]],
            [archetype2_2d[idx, 1], transported_archetype2_2d[idx, 1]],
            "k-",
            alpha=0.1,
        )

    # Set the title based on provided parameters
    plt.title(f"Target vs. Transported Target: Archetype {i} → {j}")
    plt.legend()
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.show()


@timeit
def plot_hop_distribution(C1, C2, source_idx, target_idx):
    """Plot the distribution of hop distances in a k-NN graph using Seaborn.

    Parameters:
    -----------
    C1 : torch.Tensor or numpy.ndarray
        The hop distance matrix
    C2 : torch.Tensor or numpy.ndarray
        The hop distance matrix
    source_idx : int
        Source archetype index (for title)
    target_idx : int, optional
        Target archetype index if comparing two archetypes
    figsize : tuple, optional
        Figure size (width, height)_
    palette : str, optional
        Seaborn color palette to use
    """
    ## Convert to numpy and handle infinities
    # use a random subset of the values
    random_subset_size = min(500, len(C1), len(C2))
    indices_1 = np.random.rand(len(C1)) < random_subset_size / len(C1)
    C_1_random_subset = C1[indices_1, :][:, indices_1]
    C_2_random_subset = C2[indices_1, :][:, indices_1]
    C1_np = (
        C_1_random_subset.cpu().numpy() if hasattr(C_1_random_subset, "cpu") else C_1_random_subset
    )
    C2_np = (
        C_2_random_subset.cpu().numpy() if hasattr(C_2_random_subset, "cpu") else C_2_random_subset
    )

    # Extract finite hop distances
    num_of_inf_C1 = np.sum(np.isinf(C1_np))
    num_of_inf_C2 = np.sum(np.isinf(C2_np))
    c1_hop_values = C1_np[~np.isinf(C1_np) & ~np.isnan(C1_np)].flatten()
    c2_hop_values = C2_np[~np.isinf(C2_np) & ~np.isnan(C2_np)].flatten()
    if c1_hop_values[c1_hop_values > 0].min() < 1:
        c1_hop_values = c1_hop_values * (1 / c1_hop_values[c1_hop_values > 0].min())
    if c2_hop_values[c2_hop_values > 0].min() < 1:
        c2_hop_values = c2_hop_values * (1 / c2_hop_values[c2_hop_values > 0].min())
    if any(c1_hop_values - c1_hop_values.astype(int) > 0):
        warnings.warn("Hop values are not integers (maybe due to random subset)")
    c1_hop_values = c1_hop_values.astype(int)
    c2_hop_values = c2_hop_values.astype(int)
    # Create figure
    plt.figure(figsize=(10, 6))

    # Calculate appropriate bin width using Freedman-Diaconis rule
    q25, q75 = np.percentile(c1_hop_values, [25, 75])
    bin_width = 2 * (q75 - q25) * len(c1_hop_values) ** (-1 / 3)
    n_bins = max(5, min(50, int((c1_hop_values.max() - c1_hop_values.min()) / max(0.1, bin_width))))

    # a random subset of the values
    c1_hop_values = c1_hop_values[np.random.rand(len(c1_hop_values)) < 0.1]
    c2_hop_values = c2_hop_values[np.random.rand(len(c2_hop_values)) < 0.1]
    ax = sns.histplot(
        c1_hop_values,
        bins=n_bins,
        kde=False,  # Add density curve
        stat="density",  # Show density instead of count
        color=sns.color_palette("muted")[0],
        line_kws={"linewidth": 2},
        alpha=0.6,
        discrete=True,  # Optimize for integer data
    )
    ax2 = ax.twinx()
    ax2.hist(c2_hop_values, bins=n_bins, color=sns.color_palette("muted")[1], alpha=0.6)
    median_val = np.median(c1_hop_values)
    median_val2 = np.median(c2_hop_values)
    plt.axvline(
        x=median_val,
        color="green",
        linestyle="-",
        label=f"Median Source: {median_val:.2f}",
        linewidth=1.5,
    )
    ax2.axvline(
        x=median_val2,
        color="red",
        linestyle="-",
        label=f"Median Target: {median_val2:.2f}",
        linewidth=1.5,
    )
    # Add labels and title
    plt.title(
        f"Hop Distance Distribution for Archetypes {target_idx}↔{source_idx} \
              \nNumber of infinite values in C1: {num_of_inf_C1}, Number of infinite values in C2: {num_of_inf_C2}",
        fontsize=14,
    )
    plt.xlabel("Number of Hops", fontsize=12)
    plt.ylabel("Density", fontsize=12)

    # Add detail annotations
    unique_vals_1 = np.unique(c1_hop_values)
    unique_vals_2 = np.unique(c2_hop_values)
    plt.annotate(
        f"Unique hop counts C1: {len(unique_vals_1)}\nUnique hop counts C2: {len(unique_vals_2)}\n"
        f"Min: {c1_hop_values.min():.0f}, Max: {c1_hop_values.max():.0f}\n"
        f"Std Dev: {np.std(c1_hop_values):.2f}",
        xy=(0.92, 0.75),
        xycoords="axes fraction",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.show()


def get_epsilon_and_tol(op_subsample_size, n_dim):
    if op_subsample_size < 1000:
        epsilon = 0.001  # Small datasets
    elif op_subsample_size < 10000:
        epsilon = 0.005  # Medium datasets
    else:
        epsilon = 0.01  # Large datasets

    # Further adjustment for dimensionality
    epsilon *= 1 + 0.01 * n_dim  # Slight increase for high-dimensional data
    # Based on sample size
    if op_subsample_size < 1000:
        tol = 1e-13  # Very strict for small datasets
    elif op_subsample_size < 10000:
        tol = 1e-8  # Moderately relaxed for medium datasets
    else:
        tol = 1e-6  # Further relaxed for large datasets
    return epsilon, tol


def print_top_4_most_common_distance_values(C1, C2):
    C1_unique, C1_counts = torch.unique(C1, return_counts=True)
    C2_unique, C2_counts = torch.unique(C2, return_counts=True)

    # Sort by count and get top 4
    C1_top = sorted(zip(C1_counts, C1_unique), reverse=True)[:4]
    C2_top = sorted(zip(C2_counts, C2_unique), reverse=True)[:4]

    print("\nTop 4 most common distance values in C1 (pre normalization):")
    print(
        tabulate(
            [[value, count] for count, value in C1_top], headers=["Value", "Count"], tablefmt="grid"
        )
    )

    print("\nTop 4 most common distance values in C2 (pre normalization):")
    print(
        tabulate(
            [[value, count] for count, value in C2_top], headers=["Value", "Count"], tablefmt="grid"
        )
    )


def run_gromov_wasserstein(C1, C2, p, q, epsilon, max_iter, tol, loss_type):
    gw_dist, log = ot.gromov.entropic_gromov_wasserstein2(
        C1,
        C2,
        p,
        q,
        loss_type,
        epsilon=epsilon,
        max_iter=max_iter,
        tol=tol,
        verbose=True,
        log=True,
    )
    return gw_dist, log


# read those adata_1_rna.write(f"CODEX_RNA_seq/data/processed_data/adata_rna_archetype_generated_ot_test.h5ad")
# adata_2_prot.write(f"CODEX_RNA_seq/data/processed_data/adata_prot_archetype_generated_ot_test.h5ad")
adata_1_rna = sc.read_h5ad(
    "CODEX_RNA_seq/data/processed_data/adata_rna_archetype_generated_ot_test.h5ad"
)
adata_2_prot = sc.read_h5ad(
    "CODEX_RNA_seq/data/processed_data/adata_prot_archetype_generated_ot_test.h5ad"
)
# subsample the adata_1_rna and adata_2_prot using scanpy subsample function
sc.pp.subsample(adata_1_rna, n_obs=min(len(adata_1_rna), 100000))
sc.pp.subsample(adata_2_prot, n_obs=min(len(adata_2_prot), 100000))
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

op_subsample_size = 3000
n_dim = 4
max_iter = 70  # should be higher
metric = "nn"  #  change to neighbor graph distance
loss_type = "kl_loss"
# Based on sample size

# loss_type = "square_loss"

num_archetypes = weights_rna.shape[1]
# each items in the list is an array of cells that are assigned to the archetype
rna_archetypes = []
print(num_archetypes)
# Find cells for each RNA archetype
arche_index_for_each_cell_rna = np.argmax(weights_rna, axis=1)
for i in range(num_archetypes):
    locs = arche_index_for_each_cell_rna == i
    archetype_cells = adata_1_rna.obsm["X_pca"][locs, :n_dim]
    # Ensure exactly op_subsample_size cells
    if len(archetype_cells) > op_subsample_size:
        archetype_cells = archetype_cells[:op_subsample_size]
    # else:
    #     raise ValueError(f"Archetype {i} has {len(archetype_cells)} cells, expected {op_subsample_size} cells")
    rna_archetypes.append(archetype_cells)

# Create lists to store cells for each archetype (Protein)
# each items in the list is an array of cells that are assigned to the archetype
prot_archetypes = []
# Find cells for each Protein archetype
arche_index_for_each_cell_prot = np.argmax(weights_prot, axis=1)
for i in range(num_archetypes):
    locs = arche_index_for_each_cell_prot == i
    archetype_cells = adata_2_prot.obsm["X_pca"][locs, :n_dim]
    # Ensure exactly     cells
    if len(archetype_cells) > op_subsample_size:
        archetype_cells = archetype_cells[:op_subsample_size]
    # else:
    #     raise ValueError(f"Archetype {i} has {len(archetype_cells)} cells, expected {op_subsample_size} cells")
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
# sort both useing pandas hierarchically by the first column
earchetyp_1_rna_cell_dummy = [
    pd.DataFrame(archetype).sort_values(by=0).values for archetype in earchetyp_1_rna_cell_dummy
]
eaxmple_prot_cell_dummy = [
    pd.DataFrame(archetype).sort_values(by=0).values for archetype in eaxmple_prot_cell_dummy
]

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
# reverse earchetyp_1_rna_cell_dummy dimension and remove the last dimension
earchetyp_1_rna_cell_dummy[0] = earchetyp_1_rna_cell_dummy[0][:, :-1][:, ::-1].copy()
# make not a function
archetypes_space1 = earchetyp_1_rna_cell_dummy
archetypes_space2 = eaxmple_prot_cell_dummy
# %%

"""
Match archetypes across different feature spaces using Gromov-Wasserstein Optimal Transport.
"""
print(f"sizes of the archetypes {(archetypes_space1[0].shape)} {(archetypes_space2)[0].shape}")
# Number of archetypes in each space
# Create a cost matrix between archetypes across spaces
# Compute distances between each pair of archetypes across spaces
# Before GW computation
archetypes_space1 = [
    torch.tensor(archetype, device=device, dtype=torch.float32) for archetype in archetypes_space1
]
archetypes_space2 = [
    torch.tensor(archetype, device=device, dtype=torch.float32) for archetype in archetypes_space2
]
archetypes_space1 = archetypes_space1[:2]
archetypes_space2 = archetypes_space2[:2]
num_archetypes1 = len(archetypes_space1)
num_archetypes2 = len(archetypes_space2)

cost_matrix = np.zeros((num_archetypes1, num_archetypes2))
# %%
for target_idx, archetype1 in tqdm(
    enumerate(archetypes_space1), total=num_archetypes1, desc="Computing GW distances"
):
    for source_idx, archetype2 in enumerate(archetypes_space2):
        # Visualize the archetypes
        print(
            f"archetype1.shape: {list(archetype1.shape)}, \narchetype2.shape: {list(archetype2.shape)}"
        )
        num_cells = archetype1.shape[0] + archetype2.shape[0]
        num_dim = int((archetype1.shape[1] + archetype2.shape[1]) / 2)
        epsilon, tol = get_epsilon_and_tol(num_cells, num_dim)
        print(f"epsilon: {epsilon}, tol: {tol}")
        # Skip if either archetype has no cells
        if len(archetype1) == 0 or len(archetype2) == 0:
            cost_matrix[target_idx, source_idx] = np.inf
            continue
        archetype1_np = archetype1.cpu().numpy()
        archetype2_np = archetype2.cpu().numpy()

        # Compute distance matrices within each archetype
        plot_raw_source_vs_target(archetype1_np, archetype2_np, target_idx, source_idx)
        nn_metric_kwargs = {"k": 5, "max_hops": int(3 * np.log(num_cells))}
        C1, C2 = compute_distance_matrix(
            archetype1, archetype2, metric="nn", kwargs=nn_metric_kwargs
        )
        plot_distance_heatmap(C1, target_idx, source_idx)
        plot_knn_graph(archetype1, archetype2, k=nn_metric_kwargs["k"], i=target_idx, j=source_idx)
        plot_k_distance(archetype1, k=nn_metric_kwargs["k"], i=target_idx, j=source_idx)
        plot_hop_distribution(C1, C2, source_idx, target_idx)
        # Define weights for samples (uniform weights)
        p = torch.ones(len(archetype1), device=device, dtype=torch.float32) / len(archetype1)
        q = torch.ones(len(archetype2), device=device, dtype=torch.float32) / len(archetype2)

        # Compute Gromov-Wasserstein distance
        # try UCOOT (Co-Optimal Transport) too later
        if not (torch.allclose(C1, C1.T) and torch.allclose(C2, C2.T)):
            raise ValueError("C1 or C2 is not symmetric (need to change the arg for ot)")  # why?

        gw_dist, log = run_gromov_wasserstein(C1, C2, p, q, epsilon, max_iter, tol, loss_type)
        transport_plan = log["T"]
        transport_plan_np = transport_plan.cpu().numpy()
        transformed_target_scaled = (
            transport_plan @ archetype2 / torch.sum(transport_plan, axis=1, keepdims=True)
        )
        transformed_target_scaled_np = transformed_target_scaled.cpu().numpy()
        plot_target_transport_effect(
            archetype2_np,
            transformed_target_scaled_np,
            n_components=2,
            i=target_idx,
            j=source_idx,
        )

        print_transport_plan_validation(transport_plan, p, q, target_idx, source_idx)

        plot_transport_plan_heatmap(transport_plan_np, target_idx, source_idx, gw_dist)

        plot_transport_plan_diagonal_vs_all(transport_plan_np, target_idx, source_idx)

        plot_pre_post_transport_plan(
            archetype1_np,
            archetype2_np,
            transformed_target_scaled_np,
            target_idx,
            source_idx,
        )

        plot_distribution_comparison(
            archetype1_np,
            archetype2_np,
            transformed_target_scaled_np,
            target_idx,
            source_idx,
        )

        plot_convergence(log, target_idx, source_idx)
        # Use the Gromov-Wasserstein distance as the cost
        cost_matrix[target_idx, source_idx] = gw_dist

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

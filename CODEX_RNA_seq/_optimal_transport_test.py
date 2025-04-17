# %%
import copy
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import ot
import scanpy as sc
import seaborn as sns
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def soften_matching(M, temperature=0.1):
    M_exp = np.exp(-M / temperature)
    return M_exp / M_exp.sum(axis=1, keepdims=True)


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
num_archetypes = weights_rna.shape[1]
# each items in the list is an array of cells that are assigned to the archetype
rna_archetypes = []
print(num_archetypes)
# Find cells for each RNA archetype
arche_index_for_each_cell_rna = np.argmax(weights_rna, axis=1)
for i in range(num_archetypes):
    locs = arche_index_for_each_cell_rna == i
    archetype_cells = adata_1_rna.obsm["X_pca"][locs, :2]
    # Ensure exactly 50 cells
    if len(archetype_cells) > 50:
        archetype_cells = archetype_cells[:50]
    elif len(archetype_cells) < 50:
        # If we don't have enough cells, pad with random samples from existing cells
        n_pad = 50 - len(archetype_cells)
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
    # Ensure exactly 50 cells
    if len(archetype_cells) > 50:
        archetype_cells = archetype_cells[:50]
    elif len(archetype_cells) < 50:
        # If we don't have enough cells, pad with random samples from existing cells
        n_pad = 50 - len(archetype_cells)
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
earchetyp_1_rna_cell_dummy[0] = eaxmple_prot_cell_dummy[0][:, ::-1] + 0.00000001 * np.random.normal(
    0, 0.1, eaxmple_prot_cell_dummy[0].shape
)  # PCA().fit_transform(earchetyp_1_rna_cell_dummy[0])
# Add some noise to the RNA archetypes
earchetyp_1_rna_cell_dummy[1] = eaxmple_prot_cell_dummy[1][:, ::-1] + 0.00000001 * np.random.normal(
    0, 0.1, eaxmple_prot_cell_dummy[1].shape
)
earchetyp_1_rna_cell_dummy[2] = eaxmple_prot_cell_dummy[2][:, ::-1] + 0.00000001 * np.random.normal(
    0, 0.1, eaxmple_prot_cell_dummy[2].shape
)
earchetyp_1_rna_cell_dummy[3] = eaxmple_prot_cell_dummy[3][:, ::-1] + 0.00000001 * np.random.normal(
    0, 0.1, eaxmple_prot_cell_dummy[3].shape
)
# make not a function
archetypes_space1 = earchetyp_1_rna_cell_dummy
archetypes_space2 = eaxmple_prot_cell_dummy
metric = "euclidean"
loss_type = "square_loss"
epsilon = 0.01
max_iter = 1000
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
        if archetype1.shape[1] == 2:  # Only for 2D data
            fig = plt.figure(figsize=(10, 5))

            # Source archetype
            ax1 = fig.add_subplot(121)
            ax1.scatter(
                archetype1[:, 0].cpu().numpy(),
                archetype1[:, 1].cpu().numpy(),
                c="b",
                label="Source",
            )
            ax1.set_title(f"Archetype {i} (Source)")
            # Target archetype
            ax2 = fig.add_subplot(122)
            ax2.scatter(
                archetype2[:, 0].cpu().numpy(),
                archetype2[:, 1].cpu().numpy(),
                c="r",
                label="Target",
            )
            ax2.set_title(f"Archetype {j} (Target)")
            plt.show()
            plt.close()
        # Skip if either archetype has no cells
        if len(archetype1) == 0 or len(archetype2) == 0:
            cost_matrix[i, j] = np.inf
            continue
        try:
            # Compute distance matrices within each archetype
            C1 = torch.cdist(archetype1, archetype1)
            C2 = torch.cdist(archetype2, archetype2)
            # Normalize the distance matrices
            if C1.max() > 0:
                C1 = C1 / C1.max()
            if C2.max() > 0:
                C2 = C2 / C2.max()
            # Define weights for samples (uniform weights)
            p = torch.ones(len(archetype1), device=device, dtype=torch.float32) / len(archetype1)
            q = torch.ones(len(archetype2), device=device, dtype=torch.float32) / len(archetype2)

            # Compute Gromov-Wasserstein distance
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(C1.cpu().numpy(), cmap="viridis")
            plt.title(f"Source Archetype {i} Distance Matrix")
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.imshow(C2.cpu().numpy(), cmap="viridis")
            plt.title(f"Target Archetype {j} Distance Matrix")
            plt.colorbar()

            plt.tight_layout()
            plt.show()
            plt.close()

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

            print(f"\nTransport Plan Validation ({i}→{j}):")
            print(f"Mass conservation error: {torch.abs(p.sum() - q.sum()):.2e}")
            print(
                f"Source marginal L1 error: {torch.abs(transport_plan.sum(1) - p).sum().item():.2e}"
            )
            print(
                f"Target marginal L1 error: {torch.abs(transport_plan.sum(0) - q).sum().item():.2e}"
            )
            print(
                f"Percentage of mass transported: {100*transport_plan.sum().item()/p.sum().item():.1f}%"
            )

            # Convert to numpy for visualization
            transport_plan_np = (
                transport_plan.cpu().numpy()
                if isinstance(transport_plan, torch.Tensor)
                else transport_plan
            )

            plt.figure(figsize=(10, 8))
            plt.imshow(transport_plan_np, cmap="viridis", aspect="auto")
            plt.colorbar(label="Transport mass")
            plt.title(f"GW Transport Plan: Archetype {i} ↔ {j}\nDistance: {gw_dist:.4f}")
            plt.xlabel("Target Archetype Cells")
            plt.ylabel("Source Archetype Cells")
            plt.show()
            plt.close()

            # Visualize the transported source points to see overlap
            # This applies the transport plan to transform source points
            if archetype1.shape[1] == 2 and archetype2.shape[1] == 2:
                # Convert to numpy for visualization
                archetype1_np = archetype1.cpu().numpy()
                archetype2_np = archetype2.cpu().numpy()
                transport_plan_np = transport_plan.cpu().numpy()

                # Use the transport plan to transform source points
                # Each source point is mapped as a weighted combination of target points
                transformed_source = transport_plan_np @ archetype2_np

                # Rescale transformed source to match target scale for better visualization
                # Get min/max of target data
                target_min = archetype2_np.min(axis=0)
                target_max = archetype2_np.max(axis=0)
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

                # Plot original source, transformed source, and target
                plt.figure(figsize=(15, 5))

                # Original source
                plt.subplot(1, 3, 1)
                plt.scatter(
                    archetype1_np[:, 0], archetype1_np[:, 1], c="blue", label="Source", alpha=0.7
                )
                plt.title(f"Original Source (Archetype {i})")
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
                plt.title(f"Transformed Source (Scaled)")
                plt.legend()

                # Target and scaled transformed source overlay
                plt.subplot(1, 3, 3)
                plt.scatter(
                    archetype2_np[:, 0], archetype2_np[:, 1], c="red", label="Target", alpha=0.5
                )
                plt.scatter(
                    transformed_source_scaled[:, 0],
                    transformed_source_scaled[:, 1],
                    c="green",
                    label="Transformed Source",
                    alpha=0.5,
                )

                # Add lines connecting the most relevant points
                # Find top connections based on transport plan weights
                num_connections = 50  # Number of connections to show
                # Flatten the transport plan to find top weights
                flat_indices = np.argsort(transport_plan_np.flatten())[-num_connections:]
                source_indices, target_indices = np.unravel_index(
                    flat_indices, transport_plan_np.shape
                )

                # Define a list of colors to cycle through
                line_colors = ["black", "blue", "red", "green", "brown"]

                # Draw lines between corresponding points with different colors
                for idx, (src_idx, tgt_idx) in enumerate(zip(source_indices, target_indices)):
                    color_idx = idx % len(line_colors)
                    plt.plot(
                        [transformed_source_scaled[src_idx, 0], archetype2_np[tgt_idx, 0]],
                        [transformed_source_scaled[src_idx, 1], archetype2_np[tgt_idx, 1]],
                        color=line_colors[color_idx],
                        alpha=0.4,
                        linewidth=0.8,
                    )

                plt.title(f"Overlay: Target (Arch {j}) & Transformed with Connections")
                plt.legend()

                # Also add a separate plot focusing just on the connections
                plt.figure(figsize=(8, 8))
                plt.scatter(
                    archetype2_np[:, 0], archetype2_np[:, 1], c="red", label="Target", alpha=0.5
                )
                plt.scatter(
                    transformed_source_scaled[:, 0],
                    transformed_source_scaled[:, 1],
                    c="green",
                    label="Transformed Source",
                    alpha=0.5,
                )

                # Draw lines with higher visibility on this dedicated plot, cycling through colors
                for idx, (src_idx, tgt_idx) in enumerate(zip(source_indices, target_indices)):
                    weight = transport_plan_np[src_idx, tgt_idx]
                    # Normalize line width by weight
                    line_width = 0.5 + 4.5 * weight / np.max(transport_plan_np)
                    # Select color from cycle
                    color_idx = idx % len(line_colors)
                    plt.plot(
                        [transformed_source_scaled[src_idx, 0], archetype2_np[tgt_idx, 0]],
                        [transformed_source_scaled[src_idx, 1], archetype2_np[tgt_idx, 1]],
                        color=line_colors[color_idx],
                        alpha=0.6,
                        linewidth=line_width,
                    )
                    # Add text labels showing the connection weight
                    mid_x = (transformed_source_scaled[src_idx, 0] + archetype2_np[tgt_idx, 0]) / 2
                    mid_y = (transformed_source_scaled[src_idx, 1] + archetype2_np[tgt_idx, 1]) / 2
                    # plt.text(mid_x, mid_y, f'{weight:.3f}', fontsize=8, ha='center', va='center',
                    #          bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

                plt.title(f"Top {num_connections} Connections: Archetype {i} → {j}")
                plt.legend()
                plt.tight_layout()
                plt.show()
                plt.close()

                # Add histograms comparing distributions before and after transport
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

                x_bins = np.linspace(x_min, x_max, 20)
                y_bins = np.linspace(y_min, y_max, 20)

                # Top-left: Source X distribution
                axs[0, 0].hist(
                    archetype1_np[:, 0], bins=x_bins, alpha=0.5, color="blue", label="Source"
                )
                axs[0, 0].hist(
                    transformed_source_scaled[:, 0],
                    bins=x_bins,
                    alpha=0.5,
                    color="green",
                    label="Transformed",
                )
                axs[0, 0].hist(
                    archetype2_np[:, 0], bins=x_bins, alpha=0.5, color="red", label="Target"
                )
                axs[0, 0].set_title("X Dimension Distribution")
                axs[0, 0].legend()

                # Top-right: Source Y distribution
                axs[0, 1].hist(
                    archetype1_np[:, 1], bins=y_bins, alpha=0.5, color="blue", label="Source"
                )
                axs[0, 1].hist(
                    transformed_source_scaled[:, 1],
                    bins=y_bins,
                    alpha=0.5,
                    color="green",
                    label="Transformed",
                )
                axs[0, 1].hist(
                    archetype2_np[:, 1], bins=y_bins, alpha=0.5, color="red", label="Target"
                )
                axs[0, 1].set_title("Y Dimension Distribution")
                axs[0, 1].legend()

                # Bottom-left: 2D density of source and transformed
                axs[1, 0].scatter(
                    archetype1_np[:, 0], archetype1_np[:, 1], c="blue", alpha=0.5, label="Source"
                )
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
                axs[1, 1].scatter(
                    archetype2_np[:, 0], archetype2_np[:, 1], c="red", alpha=0.5, label="Target"
                )
                axs[1, 1].set_title("Transformed vs Target")
                axs[1, 1].legend()

                plt.suptitle(f"Distribution Comparison: Archetype {i} → {j}")
                plt.tight_layout()
                plt.show()
                plt.close()

            # Plot convergence if error history is available
            if "errs" in log:
                plt.figure()
                plt.plot(log["errs"])
                plt.yscale("log")
                plt.title(f"GW Convergence: Archetype {i} ↔ {j}")
                plt.xlabel("Iteration")
                plt.ylabel("Marginal Violation (log scale)")
                plt.show()
                plt.close()
            # Use the Gromov-Wasserstein distance as the cost
            cost_matrix[i, j] = gw_dist

        except Exception as e:
            print(f"Error computing GW distance between archetype {i} and {j}: {e}")
            cost_matrix[i, j] = np.inf
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

# %%
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from bar_nick_utils import match_datasets

# %%
# Create dummy data
n_rna = 100  # smaller dataset
n_prot = 150  # larger dataset
n_archetypes = 5

# Create random archetype vectors
rna_archetypes = np.random.randn(n_rna, n_archetypes)
prot_archetypes = np.random.randn(n_prot, n_archetypes)

# Create cell types
cell_types = ["A", "B", "C", "D"]
major_cell_types = ["X", "Y"]

# Create random cell type assignments
rna_cell_types = np.random.choice(cell_types, n_rna)
rna_major_types = np.random.choice(major_cell_types, n_rna)
prot_cell_types = np.random.choice(cell_types, n_prot)
prot_major_types = np.random.choice(major_cell_types, n_prot)

# Create AnnData objects
adata_rna = AnnData(
    X=np.random.randn(n_rna, 10),  # dummy expression data
    obs=pd.DataFrame({"cell_types": rna_cell_types, "major_cell_types": rna_major_types}),
    obsm={"archetype_vec": rna_archetypes},
)

adata_prot = AnnData(
    X=np.random.randn(n_prot, 15),  # dummy expression data
    obs=pd.DataFrame({"cell_types": prot_cell_types, "major_cell_types": prot_major_types}),
    obsm={"archetype_vec": prot_archetypes},
)

# %%
# Test the match_datasets function
print("\nTesting match_datasets function...")
matched_rna, matched_prot = match_datasets(
    adata_rna, adata_prot, threshold=0.5, plot_flag=True  # arbitrary threshold for testing
)

# %%
# Print results
print("\nResults:")
print(f"Original RNA data shape: {adata_rna.shape}")
print(f"Original protein data shape: {adata_prot.shape}")
print(f"Matched RNA data shape: {matched_rna.shape}")
print(f"Matched protein data shape: {matched_prot.shape}")

# Verify sizes
print("\nVerifying output sizes...")
assert len(matched_rna) == n_rna, f"RNA data should have {n_rna} cells, got {len(matched_rna)}"
assert (
    len(matched_prot) == n_prot
), f"Protein data should have {n_prot} cells, got {len(matched_prot)}"
print("✓ Output sizes are correct")

# Verify no duplicates in matches
print("\nVerifying no duplicates in matches...")
rna_indices = matched_rna.obs.index
prot_indices = matched_prot.obs.index
print(f"Unique RNA indices: {len(set(rna_indices))} == {len(rna_indices)}")
print(f"Unique protein indices: {len(set(prot_indices))} == {len(prot_indices)}")

# Verify cell type ordering in RNA data
print("\nVerifying RNA data cell type ordering...")
print("RNA data cell types order:")
print(matched_rna.obs[["major_cell_types", "cell_types"]].head())
print("\nChecking if RNA data is sorted by cell types...")
rna_sorted = matched_rna.obs.sort_values(by=["major_cell_types", "cell_types"])
print("RNA data is sorted:", matched_rna.obs.equals(rna_sorted))

# Verify protein data ordering
print("\nVerifying protein data ordering...")
print("First 100 cells (matched portion):")
print(matched_prot.obs[["major_cell_types", "cell_types"]].head(n_rna))
print("\nLast 50 cells (unmatched portion):")
print(matched_prot.obs[["major_cell_types", "cell_types"]].tail(n_prot - n_rna))

# Verify protein data sorting
print("\nChecking protein data sorting...")
# First n_rna cells should be matched with RNA data
matched_prot_portion = matched_prot.obs.iloc[:n_rna]
print(
    "First portion (matched) is sorted:",
    matched_prot_portion.equals(
        matched_prot_portion.sort_values(by=["major_cell_types", "cell_types"])
    ),
)

# Last n_prot - n_rna cells should be sorted by cell type
unmatched_prot_portion = matched_prot.obs.iloc[n_rna:]
print(
    "Last portion (unmatched) is sorted:",
    unmatched_prot_portion.equals(
        unmatched_prot_portion.sort_values(by=["major_cell_types", "cell_types"])
    ),
)

# %%
# Additional verification
print("\nAdditional verification:")
print("1. Checking if matched cells are properly aligned...")
print(f"Number of matched cells: {n_rna}")

print("\n2. Checking if matched cells are at the beginning of protein data...")
print("First few matched protein cells:")
print(matched_prot.obs[["major_cell_types", "cell_types"]].head(n_rna))

print("\n3. Checking if unmatched cells are at the end...")
print("Last few protein cell types:")
print(matched_prot.obs[["major_cell_types", "cell_types"]].tail(n_prot - n_rna))

# %%
# Verify cell type ordering in matched portion
print("\n4. Verifying cell type ordering in matched portion...")
matched_rna_order = matched_rna.obs[["major_cell_types", "cell_types"]]
matched_prot_order = matched_prot.obs[["major_cell_types", "cell_types"]].head(n_rna)
print("RNA matched cells order:")
print(matched_rna_order)
print("\nProtein matched cells order:")
print(matched_prot_order)

# Check if the orders match
print("\n5. Checking if orders match...")
print("Orders match:", matched_rna_order.equals(matched_prot_order))

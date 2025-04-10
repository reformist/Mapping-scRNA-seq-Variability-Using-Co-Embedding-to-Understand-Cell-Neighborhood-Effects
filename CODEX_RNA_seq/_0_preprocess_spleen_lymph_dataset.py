# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: scvi
#     language: python
#     name: python3
# ---

# %%
# preprocess the cite-seq from blood

import json
import os
import sys
import seaborn as sns


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"sys.path added: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
print(f"sys.path added: {os.path.dirname(os.path.abspath(__file__))}")
print(f"Current working directory: {os.getcwd()}")

# Load config if exists
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    num_rna_cells = config["subsample"]["num_rna_cells"]
    num_protein_cells = config["subsample"]["num_protein_cells"]
    plot_flag = config["plot_flag"]
else:
    num_rna_cells = num_protein_cells = 2000
    plot_flag = True

# %%
# Imports
# %%
import importlib
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import plotting_functions as pf
import scanpy as sc
import torch
from scipy.io import mmread

import bar_nick_utils
importlib.reload(bar_nick_utils)
import covet_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"sys.path added: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
print(f"sys.path added: {os.path.dirname(os.path.abspath(__file__))}")
print(f"Current working directory: {os.getcwd()}")
importlib.reload(bar_nick_utils)
importlib.reload(covet_utils)
importlib.reload(pf)

from bar_nick_utils import clean_uns_for_h5ad


def setup_environment():
    """Setup environment variables and random seeds"""
    np.random.seed(8)
    warnings.filterwarnings("ignore")
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.max_rows", 10)
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def check_data_exists(data_dir):
    """Check if required data files exist"""
    required_files = [
        "citeseq_pbmc/pro.csv",
        "citeseq_pbmc/citeseq_rna_names.csv",
        "citeseq_pbmc/meta.csv",
        "citeseq_pbmc/orig_x.csv",
        "citeseq_pbmc/orig_y.csv",
        "citeseq_pbmc/rna_protein_correspondence.csv",
        "citeseq_pbmc/rna.txt",
        # "tonsil/tonsil_rna_counts.txt",
        # "tonsil/tonsil_rna_names.csv",
        # "tonsil/tonsil_rna_meta.csv",
    ]
    return all((data_dir / file).exists() for file in required_files)


def download_cite_seq_data(data_dir):
    """Download and extract spleen/lymph data"""
    # if check_data_exists(data_dir):
    #     print("Data files already exist, skipping download.")
    #     return

    print("Downloading CITE-seq data...")
    import requests, zipfile, io
    r = requests.get("http://stat.wharton.upenn.edu/~zongming/maxfuse/data.zip")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(str(data_dir.parent))


# fmt: off
def load_protein_data(data_dir):
    """Load and process protein data"""
    import scvi
    print("Loading protein data...")
    adata = scvi.data.spleen_lymph_cite_seq(save_path='data')

    adata_1 = adata[adata.obs['batch'] == f'SLN111-D1']
    # adata_2 = adata[adata.obs['batch'] == f'SLN111-D2']
    fraction = 1
    sc.pp.subsample(adata_1, fraction=fraction)
    # sc.pp.subsample(adata_2, fraction=fraction)
    adata_1 = adata_1[adata_1.obs['cell_types'].argsort(), :]  # sort by cell types for easier visualization
    # adata_2 = adata_2[adata_2.obs['cell_types'].argsort(), :] 
    cell_type_mapping = {
    "Activated CD4 T": "CD4 T",
    "B1 B": "B cells",
    "CD122+ CD8 T": "CD8 T",
    "CD4 T": "CD4 T",
    "CD8 T": "CD8 T",
    "Erythrocytes": "RBC",
    "GD T": "T cells",
    "ICOS-high Tregs": "CD4 T",
    "Ifit3-high B": "B cells",
    "Ifit3-high CD4 T": "CD4 T",
    "Ifit3-high CD8 T": "CD8 T",
    "Ly6-high mono": "Monocytes",
    "Ly6-low mono": "Monocytes",
    "MZ B": "B cells",
    "MZ/Marco-high macrophages": "Macrophages",
    "Mature B": "B cells",
    "Migratory DCs": "cDCs",
    "NK": "NK",
    "NKT": "T cells",
    "Neutrophils": "Neutrophils",
    "Plasma B": "B cells",
    "Red-pulp macrophages": "Macrophages",
    "Transitional B": "B cells",
    "Tregs": "Treg",
    "cDC1s": "cDCs",
    "cDC2s": "cDCs",
    "pDCs": "pDCs",
}
    # Map the specific cell types to major cell types and add as a new column in obs
    adata.obs['major_cell_types'] = pd.Categorical(adata.obs['cell_types'].map(cell_type_mapping))

    adata_1.obs['major_cell_types'] = pd.Categorical(adata_1.obs['cell_types'].map(cell_type_mapping))
    # adata_2.obs['major_cell_types'] = pd.Categorical(adata_2.obs['cell_types'].map(cell_type_mapping))

    assert set(cell_type_mapping.keys()) == set(adata.obs['cell_types'])

    # generate major cell types
    major_to_minor_dict ={}
    # from major to minor dict
    for k,v in cell_type_mapping.items():
        if v not in major_to_minor_dict:
            major_to_minor_dict[v] = [k]
        else:
            major_to_minor_dict[v].append(k)

    
    # protein_adata = adata_1

    protein_matrix = adata_1.obsm['protein_expression']

    # Optional: include obs and var names
    obs = adata_1.obs.copy()
    # var = pd.DataFrame(index=[f"protein_{i}" for i in range(protein_matrix.shape[1])])

    # Create a new AnnData object for protein data
    protein_adata= ad.AnnData(X=protein_matrix, obs=obs)

    return protein_adata


def load_rna_data(data_dir):
    """Load and process RNA data"""
    print("Loading RNA data...")
    import scvi
    print("Loading protein data...")
    adata = scvi.data.spleen_lymph_cite_seq(save_path='data')

    adata_1 = adata[adata.obs['batch'] == f'SLN111-D1']
    # adata_2 = adata[adata.obs['batch'] == f'SLN111-D2']
    fraction = 1
    sc.pp.subsample(adata_1, fraction=fraction)
    # sc.pp.subsample(adata_2, fraction=fraction)
    adata_1 = adata_1[adata_1.obs['cell_types'].argsort(), :]  # sort by cell types for easier visualization
    # adata_2 = adata_2[adata_2.obs['cell_types'].argsort(), :] 
    cell_type_mapping = {
    "Activated CD4 T": "CD4 T",
    "B1 B": "B cells",
    "CD122+ CD8 T": "CD8 T",
    "CD4 T": "CD4 T",
    "CD8 T": "CD8 T",
    "Erythrocytes": "RBC",
    "GD T": "T cells",
    "ICOS-high Tregs": "CD4 T",
    "Ifit3-high B": "B cells",
    "Ifit3-high CD4 T": "CD4 T",
    "Ifit3-high CD8 T": "CD8 T",
    "Ly6-high mono": "Monocytes",
    "Ly6-low mono": "Monocytes",
    "MZ B": "B cells",
    "MZ/Marco-high macrophages": "Macrophages",
    "Mature B": "B cells",
    "Migratory DCs": "cDCs",
    "NK": "NK",
    "NKT": "T cells",
    "Neutrophils": "Neutrophils",
    "Plasma B": "B cells",
    "Red-pulp macrophages": "Macrophages",
    "Transitional B": "B cells",
    "Tregs": "Treg",
    "cDC1s": "cDCs",
    "cDC2s": "cDCs",
    "pDCs": "pDCs",
    }
    # Map the specific cell types to major cell types and add as a new column in obs
    adata.obs['major_cell_types'] = pd.Categorical(adata.obs['cell_types'].map(cell_type_mapping))

    adata_1.obs['major_cell_types'] = pd.Categorical(adata_1.obs['cell_types'].map(cell_type_mapping))
    # adata_2.obs['major_cell_types'] = pd.Categorical(adata_2.obs['cell_types'].map(cell_type_mapping))

    assert set(cell_type_mapping.keys()) == set(adata.obs['cell_types'])

    # generate major cell types
    major_to_minor_dict ={}
    # from major to minor dict
    for k,v in cell_type_mapping.items():
        if v not in major_to_minor_dict:
            major_to_minor_dict[v] = [k]
        else:
            major_to_minor_dict[v].append(k)
            
    rna_adata = adata_1
    

    # rna_adata = ad.AnnData(rna.tocsr(), dtype=np.float32)
    # rna_adata.var_names = rna_names

    # metadata_rna = pd.read_csv(data_dir / "tonsil/tonsil_rna_meta.csv")
    # rna_adata.obs["cell_types"] = metadata_rna["cluster.info"].to_numpy()

    return rna_adata


def preprocess_rna_maxfuse(adata):
    """Preprocess RNA data using MaxFuse method"""
    print("Preprocessing RNA data...")
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)  # todo maxfuse uses 5000 may be too much
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata)
    return adata

def preprocess_protein_cite_seq(adata):
    """Preprocessing Protein data using MaxFuse method"""
    sc.pp.normalize_total(protein_adata)
    sc.pp.log1p(protein_adata)
    sc.pp.scale(protein_adata)
    return adata


def filter_and_subsample_data(adata_1, adata_2, num_rna_cells=None, num_protein_cells=None):
    """Filter and subsample data"""
    print("Filtering and subsampling data...")
    # Filter out tumor and dead cells
    adata_2 = adata_2[adata_2.obs["cell_types"] != "tumor"]
    adata_2 = adata_2[adata_2.obs["cell_types"] != "dead"]

    # Subsample cells
    subsample_n_obs_rna = min(adata_1.shape[0], num_rna_cells)
    subsample_n_obs_protein = min(adata_2.shape[0], num_protein_cells)
    if num_rna_cells is not None:
        sc.pp.subsample(adata_1, n_obs=subsample_n_obs_rna)
    if num_protein_cells is not None:
        sc.pp.subsample(adata_2, n_obs=subsample_n_obs_protein)

    # Remove NK cells
    adata_1 = adata_1[adata_1.obs["cell_types"] != "nk cells"]
    adata_2 = adata_2[adata_2.obs["cell_types"] != "nk cells"]

    # Sort by cell types
    adata_1 = adata_1[adata_1.obs["cell_types"].argsort(), :]
    adata_2 = adata_2[adata_2.obs["cell_types"].argsort(), :]

    return adata_1, adata_2


def process_spatial_data(adata):
    """Process spatial data for protein dataset"""
    print("Processing spatial data...")
    adata = preprocess_protein_cite_seq(adata)

    import numpy as np
    import pandas as pd

    # Generate random coordinates between 0 and 100
#     num_cells = adata.n_obs  # number of observations
#     x_coor = np.random.uniform(0, 100, size=num_cells)
#     y_coor = np.random.uniform(0, 100, size=num_cells)

#     spatial_coords = np.column_stack((x_coor, y_coor))

# # Assign to adata.obsm
#     adata.obsm["spatial"] = spatial_coords


    # ACTUAL SPATIAL DATA WORK

    assert adata.obs.index.is_unique
    
    major_to_minor_dict ={}
    # from major to minor dict
    cell_type_mapping = {
        "Activated CD4 T": "CD4 T",
        "B1 B": "B cells",
        "CD122+ CD8 T": "CD8 T",
        "CD4 T": "CD4 T",
        "CD8 T": "CD8 T",
        "Erythrocytes": "RBC",
        "GD T": "T cells",
        "ICOS-high Tregs": "CD4 T",
        "Ifit3-high B": "B cells",
        "Ifit3-high CD4 T": "CD4 T",
        "Ifit3-high CD8 T": "CD8 T",
        "Ly6-high mono": "Monocytes",
        "Ly6-low mono": "Monocytes",
        "MZ B": "B cells",
        "MZ/Marco-high macrophages": "Macrophages",
        "Mature B": "B cells",
        "Migratory DCs": "cDCs",
        "NK": "NK",
        "NKT": "T cells",
        "Neutrophils": "Neutrophils",
        "Plasma B": "B cells",
        "Red-pulp macrophages": "Macrophages",
        "Transitional B": "B cells",
        "Tregs": "Treg",
        "cDC1s": "cDCs",
        "cDC2s": "cDCs",
        "pDCs": "pDCs",
    }
    for k,v in cell_type_mapping.items():
        if v not in major_to_minor_dict:
            major_to_minor_dict[v] = [k]
        else:
            major_to_minor_dict[v].append(k)

    adata,horizontal_splits,vertical_splits = bar_nick_utils.add_spatial_data_to_prot(adata, major_to_minor_dict)
    adata.obsm['spatial_location'] = pd.DataFrame([adata.obs['X'],adata.obs['Y']]).T


    if plot_flag:
        sc.pl.scatter(adata[adata.obs['major_cell_types']=='B cells'], x='X', y='Y', color='cell_types', title='B Cell subtypes locations')
        sc.pl.scatter(adata[adata.obs['major_cell_types']=='CD4 T'], x='X', y='Y', color='cell_types', title='T Cell subtypes locations')
        sc.pl.scatter(adata[adata.obs['major_cell_types']=='CD8 T'], x='X', y='Y', color='cell_types', title='T Cell subtypes locations')

    adata.obsm['spatial'] = adata.obsm['spatial_location']
    
    if isinstance(adata.obsm["spatial"], pd.DataFrame):
        adata.obsm["spatial"] = adata.obsm["spatial"].values

    # Store in a dataframe and add to AnnData object
    # temp = pd.DataFrame([x_coor, y_coor], index=["x", "y"]).T
    # temp.index = adata.obs.index
    # adata.obsm["spatial_location"] = temp
    # adata.obs["X"] = x_coor
    # adata.obs["Y"] = y_coor

    # x_coor = adata.obsm["spatial"][:, 0]
    # y_coor = adata.obsm["spatial"][:, 1]
    # temp = pd.DataFrame([x_coor, y_coor], index=["x", "y"]).T
    # temp.index = adata.obs.index
    # adata.obsm["spatial_location"] = temp
    # adata.obs["X"] = x_coor
    # adata.obs["Y"] = y_coor
    return adata


def save_processed_data(adata_1, adata_2, save_dir):
    """Save processed data"""
    print("Saving processed data...")
    clean_uns_for_h5ad(adata_2)
    clean_uns_for_h5ad(adata_1)
    time_stamp = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rna_file = save_dir / f"preprocessed_adata_rna_cite_seq_{time_stamp}.h5ad"
    prot_file = save_dir / f"preprocessed_adata_prot_cite_seq_{time_stamp}.h5ad"

    print(f"\nRNA data dimensions: {adata_1.shape[0]} samples x {adata_1.shape[1]} features")
    print(f"Protein data dimensions: {adata_2.shape[0]} samples x {adata_2.shape[1]} features\n")

    adata_1.write(rna_file)
    adata_2.write(prot_file)

    print(f"Saved RNA data: {rna_file} ({rna_file.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Saved protein data: {prot_file} ({prot_file.stat().st_size / (1024*1024):.2f} MB)")


# %%
# Setup and run preprocessing
# %%
# Setup environment
device = setup_environment()

# Setup paths
root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = root_dir / "CODEX_RNA_seq" / "data" / "raw_data" / "data"
save_dir = root_dir / "CODEX_RNA_seq" / "data" / "processed_data"

# Create data directory if it doesn't exist
data_dir.mkdir(parents=True, exist_ok=True)

# %%
# Download and load data
# %%
# Download data if needed
# download_cite_seq_data(data_dir)

# Load data
protein_adata = load_protein_data(data_dir)
rna_adata = load_rna_data(data_dir)

# %%
# Plot initial data overview

protein_adata = process_spatial_data(protein_adata)

# %%
if plot_flag:
    # Compute PCA for both datasets
    sc.tl.pca(rna_adata)
    sc.tl.pca(protein_adata)

    pf.plot_data_overview(rna_adata, protein_adata)
    pf.plot_cell_type_distribution(rna_adata, protein_adata)
    pf.plot_spatial_data(protein_adata)

# %%
# Filter and subsample data
# %%
rna_adata, protein_adata = filter_and_subsample_data(
    rna_adata, protein_adata, num_rna_cells, num_protein_cells
)
# Subsample protein data to 40k cells
sc.pp.subsample(protein_adata, n_obs=8736)


# %%
# Preprocess RNA data
# %%
rna_adata = preprocess_rna_maxfuse(rna_adata)

if plot_flag:
    sc.pl.highly_variable_genes(rna_adata)


# %%
# Process spatial data
# %%
# protein_adata = process_spatial_data(protein_adata)

# %%
# Save processed data
# %%
save_processed_data(rna_adata, protein_adata, save_dir)

# %%
# Plot preprocessing results
# %%
if plot_flag:
    # Compute UMAP for both datasets
    sc.pp.neighbors(rna_adata)
    sc.tl.umap(rna_adata)
    sc.pp.neighbors(protein_adata)
    sc.tl.umap(protein_adata)

    pf.plot_preprocessing_results(rna_adata, protein_adata)

print("Preprocessing completed successfully!")

# %%

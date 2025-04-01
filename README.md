# Mapping scRNA-seq Variability Using Co-Embedding to Understand Cell Neighborhood Effects

This repository explores methods to understand and align scRNA-seq and protein-based single-cell modalities
(e.g., from CITE-seq) by jointly embedding them. The approach aims to capture and preserve cell
neighborhood structures and archetype representations, helping to reveal how local cell neighborhoods
affect variability and relationships between transcriptomic and proteomic data.

## Contents

- [Reproducing the Results](#reproducing-the-results)
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [System Requirements and Dependencies](#system-requirements-and-dependencies)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Running Model](#running-model)

## Reproducing the Results

To reproduce the provided results:

1. Follow [Installation](#installation) instructions.
2. Run `archetype_generation.ipynb` end-to-end with the same parameters.
3. Run `train_vae_with_archetypes_vectors.ipynb` to train the dual VAE model.

## Overview

This project involves:
1. **Preprocessing RNA and Protein Data:**
   - Selecting highly variable genes and normalizing/scaling the data.
   - Generating synthetic spatial coordinates for cells if needed.

2. **Archetype Analysis:**
   - Using Principal Convex Hull Analysis (PCHA) to identify "archetypes" that represent distinct biological or compositional extremes in the data.
   - Matching RNA and protein archetypes to ensure both modalities align around shared biological extremes.

3. **Co-Embedding with Dual VAEs:**
   - Training Variational Autoencoders (VAEs) for both RNA and protein data.
   - Incorporating contrastive losses and archetype-based constraints to maintain modality alignment.
   - Exploring how neighborhood context (e.g., cluster or spatial neighborhoods) affects latent space organization.

4. **Visualization and Evaluation:**
   - Using PCA, UMAP, and heatmaps to visualize embeddings and archetype relationships.
   - Evaluating alignment quality with various distance metrics and transport costs.

## Repository Structure

```
.
│  ├─ environment.yaml                    # Conda environment file listing dependencies
│  ├─ README.md                           # This README file
├─ CITE-seq_RNA_seq/
│  ├─ archetype_generation.ipynb          # Notebook for generating archetypes and analyzing them
│  ├─ train_vae_with_archetypes_vectors.ipynb # Notebook for training dual VAEs using archetype constraints
│  ├─ bar_nick_utils.py                   # Utility functions for plotting, data manipulation, etc.
│  ├─ model.pt                            # dual VAE model weights
│  ├─ data/                               # Directory containing input data and saved results
│  │  ├─ adata_rna_*.h5ad                 # Processed RNA AnnData files with timestamp
│  │  ├─ adata_prot_*.h5ad                # Processed Protein AnnData files with timestamp
│  │  ├─ adata_archetype_rna_*.h5ad       # RNA archetype embeddings AnnData files
│  │  ├─ adata_archetype_prot_*.h5ad      # Protein archetype embeddings AnnData files
│  │  ├─ spleen_lymph_cite_seq.h5ad       # input data
└─ ...
```

### Key Files and Parameters

- **archetype_generation.ipynb:**
  Jupyter notebook that:
  - Preprocesses RNA and protein data.
  - Identifies archetypes using PCHA.
  - Aligns archetypes between RNA and protein modalities.
  - Saves processed `adata_rna_*.h5ad` and `adata_prot_*.h5ad` files.

- **train_vae_with_archetypes_vectors.ipynb:**
  Jupyter notebook that:
  - Loads the preprocessed and archetype-annotated data.
  - Sets up and trains dual VAEs on RNA and protein.
  - Incorporates contrastive losses and saves trained VAEs.

- **bar_nick_utils.py:**
  Python utilities for:
  - Data preprocessing (e.g., `preprocess_rna`, `preprocess_protein`).
  - Archetype fitting and visualization (e.g., `plot_archetypes`, `compute_pairwise_kl`).
  - Saving/loading models, cleaning AnnData objects, and plotting training curves.

- **environment.yaml:**
  Conda dependencies environment needed to run the analysis.

- **data/ directory:**
  Contains:
  - Input data files (e.g., `spleen_lymph_cite_seq.h5ad`).
  - Processed outputs (e.g., `adata_rna_*.h5ad`, `adata_prot_*.h5ad`).
  - Archetype-based embeddings (`adata_archetype_rna_*.h5ad`, `adata_archetype_prot_*.h5ad`).

### Parameters Within the Code

Inside the notebooks, you will find parameters controlling:
- **Number of archetypes (k):**
  Set in `archetype_generation.ipynb` to define how many archetypes to extract.
- **Training parameters like n_epochs Batch size, learning rate, and kl_weight...**
  Set in `train_vae_with_archetypes_vectors.ipynb` when calling `rna_vae.train(...)`.
- **Contrastive weights and other loss-related parameters:**
  Adjusted in `DualVAETrainingPlan` class inside the code.

## System Requirements and Dependencies

- **Operating System:** Linux/Unix.
- **Conda**
- **Python Version:** Python 3.10.
- **CPU/GPU:** GPU recommended for VAE training.


## Installation

1. **Create Conda Environment:**
   ```bash
   conda env create -f environment.yaml
   conda activate scvi
   ```

2. **Install PyTorch (CPU or GPU version):**
   For CPU-only:
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

   For GPU (CUDA) see pytorch website
## Data Preparation

- Run `archetype_generation.ipynb` to:
  - Download `spleen_lymph_cite_seq.h5ad`
  - Preprocess the data.
  - Identify archetypes.
  - Save processed AnnData objects (RNA and Protein).

The resulting files (e.g., `adata_rna_YYYY-MM-DD-HH-MM-SS.h5ad` and `adata_prot_YYYY-MM-DD-HH-MM-SS.h5ad`) will be saved in `data/`.

## Running Model

1. **Archetype Generation:**
   - Open and run `archetype_generation.ipynb`.
   - This notebook will:
     - Load raw data.
     - Preprocess it (filtering cells, selecting HVGs).
     - Apply PCHA to identify archetypes.
     - Align RNA and protein archetypes.
     - Save processed `adata_*.h5ad` files with archetype embeddings.

2. **Training the Model:**
   - Open and run `train_vae_with_archetypes_vectors.ipynb`.
   - This notebook will:
     - Load previously saved processed data.
     - Set up SCVI models for RNA and protein.
     - Define a custom dual training plan that includes contrastive and alignment losses.
     - Train the dual VAE model.
     - Save the trained model.

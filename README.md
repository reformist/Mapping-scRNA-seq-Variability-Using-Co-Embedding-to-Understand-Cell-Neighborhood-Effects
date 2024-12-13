# Mapping scRNA-seq Variability Using Co-Embedding to Understand Cell Neighborhood Effects

This repository explores methods for mapping single-cell RNA sequencing (scRNA-seq) variability by generating and analyzing co-embeddings of RNA and protein modalities from CITE-seq data. The approach aims to understand how local neighborhoods of cells (based on both transcriptomic and proteomic information) affect their variability and relationships to defined archetypes.

## Overview

The analysis involves:
- Preprocessing scRNA-seq and protein expression data (e.g., CITE-seq data).
- Generating synthetic spatial coordinates for cells to simulate spatial organization.
- Identifying and extracting cell archetypes from the data using **PCHA** (Principal Convex Hull Analysis).
- Comparing matched vs. random alignments of archetypes and exploring the effects of cell neighborhoods on the latent representations.
- Performing dual variational autoencoder (VAE) training on both RNA and protein modalities, encouraging alignment and contrastive learning to maintain neighborhood structure.
- Visualizing and evaluating the joint embeddings to understand how cell types and archetypes are preserved and aligned across modalities.

## Installation

### 1. Conda Environment Setup

First, ensure that you have [conda](https://docs.conda.io/en/latest/miniconda.html) installed. Then, create the conda environment from the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
conda activate scvi
```
### 2. Install PyTorch
For example, if you can only use torch cpu:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Run the Jupyter Notebook
Start by running [archetype_generation.ipynb](CITE-seq_RNA_seq/archetype_generation.ipynb)
which will guide you through the process of generating cell archetypes and synthetic spatial coordinates.
and will save a new annotated dataset with the archetype embedding

Second run [train_vae_with_archetypes_vectors.ipynb](CITE-seq_RNA_seq/train_vae_with_archetypes_vectors.ipynb)
which will guide you through the process of training a dual VAE model with the archetype embedding as a matching loss across modalities
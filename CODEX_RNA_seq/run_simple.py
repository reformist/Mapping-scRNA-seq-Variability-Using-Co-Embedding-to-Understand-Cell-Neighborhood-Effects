#!/usr/bin/env python
import importlib
import os
import sys
from pathlib import Path

# Add repository root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set working directory to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cell_lists
import plotting_functions

import bar_nick_utils

importlib.reload(cell_lists)
importlib.reload(plotting_functions)
importlib.reload(bar_nick_utils)


# Get the project root directory
project_root = Path(__file__).parent.parent.absolute()
os.chdir(project_root)

# Run each script
scripts = [
    "CODEX_RNA_seq/preprocess_maxfuse_tonsil_dataset.py",
    "CODEX_RNA_seq/archetype_generation_neighbors_means_maxfuse.py",
    "CODEX_RNA_seq/prepare_data_for_training.py",
    "CODEX_RNA_seq/train_vae_with_archetypes_vectors.py",
]

for script in scripts:
    print(f"Running {script}...")
    exec(open(script).read())
    print(f"Finished {script}\n")

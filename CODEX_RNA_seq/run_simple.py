#!/usr/bin/env python
import os
import sys
from pathlib import Path

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

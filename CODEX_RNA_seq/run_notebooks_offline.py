
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import subprocess

# Define the paths to your notebooks
notebooks = [
    "/home/barroz/projects/Mapping-scRNA-seq-Variability-Using-Co-Embedding-to-Understand-Cell-Neighborhood-Effects/CODEX_RNA_seq/archetype_generation_real.ipynb",
    "/home/barroz/projects/Mapping-scRNA-seq-Variability-Using-Co-Embedding-to-Understand-Cell-Neighborhood-Effects/CODEX_RNA_seq/train_vae_with_archetypes_vectors.ipynb"
]

# notebooks = [
# '/home/barroz/projects/Mapping-scRNA-seq-Variability-Using-Co-Embedding-to-Understand-Cell-Neighborhood-Effects/CODEX_RNA_seq/zadu copy.ipynb']


# Path to Conda initialization script (adjust this path if necessary)
conda_init = "/opt/conda/etc/profile.d/conda.sh"

# Define the Conda environment name
conda_env = "scvi"

# Function to execute a notebook
def run_notebook(notebook_path):
    # Get the directory of the notebook
    notebook_dir = os.path.dirname(notebook_path)

    # Change the working directory to the notebook's folder
    os.chdir(notebook_dir)

    # Build the command to execute the notebook
    output_notebook = os.path.basename(notebook_path).replace(".ipynb", "_output.ipynb")
    command = f"""
    . {conda_init} && conda activate {conda_env} && \
    jupyter nbconvert --to notebook --execute "{notebook_path}" --output "{output_notebook}"
    """

    # Run the command and handle errors
    print(f"Running notebook: {notebook_path}")
    try:
        subprocess.run(command, shell=True, check=True, executable="/bin/bash")
        print(f"Finished running: {notebook_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error while running {notebook_path}: {e}")
        raise

# Run each notebook sequentially
for notebook in notebooks:
    run_notebook(notebook)

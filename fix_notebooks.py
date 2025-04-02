import glob
import json
import os


def remove_ids_from_notebook(notebook_path):
    try:
        with open(notebook_path, "r") as f:
            nb = json.load(f)

        # Remove 'id' from each cell
        modified = False
        for cell in nb["cells"]:
            if "id" in cell:
                del cell["id"]
                modified = True

        if modified:
            with open(notebook_path, "w") as f:
                json.dump(nb, f, indent=1)
            print(f"Fixed {notebook_path}")

    except Exception as e:
        print(f"Error processing {notebook_path}: {str(e)}")


# Find all notebooks
notebooks = glob.glob("**/*.ipynb", recursive=True)

# Process each notebook
for nb_path in notebooks:
    remove_ids_from_notebook(nb_path)

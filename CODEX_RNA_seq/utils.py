# %% Utility Functions
# This module contains utility functions for data processing and analysis.

# %% Imports and Setup
import importlib
import os
import sys

import numpy as np
from scipy.sparse import issparse

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


def select_gene_likelihood(adata):
    """Select an appropriate gene likelihood based on the data."""
    if issparse(adata.X):
        # For sparse matrices, assume it's count data (common in scRNA-seq)
        return "zinb"
    elif adata.X.min() >= 0:
        # If all values are non-negative
        if np.allclose(np.round(adata.X), adata.X):
            # If data is integer-valued (like counts)
            return "zinb"
        else:
            # If data contains non-integer values
            return "normal"
    else:
        # For data with negative values
        return "normal"

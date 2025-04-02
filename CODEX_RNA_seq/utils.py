import numpy as np
from scipy.sparse import issparse


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

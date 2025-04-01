#  code from https://github.com/dpeerlab/ENVI/tree/main
import numpy as np
import scanpy as sc
import scipy
import sklearn.neighbors


def MatSqrt(Mats):
    """
    :meta private:
    """

    e, v = np.linalg.eigh(Mats)
    e = np.where(e < 0, 0, e)
    e = np.sqrt(e)

    m, n = e.shape
    diag_e = np.zeros((m, n, n), dtype=e.dtype)
    diag_e.reshape(-1, n**2)[..., :: n + 1] = e
    return np.matmul(np.matmul(v, diag_e), v.transpose([0, 2, 1]))


def BatchKNN(data, batch, k):
    """
    :meta private:
    """

    kNNGraphIndex = np.zeros(shape=(data.shape[0], k))

    for val in np.unique(batch):
        val_ind = np.where(batch == val)[0]

        batch_knn = sklearn.neighbors.kneighbors_graph(
            data[val_ind], n_neighbors=k, mode="connectivity", n_jobs=-1
        ).tocoo()
        batch_knn_ind = np.reshape(np.asarray(batch_knn.col), [data[val_ind].shape[0], k])
        kNNGraphIndex[val_ind] = val_ind[batch_knn_ind]

    return kNNGraphIndex.astype("int")


def CalcCovMats(spatial_data, kNN, genes, spatial_key="spatial", batch_key=-1):
    """
    :meta private:
    """

    ExpData = (
        np.log(spatial_data[:, genes].X.toarray() + 1)
        if scipy.sparse.issparse(spatial_data[:, genes].X)
        else np.log(spatial_data[:, genes].X + 1)
    )

    if batch_key == -1:
        kNNGraph = sklearn.neighbors.kneighbors_graph(
            spatial_data.obsm[spatial_key],
            n_neighbors=kNN,
            mode="connectivity",
            n_jobs=-1,
        ).tocoo()
        kNNGraphIndex = np.reshape(
            np.asarray(kNNGraph.col), [spatial_data.obsm[spatial_key].shape[0], kNN]
        )
    else:
        kNNGraphIndex = BatchKNN(spatial_data.obsm[spatial_key], spatial_data.obs[batch_key], kNN)

    DistanceMatWeighted = (
        ExpData.mean(axis=0)[None, None, :] - ExpData[kNNGraphIndex[np.arange(ExpData.shape[0])]]
    )

    CovMats = np.matmul(DistanceMatWeighted.transpose([0, 2, 1]), DistanceMatWeighted) / (kNN - 1)
    CovMats = CovMats + CovMats.mean() * 0.00001 * np.expand_dims(
        np.identity(CovMats.shape[-1]), axis=0
    )
    return CovMats


def compute_covet(spatial_data, k=8, g=64, genes=[], spatial_key="spatial", batch_key=-1):
    """
    Compute niche covariance matrices for spatial data, run with scenvi.compute_covet

    :param spatial_data: (anndata) spatial data, with an obsm indicating spatial location of spot/segmented cell
    :param k: (int) number of nearest neighbours to define niche (default 8)
    :param g: (int) number of HVG to compute COVET representation on (default 64)
    :param genes: (list of str) list of genes to keep for niche covariance (default []
    :param spatial_key: (str) obsm key name with physical location of spots/cells (default 'spatial')
    :param batch_key: (str) obs key name of batch/sample of spatial data (default 'batch' if in spatial_data.obs, else -1)

    :return COVET: niche covariance matrices
    :return COVET_SQRT: matrix square-root of niche covariance matrices for approximate OT
    :return CovGenes: list of genes selected for COVET representation
    """

    if g == -1:
        CovGenes = spatial_data.var_names
    else:
        if "highly_variable" not in spatial_data.var.columns:
            if "log" in spatial_data.layers.keys():
                sc.pp.highly_variable_genes(spatial_data, n_top_genes=g, layer="log")
            elif "log1p" in spatial_data.layers.keys():
                sc.pp.highly_variable_genes(spatial_data, n_top_genes=g, layer="log1p")
            elif spatial_data.X.min() < 0:
                sc.pp.highly_variable_genes(spatial_data, n_top_genes=g)
            else:
                spatial_data.layers["log"] = (
                    np.log(spatial_data.X.toarray() + 1)
                    if scipy.sparse.issparse(spatial_data.X)
                    else np.log(spatial_data.X + 1)
                )
                sc.pp.highly_variable_genes(spatial_data, n_top_genes=g, layer="log")

        CovGenes = np.asarray(spatial_data.var_names[spatial_data.var.highly_variable])
        if len(genes) > 0:
            CovGenes = np.union1d(CovGenes, genes)

    if batch_key not in spatial_data.obs.columns:
        batch_key = -1

    COVET = CalcCovMats(
        spatial_data, k, genes=CovGenes, spatial_key=spatial_key, batch_key=batch_key
    )
    COVET_SQRT = MatSqrt(COVET)

    return (
        COVET.astype("float32"),
        COVET_SQRT.astype("float32"),
        np.asarray(CovGenes).astype("str"),
    )

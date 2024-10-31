from anndata import AnnData


def get_data_subset(adata: AnnData, column: str, group: str | int) -> AnnData:
    """Subset dask-backed data for toy GRN.

    Parameters
    ----------
    adata
        AnnData object including lazily loaded data.
    column
        Column in `adata.obs` according to which to subset the data.
    group
        Group to subset to.

    Returns
    -------
    AnnData subsetted to a specific group and read in memory.
    """
    obs_mask = adata.obs[column] == group

    adata_subset = adata[obs_mask, :]
    for parameter in ["true_beta", "true_gamma"]:
        adata_subset.var[parameter] = adata.uns[group][parameter]
    del adata_subset.uns

    return adata_subset.to_memory()

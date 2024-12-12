import numpy as np

from anndata import AnnData


def get_data_subset(adata: AnnData, column: str, group: str | int, uns_keys: list[str]) -> AnnData:
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

    adata_subset = adata[obs_mask, :].copy()
    uns = {}
    for parameter in uns_keys:
        value = adata.uns[group][parameter]
        if value.shape == (adata_subset.n_vars,):
            adata_subset.var[parameter] = value
        else:
            uns[parameter] = value
    adata_subset.uns = uns

    return adata_subset.to_memory()


# Code mostly taken from veloVI reproducibility repo
# https://yoseflab.github.io/velovi_reproducibility/estimation_comparison/simulation_w_inferred_rates.html
def set_output(adata, vae, n_samples: int = 1, batch_size: int | None = None) -> None:
    """Add inference results to adata."""
    latent_time = vae.get_latent_time(n_samples=n_samples, batch_size=batch_size)
    velocities = vae.get_velocity(n_samples=n_samples, batch_size=batch_size)

    t = latent_time.values
    scaling = 20 / t.max(0)

    adata.layers["velocity"] = velocities / scaling
    adata.layers["latent_time_velovi"] = latent_time

    rates = vae.get_rates()
    if "alpha" in rates:
        adata.var["fit_alpha"] = rates["alpha"] / scaling
    adata.var["fit_beta"] = rates["beta"] / scaling
    adata.var["fit_gamma"] = rates["gamma"] / scaling

    adata.layers["fit_t"] = latent_time * scaling[np.newaxis, :]
    adata.var["fit_scaling"] = 1.0

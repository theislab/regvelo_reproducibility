import numpy as np


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

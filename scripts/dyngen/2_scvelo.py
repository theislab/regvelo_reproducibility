# %% [markdown]
# # scVelo benchmark on dyngen data
#
# Notebook benchmarks velocity and latent time inference using scVelo on dyngen-generated data.

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd

import anndata as ad
import scvelo as scv

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import get_time_correlation, get_velocity_correlation

# %% [markdown]
# ## General settings

# %%
scv.settings.verbosity = 0

# %% [markdown]
# ## Constants

# %%
DATASET = "dyngen"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Velocity pipeline

# %%
velocity_correlation = []
time_correlation = []

for filename in (DATA_DIR / DATASET / "processed").iterdir():
    if filename.suffix != ".zarr":
        continue

    adata = ad.io.read_zarr(filename)

    # Parameter inference
    scv.tl.recover_dynamics(adata, fit_scaling=False, var_names=adata.var_names, n_jobs=1)

    # Velocity inferene
    adata.var["fit_scaling"] = 1.0
    scv.tl.velocity(adata, mode="dynamical", min_likelihood=-np.inf, min_r2=None)

    velocity_correlation.append(
        get_velocity_correlation(
            ground_truth=adata.layers["true_velocity"], estimated=adata.layers["velocity"], aggregation=np.mean
        )
    )

    ## calculate per gene latent time correlation
    time_corr = [
        get_time_correlation(ground_truth=adata.obs["true_time"], estimated=adata.layers["fit_t"][:, i])
        for i in range(adata.layers["fit_t"].shape[1])
    ]
    time_correlation.append(np.mean(time_corr))

# %% [markdown]
# ## Data saving

# %%
if SAVE_DATA:
    pd.DataFrame({"velocity": velocity_correlation, "time": time_correlation}).to_parquet(
        path=DATA_DIR / DATASET / "results" / "scvelo_correlation.parquet"
    )

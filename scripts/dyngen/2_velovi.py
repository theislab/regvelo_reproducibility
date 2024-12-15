# %% [markdown]
# # veloVI benchmark on dyngen data
#
# Notebook benchmarks velocity and latent time inference using veloVI on dyngen-generated data.

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd

import anndata as ad
from velovi import VELOVI

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import (
    get_time_correlation,
    get_velocity_correlation,
    set_output,
)

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

    VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    vae = VELOVI(adata)
    vae.train(max_epochs=1500)

    set_output(adata, vae, n_samples=30)

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
        path=DATA_DIR / DATASET / "results" / "velovi_correlation.parquet"
    )

# %%

# %% [markdown]
# # VeloVAE (fullvb) benchmark on dyngen data
#
# Notebook benchmarks velocity and latent time inference using VeloVAE (fullvb) on dyngen-generated data.

# %% [markdown]
# ## Library imports

# %%
import velovae as vv

import numpy as np
import pandas as pd
import torch

import anndata as ad
import scvelo as scv

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import get_velocity_correlation

# %% [markdown]
# ## General settings

# %%
scv.settings.verbosity = 3

# %% [markdown]
# ## Constants

# %%
torch.manual_seed(0)
np.random.seed(0)

# %%
DATASET = "dyngen"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / DATASET / "processed" / "velovae_vae").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Velocity pipeline

# %%
velocity_correlation = []

for filename in (DATA_DIR / DATASET / "processed").iterdir():
    if filename.suffix != ".zarr":
        continue

    adata = ad.io.read_zarr(filename)

    try:
        rate_prior = {"alpha": (0.0, 1.0), "beta": (0.0, 0.5), "gamma": (0.0, 0.5)}
        vae = vv.VAE(adata, tmax=20, dim_z=5, device="cuda:0", full_vb=True, rate_prior=rate_prior)
        config = {}
        vae.train(adata, config=config, plot=False, embed="pca")

        # Output velocity to adata object
        vae.save_anndata(adata, "fullvb", DATA_DIR / DATASET / "processed" / "velovae_vae", file_name="velovae.h5ad")

        adata.layers["velocity"] = adata.layers["fullvb_velocity"].copy()

        velocity_correlation.append(
            get_velocity_correlation(
                ground_truth=adata.layers["true_velocity"], estimated=adata.layers["velocity"], aggregation=np.mean
            )
        )

    except Exception as e:  # noqa: BLE001
        # Append np.nan in case of an error and optionally log the error
        print(f"An error occurred: {e}")
        velocity_correlation.append(np.nan)

# %% [markdown]
# ## Data saving

# %%
if SAVE_DATA:
    pd.DataFrame({"velocity": velocity_correlation}).to_parquet(
        path=DATA_DIR / DATASET / "results" / "velovae_fullvb_correlation.parquet"
    )

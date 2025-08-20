# %% [markdown]
# # UniTVelo benchmark on dyngen data
#
# Notebook benchmarks velocity and latent time inference using UniTVelo on dyngen-generated data.

# %% [markdown]
# ## Library imports

# %%
import os

import numpy as np
import pandas as pd

import anndata as ad
import scvelo as scv
import unitvelo as utv

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import get_velocity_correlation

# %% [markdown]
# ## General setting

# %%
scv.settings.verbosity = 3

# %%
velo_config = utv.config.Configuration()
velo_config.R2_ADJUST = True
velo_config.IROOT = None
velo_config.FIT_OPTION = "1"
velo_config.AGENES_R2 = 1
velo_config.GPU = -1

# %%
os.environ["TF_USE_LEGACY_KERAS"] = "True"

# %% [markdown]
# ## Constants

# %%
DATASET = "dyngen"

# %%
COMPLEXITY = "complexity_1"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / COMPLEXITY / "results").mkdir(parents=True, exist_ok=True)

# %%
SAVE_DATASETS = True
if SAVE_DATASETS:
    (DATA_DIR / DATASET / COMPLEXITY / "trained_unitvelo").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Velocity pipeline

# %%
velocity_correlation = []

cnt = 0
for filename in (DATA_DIR / DATASET / COMPLEXITY / "processed").iterdir():
    if filename.suffix != ".zarr":
        continue

    simulation_id = int(filename.stem.removeprefix("simulation_"))
    print(f"Run {cnt}, dataset {simulation_id}.")

    adata = ad.io.read_zarr(filename)
    adata.var["highly_variable"] = True

    adata.obs["cluster"] = "0"
    adata = utv.run_model(adata, label="cluster", config_file=velo_config)

    # save data
    adata.write_zarr(DATA_DIR / DATASET / COMPLEXITY / "trained_unitvelo" / f"trained_{simulation_id}.zarr")

    velocity_correlation.append(
        get_velocity_correlation(
            ground_truth=adata.layers["true_velocity"], estimated=adata.layers["velocity"], aggregation=np.mean
        )
    )

    cnt += 1

# %% [markdown]
# ## Data saving

# %%
if SAVE_DATA:
    pd.DataFrame({"velocity": velocity_correlation}).to_parquet(
        path=DATA_DIR / DATASET / COMPLEXITY / "results" / "unitvelo_correlation.parquet"
    )

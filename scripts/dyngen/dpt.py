# %% [markdown]
# # DPT benchmark on dyngen data
#
# Notebook benchmarks latent time inference using DPT on dyngen-generated data.

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd

import anndata as ad
import scanpy as sc

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import get_time_correlation

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
time_correlation = []

for filename in (DATA_DIR / DATASET / "processed").iterdir():
    if filename.suffix != ".zarr":
        continue

    adata = ad.io.read_zarr(filename)

    adata.uns["iroot"] = np.flatnonzero(adata.obs["true_time"] == 0)[0]

    sc.pp.neighbors(adata)
    sc.tl.diffmap(adata)
    sc.tl.dpt(adata)

    time_correlation.append(
        get_time_correlation(ground_truth=adata.obs["true_time"], estimated=adata.obs["dpt_pseudotime"].values)
    )

# %% [markdown]
# ## Data saving

# %%
if SAVE_DATA:
    pd.DataFrame({"time": time_correlation}).to_parquet(path=DATA_DIR / DATASET / "results" / "dpt_correlation.parquet")

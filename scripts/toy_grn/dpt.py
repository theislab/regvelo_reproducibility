# %% [markdown]
# # DPT benchmark on toy GRN
#
# Notbook benchmarks latent time inference using DPT with toy GRN data.

# %% [markdown]
# ## Library imports

# %%
from tqdm import tqdm

import numpy as np
import pandas as pd

import scanpy as sc

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import get_data_subset, get_time_correlation
from rgv_tools.core import read_as_dask

# %% [markdown]
# ## General settings

# %% [markdown]
# ## Constants

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / "toy_grn" / "results").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Function definitions

# %% [markdown]
# ## Data loading

# %%
adata = read_as_dask(store=DATA_DIR / "toy_grn" / "raw" / "adata.zarr", layers=[])
adata

# %% [markdown]
# ## Pseudotime pipeline

# %%
time_correlation = []

for dataset in tqdm(adata.obs["dataset"].cat.categories):
    adata_subset = get_data_subset(adata=adata, column="dataset", group=dataset, uns_keys=[])

    adata_subset.uns["iroot"] = np.flatnonzero(adata_subset.obs["true_time"] == 0)[0]

    sc.pp.neighbors(adata_subset)
    sc.tl.diffmap(adata_subset)
    sc.tl.dpt(adata_subset)

    time_correlation.append(
        get_time_correlation(
            ground_truth=adata_subset.obs["true_time"], estimated=adata_subset.obs["dpt_pseudotime"].values
        )
    )

# %% [markdown]
# ## Data saving

# %%
if SAVE_DATA:
    pd.DataFrame({"time": time_correlation}).to_parquet(
        path=DATA_DIR / "toy_grn" / "results" / "dpt_correlation.parquet"
    )

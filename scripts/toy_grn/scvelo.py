# %% [markdown]
# # scVelo benchmark on toy GRN
#
# Notbook benchmarks velocity and latent time inference using scVelo with toy GRN data.

# %% [markdown]
# ## Library imports

# %%
from tqdm import tqdm

import numpy as np
import pandas as pd

import scanpy as sc
import scvelo as scv
from velovi import preprocess_data

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import (
    get_data_subset,
    get_time_correlation,
    get_velocity_correlation,
)
from rgv_tools.core import read_as_dask

# %% [markdown]
# ## General settings

# %%
scv.settings.verbosity = 0

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
adata = read_as_dask(
    store=DATA_DIR / "toy_grn" / "raw" / "mini_adata.zarr", layers=["unspliced", "Mu", "spliced", "Ms", "true_velocity"]
)
adata

# %% [markdown]
# ## Velocity pipeline

# %%
velocity_correlation = []
time_correlation = []

parameters = []

for dataset in tqdm(adata.obs["dataset"].cat.categories):
    adata_subset = get_data_subset(adata=adata, column="dataset", group=dataset, uns_keys=["true_beta", "true_gamma"])

    # Data preprocessing
    adata_subset = preprocess_data(adata_subset, filter_on_r2=False)
    # neighbor graph with scVelo's default number of neighbors
    sc.pp.neighbors(adata_subset, n_neighbors=30)

    # Parameter inference
    scv.tl.recover_dynamics(adata_subset, fit_scaling=False, var_names=adata.var_names, n_jobs=1)

    # Velocity inferene
    adata_subset.var["fit_scaling"] = 1.0
    scv.tl.velocity(adata_subset, mode="dynamical", min_likelihood=-np.inf, min_r2=None)

    _parameters = adata_subset.var[["fit_beta", "fit_gamma"]].copy()
    _parameters["dataset"] = dataset
    _parameters.index = _parameters.index + f"-dataset_{dataset}"
    parameters.append(_parameters)
    del _parameters

    # estimated_velocity = adata_subset.layers["unspliced"] * adata_subset.var["fit_beta"].values - adata_subset.layers["spliced"] * adata_subset.var["fit_gamma"].values
    velocity_correlation.append(
        get_velocity_correlation(
            ground_truth=adata_subset.layers["true_velocity"],
            estimated=adata_subset.layers["velocity"],
            aggregation=np.mean,
        )
    )
    time_correlation.append(
        get_time_correlation(
            ground_truth=adata_subset.obs["true_time"], estimated=adata_subset.layers["fit_t"].mean(axis=1)
        )
    )

# %% [markdown]
# ## Data saving

# %%
if SAVE_DATA:
    pd.DataFrame({"velocity": velocity_correlation, "time": time_correlation}).to_parquet(
        path=DATA_DIR / "toy_grn" / "results" / "scvelo_correlation.parquet"
    )
    pd.concat(parameters).to_parquet(path=DATA_DIR / "toy_grn" / "results" / "scvelo_rates.parquet")

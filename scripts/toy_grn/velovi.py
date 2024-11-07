# %% [markdown]
# # veloVI benchmark on toy GRN
#
# Notbook benchmarks velocity and latent time inference using veloVI with toy GRN data.

# %% [markdown]
# ## Library imports

# %%
from tqdm import tqdm

import numpy as np
import pandas as pd

from velovi import preprocess_data, VELOVI

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import (
    get_data_subset,
    get_time_correlation,
    get_velocity_correlation,
    set_output,
)
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
adata = read_as_dask(
    store=DATA_DIR / "toy_grn" / "raw" / "adata.zarr", layers=["unspliced", "Mu", "spliced", "Ms", "true_velocity"]
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

    VELOVI.setup_anndata(adata_subset, spliced_layer="Ms", unspliced_layer="Mu")
    vae = VELOVI(adata_subset)
    vae.train()

    set_output(adata_subset, vae, n_samples=30)

    estimated_velocity = (
        adata_subset.layers["unspliced"] * adata_subset.var["fit_beta"].values
        - adata_subset.layers["spliced"] * adata_subset.var["fit_gamma"].values
    )
    velocity_correlation.append(
        get_velocity_correlation(
            ground_truth=adata_subset.layers["true_velocity"], estimated=estimated_velocity, aggregation=np.mean
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
        path=DATA_DIR / "toy_grn" / "results" / "velovi_correlation.parquet"
    )

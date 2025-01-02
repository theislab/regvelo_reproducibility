# %% [markdown]
# # Correlation benchmark on toy GRN
#
# Notebook benchmarks GRN inference using a correlation-based scheme on toy GRN data.

# %% [markdown]
# ## Library imports

# %%
from tqdm import tqdm

import pandas as pd

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import get_data_subset, get_grn_auroc
from rgv_tools.core import read_as_dask

# %% [markdown]
# ## Constants

# %%
DATASET = "toy_grn"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Function definitions

# %% [markdown]
# ## Data loading

# %%
adata = read_as_dask(store=DATA_DIR / DATASET / "raw" / "adata.zarr", layers=[])
adata

# %% [markdown]
# ## Velocity pipeline

# %%
grn_correlation = []

for dataset in tqdm(adata.obs["dataset"].cat.categories):
    adata_subset = get_data_subset(adata=adata, column="dataset", group=dataset, uns_keys=["true_K"])

    grn_estimate = adata_subset.to_df().corr().abs().values
    grn_correlation.append(get_grn_auroc(ground_truth=adata_subset.uns["true_K"], estimated=grn_estimate))

# %% [markdown]
# ## Data saving

# %%
if SAVE_DATA:
    pd.DataFrame({"grn": grn_correlation}).to_parquet(
        path=DATA_DIR / DATASET / "results" / "correlation_correlation.parquet"
    )

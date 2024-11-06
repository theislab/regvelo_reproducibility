# %% [markdown]
# # GRNBoost2 benchmark on toy GRN
#
# Notbook benchmarks GRN inference using GRNBoost2 with toy GRN data.

# %% [markdown]
# ## Library imports

# %%
from tqdm import tqdm

import pandas as pd

from arboreto.algo import grnboost2

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import get_data_subset, get_grn_auroc
from rgv_tools.core import read_as_dask

# %% [markdown]
# ## General settings

# %%
"""
from dask import config as cfg

cfg.set({"distributed.scheduler.worker-ttl": None})
"""

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
# ## Velocity pipeline

# %%
grn_correlation = []

for dataset in tqdm(adata.obs["dataset"].cat.categories):
    adata_subset = get_data_subset(adata=adata, column="dataset", group=dataset, uns_keys=["true_K"])

    network = grnboost2(expression_data=adata_subset.to_df(), tf_names=adata.var_names.to_list())
    grn_estimate = pd.pivot(network, index="target", columns="TF").fillna(0).values

    grn_correlation.append(get_grn_auroc(ground_truth=adata_subset.uns["true_K"], estimated=grn_estimate))

# %% [markdown]
# ## Data saving

# %%
if SAVE_DATA:
    pd.DataFrame({"grn": grn_correlation}).to_parquet(
        path=DATA_DIR / "toy_grn" / "results" / "grnboost2_correlation.parquet"
    )

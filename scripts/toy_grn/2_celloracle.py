# %% [markdown]
# # CellOracle benchmark on toy GRN
#
# Notebook benchmarks GRN inference using CellOracle on toy GRN data.

# %% [markdown]
# ## Library imports

# %%
from tqdm import tqdm

import numpy as np
import pandas as pd

import celloracle as co

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

    base_grn = np.ones((adata_subset.n_vars, adata_subset.n_vars))
    base_grn = pd.DataFrame(base_grn, columns=adata_subset.var_names)
    base_grn["peak_id"] = adata_subset.var_names.str.replace("gene", "peak")
    base_grn["gene_short_name"] = adata_subset.var_names
    base_grn = base_grn[["peak_id", "gene_short_name"] + adata_subset.var_names.to_list()]

    net = co.Net(gene_expression_matrix=adata_subset.to_df(), TFinfo_matrix=base_grn, verbose=False)
    net.fit_All_genes(bagging_number=100, alpha=1, verbose=False)
    net.updateLinkList(verbose=False)

    grn_estimate = pd.pivot(net.linkList[["source", "target", "coef_mean"]], index="target", columns="source")
    grn_estimate = grn_estimate.fillna(0).abs().values

    grn_correlation.append(get_grn_auroc(ground_truth=adata_subset.uns["true_K"], estimated=grn_estimate))

# %% [markdown]
# ## Data saving

# %%
if SAVE_DATA:
    pd.DataFrame({"grn": grn_correlation}).to_parquet(
        path=DATA_DIR / DATASET / "results" / "celloracle_correlation.parquet"
    )

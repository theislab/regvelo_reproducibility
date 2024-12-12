# %% [markdown]
# # SpliceJAC benchmark on cell cycle
#
# Notebook benchmarks GRN inference using SpliceJAC on cell cycling dataset

# %% [markdown]
# ## Library imports

# %%
import splicejac as sp

import pandas as pd

import anndata as ad

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import get_grn_auroc_cc

# %% [markdown]
# ## Constants

# %%
DATASET = "cell_cycle"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
adata = ad.io.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_processed.h5ad")
adata

# %% [markdown]
# ## GRN pipeline

# %%
## We ignore the cell label information and assume all cells is the same label due to we use cellline dataset
adata.obs["clusters"] = "0"
n = len(adata.var_names)

sp.tl.estimate_jacobian(adata, n_top_genes=adata.shape[1], min_shared_counts=0)
grn_estimate = adata.uns["average_jac"]["0"][0][0:n, n:].copy()

grn_correlation = [get_grn_auroc_cc(ground_truth=adata.varm["true_skeleton"].toarray(), estimated=grn_estimate.T)]

# %%
if SAVE_DATA:
    pd.DataFrame({"grn": grn_correlation}).to_parquet(
        path=DATA_DIR / DATASET / "results" / "splicejac_correlation.parquet"
    )

# %%

# %% [markdown]
# # Correlation benchmark on cell cycle
#
# Notebook benchmarks GRN inference using correlation on cell cycling dataset

# %% [markdown]
# ## Library imports

# %%
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
grn_estimate = adata.to_df(layer="Ms").corr().abs().values

grn_correlation = [get_grn_auroc_cc(ground_truth=adata.varm["true_skeleton"].toarray(), estimated=grn_estimate.T)]

# %%
if SAVE_DATA:
    pd.DataFrame({"grn": grn_correlation}).to_parquet(
        path=DATA_DIR / DATASET / "results" / "correlation_correlation.parquet"
    )

# %%

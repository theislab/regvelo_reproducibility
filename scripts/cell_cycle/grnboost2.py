# %% [markdown]
# # GRNBoost2 benchmark on cell cycle
#
# Notebook benchmarks GRN inference using GRNBoost2 on cell cycling dataset

# %% [markdown]
# ## Library imports

# %%
import pandas as pd

import anndata as ad
from arboreto.algo import grnboost2

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
network = grnboost2(expression_data=adata.to_df(layer="Ms"), tf_names=adata.var_names.to_list())
grn_estimate = pd.pivot(network, index="target", columns="TF").fillna(0).values

grn_correlation = [get_grn_auroc_cc(ground_truth=adata.varm["true_skeleton"].toarray(), estimated=grn_estimate.T)]

# %%
if SAVE_DATA:
    pd.DataFrame({"grn": grn_correlation}).to_parquet(
        path=DATA_DIR / DATASET / "results" / "grnboost2_correlation.parquet"
    )

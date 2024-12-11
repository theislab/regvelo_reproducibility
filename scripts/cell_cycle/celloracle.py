# %% [markdown]
# # CellOracle benchmark on cell cycle
#
# Notebook benchmarks GRN inference using CellOracle on cell cycling dataset

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd

import anndata as ad
import celloracle as co

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
base_grn = np.ones((adata.n_vars, adata.n_vars))
base_grn = pd.DataFrame(base_grn, columns=adata.var_names)
base_grn["peak_id"] = "peak_" + adata.var_names
base_grn["gene_short_name"] = adata.var_names
base_grn = base_grn[["peak_id", "gene_short_name"] + adata.var_names.to_list()]

net = co.Net(gene_expression_matrix=adata.to_df(layer="Ms"), TFinfo_matrix=base_grn, verbose=False)
net.fit_All_genes(bagging_number=100, alpha=1, verbose=False)
net.updateLinkList(verbose=False)

grn_estimate = pd.pivot(net.linkList[["source", "target", "coef_mean"]], index="target", columns="source")
grn_estimate = grn_estimate.fillna(0).abs().values

grn_correlation = [get_grn_auroc_cc(ground_truth=adata.varm["true_skeleton"].toarray(), estimated=grn_estimate.T)]

# %%
if SAVE_DATA:
    pd.DataFrame({"grn": grn_correlation}).to_parquet(
        path=DATA_DIR / DATASET / "results" / "celloracle_correlation.parquet"
    )

# %%

# %% [markdown]
# # spliceJAC benchmark on dyngen data
#
# Notebook benchmarks GRN inference using spliceJAC on dyngen-generated data.

# %% [markdown]
# ## Library imports

# %%
import splicejac as sp

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import anndata as ad
import scvi

from rgv_tools import DATA_DIR

# %% [markdown]
# ## General settings

# %%
scvi.settings.seed = 0

# %% [markdown]
# ## Constants

# %%
DATASET = "dyngen"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Velocity pipeline

# %%
grn_correlation = []

for filename in (DATA_DIR / DATASET / "processed").iterdir():
    torch.cuda.empty_cache()
    if filename.suffix != ".zarr":
        continue

    adata = ad.io.read_zarr(filename)
    n = len(adata.var_names)

    grn_true = adata.uns["true_skeleton"]
    grn_sc_true = adata.uns["true_sc_grn"]

    ## We ignore the cell label information and assume all cells is the same label
    adata.obs["clusters"] = "1"
    sp.tl.estimate_jacobian(adata, n_top_genes=adata.shape[1], min_shared_counts=0)
    grn_estimate = adata.uns["average_jac"]["1"][0][0:n, n:].copy()

    grn_auroc = []
    for cell_id in range(adata.n_obs):
        ground_truth = grn_sc_true[:, :, cell_id]

        if ground_truth.sum() > 0:
            ground_truth = ground_truth.T[np.array(grn_true.T) == 1]
            ground_truth[ground_truth != 0] = 1

            estimated = grn_estimate[np.array(grn_true.T) == 1]
            estimated = np.abs(estimated)

            number = min(10000, len(ground_truth))
            estimated, index = torch.topk(torch.tensor(estimated), number)

            grn_auroc.append(roc_auc_score(ground_truth[index], estimated))

    grn_correlation.append(np.mean(grn_auroc))

# %% [markdown]
# ## Data saving

# %%
if SAVE_DATA:
    pd.DataFrame({"grn": grn_correlation}).to_parquet(
        path=DATA_DIR / DATASET / "results" / "splicejac_correlation.parquet"
    )

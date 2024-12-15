# %% [markdown]
# # CellOracle benchmark on dyngen data
#
# Notebook benchmarks GRN inference using CellOracle on dyngen-generated data.

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import anndata as ad
import celloracle as co
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

    grn_true = adata.uns["true_skeleton"]
    grn_sc_true = adata.uns["true_sc_grn"]

    base_grn = np.ones((adata.n_vars, adata.n_vars))
    base_grn = pd.DataFrame(base_grn, columns=adata.var_names)
    base_grn["peak_id"] = "peak_" + adata.var_names
    base_grn["gene_short_name"] = adata.var_names
    base_grn = base_grn[["peak_id", "gene_short_name"] + adata.var_names.to_list()]

    net = co.Net(gene_expression_matrix=adata.to_df(layer="Ms"), TFinfo_matrix=base_grn, verbose=False)
    net.fit_All_genes(bagging_number=100, alpha=1, verbose=False)
    net.updateLinkList(verbose=False)

    grn_estimate = pd.pivot(net.linkList[["source", "target", "coef_mean"]], index="target", columns="source")
    grn_estimate = grn_estimate.fillna(0).values

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
        path=DATA_DIR / DATASET / "results" / "celloracle_correlation.parquet"
    )

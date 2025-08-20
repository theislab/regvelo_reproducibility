# %% [markdown]
# # GRNBoost2 benchmark on dyngen data
#
# Notebook benchmarks GRN inference using GRNBoost2 on dyngen-generated data.

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import anndata as ad
import scvi
from arboreto.algo import grnboost2

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
COMPLEXITY = "complexity_1"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / COMPLEXITY / "results").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Velocity pipeline

# %%
grn_correlation = []

cnt = 0
for filename in (DATA_DIR / DATASET / COMPLEXITY / "processed").iterdir():
    torch.cuda.empty_cache()
    if filename.suffix != ".zarr":
        continue

    simulation_id = int(filename.stem.removeprefix("simulation_"))
    print(f"Run {cnt}, dataset {simulation_id}.")

    adata = ad.io.read_zarr(filename)

    TF = adata.var_names[adata.var["is_tf"]]
    TF_ind = [adata.var_names.get_loc(tf) for tf in TF]

    grn_true = adata.uns["true_skeleton"][:, TF_ind]
    grn_sc_true = adata.uns["true_sc_grn"][:, TF_ind]

    network = grnboost2(expression_data=adata.to_df(layer="Ms"), tf_names=TF.tolist())
    grn_estimate = pd.pivot(network, index="target", columns="TF").fillna(0).values.T

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

            if len(np.unique(ground_truth[index])) < 2:
                print("Skipping cell due to single-class ground truth")
                grn_auroc.append(np.nan)
            else:
                grn_auroc.append(roc_auc_score(ground_truth[index], estimated))

    grn_correlation.append(np.mean(grn_auroc))
    cnt = cnt + 1

# %% [markdown]
# ## Data saving

# %%
if SAVE_DATA:
    pd.DataFrame({"grn": grn_correlation}).to_parquet(
        path=DATA_DIR / DATASET / COMPLEXITY / "results" / "grnboost2_correlation.parquet"
    )

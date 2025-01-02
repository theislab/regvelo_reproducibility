# %% [markdown]
# # RegVelo benchmark on dyngen data
#
# Notebook benchmarks velocity and latent time inference using RegVelo on dyngen-generated data.

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import anndata as ad
import scvi
from regvelo import REGVELOVI

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import (
    get_time_correlation,
    get_velocity_correlation,
    set_output,
)

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
velocity_correlation = []
time_correlation = []
grn_correlation = []

for filename in (DATA_DIR / DATASET / "processed").iterdir():
    torch.cuda.empty_cache()
    if filename.suffix != ".zarr":
        continue

    adata = ad.io.read_zarr(filename)

    W = torch.ones([adata.n_vars, adata.n_vars])
    REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    vae = REGVELOVI(adata, W=W, t_max=20)
    vae.train()

    set_output(adata, vae, n_samples=30, batch_size=adata.n_obs)

    velocity_correlation.append(
        get_velocity_correlation(
            ground_truth=adata.layers["true_velocity"], estimated=adata.layers["velocity"], aggregation=np.mean
        )
    )

    ## calculate per gene latent time correlation
    time_corr = [
        get_time_correlation(ground_truth=adata.obs["true_time"], estimated=adata.layers["fit_t"][:, i])
        for i in range(adata.layers["fit_t"].shape[1])
    ]
    time_correlation.append(np.mean(time_corr))

    grn_true = adata.uns["true_skeleton"]
    grn_sc_true = adata.uns["true_sc_grn"]

    grn_estimate = vae.module.v_encoder.GRN_Jacobian2(torch.tensor(adata.layers["Ms"]).to("cuda:0"))
    grn_estimate = grn_estimate.cpu().detach().numpy()

    grn_auroc = []
    for cell_id in range(adata.n_obs):
        ground_truth = grn_sc_true[:, :, cell_id]

        if ground_truth.sum() > 0:
            ground_truth = ground_truth.T[np.array(grn_true.T) == 1]
            ground_truth[ground_truth != 0] = 1

            estimated = grn_estimate[cell_id, :, :][np.array(grn_true.T) == 1]
            estimated = np.abs(estimated)

            number = min(10000, len(ground_truth))
            estimated, index = torch.topk(torch.tensor(estimated), number)

            grn_auroc.append(roc_auc_score(ground_truth[index], estimated))

    grn_correlation.append(np.mean(grn_auroc))

# %% [markdown]
# ## Data saving

# %%
if SAVE_DATA:
    pd.DataFrame({"velocity": velocity_correlation, "time": time_correlation, "grn": grn_correlation}).to_parquet(
        path=DATA_DIR / DATASET / "results" / "regvelo_correlation.parquet"
    )

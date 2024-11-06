# %% [markdown]
# # RegVelo benchmark on toy GRN
#
# Notbook benchmarks velocity, latent time and GRN inference using RegVelo with toy GRN data.

# %% [markdown]
# ## Library imports

# %%
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

from regvelo import REGVELOVI
from velovi import preprocess_data

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import (
    get_data_subset,
    get_grn_auroc,
    get_time_correlation,
    get_velocity_correlation,
    set_output,
)
from rgv_tools.core import read_as_dask

# %% [markdown]
# ## General settings

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
adata = read_as_dask(
    store=DATA_DIR / "toy_grn" / "raw" / "adata.zarr", layers=["unspliced", "Mu", "spliced", "Ms", "true_velocity"]
)
adata

# %% [markdown]
# ## Velocity pipeline

# %%
velocity_correlation = []
time_correlation = []
grn_correlation = []

parameters = []

for dataset in tqdm(adata.obs["dataset"].cat.categories):
    adata_subset = get_data_subset(
        adata=adata, column="dataset", group=dataset, uns_keys=["true_beta", "true_gamma", "true_K"]
    )
    adata_subset.uns["regulators"] = adata_subset.var.index.values
    adata_subset.uns["targets"] = adata_subset.var.index.values
    adata_subset.uns["skeleton"] = np.ones((adata_subset.n_vars, adata_subset.n_vars))
    adata_subset.uns["network"] = np.ones((adata_subset.n_vars, adata_subset.n_vars))

    # Data preprocessing
    adata_subset = preprocess_data(adata_subset, filter_on_r2=False)

    W = adata_subset.uns["skeleton"].copy()
    W = torch.tensor(W).int()

    REGVELOVI.setup_anndata(adata_subset, spliced_layer="Ms", unspliced_layer="Mu")
    vae = REGVELOVI(adata_subset, W=W, lam2=1, soft_constraint=False, simple_dynamics=True)
    vae.train()

    set_output(adata_subset, vae, n_samples=30, batch_size=adata.n_obs)
    _parameters = adata_subset.var[["fit_beta", "fit_gamma"]].copy()
    _parameters["dataset"] = dataset
    _parameters.index = _parameters.index + f"-dataset_{dataset}"
    parameters.append(_parameters)
    del _parameters

    # estimated_velocity = adata_subset.layers["unspliced"] * adata_subset.var["fit_beta"].values - adata_subset.layers["spliced"] * adata_subset.var["fit_gamma"].values
    velocity_correlation.append(
        get_velocity_correlation(
            ground_truth=adata_subset.layers["true_velocity"],
            estimated=adata_subset.layers["velocity"],
            aggregation=np.mean,
        )
    )
    time_correlation.append(
        get_time_correlation(
            ground_truth=adata_subset.obs["true_time"], estimated=adata_subset.layers["fit_t"].mean(axis=1)
        )
    )

    grn_estimate = vae.module.v_encoder.GRN_Jacobian(torch.tensor(adata_subset.layers["spliced"].mean(0)).to("cuda:0"))
    grn_estimate = grn_estimate.cpu().detach().numpy()
    grn_correlation.append(get_grn_auroc(ground_truth=adata_subset.uns["true_K"], estimated=grn_estimate))

# %% [markdown]
# ## Data saving

# %%
if SAVE_DATA:
    pd.DataFrame({"velocity": velocity_correlation, "time": time_correlation, "grn": grn_correlation}).to_parquet(
        path=DATA_DIR / "toy_grn" / "results" / "regvelo_correlation.parquet"
    )
    pd.concat(parameters).to_parquet(path=DATA_DIR / "toy_grn" / "results" / "regvelo_rates.parquet")

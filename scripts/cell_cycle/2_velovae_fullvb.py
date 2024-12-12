# %% [markdown]
# # veloVAE (fullvb) benchmark on cell cycle data
#
# Notebook benchmarks velocity, latent time inference, and cross boundary correctness using veloVAE (fullvb) on cell cycle data.

# %%
import velovae as vv

import numpy as np
import pandas as pd
import torch

import anndata as ad
import scvelo as scv
from cellrank.kernels import VelocityKernel

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import get_time_correlation

# %% [markdown]
# ## General settings

# %%
scv.settings.verbosity = 3

# %% [markdown]
# ## Constants

# %%
torch.manual_seed(0)
np.random.seed(0)

# %%
DATASET = "cell_cycle"

# %%
STATE_TRANSITIONS = [("G1", "S"), ("S", "G2M")]

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / DATASET / "processed" / "velovae_vae").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
adata = ad.io.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_processed.h5ad")
adata

# %% [markdown]
# ## Velocity pipeline

# %%
rate_prior = {"alpha": (0.0, 1.0), "beta": (0.0, 0.5), "gamma": (0.0, 0.5)}
full_vb = vv.VAE(adata, tmax=20, dim_z=5, device="cuda:0", full_vb=True, rate_prior=rate_prior)
config = {}
full_vb.train(adata, config=config, plot=False, embed="pca")

if SAVE_DATA:
    full_vb.save_anndata(
        adata, "fullvb", DATA_DIR / DATASET / "processed" / "velovae_vae", file_name="velovae_fullvb.h5ad"
    )

# %%
time_correlation = [get_time_correlation(ground_truth=adata.obs["fucci_time"], estimated=adata.obs["fullvb_time"])]

# %%
scv.tl.velocity_graph(adata, vkey="fullvb_velocity", n_jobs=1)
scv.tl.velocity_confidence(adata, vkey="fullvb_velocity")

# %% [markdown]
# ## Cross-boundary correctness

# %%
vk = VelocityKernel(adata, vkey="fullvb_velocity").compute_transition_matrix()

cluster_key = "phase"
rep = "X_pca"

score_df = []
for source, target in STATE_TRANSITIONS:
    cbc = vk.cbc(source=source, target=target, cluster_key=cluster_key, rep=rep)

    score_df.append(
        pd.DataFrame(
            {
                "State transition": [f"{source} - {target}"] * len(cbc),
                "CBC": cbc,
            }
        )
    )
score_df = pd.concat(score_df)

# %% [markdown]
# ## Data saving

# %%
if SAVE_DATA:
    pd.DataFrame({"time": time_correlation}, index=adata.obs_names).to_parquet(
        path=DATA_DIR / DATASET / "results" / "velovae_fullvb_correlation.parquet"
    )
    adata.obs[["velocity_confidence"]].to_parquet(
        path=DATA_DIR / DATASET / "results" / "velovae_fullvb_confidence.parquet"
    )
    score_df.to_parquet(path=DATA_DIR / DATASET / "results" / "velovae_fullvb_cbc.parquet")

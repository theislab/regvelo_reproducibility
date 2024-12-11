# %% [markdown]
# # UniTVelo benchmark on cell cycle data
#
# Notebook benchmarks velocity, latent time inference, and cross boundary correctness using UniTVelo on cell cycle data.

# %%
import os

import pandas as pd

import anndata as ad
import scvelo as scv
import unitvelo as utv
from cellrank.kernels import VelocityKernel

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import get_time_correlation

# %% [markdown]
# ## General settings

# %%
scv.settings.verbosity = 3

# %%
velo_config = utv.config.Configuration()
velo_config.R2_ADJUST = True
velo_config.IROOT = None
velo_config.FIT_OPTION = "1"
velo_config.AGENES_R2 = 1
velo_config.GPU = -1

# %%
os.environ["TF_USE_LEGACY_KERAS"] = "True"

# %% [markdown]
# ## Constants

# %%
DATASET = "cell_cycle"

# %%
STATE_TRANSITIONS = [("G1", "S"), ("S", "G2M")]

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
adata = ad.io.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_processed.h5ad")
scv.pp.moments(adata, n_pcs=None, n_neighbors=None)  ## reconstruct Mu and Ms due to veloVAE run on continues space
adata

# %% [markdown]
# ## Velocity pipeline

# %%
adata.obs["cluster"] = "0"
adata = utv.run_model(adata, label="cluster", config_file=velo_config)

# %%
time_correlation = [get_time_correlation(ground_truth=adata.obs["fucci_time"], estimated=adata.obs["latent_time"])]

# %%
scv.tl.velocity_graph(adata, vkey="velocity", n_jobs=1)
scv.tl.velocity_confidence(adata, vkey="velocity")

# %% [markdown]
# ## Cross-boundary correctness

# %%
vk = VelocityKernel(adata, vkey="velocity").compute_transition_matrix()

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
        path=DATA_DIR / DATASET / "results" / "unitvelo_correlation.parquet"
    )
    adata.obs[["velocity_confidence"]].to_parquet(path=DATA_DIR / DATASET / "results" / "unitvelo_confidence.parquet")
    score_df.to_parquet(path=DATA_DIR / DATASET / "results" / "unitvelo_cbc.parquet")

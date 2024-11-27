# %% [markdown]
# # RegVelo benchmark on cell cycle data
#
# Notebook benchmarks velocity, latent time and GRN inference, and cross boundary correctness using RegVelo on cell cycle data.

# %% [markdown]
# ## Library imports

# %%
import pandas as pd
import torch

import anndata as ad
import scvelo as scv
from cellrank.kernels import VelocityKernel
from regvelo import REGVELOVI

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import get_grn_auroc, get_time_correlation, set_output

# %% [markdown]
# ## General settings

# %%
scv.settings.verbosity = 3

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
adata

# %% [markdown]
# ## Velocity pipeline

# %%
W = torch.ones((adata.n_vars, adata.n_vars), dtype=int)

REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
vae = REGVELOVI(adata, W=W)
vae.train()

# %%
set_output(adata, vae, n_samples=30, batch_size=adata.n_obs)

# %%
time_correlation = [
    get_time_correlation(ground_truth=adata.obs["fucci_time"], estimated=adata.layers["fit_t"].mean(axis=1))
]

# %%
grn_estimate = vae.module.v_encoder.GRN_Jacobian(torch.tensor(adata.layers["spliced"].mean(0)).to("cuda:0"))
grn_estimate = grn_estimate.cpu().detach().numpy()
grn_correlation = [get_grn_auroc(ground_truth=adata.varm["true_skeleton"].toarray(), estimated=grn_estimate)]

# %%
scv.tl.velocity_graph(adata, vkey="velocity", n_jobs=1)
scv.tl.velocity_confidence(adata, vkey="velocity")

# %% [markdown]
# ## Cross-boundary correctness

# %%
vk = VelocityKernel(adata).compute_transition_matrix()

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
        path=DATA_DIR / DATASET / "results" / "velovi_correlation.parquet"
    )
    adata.obs[["velocity_confidence"]].to_parquet(path=DATA_DIR / DATASET / "results" / "velovi_confidence.parquet")
    score_df.to_parquet(path=DATA_DIR / DATASET / "results" / "velovi_cbc.parquet")

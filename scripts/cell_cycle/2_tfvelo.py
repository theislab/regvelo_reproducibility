# %% [markdown]
# # TFvelo benchmark on cell cycle data
#
# Notebook benchmarks velocity, latent time inference, and cross boundary correctness using TFvelo on cell cycle data.

# %% [markdown]
# ## Library imports

# %%
import TFvelo as TFv

import numpy as np
import pandas as pd

import anndata as ad
import scvelo as scv
from cellrank.kernels import VelocityKernel

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import get_grn_auroc_cc, get_time_correlation

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
#
# TFvelo preprocessing

# %%
if "spliced" in adata.layers:
    adata.layers["total"] = adata.layers["spliced"].todense() + adata.layers["unspliced"].todense()
elif "new" in adata.layers:
    adata.layers["total"] = np.array(adata.layers["total"].todense())
else:
    adata.layers["total"] = adata.X
adata.layers["total_raw"] = adata.layers["total"].copy()
n_cells, n_genes = adata.X.shape

# %%
TFv.pp.moments(adata, n_pcs=30)

TFv.pp.get_TFs(adata, databases="all")
adata.uns["genes_pp"] = np.array(adata.var_names)

# %%
TFv.tl.recover_dynamics(
    adata,
    n_jobs=4,
    max_iter=20,
    var_names="all",
    WX_method="lsq_linear",
    WX_thres=20,
    n_top_genes=2000,
    fit_scaling=True,
    use_raw=0,
    init_weight_method="correlation",
    n_time_points=1000,
)

# %%
adata.layers["fit_t"] = np.nan_to_num(adata.layers["fit_t"], nan=0)
time_correlation = [
    get_time_correlation(ground_truth=adata.obs["fucci_time"], estimated=adata.layers["fit_t"].mean(axis=1))
]

# %%
grn_estimate = pd.DataFrame(0, index=adata.var_names, columns=adata.var_names)
grn_estimate.loc[:, adata.uns["all_TFs"]] = adata.varm["fit_weights_final"]
grn_estimate = np.array(grn_estimate)
grn_correlation = [
    get_grn_auroc_cc(ground_truth=adata.varm["true_skeleton"].toarray(), estimated=np.abs(grn_estimate).T)
]

# %%
scv.tl.velocity_graph(adata, vkey="velocity", n_jobs=1)
scv.tl.velocity_confidence(adata, vkey="velocity")

# %% [markdown]
# ## Cross-boundary correctness

# %%
vk = VelocityKernel(adata, vkey="velocity", xkey="M_total").compute_transition_matrix()

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
        path=DATA_DIR / DATASET / "results" / "tfvelo_correlation.parquet"
    )
    pd.DataFrame({"grn": grn_correlation}).to_parquet(
        path=DATA_DIR / DATASET / "results" / "tfvelo_grn_correlation.parquet"
    )
    adata.obs[["velocity_confidence"]].to_parquet(path=DATA_DIR / DATASET / "results" / "tfvelo_confidence.parquet")
    score_df.to_parquet(path=DATA_DIR / DATASET / "results" / "tfvelo_cbc.parquet")

# %% [markdown]
# # RegVelo-based analyis of zebrafish data
#
# Notebook runs RegVelo on zebrafish dataset.

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd
import scipy
import torch

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

import cellrank as cr
import scanpy as sc
import scvi
from regvelo import REGVELOVI

from rgv_tools import DATA_DIR, FIG_DIR
from rgv_tools.benchmarking import set_output

# %% [markdown]
# ## General settings

# %%
scvi.settings.seed = 0

# %% [markdown]
# ## Constants

# %%
DATASET = "zebrafish"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)

# %%
SAVE_FIGURES = False
if SAVE_FIGURES:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)

# %%
TERMINAL_STATES = [
    "mNC_head_mesenchymal",
    "mNC_arch2",
    "mNC_hox34",
    "Pigment",
]


# %% [markdown]
# ## Function definitions


# %%
def min_max_scaling(data):
    """Compute min and max values for each feature."""
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)

    # Perform min-max scaling
    scaled_data = (data - min_vals) / (max_vals - min_vals)

    return scaled_data


# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed.h5ad")

# %% [markdown]
# ## Velocity pipeline

# %%
# Prepare skeleton
W = adata.uns["skeleton"].copy()
W = torch.tensor(np.array(W)).int()

# Prepare TF
TF = adata.var_names[adata.var["TF"]]

# Prepare model
REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
vae = REGVELOVI(adata, W=W.T, regulators=TF, soft_constraint=False)

# %%
vae.train()

# %%
set_output(adata, vae, n_samples=30, batch_size=adata.n_obs)

# %% [markdown]
# ## Latent time analysis

# %%
adata.obs["latent_time"] = min_max_scaling(adata.layers["fit_t"].mean(axis=1))

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sc.pl.umap(adata=adata, color="latent_time", title="", frameon=False, legend_fontsize=14, cmap="magma", ax=ax)

    if SAVE_FIGURES:
        fig.savefig(FIG_DIR / DATASET / "latent_time.svg", format="svg", transparent=True, bbox_inches="tight")

# %%
df = pd.DataFrame({"stage": adata.obs["stage"].tolist(), "latent_time": adata.obs["latent_time"].tolist()})

# %%
adata.obs["stage_num"] = 0
adata.obs["stage_num"][adata.obs["stage"].isin(["3ss"])] = 3
adata.obs["stage_num"][adata.obs["stage"].isin(["6-7ss"])] = 6.5
adata.obs["stage_num"][adata.obs["stage"].isin(["10ss"])] = 10
adata.obs["stage_num"][adata.obs["stage"].isin(["12-13ss"])] = 12.5
adata.obs["stage_num"][adata.obs["stage"].isin(["17-18ss"])] = 17.5
adata.obs["stage_num"][adata.obs["stage"].isin(["21-22ss"])] = 21.5

# %%
scipy.stats.spearmanr(adata.obs["stage_num"].tolist(), adata.obs["latent_time"].tolist())

# %%
loc = [0, 1, 2, 3, 4, 5]
label = ["3", "6-7", "10", "12-13", "17-18", "21-22"]

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["3ss", "6-7ss", "10ss", "12-13ss", "17-18ss", "21-22ss"]
    flierprops = {"marker": "D", "markerfacecolor": "black", "markersize": 3, "linestyle": "none"}
    sns.boxplot(data=df, x="stage", y="latent_time", order=order, color="grey", ax=ax, flierprops=flierprops)
    # Set labels and title
    plt.ylabel("Stage", fontsize=16)
    plt.xlabel("Estimated latent time", fontsize=16)

    plt.xticks(ticks=loc, labels=label)  # Replace ticks with new labels
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.show()

    if SAVE_FIGURES:
        fig.savefig(FIG_DIR / DATASET / "latent_time_boxplot.svg", format="svg", transparent=True, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## CellRank pipeline

# %%
vk = cr.kernels.VelocityKernel(adata)
vk.compute_transition_matrix()
ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()

kernel = 0.8 * vk + 0.2 * ck

# %%
estimator = cr.estimators.GPCCA(kernel)
estimator.compute_macrostates(n_states=8, n_cells=30, cluster_key="cell_type")
estimator.set_terminal_states(TERMINAL_STATES)

# %%
estimator.plot_macrostates(which="terminal", basis="umap", legend_loc="right", s=100)

# %% [markdown]
# ## Save dataset

# %%
if SAVE_DATA:
    adata.write_h5ad(DATA_DIR / DATASET / "processed" / "adata_run_regvelo.h5ad")
    vae.save(DATA_DIR / DATASET / "processed" / "rgv_model")

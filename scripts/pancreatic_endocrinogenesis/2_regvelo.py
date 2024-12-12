# %% [markdown]
# # RegVelo-based analyis of pancreatic endocrine data
#
# Notebook runs RegVelo on pancreatic endocrine dataset.

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import torch

import matplotlib.pyplot as plt
import mplscience

import cellrank as cr
import scanpy as sc
import scvelo as scv
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
DATASET = "pancreatic_endocrine"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)

# %%
SAVE_FIGURES = False
if SAVE_FIGURES:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)

# %%
TERMINAL_STATES = ["Alpha", "Beta", "Delta", "Epsilon"]


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
TF = adata.var_names[adata.var["tf"]]

# Prepare model
REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
vae = REGVELOVI(adata, W=W.T, regulators=TF)

# %%
vae.train()

# %%
set_output(adata, vae, n_samples=30, batch_size=adata.n_obs)

# %% [markdown]
# ## Visualize trajectory

# %%
scv.tl.velocity_graph(adata)

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.velocity_embedding_stream(adata, basis="umap", title="", legend_loc="lower right", ax=ax)

if SAVE_FIGURES:
    fig.savefig(FIG_DIR / DATASET / "intro_vector_field.svg", format="svg", transparent=True, bbox_inches="tight")

# %% [markdown]
# ## CellRank pipeline

# %%
vk = cr.kernels.VelocityKernel(adata)
vk.compute_transition_matrix()
ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()

kernel = 0.8 * vk + 0.2 * ck

# %%
estimator = cr.estimators.GPCCA(kernel)
estimator.compute_macrostates(n_states=7, cluster_key="clusters")
estimator.set_terminal_states(TERMINAL_STATES)

# %%
estimator.plot_macrostates(which="terminal", basis="umap", legend_loc="right", s=100)

# %% [markdown]
# ## Data saving

# %%
if SAVE_DATA:
    adata.write_h5ad(DATA_DIR / DATASET / "processed" / "adata_run_regvelo.h5ad")
    vae.save(DATA_DIR / DATASET / "processed" / "rgv_model")

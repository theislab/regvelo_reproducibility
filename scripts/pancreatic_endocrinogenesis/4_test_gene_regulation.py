# %% [markdown]
# # Regulation is important for terminal state identification in pancreatic endocrine
#
# Test the gene regulation roles in predicting terminal states

# %% [markdown]
# ## Library imports

# %%
import random

import pandas as pd
import torch

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

import cellrank as cr
import scanpy as sc
import scvelo as scv
from regvelo import REGVELOVI

from rgv_tools import DATA_DIR, FIG_DIR
from rgv_tools.benchmarking import set_output

# %% [markdown]
# ## General settings

# %%
plt.rcParams["svg.fonttype"] = "none"
sns.reset_defaults()
sns.reset_orig()
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=14, color_map="viridis")

# %% [markdown]
# ## Constants

# %%
DATASET = "pancreatic_endocrinogenesis"

# %%
SAVE_FIGURES = True
if SAVE_FIGURES:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)

# %%
TERMINAL_STATES = ["Beta", "Alpha", "Delta", "Epsilon"]

# %%
N_STATES = 7

# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed.h5ad")

# %% [markdown]
# ## Model loading

# %%
vae = REGVELOVI.load(DATA_DIR / DATASET / "processed" / "rgv_model", adata)

# %% [markdown]
# ## Delete regulation

# %%
adata_no_regulation = adata.copy()
vae.module.v_encoder.fc1.weight.data = vae.module.v_encoder.fc1.weight.data * 0
set_output(adata_no_regulation, vae, n_samples=30, batch_size=adata.n_obs)

# %%
scv.tl.velocity_graph(adata_no_regulation)

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(4, 3))
    scv.pl.velocity_embedding_stream(adata_no_regulation, basis="umap", title="", legend_loc=False, ax=ax)

if SAVE_FIGURES:
    fig.savefig(
        FIG_DIR / DATASET / "no_regulation_vector_field.svg", format="svg", transparent=True, bbox_inches="tight"
    )

# %% [markdown]
# ### Predict terminal states

# %%
vk = cr.kernels.VelocityKernel(adata_no_regulation)
vk.compute_transition_matrix()
ck = cr.kernels.ConnectivityKernel(adata_no_regulation).compute_transition_matrix()
estimator = cr.estimators.GPCCA(0.8 * vk + 0.2 * ck)
## evaluate the fate prob on original space
estimator.compute_macrostates(n_states=N_STATES, cluster_key="clusters")

# %%
estimator.set_terminal_states(list(set(estimator.macrostates.cat.categories.tolist()).intersection(TERMINAL_STATES)))

# %%
estimator.plot_macrostates(which="terminal", basis="umap", legend_loc="right", s=100)

# %% [markdown]
# ## Randomize regulation

# %%
vae = REGVELOVI.load(DATA_DIR / DATASET / "processed" / "rgv_model", adata)
w = vae.module.v_encoder.fc1.weight.data.detach().clone().cpu().numpy()

# Shuffle genes to randomize weights
gene = adata.var.index.tolist()
random.shuffle(gene)
w = pd.DataFrame(w, index=gene, columns=gene)
w = w.loc[adata.var.index, adata.var.index]

# Convert back to tensor and move to GPU
w = torch.tensor(w.values, device="cuda:0")

# %%
vae.module.v_encoder.fc1.weight.data = w

# %%
adata_random_regulation = adata.copy()
set_output(adata_random_regulation, vae, n_samples=30, batch_size=adata.n_obs)

# %%
scv.tl.velocity_graph(adata_random_regulation)

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(4, 3))
    scv.pl.velocity_embedding_stream(adata_random_regulation, basis="umap", title="", legend_loc=False, ax=ax)

if SAVE_FIGURES:
    fig.savefig(
        FIG_DIR / DATASET / "randomize_regulation_vector_field.svg", format="svg", transparent=True, bbox_inches="tight"
    )

# %% [markdown]
# ### Predict terminal states

# %%
vk = cr.kernels.VelocityKernel(adata_random_regulation).compute_transition_matrix()
ck = cr.kernels.ConnectivityKernel(adata_random_regulation).compute_transition_matrix()
estimator = cr.estimators.GPCCA(0.8 * vk + 0.2 * ck)
## evaluate the fate prob on original space
estimator.compute_macrostates(n_states=N_STATES, cluster_key="clusters")

# %%
estimator.set_terminal_states(list(set(estimator.macrostates.cat.categories.tolist()).intersection(TERMINAL_STATES)))

# %%
estimator.plot_macrostates(which="terminal", basis="umap", legend_loc="right", s=100)

# %% [markdown]
# # scVelo-based analysis of hematopoiesis dataset
#
# Notebook runs the scvelo model on the hematopoiesis dataset.

# %% [markdown]
# ## Library imports

# %%
import numpy as np

import anndata as ad
import cellrank as cr
import scanpy as sc
import scvelo as scv

from rgv_tools import DATA_DIR

# %% [markdown]
# ## General settings

# %%
sc.settings.verbosity = 2
scv.settings.verbosity = 3

# %% [markdown]
# ## Constants

# %%
DATASET = "hematopoiesis"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

# %%
TERMINAL_STATES = ["Mon", "Meg", "Bas", "Ery"]

# %% [markdown]
# ## Data loading

# %%
adata = ad.io.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed.h5ad")
adata_full = ad.io.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed_full.h5ad")

# %% [markdown]
# ## Run scVelo

# %%
velocity_genes = adata.var["velocity_genes"].copy()

# %%
scv.tl.recover_dynamics(adata, fit_scaling=False, var_names=adata.var_names)
adata.var["fit_scaling"] = 1.0

# %%
scv.tl.velocity(adata, mode="dynamical", min_likelihood=-np.inf, min_r2=None)

# %%
adata.var["velocity_genes"] = velocity_genes

# %% [markdown]
# ## Calculate lineage fate probabilities and identify fate-associated genes

# %%
vk = cr.kernels.VelocityKernel(adata)
vk.compute_transition_matrix()
estimator = cr.estimators.GPCCA(vk)  ## We used vk here due to we want to benchmark on velocity

estimator.compute_macrostates(n_states=5, cluster_key="cell_type")
estimator.set_terminal_states(TERMINAL_STATES)

estimator.compute_fate_probabilities()
estimator.adata = adata_full.copy()
scv_ranking = estimator.compute_lineage_drivers(return_drivers=True, cluster_key="cell_type")

scv_ranking = scv_ranking.loc[:, ["Ery_corr", "Mon_corr", "Ery_pval", "Mon_pval"]]

# %% [markdown]
# ## Save dataset

# %% [markdown]
# Recalculate PCA for downstream CBC computation, as velocity is derived from the moment matrices.

# %%
sc.tl.pca(adata, layer="Ms")

# %% [markdown]
# Save adata with velocity layer

# %%
if SAVE_DATA:
    adata.write_h5ad(DATA_DIR / DATASET / "processed" / "adata_run_scvelo.h5ad")

# %% [markdown]
# Save uncertainty and gene ranking results

# %%
if SAVE_DATA:
    scv_ranking.to_csv(DATA_DIR / DATASET / "results" / "scv_ranking.csv")

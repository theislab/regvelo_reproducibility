# %% [markdown]
# # scVelo-based analyis of pancreatic endocrine data
#
# Notebook runs scVelo's dynamical model on pancreatic endocrine dataset.

# %% [markdown]
# ## Library imports

# %%
import numpy as np

import scanpy as sc
import scvelo as scv

from rgv_tools import DATA_DIR

# %% [markdown]
# ## Constants

# %%
DATASET = "pancreatic_endocrinogenesis"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed_filtered.h5ad")

# %% [markdown]
# ## Velocity pipeline

# %%
scv.tl.recover_dynamics(adata, fit_scaling=False, var_names=adata.var_names)
adata.var["fit_scaling"] = 1.0

# %%
scv.tl.velocity(adata, mode="dynamical", min_likelihood=-np.inf, min_r2=None)

# %% [markdown]
# ## Save dataset

# %%
if SAVE_DATA:
    adata.write_h5ad(DATA_DIR / DATASET / "processed" / "adata_run_scvelo.h5ad")

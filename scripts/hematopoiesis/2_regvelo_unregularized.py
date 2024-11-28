# %% [markdown]
# # RegVelo-based analysis of hematopoiesis dataset
#
# Notebook runs the regvelo model on the hematopoiesis dataset.

# %% [markdown]
# ## Library imports

# %%
import scvi

import numpy as np
import torch

import scanpy as sc
from regvelo import REGVELOVI

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import set_output

# %% [markdown]
# ## General settings

# %%
scvi.settings.seed = 0

# %% [markdown]
# ## Constants

# %%
DATASET = "hematopoiesis"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed.h5ad")

# %% [markdown]
# ## Run RegVelo

# %%
## prepare skeleton
W = adata.uns["skeleton"].copy()
W = torch.tensor(np.array(W)).int()

## prepare TF
TF = adata.var_names[adata.var["TF"]]

## prepare model
REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
vae = REGVELOVI(adata, W=W.T, regulators=TF)

# %%
vae.train()

# %%
set_output(adata, vae, n_samples=30, batch_size=adata.n_obs)

# %% [markdown]
# ## Save dataset

# %% [markdown]
# Recalculate the PCA downstream CBC calculation because velocity is calculated from the moment matrices

# %%
sc.tl.pca(adata, layer="Ms")

# %% [markdown]
# Save adata with velocity layer

# %%
if SAVE_DATA:
    adata.write_h5ad(DATA_DIR / DATASET / "processed" / "adata_run_regvelo_unregularized.h5ad")

# %%

# %%
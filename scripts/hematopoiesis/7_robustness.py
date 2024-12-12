# %% [markdown]
# # Repeat run RegVelo and veloVI
#
# Run ten repeats to evaluate estimation robustness of regvelo and velovi

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import torch

import scanpy as sc
import scvi
from regvelo import REGVELOVI
from velovi import VELOVI

from rgv_tools import DATA_DIR, FIG_DIR
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

# %%
SAVE_FIGURES = True
if SAVE_FIGURES:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)

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

# %%
# Train model 10 times
for nrun in range(10):
    ## Running regvelo
    REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    vae = REGVELOVI(adata, W=W.T, regulators=TF, lam2=1)
    vae.train()
    set_output(adata, vae, n_samples=30, batch_size=adata.n_obs)

    path = DATA_DIR / DATASET / "processed"
    adata_name = "rgv_adata_runs_" + str(nrun) + ".h5ad"
    adata.write_h5ad(path / adata_name)

    ## Running veloVI
    VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    vae = VELOVI(adata)
    vae.train()
    set_output(adata, vae, n_samples=30)

    path = DATA_DIR / DATASET / "processed"
    vi_model_name = "vi_adata_runs_" + str(nrun) + ".h5ad"
    adata.write_h5ad(path / vi_model_name)

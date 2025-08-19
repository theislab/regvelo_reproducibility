# %% [markdown]
# # RegVelo-based analysis on murine NCC cell
#
# We run RegVelo on each scale dataset.

# %%
import numpy as np
import torch

import scanpy as sc
import scvi
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
DATASET = "mouse_neural_crest"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Function definations


# %%
def min_max_scaling(data):
    """Compute min and max values for each feature."""
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)

    # Perform min-max scaling
    scaled_data = (data - min_vals) / (max_vals - min_vals)

    return scaled_data


# %% [markdown]
# ## Velocity pipeline

# %%
for i in range(1, 5):
    adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / f"adata_stage{i}_processed_velo_all_regulons.h5ad")

    # Prepare skeleton
    scvi.settings.seed = 0
    W = adata.uns["skeleton"].copy()
    W = torch.tensor(np.array(W)).int()

    # Prepare TF
    TF = adata.var_names[adata.var["TF"]]

    # Prepare model
    REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    vae = REGVELOVI(adata, W=W.T, regulators=TF)

    vae.train()
    set_output(adata, vae, n_samples=30, batch_size=adata.n_obs)

    if SAVE_DATA:
        adata.write_h5ad(DATA_DIR / DATASET / "processed" / f"adata_run_stage_{i}_regvelo_all_regulons.h5ad")
        vae.save(DATA_DIR / DATASET / "processed" / f"rgv_run_stage_{i}_all_regulons")

# %%

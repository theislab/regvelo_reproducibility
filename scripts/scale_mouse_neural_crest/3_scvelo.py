# %% [markdown]
# # scVelo application on murine neural crest data

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
DATASET = "mouse_neural_crest"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Velocity pipeline

# %%
for i in range(1, 5):
    adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / f"adata_stage{i}_processed_velo_all_regulons.h5ad")

    scv.tl.recover_dynamics(adata, fit_scaling=False, var_names=adata.var_names)
    adata.var["fit_scaling"] = 1.0

    scv.tl.velocity(adata, mode="dynamical", min_likelihood=-np.inf, min_r2=None)

    if SAVE_DATA:
        adata.write_h5ad(DATA_DIR / DATASET / "processed" / f"adata_run_stage_{i}_scvelo_all_regulons.h5ad")

# %%

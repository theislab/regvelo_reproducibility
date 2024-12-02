# %% [markdown]
# # Repeative run RegVelo for perturbation analysis

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import scvelo as scv
from regvelo import REGVELOVI

from rgv_tools import DATA_DIR, FIG_DIR

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
DATASET = "pancreatic_endocrine"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / DATASET / "processed" / "cell_cycle_repeat_runs").mkdir(parents=True, exist_ok=True)

# %%
SAVE_FIGURES = False
if SAVE_FIGURES:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed_filtered.h5ad")
TF = adata.var_names[adata.var["tf"]]

# Prepare skeleton
W = adata.uns["skeleton"].copy()
W = torch.tensor(np.array(W)).int()

# Prepare TF
TF = adata.var_names[adata.var["tf"]]

# %% [markdown]
# ### Repeat run model
#
# Under `soft_mode` due to the number of gene regulation parameter need to be estimated, we can repeat run models for five times, and aggregate inferred GRN to get robust estimation

# %%
## repeat models
for nrun in range(15):
    print("training model...")
    REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    vae = REGVELOVI(adata, W=W.T, regulators=TF)
    vae.train()

    print("save model...")
    model_name = "rgv_model_" + str(nrun)
    model = DATA_DIR / DATASET / "processed" / "perturb_repeat_runs" / model_name
    vae.save(model)

# %%

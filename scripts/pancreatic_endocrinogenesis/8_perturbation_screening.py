# %% [markdown]
# # Perturbation prediction on pancreatic endocrine via RegVelo
#
# Using RegVelo to perform perturbation prediction, under `soft_mode` each run will repeat run 15 models and aggregate prediction results

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd
import torch

import scanpy as sc
import scvi

from rgv_tools import DATA_DIR
from rgv_tools.perturbation import get_list_name, TFScanning

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
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

# %%
TERMINAL_STATES = ["Alpha", "Beta", "Delta", "Epsilon"]

# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed.h5ad")

# %% [markdown]
# ## Perturbation screening

# %%
## prepare skeleton
W = adata.uns["skeleton"].copy()
W = torch.tensor(np.array(W)).int()

## prepare TF
TF = adata.var_names[adata.var["TF"]]

# %%
### repeat run the model to get aggregate performance
for nrun in range(0, 15):
    model_name = "rgv_model_" + str(nrun)
    coef_name = "coef_" + str(nrun)
    pval_name = "pval_" + str(nrun)

    model = DATA_DIR / DATASET / "processed" / "perturb_repeat_runs" / model_name
    coef_save = DATA_DIR / DATASET / "results" / coef_name
    pval_save = DATA_DIR / DATASET / "results" / pval_name

    print("inferring perturbation...")

    perturb_screening = TFScanning(model, adata, 7, "clusters", TERMINAL_STATES, TF, 0, method="t-statistics")
    coef = pd.DataFrame(np.array(perturb_screening["coefficient"]))
    coef.index = perturb_screening["TF"]
    coef.columns = get_list_name(perturb_screening["coefficient"][0])

    pval = pd.DataFrame(np.array(perturb_screening["pvalue"]))
    pval.index = perturb_screening["TF"]
    pval.columns = get_list_name(perturb_screening["pvalue"][0])

    rows_with_nan = coef.isna().any(axis=1)
    # Set all values in those rows to NaN
    coef.loc[rows_with_nan, :] = np.nan
    pval.loc[rows_with_nan, :] = np.nan

    coef.to_csv(coef_save)
    pval.to_csv(pval_save)

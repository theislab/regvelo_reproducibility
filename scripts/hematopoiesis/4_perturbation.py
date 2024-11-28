# %% [markdown]
# # Perturbation prediction on hematopoiesis dataset via RegVelo
#
# Using RegVelo to perform perturbation prediction, under `soft_mode` each run will repeat run 5 models and aggregate prediction results

# %% [markdown]
# ## Library imports

# %%
import shutil

import numpy as np
import pandas as pd
import torch

import scanpy as sc
import scvi
from regvelo import REGVELOVI

from rgv_tools import DATA_DIR, FIG_DIR
from rgv_tools.perturbation import get_list_name, TFScanning

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
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

SAVE_FIGURES = True
if SAVE_FIGURES:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)

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
terminal_states = ["Meg", "Mon", "Bas", "Ery"]
for nrun in range(0, 15):
    print("training model...")
    REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    vae = REGVELOVI(adata, W=W.T, regulators=TF, lam2=1)

    torch.cuda.empty_cache()
    vae.train()

    print("save model...")

    model_name = "rgv_model_" + str(nrun)
    coef_name = "coef_" + str(nrun)
    pval_name = "pval_" + str(nrun)

    model = DATA_DIR / DATASET / "processed" / "perturb_repeat_runs" / model_name
    coef_save = DATA_DIR / DATASET / "results" / coef_name
    pval_save = DATA_DIR / DATASET / "results" / pval_name

    vae.save(model)

    print("inferring perturbation...")

    while True:
        try:
            perturb_screening = TFScanning(model, adata, 6, "cell_type", terminal_states, TF, 0)
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

            break
        except:  # noqa E722
            # If an error is raised, increment a and try again, and need to recompute double knock-out reults
            print("perturbation screening has error, retraining model...")
            shutil.rmtree(model)
            REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
            vae = REGVELOVI(adata, W=W.T, regulators=TF, lam2=1)
            vae.train()
            print("save model...")
            vae.save(model)

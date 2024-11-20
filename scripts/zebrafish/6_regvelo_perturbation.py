# %% [markdown]
# # Using RegVelo to predict perturbation

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd

import scanpy as sc

from rgv_tools import DATA_DIR
from rgv_tools.perturbation import (
    get_list_name,
    Multiple_TFScanning,
    split_elements,
    TFScanning,
)

# %% [markdown]
# ## Constants

# %%
DATASET = "zebrafish"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed.h5ad")

# %% [markdown]
# ## Perturbation prediction (single)

# %%
terminal_states = [
    "mNC_head_mesenchymal",
    "mNC_arch2",
    "mNC_hox34",
    "Pigment",
]

gene_list = ["elk3", "erf", "etv2", "fli1a", "mitfa", "nr2f5", "rarga", "rxraa", "smarcc1a", "tfec", "nr2f2"]
gene_list = set(gene_list).intersection(adata.var_names)
gene_list = list(gene_list)

# %%
for nrun in range(3):
    model_name = "rgv_model_" + str(nrun)
    coef_name = "coef_single_regvelo_" + str(nrun)
    pval_name = "pval_single_regvelo_" + str(nrun)

    model = DATA_DIR / DATASET / "processed" / "perturb_repeat_runs" / model_name
    coef_save = DATA_DIR / DATASET / "results" / coef_name
    pval_save = DATA_DIR / DATASET / "results" / pval_name

    ## Perturbation
    d = TFScanning(model, adata, 8, "cell_type", terminal_states, gene_list, 0)

    coef = pd.DataFrame(np.array(d["coefficient"]))
    coef.index = d["TF"]
    coef.columns = get_list_name(d["coefficient"][0])
    pval = pd.DataFrame(np.array(d["pvalue"]))
    pval.index = d["TF"]
    pval.columns = get_list_name(d["pvalue"][0])
    coef = coef.loc[gene_list, :]

    coef.to_csv(coef_save)
    pval.to_csv(pval_save)

# %% [markdown]
# ## Perturbation prediction (Multiple)

# %%
multiple_ko = ["fli1a_elk3", "mitfa_tfec", "tfec_mitfa_bhlhe40", "fli1a_erf_erfl3", "erf_erfl3"]
multiple_ko_list = split_elements(multiple_ko)

# %%
for nrun in [0, 1, 2]:
    model_name = "rgv_model_" + str(nrun)
    coef_name = "coef_multiple_regvelo_" + str(nrun)
    pval_name = "pval_multiple_regvelo_" + str(nrun)

    model = DATA_DIR / DATASET / "processed" / "perturb_repeat_runs" / model_name
    coef_save = DATA_DIR / DATASET / "results" / coef_name
    pval_save = DATA_DIR / DATASET / "results" / pval_name

    ## Perturbatiom
    d = Multiple_TFScanning(model, adata, 8, "cell_type", terminal_states, multiple_ko_list, 0)
    coef = pd.DataFrame(np.array(d["coefficient"]))
    coef.index = d["TF"]
    coef.columns = get_list_name(d["coefficient"][0])
    pval = pd.DataFrame(np.array(d["pvalue"]))
    pval.index = d["TF"]
    pval.columns = get_list_name(d["pvalue"][0])

    coef.to_csv(coef_save)
    pval.to_csv(pval_save)

# %% [markdown]
# # Dynamo-based perturbation analysis

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd

import dynamo as dyn
import scanpy as sc
from dynamo.preprocessing import Preprocessor

from rgv_tools import DATA_DIR
from rgv_tools.perturbation import (
    delta_to_probability,
    density_likelihood_dyn,
    get_list_name,
    Multiple_TFScanning_perturbation_dyn,
    split_elements,
    TFScanning_perturbation_dyn,
)

# %% [markdown]
# ## Constants

# %%
DATASET = "zebrafish"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

# %%
genes = ["nr2f5", "sox9b", "twist1b", "ets1"]

TERMINAL_STATES_KO = [
    "mNC_head_mesenchymal",
    "mNC_arch2",
    "mNC_hox34",
    "Pigment_gch2",
]

TERMINAL_STATES_PERTURB = [
    "mNC_head_mesenchymal",
    "mNC_arch2",
    "mNC_hox34",
    "Pigment",
]

# %%
single_ko = ["rarga", "rxraa", "nr2f5", "fli1a", "tfec", "elk3", "mitfa", "ets1", "nr2f2", "elf1", "ebf3a"]
multiple_ko = ["fli1a_elk3", "tfec_mitfa_bhlhe40", "mitfa_tfec", "mitfa_tfec_tfeb"]

# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed_cr.h5ad")

# %%
adata.X = adata.layers["matrix"].copy()

# %% [markdown]
# ## Processing by dynamo

# %%
preprocessor = Preprocessor()
preprocessor.preprocess_adata(adata, recipe="monocle")

# %%
dyn.tl.dynamics(adata)

# %%
dyn.tl.reduceDimension(adata)

# %%
dyn.tl.cell_velocities(adata, basis="pca")

# %%
dyn.vf.VectorField(adata, basis="pca")

# %%
adata_perturb = adata.copy()

# %% [markdown]
# ## Perturbation prediction
#
# Function based perturbation

# %%
single_ko = set(single_ko).intersection(adata.var_names)
single_ko = list(single_ko)

# %%
start_indices = np.where(adata.obs["cell_type"].isin(["NPB_nohox"]))[0]

# %%
cand_list = single_ko + multiple_ko

# %%
dl_score_all = []
dl_sig_all = []
for TF in cand_list:
    TF_list = split_elements([TF])[0]
    dl_score, dl_sig, _, _ = density_likelihood_dyn(
        adata, TF_list, start_indices, TERMINAL_STATES_KO, n_simulations=1000
    )
    dl_score_all.append(dl_score)
    dl_sig_all.append(dl_sig)

# %%
pred_m_single = pd.DataFrame(
    np.array(dl_score_all[: len(single_ko)]), index=cand_list[: len(single_ko)], columns=TERMINAL_STATES_KO
)
pred_m_multiple = pd.DataFrame(
    np.array(dl_score_all[len(single_ko) :]), index=cand_list[len(single_ko) :], columns=TERMINAL_STATES_KO
)

# %%
pval_m_single = pd.DataFrame(
    np.array(dl_sig_all[: len(single_ko)]), index=cand_list[: len(single_ko)], columns=TERMINAL_STATES_KO
)
pval_m_multiple = pd.DataFrame(
    np.array(dl_sig_all[len(single_ko) :]), index=cand_list[len(single_ko) :], columns=TERMINAL_STATES_KO
)

# %%
## Perform KO single screening using function based perturbation
coef_KO = delta_to_probability(pred_m_single, k=0.005)

## Perform KO multiple screening using function based perturbation
coef_KO_multiple = delta_to_probability(pred_m_multiple, k=0.005)

# %%
coef_KO = coef_KO.loc[single_ko, TERMINAL_STATES_KO]
coef_KO_multiple = coef_KO_multiple.loc[multiple_ko, TERMINAL_STATES_KO]

pval_KO = pval_m_single.loc[single_ko, TERMINAL_STATES_KO]
pval_KO_multiple = pval_m_multiple.loc[multiple_ko, TERMINAL_STATES_KO]

# %% [markdown]
# ## Gene expression perturbation

# %%
## Dynamo (perturbation)
perturbation_dyn = TFScanning_perturbation_dyn(adata, 8, "cell_type", TERMINAL_STATES_KO, single_ko)

# %%
## Dynamo (perturbation) in multiple
multiple_ko = ["mitfa_tfec_tfeb", "fli1a_elk3", "mitfa_tfec", "tfec_mitfa_bhlhe40"]
multiple_ko_list = split_elements(multiple_ko)
perturbation_dyn_multiple = Multiple_TFScanning_perturbation_dyn(
    adata, 8, "cell_type", TERMINAL_STATES_KO, multiple_ko_list
)

# %%
## Perform KO screening using function based perturbation
coef_perturb = pd.DataFrame(np.array(perturbation_dyn["coefficient"]))
coef_perturb.index = perturbation_dyn["TF"]
coef_perturb.columns = get_list_name(perturbation_dyn["coefficient"][0])
coef_perturb = coef_perturb.loc[single_ko, TERMINAL_STATES_PERTURB]

## Perform perturbation screening using gene expression based perturbation
coef_perturb_multiple = pd.DataFrame(np.array(perturbation_dyn_multiple["coefficient"]))
coef_perturb_multiple.index = perturbation_dyn_multiple["TF"]
coef_perturb_multiple.columns = get_list_name(perturbation_dyn_multiple["coefficient"][0])
coef_perturb_multiple = coef_perturb_multiple.loc[multiple_ko, TERMINAL_STATES_PERTURB]

# %%
if SAVE_DATA:
    coef_KO.to_csv(DATA_DIR / DATASET / "results" / "dynamo_KO_single.csv")
    coef_KO_multiple.to_csv(DATA_DIR / DATASET / "results" / "dynamo_KO_multiple.csv")
    pval_KO.to_csv(DATA_DIR / DATASET / "results" / "dynamo_KO_single_pval.csv")
    pval_KO_multiple.to_csv(DATA_DIR / DATASET / "results" / "dynamo_KO_multiple_pval.csv")

    coef_perturb.to_csv(DATA_DIR / DATASET / "results" / "dynamo_perturb_single.csv")
    coef_perturb_multiple.to_csv(DATA_DIR / DATASET / "results" / "dynamo_perturb_multiple.csv")

# %%

# %%

# %%

# %%

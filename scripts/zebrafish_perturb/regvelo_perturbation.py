# %% [markdown]
# # Compute density depletion likelihood with RegVelo
#
# Using regvelo to predict the density change likelihood

# %% [markdown]
# ## Library imports

# %%
import cellrank as cr
import scvelo as scv
import scanpy as sc

import scipy
import numpy as np
import pandas as pd

import scvi
from anndata import AnnData
from regvelo import REGVELOVI
from typing import List

from collections import Counter

import mplscience
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

from rgv_tools.perturbation import markov_density_simulation, delta_to_probability, smooth_score, density_likelihood
from rgv_tools.perturbation import in_silico_block_simulation, split_elements
from rgv_tools.benchmarking import set_output
from rgv_tools import DATA_DIR, FIG_DIR

# %% [markdown]
# ## General settings

# %%
sc.settings.verbosity = 2
scv.settings.verbosity = 3

# %%
plt.rcParams["svg.fonttype"] = "none"

# %%
scvi.settings.seed = 0

# %% [markdown]
# ## Constants

# %%
DATASET = "zebrafish"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)
    for nrun in range(3):
        (DATA_DIR / DATASET / "processed" / ("runs" + str(nrun + 1))).mkdir(parents=True, exist_ok=True)

    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)
    for nrun in range(3):
        (DATA_DIR / DATASET / "results" / ("runs" + str(nrun + 1))).mkdir(parents=True, exist_ok=True)

# %%
single_ko = ["rarga", "rxraa", "nr2f5", "fli1a", "tfec", "elk3", "mitfa", "ets1", "nr2f2", "elf1", "ebf3a"]
multiple_ko = ["fli1a_elk3", "tfec_mitfa_bhlhe40", "mitfa_tfec", "mitfa_tfec_tfeb"]

# %%
terminal_states = ["mNC_arch2", "mNC_head_mesenchymal", "mNC_hox34", "Pigment"]
terminal_states_compare = ["mNC_arch2", "mNC_head_mesenchymal", "mNC_hox34", "Pigment_gch2"]

# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / "processed" / "adata_preprocessed.h5ad")

# %%
model_list = [
    DATA_DIR / DATASET / "processed" / "perturb_repeat_runs" / "rgv_model_0",
    DATA_DIR / DATASET / "processed" / "perturb_repeat_runs" / "rgv_model_1",
    DATA_DIR / DATASET / "processed" / "perturb_repeat_runs" / "rgv_model_2",
]

# %%
for nrun in range(3):
    model = model_list[nrun]
    vae = REGVELOVI.load(model, adata)
    set_output(adata, vae, n_samples=30, batch_size=adata.n_obs)
    adata.write_h5ad(DATA_DIR / DATASET / "processed" / ("runs" + str(nrun + 1)) / "control.h5ad")

    for TF in single_ko + multiple_ko:
        TF_list = split_elements([TF])[0]
        adata_target_perturb, reg_vae_perturb = in_silico_block_simulation(model, adata, TF_list)
        adata_target_perturb.write_h5ad(DATA_DIR / DATASET / "processed" / ("runs" + str(nrun + 1)) / f"{TF}.h5ad")

# %% [markdown]
# ## Repeats run regvelo

# %%
dl_all_list = []
dl_sig_all_list = []
for nrun in range(3):
    adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / ("runs" + str(nrun + 1)) / "control.h5ad")
    adata.obs["cell_type_old"] = adata.obs["cell_type"].copy()
    start_indices = np.where(adata.obs["cell_type"].isin(["NPB_nohox"]))[0]

    annotation = pd.read_csv(DATA_DIR / DATASET / "processed" / "annotation.csv", index_col=0)
    sl = annotation.index[annotation["pigment_annotation"] == "Pigment_gch2"].tolist()
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
    adata.obs.loc[[i.replace("-1", "") for i in sl], "cell_type"] = "Pigment_gch2"
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    del adata.uns["cell_type_colors"]

    vk = cr.kernels.VelocityKernel(adata)
    vk.compute_transition_matrix()
    ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()

    kernel = 0.8 * vk + 0.2 * ck
    # kernel = vk

    estimator = cr.estimators.GPCCA(kernel)
    estimator.compute_macrostates(n_states=8, cluster_key="cell_type_old")
    estimator.set_terminal_states(terminal_states)

    adata.obs["term_states_fwd"] = adata.obs["term_states_fwd"].astype("str")
    adata.obs["term_states_fwd"][adata.obs["term_states_fwd"] == "Pigment"] = adata.obs["cell_type"][
        adata.obs["term_states_fwd"] == "Pigment"
    ].tolist()
    adata.obs["term_states_fwd"][adata.obs["term_states_fwd"] == "Pigment"]

    dl_all = []
    dl_sig_all = []
    for TF in single_ko + multiple_ko:
        adata_perturb = sc.read_h5ad(DATA_DIR / DATASET / "processed" / ("runs" + str(nrun + 1)) / f"{TF}.h5ad")
        adata_perturb.obs["term_states_fwd"] = adata.obs["term_states_fwd"].copy()
        dl_score, dl_sig, cont_sim, pert_sim = density_likelihood(
            adata, adata_perturb, start_indices, terminal_states_compare, n_simulations=1000
        )
        dl_all.append(dl_score)
        dl_sig_all.append(dl_sig)

    dl_all_list.append(dl_all)
    dl_sig_all_list.append(dl_sig_all)

# %% [markdown]
# ## Save data

# %%
for nrun in range(len(dl_all_list)):
    dl_all = dl_all_list[nrun]
    dl_sig_all = dl_sig_all_list[nrun]

    pred_m_single = pd.DataFrame(np.array(dl_all[: len(single_ko)]), index=single_ko, columns=terminal_states_compare)
    pred_m_multiple = pd.DataFrame(
        np.array(dl_all[len(single_ko) :]), index=multiple_ko, columns=terminal_states_compare
    )
    pval_m_single = pd.DataFrame(
        np.array(dl_sig_all[: len(single_ko)]), index=single_ko, columns=terminal_states_compare
    )
    pval_m_multiple = pd.DataFrame(
        np.array(dl_sig_all[len(single_ko) :]), index=multiple_ko, columns=terminal_states_compare
    )

    if SAVE_DATA:
        pred_m_single.to_csv(DATA_DIR / DATASET / "results" / ("runs" + str(nrun + 1)) / "regvelo_single.csv")
        pred_m_multiple.to_csv(DATA_DIR / DATASET / "results" / ("runs" + str(nrun + 1)) / "regvelo_multiple.csv")
        pval_m_single.to_csv(DATA_DIR / DATASET / "results" / ("runs" + str(nrun + 1)) / "regvelo_single_pval.csv")
        pval_m_multiple.to_csv(DATA_DIR / DATASET / "results" / ("runs" + str(nrun + 1)) / "regvelo_multiple_pval.csv")

# %%
if SAVE_DATA:
    adata.write_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed_cr.h5ad")

# %%

# %%

# %%

# %%

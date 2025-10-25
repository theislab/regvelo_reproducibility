# %% [markdown]
# # Run simple simulation via perturb nr2f5 targets

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd
import scipy
import torch
import random
import anndata as ad

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns
import sklearn

import cellrank as cr
import scanpy as sc
import scvi
import regvelo
from regvelo import REGVELOVI
import regvelo as rgv

from rgv_tools import DATA_DIR, FIG_DIR
from rgv_tools.benchmarking import set_output
from rgv_tools.perturbation import in_silico_block_simulation
from rgv_tools.perturbation import inferred_GRN, abundance_test
from rgv_tools.perturbation import get_list_name, TFScanning

# %% [markdown]
# ## General settings

# %%
# %matplotlib inline

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
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

# %%
SAVE_FIGURES = False
if SAVE_FIGURES:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)

# %%
genes = ["nr2f5", "sox9b", "twist1b", "ets1"]

# %%
TERMINAL_STATES = [
    "mNC_head_mesenchymal",
    "mNC_arch2",
    "mNC_hox34",
    "Pigment",
]

# %%
MODEL = DATA_DIR / DATASET / "processed" / "rgv_model"


# %% [markdown]
# ## Define functions


# %%
def cosine_similarity(vector_a, vector_b):
    """
    Calculate the cosine similarity between two vectors.

    Args:
        vector_a (np.array): First vector.
        vector_b (np.array): Second vector.

    Returns:
        float: Cosine similarity between vector_a and vector_b.
    """
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    return dot_product / (norm_a * norm_b)


# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed.h5ad")

# %% [markdown]
# ## Load model

# %%
# Prepare skeleton
W = adata.uns["skeleton"].copy()
W = torch.tensor(np.array(W)).int()

# Prepare TF
TF = adata.var_names[adata.var["TF"]]

# Prepare model
REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
vae = REGVELOVI(adata, W=W.T, regulators=TF, soft_constraint=False)

# %%
vae.train()

# %%
vae.save(MODEL)

# %%
vae = REGVELOVI.load(MODEL, adata)

# %%
set_output(adata, vae, n_samples=30, batch_size=adata.n_obs)

# %%
TERMINAL_STATES = ["mNC_head_mesenchymal", "mNC_arch2", "mNC_hox34", "Pigment"]
vk = cr.kernels.VelocityKernel(adata).compute_transition_matrix()
ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()

estimator = cr.estimators.GPCCA(vk)

## evaluate the fate prob on original space
estimator.compute_macrostates(n_states=7, cluster_key="cell_type")
estimator.set_terminal_states(TERMINAL_STATES)
estimator.compute_fate_probabilities()
estimator.plot_fate_probabilities(same_plot=False)

# %%
lineage_genes = estimator.compute_lineage_drivers(return_drivers=True, cluster_key="cell_type")

# %%
adata_perturb_dict = {}
reg_vae_perturb_dict = {}
cand_list = ["nr2f5", "ets1", "sox9b", "twist1b"]

for TF in cand_list:
    adata_target_perturb, reg_vae_perturb = in_silico_block_simulation(model=MODEL, adata=adata, gene=TF, cutoff=0)
    adata_perturb_dict[TF] = adata_target_perturb
    reg_vae_perturb_dict[TF] = reg_vae_perturb

# %%
ct_indices = {
    ct: adata.obs["term_states_fwd"][adata.obs["term_states_fwd"] == ct].index.tolist() for ct in TERMINAL_STATES
}

# Computing states transition probability for perturbed systems
for TF, adata_target_perturb in adata_perturb_dict.items():
    vkp = cr.kernels.VelocityKernel(adata_target_perturb).compute_transition_matrix()
    ck = cr.kernels.ConnectivityKernel(adata_target_perturb).compute_transition_matrix()

    estimator = cr.estimators.GPCCA(vkp)
    estimator.set_terminal_states(ct_indices)
    estimator.compute_fate_probabilities()

    adata_perturb_dict[TF] = adata_target_perturb

# %% [markdown]
# ## Select top-20 targets

# %%
W = pd.DataFrame(vae.module.v_encoder.fc1.weight.data.cpu().numpy(), index=adata.var_names, columns=adata.var_names)

# %%
targets = W.sort_values(by="nr2f5", ascending=False).loc[:, "nr2f5"]
targets = targets[:20].index

# %%
targets

# %% [markdown]
# Both `serpinh1b` and `alcama` genes are known functional related to skeleton development, in which closely relate to facial mesenchymal
#
# alcama: https://pmc.ncbi.nlm.nih.gov/articles/PMC3036164/
# serpinh1b: https://www.sciencedirect.com/science/article/pii/S0925477315000489?utm_source=chatgpt.com

# %%
Wp_p = W.copy()
for g in ["alcama", "serpinh1b"]:
    Wp_p.loc[g, "nr2f5"] = 0

# %%
reg_vae_perturb_dict["nr2f5"].module.v_encoder.fc1.weight.data = torch.tensor(np.array(Wp_p), device="cuda:0")

# %%
set_output(adata_perturb_dict["nr2f5"], reg_vae_perturb_dict["nr2f5"], n_samples=30, batch_size=adata.n_obs)

# %% [markdown]
# ## Run ODE simulation

# %%
adata_perturb_dict["nr2f5"].layers["fit_s"], adata_perturb_dict["nr2f5"].layers["fit_u"] = reg_vae_perturb_dict[
    "nr2f5"
].rgv_expression_fit(n_samples=30)

# %%
adata.layers["fit_s"], adata.layers["fit_u"] = vae.rgv_expression_fit(n_samples=30)

# %% [markdown]
# ## Calculating similarity

# %%
velo = adata_perturb_dict["nr2f5"][adata.obs_names[adata.obs["cell_type"] == "NPB_nohox"]].layers["velocity"]
gex = adata[adata.obs_names[adata.obs["cell_type"] == "NPB_nohox"]].layers["fit_s"]

# %%
gex = gex + velo

cor_pert = []
for i in range(47):
    cor_pert.append(
        cosine_similarity(
            gex[i,], adata[adata.obs["term_states_fwd"] == "mNC_head_mesenchymal"].layers["fit_s"].mean(0)
        )
    )

# %%
velo = adata[adata.obs_names[adata.obs["cell_type"] == "NPB_nohox"]].layers["velocity"]
gex = adata[adata.obs_names[adata.obs["cell_type"] == "NPB_nohox"]].layers["Ms"]

# %%
gex = gex + velo

cor_raw = []
for i in range(47):
    cor_raw.append(
        cosine_similarity(
            gex[i,], adata[adata.obs["term_states_fwd"] == "mNC_head_mesenchymal"].layers["fit_s"].mean(0)
        )
    )

# %%
df = pd.DataFrame(
    {
        "Cosine similarity": np.concatenate([cor_pert, cor_raw]),
        "Type": ["Perturbed"] * len(cor_pert) + ["Raw"] * len(cor_raw),
    }
)

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(3, 3))

    sns.violinplot(
        data=df,
        ax=ax,
        # orient="h",
        x="Type",
        y="Cosine similarity",
        hue="Type",
        order=["Raw", "Perturbed"],
    )

    if SAVE_FIGURES:
        fig.savefig(FIG_DIR / DATASET / "GEX_prediction.svg", format="svg", transparent=True, bbox_inches="tight")
    plt.xlabel("")

# %%
scipy.stats.ttest_ind(cor_raw, cor_pert, alternative="greater")

# %% [markdown]
# ## Visualize the cell fate probability change after entire nr2f5 regulon depletion

# %%
score = np.array(adata_perturb_dict["nr2f5"].obsm["lineages_fwd"]["mNC_head_mesenchymal"]).reshape(-1) - np.array(
    adata.obsm["lineages_fwd"]["mNC_head_mesenchymal"]
).reshape(-1)

# %%
adata.obs["score"] = score

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(4, 3))
    sc.pl.umap(
        adata=adata,
        color="score",
        title="",
        cmap="vlag",
        vcenter=0,
        ax=ax,
        vmax=0.075,
        frameon=False,
        legend_fontsize=14,
    )

    if SAVE_FIGURES:
        fig.savefig(FIG_DIR / DATASET / "nr2f5_perturbation.svg", format="svg", transparent=True, bbox_inches="tight")
    plt.show

# %% [markdown]
# ## Visualize trajectory

# %%
wt = adata.copy()
nr2f5 = adata_perturb_dict["nr2f5"].copy()

# %%
gene = "serpinh1b"
X = np.zeros((adata.shape[0], 3))
X[:, 0] = wt[:, "nr2f5"].layers["Ms"].reshape(-1)
X[:, 1] = wt[:, gene].layers["fit_u"].reshape(-1)
X[:, 2] = nr2f5[:, gene].layers["fit_u"].reshape(-1)

# %%
var = pd.DataFrame(
    {
        "gene_name": ["nr2f5", gene, f"{gene}_pert"],
    },
    index=["nr2f5", gene, f"{gene}_pert"],
)

# %%
adata_plot = ad.AnnData(X=X, obs=adata.obs.copy(), var=var)

# %%
adata_plot.obsm = adata.obsm.copy()
adata_plot.uns = adata.uns.copy()
adata_plot.obsp = adata.obsp.copy()

# %%
adata_plot

# %%
model = cr.models.GAMR(adata_plot, n_knots=10, smoothing_penalty=10.0)

# %%
cr.pl.gene_trends(
    adata_plot,
    model=model,
    lineages="mNC_head_mesenchymal",
    data_key="X",
    genes=var.index.tolist(),
    same_plot=True,
    ncols=3,
    time_key="latent_time",
    hide_cells=True,
    weight_threshold=(1e-3, 1e-3),
    legend_loc="none",
    sharey=True,
    figsize=(6, 2),
)

# %%
gene = "alcama"
X = np.zeros((adata.shape[0], 3))
X[:, 0] = wt[:, "nr2f5"].layers["Ms"].reshape(-1)
X[:, 1] = wt[:, gene].layers["fit_u"].reshape(-1)
X[:, 2] = nr2f5[:, gene].layers["fit_u"].reshape(-1)

# %%
var = pd.DataFrame(
    {
        "gene_name": ["nr2f5", gene, f"{gene}_pert"],
    },
    index=["nr2f5", gene, f"{gene}_pert"],
)

# %%
adata_plot = ad.AnnData(X=X, obs=adata.obs.copy(), var=var)

# %%
adata_plot.obsm = adata.obsm.copy()
adata_plot.uns = adata.uns.copy()
adata_plot.obsp = adata.obsp.copy()

# %%
adata_plot

# %%
model = cr.models.GAMR(adata_plot, n_knots=10, smoothing_penalty=10.0)

# %%
cr.pl.gene_trends(
    adata_plot,
    model=model,
    lineages="mNC_head_mesenchymal",
    data_key="X",
    genes=var.index.tolist(),
    same_plot=True,
    ncols=3,
    time_key="latent_time",
    hide_cells=True,
    weight_threshold=(1e-3, 1e-3),
    legend_loc="none",
    sharey=True,
    figsize=(6, 2),
)

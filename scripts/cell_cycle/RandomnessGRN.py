# %% [markdown]
# # Regulation is important for velocity inference of RegVelo

# %% [markdown]
# ## Library imports

# %%
import os
import sys

import scvi
from paths import DATA_DIR, FIG_DIR

# %%
from regvelo import REGVELOVI

import numpy as np
import pandas as pd

# %%
import scipy
from scipy.stats import ttest_ind

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

import scanpy as sc
import scvelo as scv
import torch
import unitvelo as utv

# from _calculation import get_gams
sys.path.append("../..")


# %% [markdown]
# ## General settings

# %%
scvi.settings.dl_pin_memory_gpu_training = False

# %%
plt.rcParams["svg.fonttype"] = "none"

# %%
sns.reset_defaults()
sns.reset_orig()
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=14, color_map="viridis")

# %%
SAVE_FIGURES = True
if SAVE_FIGURES:
    os.makedirs(FIG_DIR / "cell_cycle", exist_ok=True)

SAVE_DATASETS = True
if SAVE_DATASETS:
    os.makedirs(DATA_DIR / "cell_cycle", exist_ok=True)


# %% [markdown]
# ## Function definitions


# %% [markdown]
# ### Note:
# ##### The function `add_regvelo_outputs_to_adata`,`add_significance`, `get_significance` are adapted on an an implementation by Adam Gayoso, Philipp Weilier and Justing Hong from their repository: https://github.com/YosefLab/velovi_reproducibility with License: BSD-3-Clause license


# %%
def add_significance(ax, left: int, right: int, significance: str, level: int = 0, **kwargs):
    """TODO."""
    bracket_level = kwargs.pop("bracket_level", 1)
    bracket_height = kwargs.pop("bracket_height", 0.02)
    text_height = kwargs.pop("text_height", 0.01)

    bottom, top = ax.get_ylim()
    y_axis_range = top - bottom

    bracket_level = (y_axis_range * 0.07 * level) + top * bracket_level
    bracket_height = bracket_level - (y_axis_range * bracket_height)

    ax.plot([left, left, right, right], [bracket_height, bracket_level, bracket_level, bracket_height], **kwargs)

    ax.text(
        (left + right) * 0.5,
        bracket_level + (y_axis_range * text_height),
        significance,
        ha="center",
        va="bottom",
        c="k",
    )


# %%
def get_significance(pvalue):
    """TODO."""
    if pvalue < 0.001:
        return "***"
    elif pvalue < 0.01:
        return "**"
    elif pvalue < 0.1:
        return "*"
    else:
        return "n.s."


# %%
def add_regvelo_outputs_to_adata(adata_raw, vae, filter=False):
    """TODO."""
    latent_time = vae.get_latent_time(n_samples=30, batch_size=adata_raw.shape[0])
    velocities = vae.get_velocity(n_samples=30, batch_size=adata_raw.shape[0])

    t = latent_time
    scaling = 20 / t.max(0)
    adata = adata_raw[:, vae.module.target_index].copy()

    adata.layers["velocity"] = velocities / scaling
    adata.layers["latent_time_regvelo"] = latent_time

    adata.layers["fit_t"] = latent_time.values * scaling[np.newaxis, :]
    adata.var["fit_scaling"] = 1.0

    ## calculate likelihood

    return adata


def GRN_Jacobian(reg_vae, Ms):
    """TODO."""
    reg_vae.module.v_encoder.fc1.weight.detach()
    reg_vae.module.v_encoder.fc1.bias.detach()
    reg_vae.module.v_encoder.alpha_unconstr_max.detach()
    ## calculate the jacobian matrix respect to each cell
    Jaco_m = []
    for i in range(Ms.shape[0]):
        s = Ms[i, :]
        Jaco_m.append(
            reg_vae.module.v_encoder.GRN_Jacobian(
                torch.tensor(s[reg_vae.module.v_encoder.regulator_index]).to("cuda:0")
            ).detach()
        )
    Jaco_m = torch.stack(Jaco_m, 2)
    return Jaco_m


# %% [markdown]
# ## Data loading

# %%
adata = sc.read(DATA_DIR / "cell_cycle" / "cell_cycle_processed.h5ad")

# %%
adata.uns["regulators"] = adata.var.index.values
adata.uns["targets"] = adata.var.index.values
adata.uns["skeleton"] = np.ones((len(adata.var.index), len(adata.var.index)))
adata.uns["network"] = np.ones((len(adata.var.index), len(adata.var.index)))
adata

# %%
reg_vae = REGVELOVI.load(DATA_DIR / "cell_cycle" / "model_1", adata)

# %% [markdown]
# ## Fix the learned GRN (30 times)

# %%
REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")

# %%
label = "phase"
cluster_edges = [("G1", "S-ph"), ("S-ph", "G2M")]

# %%
W = adata.uns["skeleton"].copy()
W = torch.tensor(np.array(W)).int()

# %%
score_v0 = []
score_t0 = []
for _nrun in range(30):
    reg_vae_r1 = REGVELOVI(adata, W=W.T * 0, soft_constraint=False)
    reg_vae_r1.module.v_encoder.fc1.weight.data = reg_vae.module.v_encoder.fc1.weight.data.detach().cpu().clone()
    reg_vae_r1.module.v_encoder.fc1.bias.data = reg_vae.module.v_encoder.fc1.bias.data.detach().cpu().clone()
    reg_vae_r1.train()

    adata_target = add_regvelo_outputs_to_adata(adata, reg_vae_r1)

    scv.tl.velocity_graph(adata_target)
    adata_target.obsm["X_pca"] = adata.obsm["X_pca"].copy()
    scv.tl.velocity_embedding(adata_target, basis="pca")

    regvi = utv.evaluate(adata_target, cluster_edges, label, "velocity", x_emb="X_pca")
    score_v0.append(
        np.mean(
            [
                np.mean(regvi["Cross-Boundary Direction Correctness (A->B)"][("G1", "S-ph")]),
                np.mean(regvi["Cross-Boundary Direction Correctness (A->B)"][("S-ph", "G2M")]),
            ]
        )
    )
    print(score_v0[len(score_v0) - 1])

    ## calculate latent time correlation
    adata_target.obs["latent_time"] = np.mean(adata_target.layers["fit_t"], axis=1)
    score_t0.append(scipy.stats.spearmanr(adata_target.obs["latent_time"], adata_target.obs["fucci_time"])[0])
    print(score_t0[len(score_t0) - 1])

# %%
score_v = []
dfs = []
score_t = []
for _nrun in range(30):
    original_tensor = reg_vae.module.v_encoder.fc1.weight.data.detach().cpu().clone()
    original_shape = original_tensor.shape
    # Flatten the tensor
    flattened_tensor = original_tensor.flatten()
    # Generate a random permutation of indices
    permutation = torch.randperm(flattened_tensor.size(0))
    # Shuffle the flattened tensor using the permutation
    shuffled_flattened_tensor = flattened_tensor[permutation]

    permutation = torch.randperm(flattened_tensor.size(0))
    shuffled_flattened_tensor = shuffled_flattened_tensor[permutation]
    # Reshape the shuffled tensor back to the original shape
    shuffled_tensor = shuffled_flattened_tensor.reshape(original_shape)

    reg_vae_r1 = REGVELOVI(adata, W=W.T * 0, soft_constraint=False)
    reg_vae_r1.module.v_encoder.fc1.weight.data = shuffled_tensor
    reg_vae_r1.module.v_encoder.fc1.bias.data = reg_vae.module.v_encoder.fc1.bias.data.detach().cpu().clone()
    reg_vae_r1.train()

    adata_target = add_regvelo_outputs_to_adata(adata, reg_vae_r1)

    scv.tl.velocity_graph(adata_target)
    adata_target.obsm["X_pca"] = adata.obsm["X_pca"].copy()
    scv.tl.velocity_embedding(adata_target, basis="pca")

    regvi = utv.evaluate(adata_target, cluster_edges, label, "velocity", x_emb="X_pca")
    score_v.append(
        np.mean(
            [
                np.mean(regvi["Cross-Boundary Direction Correctness (A->B)"][("G1", "S-ph")]),
                np.mean(regvi["Cross-Boundary Direction Correctness (A->B)"][("S-ph", "G2M")]),
            ]
        )
    )
    print(score_v[len(score_v) - 1])

    ## calculate latent time correlation
    adata_target.obs["latent_time"] = np.mean(adata_target.layers["fit_t"], axis=1)
    score_t.append(scipy.stats.spearmanr(adata_target.obs["latent_time"], adata_target.obs["fucci_time"])[0])
    print(score_t[len(score_t) - 1])

# %%
score_v2 = []
score_t2 = []
for _nrun in range(30):
    reg_vae_r1 = REGVELOVI(adata, W=W.T * 0, soft_constraint=False)
    reg_vae_r1.module.v_encoder.fc1.weight.data = reg_vae.module.v_encoder.fc1.weight.data.detach().cpu().clone() * 0
    reg_vae_r1.module.v_encoder.fc1.bias.data = reg_vae.module.v_encoder.fc1.bias.data.detach().cpu().clone()
    reg_vae_r1.train()

    adata_target = add_regvelo_outputs_to_adata(adata, reg_vae_r1)

    scv.tl.velocity_graph(adata_target)
    adata_target.obsm["X_pca"] = adata.obsm["X_pca"].copy()
    scv.tl.velocity_embedding(adata_target, basis="pca")

    regvi = utv.evaluate(adata_target, cluster_edges, label, "velocity", x_emb="X_pca")
    score_v2.append(
        np.mean(
            [
                np.mean(regvi["Cross-Boundary Direction Correctness (A->B)"][("G1", "S-ph")]),
                np.mean(regvi["Cross-Boundary Direction Correctness (A->B)"][("S-ph", "G2M")]),
            ]
        )
    )
    print(score_v2[len(score_v2) - 1])

    ## calculate latent time correlation
    adata_target.obs["latent_time"] = np.mean(adata_target.layers["fit_t"], axis=1)
    score_t2.append(scipy.stats.spearmanr(adata_target.obs["latent_time"], adata_target.obs["fucci_time"])[0])
    print(score_t2[len(score_t2) - 1])

# %%
df = pd.DataFrame(
    {"CBC score": score_v0 + score_v + score_v2, "Model": ["Original"] * 30 + ["Random"] * 30 + ["No regulation"] * 30}
)

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(3, 4))

    sns.violinplot(data=df, x="Model", y="CBC score", ax=ax)

    ttest_res = ttest_ind(score_v0, score_v, equal_var=False, alternative="greater")
    significance = get_significance(ttest_res.pvalue)
    add_significance(
        ax=ax,
        left=0,
        right=1,
        significance=significance,
        lw=1,
        bracket_level=1.05,
        c="k",
        level=0,
    )

    ttest_res = ttest_ind(score_v0, score_v2, equal_var=False, alternative="greater")
    significance = get_significance(ttest_res.pvalue)
    add_significance(ax=ax, left=0, right=2, significance=significance, lw=1, c="k", level=2, bracket_level=0.95)

    y_min, y_max = ax.get_ylim()
    ax.set_ylim([y_min, y_max + 0.02])

    if SAVE_FIGURES:
        fig.savefig(FIG_DIR / "cell_cycle" / "CBC_score_GRN.svg", format="svg", transparent=True, bbox_inches="tight")

# %%
df = pd.DataFrame(
    {
        "Latent time correlation": score_t0 + score_t + score_t2,
        "Model": ["Original"] * 30 + ["Random"] * 30 + ["No regulation"] * 30,
    }
)

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(3, 4))

    sns.violinplot(data=df, x="Model", y="Latent time correlation", ax=ax)

    ttest_res = ttest_ind(score_v0, score_v, equal_var=False, alternative="greater")
    significance = get_significance(ttest_res.pvalue)
    add_significance(
        ax=ax,
        left=0,
        right=1,
        significance=significance,
        lw=1,
        bracket_level=1.05,
        c="k",
        level=0,
    )

    ttest_res = ttest_ind(score_v0, score_v2, equal_var=False, alternative="greater")
    significance = get_significance(ttest_res.pvalue)
    add_significance(ax=ax, left=0, right=2, significance=significance, lw=1, c="k", level=2, bracket_level=0.95)

    y_min, y_max = ax.get_ylim()
    ax.set_ylim([y_min, y_max + 0.02])

    if SAVE_FIGURES:
        fig.savefig(FIG_DIR / "cell_cycle" / "Time_GRN.svg", format="svg", transparent=True, bbox_inches="tight")

# %%

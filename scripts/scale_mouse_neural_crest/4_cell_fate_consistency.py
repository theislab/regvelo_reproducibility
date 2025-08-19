# %% [markdown]
# # Evaluate consistency of velocity and cell fate probability across scale

# %%
import numpy as np
import pandas as pd
import torch
import cellrank as cr
import scipy

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

import scanpy as sc
import scvi
from regvelo import REGVELOVI

from rgv_tools import DATA_DIR, FIG_DIR

# %% [markdown]
# ## General settings

# %%
# %matplotlib inline

# %%
scvi.settings.seed = 0

# %% [markdown]
# ## Constants

# %%
DATASET = "scale_murine"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

# %%
SAVE_FIGURES = True
if SAVE_FIGURES:
    (FIG_DIR).mkdir(parents=True, exist_ok=True)

# %%
TERMINAL_STATES = ["Mesenchyme", "Sensory_1", "Sensory_2"]

# %%
TERMINAL_STATES_ALL = [
    "Melanocytes",
    "enFib",
    "SC",
    "Mesenchyme",
    "Sensory_1",
    "Sensory_2",
    "ChC",
    "SatGlia",
    "Gut_glia",
    "Gut_neuron",
    "Symp",
    "BCC",
]

# %%
TERMINAL_STATES_ALL = (
    TERMINAL_STATES_ALL
    + [i + "_1" for i in TERMINAL_STATES_ALL]
    + [i + "_2" for i in TERMINAL_STATES_ALL]
    + [i + "_3" for i in TERMINAL_STATES_ALL]
)

# %% [markdown]
# ## Classify sensory neurons into two

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / f"adata_run_stage_2_regvelo_all_regulons.h5ad")

# %%
sc.pl.umap(adata, color="assignments")

# %%
sensory = adata[adata.obs["assignments"].isin(["Sensory"])].copy()

# %%
# sc.pp.neighbors(sensory)
sc.tl.leiden(sensory, resolution=0.1)
sc.tl.umap(sensory)

# %%
sensory

# %%
sensory.obs["assignments"] = sensory.obs["assignments"].astype(str)
# Assign "Sensory_1" where leiden == "1"
sensory.obs.loc[sensory.obs["leiden"] == "1", "assignments"] = "Sensory_1"
# Assign "Sensory_2" where leiden is "2" or "0"
sensory.obs.loc[sensory.obs["leiden"].isin(["2", "0"]), "assignments"] = "Sensory_2"

# %%
adata.obs["assignments"] = adata.obs["assignments"].astype(str)
adata.obs.loc[sensory.obs_names.tolist(), "assignments"] = sensory.obs["assignments"]

# %%
adata.obs.loc[sensory.obs_names.tolist(), "assignments"]

# %%
adata

# %% [markdown]
# ## Measuring cell fate probabilities across different scale

# %%
fate_probs = []

for scale in [2, 3, 4, 5]:
    adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / f"adata_run_stage_{scale}_regvelo_all_regulons.h5ad")
    adata.obs["assignments"] = adata.obs["assignments"].astype(str)
    adata.obs.loc[sensory.obs_names.tolist(), "assignments"] = sensory.obs["assignments"]
    adata.obs["assignments"] = adata.obs["assignments"].astype("category")
    del adata.uns["assignments_colors"]

    vk = cr.kernels.VelocityKernel(adata)
    vk.compute_transition_matrix()
    ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()

    kernel = 0.8 * vk + 0.2 * ck
    # kernel = vk

    ## evaluate the fate prob on original space
    estimator = cr.estimators.GPCCA(kernel)

    n_states = 3
    for nround in range(100):
        estimator.compute_macrostates(n_states=n_states, cluster_key="assignments")
        if len(set(np.unique(estimator.macrostates.tolist())).intersection(TERMINAL_STATES)) == 3:
            estimator.set_terminal_states(
                list(set(np.unique(estimator.macrostates.tolist())).intersection(TERMINAL_STATES_ALL))
            )
            # estimator.plot_macrostates(which="terminal", discrete=True, legend_loc="on data", s=100)
            estimator.compute_fate_probabilities(solver="direct")
            print(str(n_states) + " works!")
            break
        n_states += 1

    fate_probs.append(
        pd.DataFrame(
            adata.obsm["lineages_fwd"], columns=adata.obsm["lineages_fwd"].names.tolist(), index=adata.obs_names
        )
    )

# %%
fate_probs2 = []

for scale in [2, 3, 4, 5]:
    adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / f"adata_run_stage_{scale}_velovi_all_regulons.h5ad")
    adata.obs["assignments"] = adata.obs["assignments"].astype(str)
    adata.obs.loc[sensory.obs_names.tolist(), "assignments"] = sensory.obs["assignments"]
    adata.obs["assignments"] = adata.obs["assignments"].astype("category")
    del adata.uns["assignments_colors"]

    vk = cr.kernels.VelocityKernel(adata)
    vk.compute_transition_matrix()
    ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()

    kernel = 0.8 * vk + 0.2 * ck
    # kernel = vk

    ## evaluate the fate prob on original space
    estimator = cr.estimators.GPCCA(kernel)

    n_states = 3
    for nround in range(100):
        estimator.compute_macrostates(n_states=n_states, cluster_key="assignments")
        if len(set(np.unique(estimator.macrostates.tolist())).intersection(TERMINAL_STATES)) == 3:
            estimator.set_terminal_states(
                list(set(np.unique(estimator.macrostates.tolist())).intersection(TERMINAL_STATES_ALL))
            )
            # estimator.plot_macrostates(which="terminal", discrete=True, legend_loc="on data", s=100)
            estimator.compute_fate_probabilities(solver="direct")
            print(str(n_states) + " works!")
            break
        n_states += 1

    fate_probs2.append(
        pd.DataFrame(
            adata.obsm["lineages_fwd"], columns=adata.obsm["lineages_fwd"].names.tolist(), index=adata.obs_names
        )
    )

# %% [markdown]
# ## Calculate consistency

# %%
cf_consis_rgv = []

for ct in ["Mesenchyme", "Sensory_1", "Sensory_2"]:
    scale_consis = pd.DataFrame(0, index=[1, 2, 3, 4], columns=[1, 2, 3, 4])
    for i in [1, 2, 3, 4]:
        for j in [1, 2, 3, 4]:
            cells = list(set(fate_probs[i - 1].index.tolist()).intersection(fate_probs[j - 1].index.tolist()))
            scale_consis.loc[i, j] = scipy.stats.spearmanr(
                fate_probs[i - 1].loc[cells, ct], fate_probs[j - 1].loc[cells, ct]
            )[0]

    rows, cols = np.triu_indices(scale_consis.shape[0], k=1)
    cf_consis_rgv += np.array(scale_consis)[rows, cols].tolist()
    print(
        "median:"
        + str(np.median(np.array(scale_consis)[rows, cols].tolist()))
        + ",min:"
        + str(np.min(np.array(scale_consis)[rows, cols].tolist()))
    )
    print("std:" + str(np.std(np.array(scale_consis)[rows, cols])))

# %%
cf_consis_vi = []

for ct in ["Mesenchyme", "Sensory_1", "Sensory_2"]:
    scale_consis = pd.DataFrame(0, index=[1, 2, 3, 4], columns=[1, 2, 3, 4])
    for i in [1, 2, 3, 4]:
        for j in [1, 2, 3, 4]:
            cells = list(set(fate_probs2[i - 1].index.tolist()).intersection(fate_probs2[j - 1].index.tolist()))
            scale_consis.loc[i, j] = scipy.stats.spearmanr(
                fate_probs2[i - 1].loc[cells, ct], fate_probs2[j - 1].loc[cells, ct]
            )[0]

    rows, cols = np.triu_indices(scale_consis.shape[0], k=1)
    cf_consis_vi += np.array(scale_consis)[rows, cols].tolist()
    print(
        "median:"
        + str(np.median(np.array(scale_consis)[rows, cols].tolist()))
        + ",min:"
        + str(np.min(np.array(scale_consis)[rows, cols].tolist()))
    )
    print("std:" + str(np.std(np.array(scale_consis)[rows, cols])))

# %%
import scipy

# %%
scipy.stats.ttest_ind(
    cf_consis_rgv,
    cf_consis_vi,
    equal_var=False,
    alternative="greater",
)

# %%
np.mean(cf_consis_rgv)

# %%
np.std(cf_consis_rgv)

# %%
np.mean(cf_consis_vi)

# %%
np.std(cf_consis_vi)

# %%
dfs = []

g_df = pd.DataFrame({"Cell fate probabilities consistency": cf_consis_rgv})
g_df["Method"] = "RegVelo"
dfs.append(g_df)

g_df = pd.DataFrame({"Cell fate probabilities consistency": cf_consis_vi})
g_df["Method"] = "veloVI"
dfs.append(g_df)

df = pd.concat(dfs, axis=0)
df["Method"] = df["Method"].astype("category")

# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(2.5, 3))
    pal = {"RegVelo": "#0173b2", "veloVI": "#de8f05"}

    sns.violinplot(
        data=df,
        ax=ax,
        # orient="h",
        x="Method",
        y="Cell fate probabilities consistency",
        order=["RegVelo", "veloVI"],
        cut=0,
        palette=pal,
    )
    # plt.legend(title='', loc='lower center', bbox_to_anchor=(0.5, -0.6), ncol=3)
    # ax.set_yticks([0.7, 0.8, 0.9,1.0])
    # ax.set_yticklabels([0.7, 0.8, 0.9,1.0])
    plt.xlabel("")
    ax.set_ylim(0, 1)

    if SAVE_FIGURES:
        fig.savefig(FIG_DIR / DATASET / "consist_robustness.svg", format="svg", transparent=True, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## Plot heatmap

# %%
for ct in ["Mesenchyme", "Sensory_1", "Sensory_2"]:
    scale_consis = pd.DataFrame(0, index=[1, 2, 3, 4], columns=[1, 2, 3, 4])
    for i in [1, 2, 3, 4]:
        for j in [1, 2, 3, 4]:
            cells = list(set(fate_probs[i - 1].index.tolist()).intersection(fate_probs[j - 1].index.tolist()))
            scale_consis.loc[i, j] = scipy.stats.spearmanr(
                fate_probs[i - 1].loc[cells, ct], fate_probs[j - 1].loc[cells, ct]
            )[0]

    with mplscience.style_context():
        sns.set_style(style="whitegrid")
        fig, ax = plt.subplots(figsize=(4, 3))

        sns.heatmap(
            scale_consis,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            square=True,
            cbar=True,
            cmap="Blues",
            vmin=0,
            vmax=1,
            ax=ax,
        )

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()

        if SAVE_FIGURES:
            plt.savefig(FIG_DIR / f"heatmap_{ct}_final.svg", format="svg", transparent=True, bbox_inches="tight")
        plt.show()

# %%

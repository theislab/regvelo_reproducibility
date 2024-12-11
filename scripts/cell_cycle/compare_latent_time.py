# %% [markdown]
# # Comparing regvelo and velovi inferred latent time
#
# Notebook compare latent time inference

# %% [markdown]
# ## Library imports

# %%
import random

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

import anndata as ad
import scvelo as scv
import scvi
from regvelo import REGVELOVI
from velovi import VELOVI

from rgv_tools import DATA_DIR, FIG_DIR
from rgv_tools.benchmarking import set_output
from rgv_tools.core import METHOD_PALETTE
from rgv_tools.plotting import get_significance

# %% [markdown]
# ## General settings

# %%
scvi.settings.dl_pin_memory_gpu_training = False


# %% [markdown]
# ## Function defination


# %%
def compute_confidence(adata, vkey="velocity"):
    """Compute confidence."""
    adata.layers[vkey]
    scv.tl.velocity_graph(adata, vkey=vkey, n_jobs=1)
    scv.tl.velocity_confidence(adata, vkey=vkey)

    g_df = pd.DataFrame()
    g_df["Latent time consistency"] = adata.obs[f"{vkey}_confidence"].to_numpy().ravel()

    return g_df


# %% [markdown]
# ## Constants

# %%
DATASET = "cell_cycle"

# %%
significance_palette = {"n.s.": "#dedede", "*": "#90BAAD", "**": "#A1E5AB", "***": "#ADF6B1"}

# %%
STATE_TRANSITIONS = [("G1", "S"), ("S", "G2M")]

# %%
SAVE_DATA = True
SAVE_FIGURES = True

if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)
if SAVE_FIGURES:
    (FIG_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
adata = ad.io.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_processed.h5ad")
adata

# %% [markdown]
# ## Model loading

# %%
vae = REGVELOVI.load(DATA_DIR / DATASET / "regvelo_model", adata)

# %%
set_output(adata, vae, n_samples=30, batch_size=adata.n_obs)
adata_regvelo = adata.copy()

# %% [markdown]
# ## Running veloVI as baseline

# %%
VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
vae = VELOVI(adata)
vae.train(max_epochs=1500)

# %%
set_output(adata, vae, n_samples=30)
adata_velovi = adata.copy()

# %% [markdown]
# ## replacing velocity as the latent time

# %%
dfs = []

g_df = compute_confidence(adata_regvelo, vkey="fit_t")
g_df["Dataset"] = "Cell cycle"
g_df["Method"] = "regvelo"
dfs.append(g_df)

g_df = compute_confidence(adata_velovi, vkey="fit_t")
g_df["Dataset"] = "Cell cycle"
g_df["Method"] = "velovi"
dfs.append(g_df)

conf_df = pd.concat(dfs, axis=0)

# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(3, 3))

    sns.violinplot(
        data=conf_df,
        ax=ax,
        # orient="h",
        x="Method",
        y="Latent time consistency",
        order=["regvelo", "velovi"],
        palette=METHOD_PALETTE,
    )
    # plt.legend(title='', loc='lower center', bbox_to_anchor=(0.5, -0.6), ncol=3)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([0.25, 0.5, 0.75, 1.0])
    plt.xlabel("")

    if SAVE_FIGURES:
        fig.savefig(
            FIG_DIR / DATASET / "Latent_time_consistency.svg", format="svg", transparent=True, bbox_inches="tight"
        )
    plt.show()

# %% [markdown]
# ## Calculate fold change

# %%
graph = adata.obsp["connectivities"].A

# %%
Time_FC = []
for i in range(graph.shape[0]):
    v = adata_regvelo[i].layers["fit_t"]
    m = adata_regvelo[graph[i, :] != 0].layers["fit_t"]
    cos_similarities = cosine_similarity(v, m)

    ## randomly sample each number of cells
    indices = random.sample(range(0, adata.shape[0]), m.shape[0])
    m = adata_regvelo[indices].layers["fit_t"]
    cos_similarities_random = cosine_similarity(v, m)

    FC = np.mean(cos_similarities) / np.mean(cos_similarities_random)
    Time_FC.append(FC)

# %%
b = np.zeros(len(Time_FC))
_, p_value = wilcoxon(Time_FC, b, alternative="greater")
p_value

# %%
b = np.zeros(len(Time_FC))
_, p_value = wilcoxon(Time_FC, b, alternative="greater")
significance = get_significance(p_value)
palette = significance_palette[significance]

# Step 3: Create the boxplot
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, axes = plt.subplots(figsize=(2, 3))

    # Step 4: Color based on significance

    sns.violinplot(data=Time_FC, color=palette)

    # Add titles and labels
    plt.xlabel("Ery")
    plt.ylabel("Log ratio")

    fig.tight_layout()
    plt.show()

    if SAVE_FIGURES:
        fig.savefig(
            FIG_DIR / DATASET / "Latent_time_consistency_test.svg",
            format="svg",
            transparent=True,
            bbox_inches="tight",
        )

# %%

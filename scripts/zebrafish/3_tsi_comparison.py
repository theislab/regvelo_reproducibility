# %% [markdown]
# # Terminal state identification
#
# Notebook compares model performance based on for terminal state identification.

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from scipy.stats import ttest_ind

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns
from matplotlib import rcParams

import cellrank as cr
import scanpy as sc
import scvelo as scv

from rgv_tools import DATA_DIR, FIG_DIR
from rgv_tools.benchmarking import stair_vec, TSI_score
from rgv_tools.plotting._significance import add_significance, get_significance

# %% [markdown]
# ## General setting

# %%
pl.seed_everything(0)

# %%
plt.rcParams["svg.fonttype"] = "none"
sns.reset_defaults()
sns.reset_orig()
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=14, color_map="viridis")

# %% [markdown]
# ## Constants

# %%
DATASET = "zebrafish"

SAVE_FIGURES = False
if SAVE_FIGURES:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)

FIGURE_FORMATE = "svg"

# %%
VELOCITY_METHODS = ["regvelo", "scvelo", "velovi"]

# %% [markdown]
# ## Data loading

# %%
vks = {}

for method in VELOCITY_METHODS:
    adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / f"adata_run_{method}.h5ad")
    ## construct graph
    vk = cr.kernels.VelocityKernel(adata).compute_transition_matrix()
    vks[method] = vk

# %% [markdown]
# ## Terminal state identification

# %%
terminal_states = [
    "mNC_head_mesenchymal",
    "mNC_arch2",
    "mNC_hox34",
    "Pigment",
]

# define threshold from 0.1 to 1
points = np.linspace(0.1, 1, 21)[:20]

# %%
estimators = {}
tsi = {}

# %%
for method in VELOCITY_METHODS:
    estimators[method] = cr.estimators.GPCCA(vks[method])
    tsi[method] = TSI_score(adata, points, "cell_type", terminal_states, estimators[method])

# %%
df = pd.DataFrame(
    {
        "TSI": tsi["regvelo"] + tsi["velovi"] + tsi["scvelo"],
        "Model": ["RegVelo"] * 20 + ["veloVI"] * 20 + ["scVelo"] * 20,
    }
)

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(3, 4))

    sns.barplot(data=df, x="Model", y="TSI", palette="colorblind", ax=ax)

    ttest_res = ttest_ind(tsi["regvelo"], tsi["velovi"], alternative="greater")
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

    ttest_res = ttest_ind(tsi["regvelo"], tsi["scvelo"], alternative="greater")
    significance = get_significance(ttest_res.pvalue)
    add_significance(ax=ax, left=0, right=2, significance=significance, lw=1, c="k", level=2, bracket_level=0.9)

    y_min, y_max = ax.get_ylim()
    ax.set_ylim([y_min, y_max + 0.02])

    if SAVE_FIGURES:
        fig.savefig(FIG_DIR / DATASET / "tsi_benchmark.svg", format="svg", transparent=True, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## Visualize terminal states

# %%
pre_value_rgv = stair_vec(adata, estimators["regvelo"], 0.8, terminal_states, "cell_type")
pre_value_scv = stair_vec(adata, estimators["scvelo"], 0.8, terminal_states, "cell_type")
pre_value_vi = stair_vec(adata, estimators["velovi"], 0.8, terminal_states, "cell_type")

# %% [markdown]
# ## Plotting

# %%
df = pd.DataFrame(
    {
        "number_macrostate": range(0, 12),
        "RegVelo": [0] + pre_value_rgv,
        "veloVI": [0] + pre_value_vi,
        "scVelo": [0] + pre_value_scv,
    }
)

# %%
df = pd.melt(df, ["number_macrostate"])
colors = sns.color_palette("colorblind", n_colors=3)
colors = colors + [(0.8274509803921568, 0.8274509803921568, 0.8274509803921568)]

# %%
# Set figure size
with mplscience.style_context():
    sns.set_style(style="whitegrid")

    rcParams["figure.figsize"] = 4, 3

    # Plot the grid plot
    ax = sns.lineplot(
        x="number_macrostate",
        y="value",
        hue="variable",
        style="variable",
        palette=colors,
        drawstyle="steps-post",
        data=df,
        linewidth=3,
    )

    # Set labels and titles
    ax.set(ylabel="Number of correct predictions")
    plt.xlabel("Number of macrostates", fontsize=14)
    plt.ylabel("Identified terminal states", fontsize=14)

    # Customize tick parameters for better readability
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_xticklabels([0, 2, 4, 6, 8, 10])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), shadow=True, ncol=4, fontsize=14)

    if SAVE_FIGURES:
        plt.savefig(
            FIG_DIR / DATASET / "state_identification_update.svg", format="svg", transparent=True, bbox_inches="tight"
        )
    plt.show()

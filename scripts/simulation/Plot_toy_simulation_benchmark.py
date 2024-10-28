# %% [markdown]
# # Plot the results of Toy simulate results

# %% [markdown]
# ## Library imports

# %%
import os

from paths import DATA_DIR, FIG_DIR

# %%
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

import scvelo as scv

# %% [markdown]
# ## General setting

# %%
sns.reset_defaults()
sns.reset_orig()
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=20, color_map="viridis")
plt.rcParams["svg.fonttype"] = "none"

# %%
SAVE_FIGURES = True
if SAVE_FIGURES:
    os.makedirs(FIG_DIR / "simulation" / "toy_GRN", exist_ok=True)

SAVE_DATASETS = True
if SAVE_DATASETS:
    os.makedirs(DATA_DIR / "simulation" / "toy_GRN", exist_ok=True)

# %%
plt.rcParams["svg.fonttype"] = "none"
mpl.rcParams.update({"font.size": 14})


# %% [markdown]
# ## Function definations


# %%
def get_significance(pvalue):
    if pvalue < 0.001:
        return "***"
    elif pvalue < 0.01:
        return "**"
    elif pvalue < 0.1:
        return "*"
    else:
        return "n.s."


def add_significance2(ax, bottom: int, top: int, significance: str, level: int = 0, **kwargs):
    bracket_level = kwargs.pop("bracket_level", 1)
    bracket_height = kwargs.pop("bracket_height", 0.02)
    text_height = kwargs.pop("text_height", 0.01)

    left, right = ax.get_xlim()
    x_axis_range = right - left

    bracket_level = (x_axis_range * 0.07 * level) + right * bracket_level
    bracket_height = bracket_level - (x_axis_range * bracket_height)

    ax.plot([bracket_height, bracket_level, bracket_level, bracket_height], [bottom, bottom, top, top], **kwargs)

    ax.text(
        bracket_level + (x_axis_range * text_height),
        (bottom + top) * 0.5,
        significance,
        va="center",
        ha="left",
        c="k",
        rotation=90,
    )


# %% [markdown]
# ## Import datasets

# %% [markdown]
# ### latent time correlation benchmark

# %%
address = DATA_DIR / "simulation" / "toy_GRN" / "latent_time_benchmark_result.csv"
dat = pd.read_csv(address, index_col=0)

# %%
dat

# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(3, 2))
    # Set font size for all elements
    colors = sns.color_palette("colorblind", n_colors=3)
    colors = colors + ["lightgrey"] * 1

    sns.violinplot(y="Model", x="Time", data=dat, palette=colors, ax=ax)

    ttest_res = ttest_ind(
        dat.loc[dat.loc[:, "Model"] == "RegVelo", "Time"],
        dat.loc[dat.loc[:, "Model"] == "scVelo", "Time"],
        equal_var=False,
        alternative="greater",
    )
    significance = get_significance(ttest_res.pvalue)
    add_significance2(
        ax=ax,
        bottom=0,
        top=2,
        significance=significance,
        lw=1,
        bracket_level=1.05,
        c="k",
        level=0,
    )

    plt.ylabel("")
    plt.xlabel("Spearman correlation")

    if SAVE_FIGURES:
        fig.savefig(
            FIG_DIR / "simulation" / "toy_GRN" / "latent_time.svg", format="svg", transparent=True, bbox_inches="tight"
        )

# %% [markdown]
# ### GRN benchmark

# %%
address = DATA_DIR / "simulation" / "toy_GRN" / "GRN_benchmark_result.csv"
dat = pd.read_csv(address, index_col=0)

# %%
dat

# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(3, 2))

    mpl.rcParams.update({"font.size": 14})

    # Then, create the grouped boxplot
    sns.violinplot(y="Model", x="GRN", data=dat, color="lightpink", ax=ax)

    ttest_res = ttest_ind(
        dat.loc[dat.loc[:, "Model"] == "RegVelo", "GRN"],
        dat.loc[dat.loc[:, "Model"] == "Correlation", "GRN"],
        equal_var=False,
        alternative="greater",
    )
    significance = get_significance(ttest_res.pvalue)
    add_significance2(
        ax=ax,
        bottom=0,
        top=1,
        significance=significance,
        lw=1,
        bracket_level=1.05,
        c="k",
        level=0,
    )

    plt.ylabel("")
    plt.xlabel("AUROC")

    if SAVE_FIGURES:
        fig.savefig(
            FIG_DIR / "simulation" / "toy_GRN" / "GRN_benchmark.svg",
            format="svg",
            transparent=True,
            bbox_inches="tight",
        )

# %% [markdown]
# ### Velocity correlation

# %%
address = DATA_DIR / "simulation" / "toy_GRN" / "velocity_benchmark.csv"
dat = pd.read_csv(address, index_col=0)

# %%
velo_rgv = dat["RegVelo"]
velo_velovi = dat["veloVI"]
velo_scv = dat["scVelo"]

# %%
dat = pd.DataFrame(
    {
        "Velo_cor": np.array(velo_rgv).tolist() + np.array(velo_velovi).tolist() + np.array(velo_scv).tolist(),
        "Model": ["RegVelo"] * 100 + ["veloVI"] * 100 + ["scVelo"] * 100,
    }
)

# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(3, 2))
    # pal = {"RegVelo":"#f3e1eb","veloVI":"#b5bbe3","scVelo":"#0fcfc0"}
    sns.violinplot(data=dat, y="Model", x="Velo_cor", ax=ax)
    plt.ylabel("")
    plt.xlabel("Pearson correlation")

    ttest_res = ttest_ind(
        velo_rgv,
        velo_velovi,
        alternative="greater",
    )
    significance = get_significance(ttest_res.pvalue)
    add_significance2(
        ax=ax,
        bottom=0,
        top=1,
        significance=significance,
        lw=1,
        bracket_level=1.05,
        c="k",
        level=0,
    )

    ttest_res = ttest_ind(
        velo_rgv,
        velo_scv,
        alternative="greater",
    )
    significance = get_significance(ttest_res.pvalue)
    add_significance2(
        ax=ax,
        bottom=0,
        top=2,
        significance=significance,
        lw=1,
        bracket_level=1.05,
        c="k",
        level=0,
    )

    if SAVE_FIGURES:
        fig.savefig(
            FIG_DIR / "simulation" / "toy_GRN" / "Velocity_benchmark.svg",
            format="svg",
            transparent=True,
            bbox_inches="tight",
        )

# %%

# %%

# %%

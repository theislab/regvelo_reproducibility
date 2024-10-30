# %% [markdown]
# # Plot the results of Toy simulate results

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

import scvelo as scv

from rgv_tools import DATA_DIR, FIG_DIR

# %% [markdown]
# ## General settings

# %%
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["svg.fonttype"] = "none"

# %%
sns.reset_defaults()
sns.reset_orig()

# %%
mpl.rcParams.update({"font.size": 14})


# %%
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=20, color_map="viridis")

# %%
SAVE_FIGURES = True
if SAVE_FIGURES:
    (FIG_DIR / "simulation" / "toy_GRN").mkdir(parents=True, exist_ok=True)

SAVE_DATASETS = True
if SAVE_DATASETS:
    (DATA_DIR / "simulation" / "toy_GRN").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Function definitions


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
def add_significance2(ax, bottom: int, top: int, significance: str, level: int = 0, **kwargs):
    """TODO."""
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
# ## Data loading

# %% [markdown]
# ### latent time correlation benchmark

# %%
latent_time_df = pd.read_csv(DATA_DIR / "simulation" / "toy_GRN" / "latent_time_benchmark_result.csv", index_col=0)
latent_time_df.head()

# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(3, 2))
    # Set font size for all elements
    colors = sns.color_palette("colorblind", n_colors=3)
    colors = colors + ["lightgrey"] * 1

    sns.violinplot(y="Model", x="Time", data=latent_time_df, palette=colors, ax=ax)

    ttest_res = ttest_ind(
        latent_time_df.loc[latent_time_df.loc[:, "Model"] == "RegVelo", "Time"],
        latent_time_df.loc[latent_time_df.loc[:, "Model"] == "scVelo", "Time"],
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
grn_df = pd.read_csv(DATA_DIR / "simulation" / "toy_GRN" / "GRN_benchmark_result.csv", index_col=0)
grn_df.head()

# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(3, 2))

    mpl.rcParams.update({"font.size": 14})

    # Then, create the grouped boxplot
    sns.violinplot(y="Model", x="GRN", data=grn_df, color="lightpink", ax=ax)

    ttest_res = ttest_ind(
        grn_df.loc[grn_df.loc[:, "Model"] == "RegVelo", "GRN"],
        grn_df.loc[grn_df.loc[:, "Model"] == "Correlation", "GRN"],
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
velo_df = pd.read_csv(DATA_DIR / "simulation" / "toy_GRN" / "velocity_benchmark.csv", index_col=0)
velo_df.head()

# %%
velo_rgv = velo_df["RegVelo"]
velo_velovi = velo_df["veloVI"]
velo_scv = velo_df["scVelo"]

# %%
velo_df = pd.DataFrame(
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
    sns.violinplot(data=velo_df, y="Model", x="Velo_cor", ax=ax)
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

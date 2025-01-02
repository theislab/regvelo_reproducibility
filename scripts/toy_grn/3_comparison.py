# %% [markdown]
# # Performance comparison of inference on toy GRN data
#
# Notebook compares metrics for velocity, latent time and GRN inference across different methods applied to toy GRN data.

# %% [markdown]
# ## Library imports

# %%
import pandas as pd
from scipy.stats import ttest_ind

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

from rgv_tools import DATA_DIR, FIG_DIR
from rgv_tools.core import METHOD_PALETTE
from rgv_tools.plotting._significance import add_significance, get_significance

# %% [markdown]
# ## General settings

# %%
SAVE_FIGURES = False
if SAVE_FIGURES:
    (FIG_DIR / "toy_grn").mkdir(parents=True, exist_ok=True)

FIGURE_FORMATE = "svg"

# %% [markdown]
# ## Constants

# %%
VELOCITY_METHODS = ["regvelo", "velovi", "scvelo"]
TIME_METHODS = ["regvelo", "velovi", "scvelo", "dpt"]
GRN_METHODS = ["regvelo", "correlation", "grnboost2", "celloracle"]

# %% [markdown]
# ## Data loading

# %%
correlation_df = []

for method in VELOCITY_METHODS:
    df = pd.read_parquet(DATA_DIR / "toy_grn" / "results" / f"{method}_correlation.parquet")
    df.columns = f"{method}_" + df.columns
    correlation_df.append(df)
del df

for method in TIME_METHODS:
    if method in VELOCITY_METHODS:
        continue
    df = pd.read_parquet(DATA_DIR / "toy_grn" / "results" / f"{method}_correlation.parquet")
    df.columns = f"{method}_" + df.columns
    correlation_df.append(df)

for method in GRN_METHODS:
    if method in VELOCITY_METHODS + TIME_METHODS:
        continue
    df = pd.read_parquet(DATA_DIR / "toy_grn" / "results" / f"{method}_correlation.parquet")
    df.columns = f"{method}_" + df.columns
    correlation_df.append(df)

correlation_df = pd.concat(correlation_df, axis=1)
correlation_df.head()

# %% [markdown]
# ## Analysis

# %% [markdown]
# ### Velocity

# %%
df = correlation_df.loc[:, correlation_df.columns.str.contains("velocity")]
df.columns = df.columns.str.removesuffix("_velocity")
df = pd.melt(df, var_name="method", value_name="correlation")

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(
        data=df, x="correlation", y="method", hue="method", order=VELOCITY_METHODS, palette=METHOD_PALETTE, ax=ax
    )

    ttest_res = ttest_ind(
        correlation_df["regvelo_velocity"],
        correlation_df["velovi_velocity"],
        equal_var=False,
        alternative="greater",
    )
    significance = get_significance(pvalue=ttest_res.pvalue)
    add_significance(
        ax=ax,
        left=0,
        right=1,
        significance=significance,
        lw=1,
        bracket_level=1.05,
        c="k",
        level=0,
        orientation="vertical",
    )

    ttest_res = ttest_ind(
        correlation_df["regvelo_velocity"],
        correlation_df["scvelo_velocity"],
        equal_var=False,
        alternative="greater",
    )
    significance = get_significance(pvalue=ttest_res.pvalue)
    add_significance(
        ax=ax,
        left=0,
        right=2,
        significance=significance,
        lw=1,
        bracket_level=1.05,
        c="k",
        level=0,
        orientation="vertical",
    )

    ax.set(
        xlabel="Pearson correlation",
        ylabel="Method",
        yticks=ax.get_yticks(),
        yticklabels=["RegVelo", "veloVI", "scVelo"],
    )

    if SAVE_FIGURES:
        fig.savefig(
            FIG_DIR / "toy_GRN" / "velocity_benchmark.svg",
            format="svg",
            transparent=True,
            bbox_inches="tight",
        )

    plt.show()

# %% [markdown]
# ### Latent time

# %%
df = correlation_df.loc[:, correlation_df.columns.str.contains("time")]
df.columns = df.columns.str.removesuffix("_time")
df = pd.melt(df, var_name="method", value_name="correlation")

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(
        data=df, x="correlation", y="method", hue="method", order=TIME_METHODS, palette=METHOD_PALETTE, ax=ax
    )

    ttest_res = ttest_ind(
        correlation_df["regvelo_time"],
        correlation_df["velovi_time"],
        equal_var=False,
        alternative="greater",
    )
    significance = get_significance(pvalue=ttest_res.pvalue)
    add_significance(
        ax=ax,
        left=0,
        right=1,
        significance=significance,
        lw=1,
        bracket_level=1.05,
        c="k",
        level=0,
        orientation="vertical",
    )

    ttest_res = ttest_ind(
        correlation_df["velovi_time"],
        correlation_df["scvelo_time"],
        equal_var=False,
        alternative="greater",
    )
    significance = get_significance(pvalue=ttest_res.pvalue)
    add_significance(
        ax=ax,
        left=0,
        right=2,
        significance=significance,
        lw=1,
        bracket_level=1.05,
        c="k",
        level=0,
        orientation="vertical",
    )

    ax.set(
        xlabel="Spearman correlation",
        ylabel="Method",
        yticks=ax.get_yticks(),
        yticklabels=["RegVelo", "veloVI", "scVelo", "DPT"],
    )

    if SAVE_FIGURES:
        fig.savefig(
            FIG_DIR / "toy_GRN" / "time_benchmark.svg",
            format="svg",
            transparent=True,
            bbox_inches="tight",
        )

    plt.show()

# %% [markdown]
# ### GRN

# %%
df = correlation_df.loc[:, correlation_df.columns.str.contains("grn")]
df.columns = df.columns.str.removesuffix("_grn")
df = pd.melt(df, var_name="method").rename(columns={"value": "correlation"})

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(data=df, x="correlation", y="method", hue="method", order=GRN_METHODS, palette=METHOD_PALETTE, ax=ax)

    ttest_res = ttest_ind(
        correlation_df["regvelo_grn"],
        correlation_df["correlation_grn"],
        equal_var=False,
        alternative="greater",
    )
    significance = get_significance(pvalue=ttest_res.pvalue)
    add_significance(
        ax=ax,
        left=0,
        right=1,
        significance=significance,
        lw=1,
        bracket_level=1.05,
        c="k",
        level=0,
        orientation="vertical",
    )

    ax.set(
        xlabel="AUROC",
        ylabel="Method",
        yticks=ax.get_yticks(),
        yticklabels=["RegVelo", "Correlation", "GRNBoost2", "CellOracle"],
    )

    if SAVE_FIGURES:
        fig.savefig(
            FIG_DIR / "toy_GRN" / "grn_benchmark.svg",
            format="svg",
            transparent=True,
            bbox_inches="tight",
        )

    plt.show()

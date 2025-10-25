# %% [markdown]
# # Benchmark GRN inference

# %% [markdown]
# ## Library imports

# %%
import pandas as pd

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

from rgv_tools import DATA_DIR, FIG_DIR

# %% [markdown]
# ## Constants

# %%
DATASET = "mHSPC"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

SAVE_FIGURE = True
if SAVE_FIGURE:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
tfv_score = pd.read_csv(DATA_DIR / DATASET / "results" / "GRN_benchmark_tfv.csv")
rgv_score = pd.read_csv(DATA_DIR / DATASET / "results" / "GRN_benchmark_rgv.csv")
grn_score = pd.read_csv(DATA_DIR / DATASET / "results" / "GRN_benchmark.csv")

# %%
df = pd.concat([tfv_score, rgv_score, grn_score], ignore_index=True)

# %% [markdown]
# ## Plot the benchmark

# %%
df

# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(3, 3), sharey=True)
    # Plot the second Seaborn plot on the first subplot
    sns.barplot(
        y="Method",
        x="AUC",
        data=df,
        capsize=0.1,
        color="grey",
        order=["regvelo", "Corr", "CellOracle", "GRNBoost2", "tfvelo"],
        ax=ax,
    )
    sns.stripplot(
        y="Method",
        x="AUC",
        data=df,
        order=["regvelo", "Corr", "CellOracle", "GRNBoost2", "tfvelo"],
        color="black",
        size=3,
        jitter=True,
        ax=ax,
    )
    ax.set_xlabel("AUC", fontsize=14)
    ax.set_ylabel("")
    plt.xlim(0.5, 0.7)
    plt.show()

    if SAVE_FIGURE:
        fig.savefig(FIG_DIR / DATASET / "AUC.svg", format="svg", transparent=True, bbox_inches="tight")

# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(3, 3), sharey=True)
    # Plot the second Seaborn plot on the first subplot
    sns.barplot(
        y="Method",
        x="EPR",
        data=df,
        capsize=0.1,
        color="grey",
        order=["regvelo", "Corr", "CellOracle", "GRNBoost2", "tfvelo"],
        ax=ax,
    )
    sns.stripplot(
        y="Method",
        x="EPR",
        data=df,
        order=["regvelo", "Corr", "CellOracle", "GRNBoost2", "tfvelo"],
        color="black",
        size=3,
        jitter=True,
        ax=ax,
    )
    ax.set_xlabel("EPR", fontsize=14)
    ax.set_ylabel("")
    plt.xlim(
        1.0,
    )
    plt.show()

    if SAVE_FIGURE:
        fig.savefig(FIG_DIR / DATASET / "EPR.svg", format="svg", transparent=True, bbox_inches="tight")

# %%

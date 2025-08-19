# %% [markdown]
# # Benchmark transition from hub cells to terminal states
#
# For simplicity of demonstration, we present the analysis steps for scale-1; the analysis for the other scales is identical.

# %%
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel
from itertools import chain

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns
from matplotlib import rcParams

import cellrank as cr
import scanpy as sc
import scvelo as scv
import scvi
from tqdm import tqdm

from rgv_tools import DATA_DIR, FIG_DIR
from rgv_tools.plotting._significance import add_significance, get_significance

# %% [markdown]
# # General setting

# %%
plt.rcParams["svg.fonttype"] = "none"
sns.reset_defaults()
sns.reset_orig()
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=14, color_map="viridis")

# %% [markdown]
# ## Constants

# %%
DATASET = "mouse_neural_crest"

SAVE_FIGURES = True
if SAVE_FIGURES:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET).mkdir(parents=True, exist_ok=True)

FIGURE_FORMATE = "svg"

# %%
VELOCITY_METHODS = ["regvelo", "scvelo", "velovi"]

# %%
TERMINAL_STATE = ["Melanocytes", "enFib", "SC", "Sensory", "ChC", "SatGlia", "Gut_glia", "Gut_neuron", "Symp", "BCC"]

# %%
STATE_TRANSITIONS_RAW = [
    ("hub", "Sensory"),
    ("hub", "SatGlia"),
    ("hub", "SC"),
    ("hub", "Gut_glia"),
    ("hub", "Gut_neuron"),
    ("hub", "ChC"),
]

# %%
scale_level = 2

# %% [markdown]
# ## Data loading

# %%
vks = {}

for method in VELOCITY_METHODS:
    adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / f"adata_run_stage_{scale_level}_{method}_all_regulons.h5ad")

    adata.obs["Hub"] = (~adata.obs["conflict"].astype(bool)) & (adata.obs["assignments"] == "none")
    adata.obs["assignments"] = adata.obs["assignments"].astype(str)
    adata.obs["assignments"][adata.obs["Hub"]] = "hub"

    print(adata)
    ## construct graph
    vk = cr.kernels.VelocityKernel(adata).compute_transition_matrix()
    vks[method] = vk

# %% [markdown]
# ## Filter out the transition

# %%
STATE_TRANSITIONS = []

for i in range(len(STATE_TRANSITIONS_RAW)):
    if STATE_TRANSITIONS_RAW[i][1] in np.unique(adata.obs["assignments"]):
        STATE_TRANSITIONS.append(STATE_TRANSITIONS_RAW[i])

# %%
STATE_TRANSITIONS_RAW = [f"{a} - {b}" for a, b in STATE_TRANSITIONS_RAW]

# %%
STATE_TRANSITIONS

# %% [markdown]
# ## Measuring transport from hub to the terminal cell states

# %%
STATE_TRANSITIONS

# %%
cluster_key = "assignments"
rep = "X_pca"

score_df_rgv_vs_scv = []
score_df_rgv_vs_vi = []

cbc_rgv_f = []
cbc_scv_f = []
cbc_vi_f = []
for source, target in tqdm(STATE_TRANSITIONS):
    cbc_rgv = vks["regvelo"].cbc(source=source, target=target, cluster_key=cluster_key, rep=rep)
    cbc_scv = vks["scvelo"].cbc(source=source, target=target, cluster_key=cluster_key, rep=rep)
    cbc_vi = vks["velovi"].cbc(source=source, target=target, cluster_key=cluster_key, rep=rep)

    score_df_rgv_vs_scv.append(
        pd.DataFrame(
            {
                "State transition": [f"{source} - {target}"] * len(cbc_rgv),
                "Log ratio": np.log((cbc_rgv + 1) / (cbc_scv + 1)),
            }
        )
    )

    score_df_rgv_vs_vi.append(
        pd.DataFrame(
            {
                "State transition": [f"{source} - {target}"] * len(cbc_rgv),
                "Log ratio": np.log((cbc_rgv + 1) / (cbc_vi + 1)),
            }
        )
    )

    cbc_rgv_f.append(cbc_rgv)
    cbc_scv_f.append(cbc_scv)
    cbc_vi_f.append(cbc_vi)

score_df_rgv_vs_scv_forward = pd.concat(score_df_rgv_vs_scv)
score_df_rgv_vs_vi_forward = pd.concat(score_df_rgv_vs_vi)

# %% [markdown]
# ## Comparing overall transition

# %%
dfs = []

g_df = pd.DataFrame({"CBC_forward": list(chain(*cbc_rgv_f))})
g_df["Method"] = "RegVelo"
dfs.append(g_df)

g_df = pd.DataFrame({"CBC_forward": list(chain(*cbc_vi_f))})
g_df["Method"] = "veloVI"
dfs.append(g_df)

g_df = pd.DataFrame({"CBC_forward": list(chain(*cbc_scv_f))})
g_df["Method"] = "scVelo"
dfs.append(g_df)

velo_df = pd.concat(dfs, axis=0)
velo_df["Method"] = velo_df["Method"].astype("category")

# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(3, 4))

    sns.violinplot(data=velo_df, x="Method", y="CBC_forward", palette="colorblind", ax=ax)

    ttest_res = ttest_ind(list(chain(*cbc_rgv_f)), list(chain(*cbc_vi_f)), equal_var=False, alternative="greater")
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

    ttest_res = ttest_ind(list(chain(*cbc_rgv_f)), list(chain(*cbc_scv_f)), equal_var=False, alternative="greater")
    significance = get_significance(ttest_res.pvalue)
    add_significance(
        ax=ax,
        left=0,
        right=2,
        significance=significance,
        lw=1,
        bracket_level=1.05,
        c="k",
        level=0,
    )

    y_min, y_max = ax.get_ylim()
    ax.set_ylim([y_min, y_max + 0.02])
    plt.show()

# %%
if SAVE_DATA:
    velo_df.to_csv(DATA_DIR / DATASET / "velo_df_scale_1.csv")

# %% [markdown]
# ## Visualize all transition results

# %%
df_all = []
for i in range(1, 5):
    df = pd.read_csv(DATA_DIR / DATASET / f"velo_df_scale_{i}.csv", index_col=0)

    df["scale"] = str(i)
    df_all.append(df)

df = pd.concat(df_all, axis=0)
df

# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 2.5))

    # Plot the barplot without error bars
    sns.barplot(data=df, x="scale", y="CBC_forward", hue="Method", ax=ax)

    # Add jittered dots
    # sns.stripplot(data=df, y="scale", x="AUROC", hue="method", dodge=True, color="black", ax=ax, jitter=True)

    # Remove the duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[3:6], labels[3:6], bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=2)

    # Customize labels and other settings
    ax.set(xlabel="", ylabel="CBC (hub -> terminal)")
    ax.set_xlabel(xlabel="Scale", fontsize=13)
    ax.set_ylim(0, 1)

    if SAVE_FIGURES:
        plt.savefig(FIG_DIR / DATASET / "CBC_benchmark.svg", format="svg", transparent=True, bbox_inches="tight")
    plt.show()

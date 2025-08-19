# %% [markdown]
# # Benchmark transcription rate prediction
#
# Notebooks for benchmarking transcription rate prediction on metabolic labelled cell cycle datasets

# %% [markdown]
# ## Library imports

# %%
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import scipy
import anndata as ad
import scanpy as sc
import scvelo as scv
from anndata import AnnData
from velovi import preprocess_data

from regvelo import REGVELOVI
import torch

from scipy.stats import ttest_ind

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns
from matplotlib.colors import to_hex

from rgv_tools import DATA_DIR, FIG_DIR
from rgv_tools.plotting._significance import add_significance, get_significance

# %% [markdown]
# ## Constants

# %%
DATASET = "cell_cycle_rpe1"

# %%
SAVE_DATA = True
SAVE_FIGURES = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_processed.h5ad")

# %%
alpha = pd.read_csv(
    DATA_DIR / DATASET / "raw" / "aax3072_table-s1.csv", index_col=2, header=1
)  # downloaded from the supplementary of https://www.science.org/doi/10.1126/science.aax3072

# %%
alpha = alpha.filter(regex="^norm_kappa_")
alpha

# %%
alpha.columns = np.array([i.replace("norm_kappa_", "") for i in alpha.columns], dtype=np.float32)

# %%
pos = adata.obs["cell_cycle_position"].unique()
GEX = []

for i in pos:
    gex = adata[adata.obs["cell_cycle_position"] == i].X.A.mean(axis=0)
    GEX.append(gex)

# %%
GEX = np.array(GEX).T

# %%
GEX = pd.DataFrame(GEX, index=adata.var_names.tolist())

# %%
GEX

# %%
gs = list(set(alpha.index.tolist()).intersection(adata.var_names.tolist()))

# %%
GEX.loc[gs, :]

# %%
GEX.columns = pos

# %%
GEX

# %%
alpha = alpha.loc[:, pos]

# %%
alpha

# %%
cor_gex = []
for i in range(len(gs)):
    cor_gex.append(scipy.stats.spearmanr(alpha.loc[gs, :].iloc[i, :], GEX.loc[gs, :].iloc[i, :])[0])

# %% [markdown]
# ## Estimate transcription rate

# %%
vae = REGVELOVI.load(DATA_DIR / DATASET / "processed" / "regvelo_model", adata)

# %%
fit_s, fit_u = vae.rgv_expression_fit(n_samples=30, return_numpy=False)

# %%
s = torch.tensor(np.array(fit_s)).to("cuda:0")
alpha_pre = vae.module.v_encoder.transcription_rate(s)

# %%
alpha_pre = pd.DataFrame(alpha_pre.cpu().detach().numpy(), index=fit_s.index, columns=fit_s.columns)

# %%
alpha_pre = alpha_pre.loc[:, gs].T

# %%
alpha_pre_m = []
for i in pos:
    pre = alpha_pre.loc[:, adata.obs["cell_cycle_position"] == i].mean(axis=1)
    alpha_pre_m.append(pre)

# %%
alpha_pre_m = np.array(alpha_pre_m).T
alpha_pre_m = pd.DataFrame(alpha_pre_m, index=gs)
alpha_pre_m.columns = pos

# %%
cor_rgv = []
for i in range(75):
    cor_rgv.append(scipy.stats.spearmanr(alpha.loc[gs, :].iloc[i, :], alpha_pre_m.iloc[i, :])[0])

# %%
alpha_pre_m_rgv = alpha_pre_m.copy()

# %% [markdown]
# ## Estimating transcription rate with celldancer

# %%
alpha_pre = pd.read_csv(DATA_DIR / DATASET / "processed" / "celldancer_alpha_estimate.csv", index_col=0)

# %%
alpha_pre = alpha_pre.loc[:, gs].T

# %%
alpha_pre_m = []
for i in pos:
    pre = alpha_pre.loc[:, adata.obs["cell_cycle_position"] == i].mean(axis=1)
    alpha_pre_m.append(pre)

# %%
alpha_pre_m = np.array(alpha_pre_m).T
alpha_pre_m = pd.DataFrame(alpha_pre_m, index=gs)
alpha_pre_m.columns = pos

# %%
alpha_pre_m

# %%
cor_cd = []
for i in range(75):
    cor_cd.append(scipy.stats.spearmanr(alpha.loc[gs, :].iloc[i, :], alpha_pre_m.iloc[i, :])[0])

# %%
np.mean(cor_cd)

# %% [markdown]
# ## Violinplot

# %%
dfs = []

g_df = pd.DataFrame({"Spearman correlation": cor_cd})
g_df["Method"] = "cellDancer"
dfs.append(g_df)

g_df = pd.DataFrame({"Spearman correlation": cor_rgv})
g_df["Method"] = "RegVelo"
dfs.append(g_df)

g_df = pd.DataFrame({"Spearman correlation": cor_gex})
g_df["Method"] = "GEX"
dfs.append(g_df)

df = pd.concat(dfs, axis=0)
df["Method"] = df["Method"].astype("category")

# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(3, 3))
    # pal = {"RegVelo": "#0173b2", "veloVI": "#de8f05"}

    sns.violinplot(
        data=df,
        ax=ax,
        # orient="h",
        x="Method",
        y="Spearman correlation",
        order=["RegVelo", "GEX", "cellDancer"],
        color="lightpink",
    )

    ttest_res = ttest_ind(
        cor_rgv,
        cor_gex,
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
    )

    ttest_res = ttest_ind(
        cor_rgv,
        cor_cd,
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
    )

    plt.xlabel("")

    if SAVE_FIGURES:
        fig.savefig(FIG_DIR / DATASET / "corr_sp.svg", format="svg", transparent=True, bbox_inches="tight")
    plt.show()

# %%

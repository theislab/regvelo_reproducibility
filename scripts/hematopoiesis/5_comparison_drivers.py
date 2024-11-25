# %% [markdown]
# # Comparison of driver TF ranking performance
#
# Notebook compares the performance of RegVelo (CR), RegVelo (perturbation), and Dynamo in predicting driver genes.

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

import cellrank as cr
import scanpy as sc
from regvelo import REGVELOVI

from rgv_tools import DATA_DIR, FIG_DIR
from rgv_tools.benchmarking import set_output
from rgv_tools.core import METHOD_PALETTE_DRIVER

# %% [markdown]
# ## General settings

# %% [markdown]
# ## Constants

# %%
DATASET = "hematopoiesis"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

SAVE_FIGURES = True
if SAVE_FIGURES:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)

# %%
Mon_driver = ["SPI1", "TCF4", "STAT6", "MEF2C"]
Ery_driver = ["NFIA", "GATA1", "TAL1", "GFI1B", "LMO2"]

# %%
terminal_states = ["Meg", "Mon", "Bas", "Ery"]

# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed.h5ad")
TF = adata.var_names[adata.var["TF"]]

# %%
coef = pd.read_csv(DATA_DIR / DATASET / "results" / "coef_0", index_col=0)
coef2 = pd.read_csv(DATA_DIR / DATASET / "results" / "coef_1", index_col=0)
coef3 = pd.read_csv(DATA_DIR / DATASET / "results" / "coef_2", index_col=0)
coef4 = pd.read_csv(DATA_DIR / DATASET / "results" / "coef_3", index_col=0)
coef5 = pd.read_csv(DATA_DIR / DATASET / "results" / "coef_4", index_col=0)

coef6 = pd.read_csv(DATA_DIR / DATASET / "results" / "coef_5", index_col=0)
coef7 = pd.read_csv(DATA_DIR / DATASET / "results" / "coef_6", index_col=0)
coef8 = pd.read_csv(DATA_DIR / DATASET / "results" / "coef_7", index_col=0)
coef9 = pd.read_csv(DATA_DIR / DATASET / "results" / "coef_8", index_col=0)
coef10 = pd.read_csv(DATA_DIR / DATASET / "results" / "coef_9", index_col=0)

coef11 = pd.read_csv(DATA_DIR / DATASET / "results" / "coef_10", index_col=0)
coef12 = pd.read_csv(DATA_DIR / DATASET / "results" / "coef_11", index_col=0)
coef13 = pd.read_csv(DATA_DIR / DATASET / "results" / "coef_12", index_col=0)
coef14 = pd.read_csv(DATA_DIR / DATASET / "results" / "coef_13", index_col=0)
coef15 = pd.read_csv(DATA_DIR / DATASET / "results" / "coef_14", index_col=0)

# %%
HSC_Mon_ranking = pd.read_csv(DATA_DIR / DATASET / "results" / "HSC_Mon_ranking.csv", index_col=0)
HSC_Ery_ranking = pd.read_csv(DATA_DIR / DATASET / "results" / "HSC_Ery_ranking.csv", index_col=0)

# %% [markdown]
# ## RegVelo's in silico perturbation driver identification

# %%
ery_auc_rgv = []
mon_auc_rgv = []

# %%
dfs = [coef, coef2, coef3, coef4, coef5]
coef_all_rep1 = pd.concat(dfs).groupby(level=0).mean()

dfs = [coef6, coef7, coef8, coef9, coef10]
coef_all_rep2 = pd.concat(dfs).groupby(level=0).mean()

dfs = [coef11, coef12, coef13, coef14, coef15]
coef_all_rep3 = pd.concat(dfs).groupby(level=0).mean()

# %%
driver = Ery_driver
cell_fate = "Ery"

# %%
ID = set(driver).intersection(set(coef_all_rep1.index.tolist()))
label = np.zeros(coef_all_rep1.shape[0])
label[[i for i in range(coef_all_rep1.shape[0]) if coef_all_rep1.index.tolist()[i] in list(ID)]] = 1

# %%
# Calculate AUROC
score_raw = coef_all_rep1.copy().loc[:, cell_fate]
score_raw[np.isnan(score_raw)] = 0
auroc_ery = roc_auc_score(label, (score_raw))
ery_auc_rgv.append(auroc_ery)

score_raw = coef_all_rep2.copy().loc[:, cell_fate]
score_raw[np.isnan(score_raw)] = 0
auroc_ery = roc_auc_score(label, (score_raw))
ery_auc_rgv.append(auroc_ery)

score_raw = coef_all_rep3.copy().loc[:, cell_fate]
score_raw[np.isnan(score_raw)] = 0
auroc_ery = roc_auc_score(label, (score_raw))
ery_auc_rgv.append(auroc_ery)
ery_auc_rgv

# %%
driver = Mon_driver
cell_fate = "Mon"

ID = set(driver).intersection(set(coef_all_rep1.index.tolist()))
label = np.zeros(coef_all_rep1.shape[0])
label[[i for i in range(coef_all_rep1.shape[0]) if coef_all_rep1.index.tolist()[i] in list(ID)]] = 1

# Calculate AUROC
score_raw = coef_all_rep1.copy().loc[:, cell_fate]
score_raw[np.isnan(score_raw)] = 0
auroc_mon = roc_auc_score(label, (score_raw))
mon_auc_rgv.append(auroc_mon)

score_raw = coef_all_rep2.copy().loc[:, cell_fate]
score_raw[np.isnan(score_raw)] = 0
auroc_mon = roc_auc_score(label, (score_raw))
mon_auc_rgv.append(auroc_mon)

score_raw = coef_all_rep3.copy().loc[:, cell_fate]
score_raw[np.isnan(score_raw)] = 0
auroc_mon = roc_auc_score(label, (score_raw))
mon_auc_rgv.append(auroc_mon)
mon_auc_rgv

# %% [markdown]
# ## CellRank's driver identification

# %%
ery_auc_cr = []
mon_auc_cr = []

## Ery driver ranking
for method in ["0", "1", "2"]:
    vae = REGVELOVI.load(DATA_DIR / DATASET / "processed" / "perturb_repeat_runs" / f"rgv_model_{method}", adata)
    set_output(adata, vae, n_samples=30, batch_size=adata.n_obs)

    ## Using CellRank identify driver
    vk = cr.kernels.VelocityKernel(adata)
    vk.compute_transition_matrix()
    ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
    g = cr.estimators.GPCCA(0.8 * vk + 0.2 * ck)
    ## evaluate the fate prob on original space
    g.compute_macrostates(n_states=6, cluster_key="cell_type")
    g.set_terminal_states(terminal_states)
    g.compute_fate_probabilities()
    df = g.compute_lineage_drivers(cluster_key="cell_type")

    df = df.loc[TF, :].copy()

    driver = Ery_driver
    cell_fate = "Ery" + "_corr"

    ID = set(driver).intersection(set(df.index.tolist()))
    label = np.zeros(df.shape[0])
    label[[i for i in range(df.shape[0]) if df.index.tolist()[i] in list(ID)]] = 1

    # Calculate AUROC
    score_raw = df.copy().loc[:, cell_fate]
    score_raw[np.isnan(score_raw)] = 0
    auroc_ery = roc_auc_score(label, (score_raw))
    ery_auc_cr.append(auroc_ery)

    ## Mon driver ranking
    driver = Mon_driver
    cell_fate = "Mon" + "_corr"

    ID = set(driver).intersection(set(df.index.tolist()))
    label = np.zeros(df.shape[0])
    label[[i for i in range(df.shape[0]) if df.index.tolist()[i] in list(ID)]] = 1

    # Calculate AUROC
    score_raw = df.copy().loc[:, cell_fate]
    score_raw[np.isnan(score_raw)] = 0
    auroc_mon = roc_auc_score(label, (score_raw))
    mon_auc_cr.append(auroc_mon)

# %% [markdown]
# ## Dynamo's driver identification

# %%
HSC_Ery_ranking["filter"] = [i in TF for i in list(HSC_Ery_ranking["all"])]
HSC_Mon_ranking["filter"] = [i in TF for i in list(HSC_Mon_ranking["all"])]

# %%
HSC_Ery_ranking = HSC_Ery_ranking.query("filter == True")
HSC_Mon_ranking = HSC_Mon_ranking.query("filter == True")

# %%
ID = set(Ery_driver).intersection(set(HSC_Ery_ranking.iloc[:, 0].tolist()))
label = np.zeros(HSC_Ery_ranking.shape[0])
label[[i for i in range(HSC_Ery_ranking.shape[0]) if HSC_Ery_ranking.iloc[:, 0].tolist()[i] in list(ID)]] = 1

score = -1 * np.array(range(HSC_Ery_ranking.shape[0]))
auroc_ery_dynamo = roc_auc_score(label, score)
print("AUROC:", auroc_ery_dynamo)

# %%
ID = set(Mon_driver).intersection(set(HSC_Mon_ranking.iloc[:, 0].tolist()))
label = np.zeros(HSC_Mon_ranking.shape[0])
label[[i for i in range(HSC_Mon_ranking.shape[0]) if HSC_Mon_ranking.iloc[:, 0].tolist()[i] in list(ID)]] = 1

score = -1 * np.array(range(HSC_Mon_ranking.shape[0]))
auroc_mon_dynamo = roc_auc_score(label, score)
print("AUROC:", auroc_mon_dynamo)

# %% [markdown]
# ## Plot driver ranking results

# %%
data = pd.DataFrame(
    {
        "AUROC": ery_auc_rgv + mon_auc_rgv + ery_auc_cr + mon_auc_cr + [auroc_ery_dynamo, auroc_mon_dynamo],
        "Terminal state": ["Ery"] * 3 + ["Mon"] * 3 + ["Ery"] * 3 + ["Mon"] * 3 + ["Ery", "Mon"],
        "Method": ["RegVelo(PS)"] * 6 + ["RegVelo(CR)"] * 6 + ["dynamo(LAP)"] * 2,
    }
)

# %%
with mplscience.style_context():
    pal = METHOD_PALETTE_DRIVER
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(4, 3))

    sns.barplot(x="Terminal state", y="AUROC", hue="Method", data=data, ci="sd", capsize=0.1, errwidth=2, palette=pal)
    sns.stripplot(
        x="Terminal state", y="AUROC", hue="Method", data=data, dodge=True, jitter=True, color="black", alpha=0.7
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[3:6], labels[3:6], bbox_to_anchor=(0.5, -0.44), loc="lower center", ncol=2, fontsize=14)

    plt.ylabel("AUROC", fontsize=14)
    plt.xlabel("", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if SAVE_FIGURES:
        plt.savefig(FIG_DIR / DATASET / "driver_ranking.svg", format="svg", transparent=True, bbox_inches="tight")
    plt.show()

# %%

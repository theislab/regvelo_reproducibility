# %% [markdown]
# # Evaluate RegVelo predicted E2f1 downstreamed targets
#
# Notebooks for E2f1 regulatory network analysis

# %% [markdown]
# ## Library imports

# %%
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import torch
from sklearn import metrics

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

import scanpy as sc
import scvelo as scv
import scvi
from regvelo import REGVELOVI

from rgv_tools import DATA_DIR, FIG_DIR

# %% [markdown]
# ## General settings

# %%
scvi.settings.seed = 0

# %%
plt.rcParams["svg.fonttype"] = "none"
sns.reset_defaults()
sns.reset_orig()
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=14, color_map="viridis")

# %% [markdown]
# ## Constants

# %%
DATASET = "pancreatic_endocrine"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / DATASET / "processed" / "cell_cycle_repeat_runs").mkdir(parents=True, exist_ok=True)

# %%
SAVE_FIGURES = False
if SAVE_FIGURES:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Function definations


# %%
def calculate_aurocs(targets, thresholds):
    """Calculate AUROC."""
    auroc_scores_1 = []
    auroc_scores_2 = []

    for threshold in thresholds:
        # Convert continuous ground truth to binary based on threshold
        chip_targets = np.array(chip_target.index.tolist())[chip_target.iloc[:, 0] > threshold].tolist()
        targets.loc[:, "gt"] = 0
        targets.loc[:, "gt"][targets.index.isin(chip_targets)] = 1

        fpr, tpr, thresholds = metrics.roc_curve(targets.loc[:, "gt"], targets.loc[:, "prior"])
        auroc_1 = metrics.auc(fpr, tpr)
        fpr, tpr, thresholds = metrics.roc_curve(targets.loc[:, "gt"], targets.loc[:, "weight"])
        auroc_2 = metrics.auc(fpr, tpr)

        auroc_scores_1.append(auroc_1)
        auroc_scores_2.append(auroc_2)

    return auroc_scores_1, auroc_scores_2


# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed.h5ad")

# Prepare skeleton
W = adata.uns["skeleton"].copy()
W = torch.tensor(np.array(W)).int()

# Prepare TF
TF = adata.var_names[adata.var["TF"]]

# %% [markdown]
# Loading E2f1 ChIP-seq targets as ground truth

# %%
chip_target = pd.read_csv(DATA_DIR / DATASET / "raw" / "E2f1_target.csv", index_col=0, sep="\t")

## filter targets only keep significant edges
chip_targets = np.array(chip_target.index.tolist())[chip_target.iloc[:, 0] > 200].tolist()

# %%
scv.tl.score_genes_cell_cycle(adata)

# %% [markdown]
# ### Repeat run model
#
# Under `soft_mode` due to the number of gene regulation parameter need to be estimated, we can repeat run models for five times, and aggregate inferred GRN to get robust estimation

# %%
SELECT_CELLS = (adata.obs["clusters"] == "Ductal") & (adata.obs["phase"] == "S")

# %%
## repeat models
for nrun in range(5):
    print("training model...")
    REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    vae = REGVELOVI(adata, W=W.T, regulators=TF)
    vae.train()

    print("save model...")
    model_name = "rgv_model_" + str(nrun)
    model = DATA_DIR / DATASET / "processed" / "cell_cycle_repeat_runs" / model_name
    vae.save(model)

# %%
grns = []
for nrun in range(0, 5):
    model = DATA_DIR / DATASET / "processed" / "cell_cycle_repeat_runs" / f"rgv_model_{nrun}"
    ### load model
    vae = REGVELOVI.load(model, adata)
    # grn1 = reg_vae.module.v_encoder.fc1.weight.data.cpu().detach().clone()
    grns.append(
        vae.module.v_encoder.GRN_Jacobian(torch.tensor(adata.layers["Ms"][SELECT_CELLS, :]).to("cuda:0"))
        .cpu()
        .detach()
        .clone()
    )

GRN = torch.mean(torch.stack(grns), dim=0).clone()

# %% [markdown]
# ## Targets analysis

# %%
targets = GRN[:, [i == "E2f1" for i in adata.var.index]].detach().cpu().numpy()

## load prior GRN
prior = vae.module.v_encoder.mask_m_raw[:, [i == "E2f1" for i in adata.var.index]].detach().cpu().numpy()

# %%
targets = pd.DataFrame(targets, index=adata.var.index)
targets.loc[:, "weight"] = targets.iloc[:, 0].abs()
targets.loc[:, "prior"] = prior

# %% [markdown]
# ### Define ground truth

# %%
targets.loc[:, "gt"] = 0
targets.loc[:, "gt"][targets.index.isin(chip_targets)] = 1

# %% [markdown]
# ### Evaluate performance

# %%
fpr, tpr, thresholds = metrics.roc_curve(targets.loc[:, "gt"], targets.loc[:, "prior"])
metrics.auc(fpr, tpr)

# %%
fpr, tpr, thresholds = metrics.roc_curve(targets.loc[:, "gt"], targets.loc[:, "weight"])
metrics.auc(fpr, tpr)

# %%
# Define thresholds using quantiles
np.random.seed(0)
num_thresholds = 100
quantile_99 = np.quantile(chip_target.iloc[:, 0], 0.99)
thresholds = np.linspace(0, quantile_99, num_thresholds)

# Calculate AUROC scores for different thresholds
auroc_scores_prior, auroc_scores_regvelo = calculate_aurocs(targets, thresholds)

plt.rcParams["svg.fonttype"] = "none"
with mplscience.style_context():
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(4, 2))
    data = pd.DataFrame(
        {
            "AUROC": auroc_scores_prior + auroc_scores_regvelo,
            "GRN": ["prior network"] * len(auroc_scores_prior)
            + ["RegVelo inferred network"] * len(auroc_scores_regvelo),
        }
    )

    # Plot the boxplot
    sns.boxplot(y="GRN", x="AUROC", data=data)

    plt.xlabel("GRN")
    plt.ylabel("AUROC Score")
    plt.title("E2f1 targets prediction")

    if SAVE_FIGURES:
        fig.savefig(FIG_DIR / DATASET / "GRN_boxplot.svg", format="svg", transparent=True, bbox_inches="tight")

# %%
scipy.stats.wilcoxon(auroc_scores_prior, auroc_scores_regvelo)

# %% [markdown]
# ## Visualize GRN

# %%
GRN_visualize = targets.sort_values("weight", ascending=False).iloc[:20, :]

# %%
graph = pd.DataFrame(
    {"from": ["E2f1"] * 20, "to": GRN_visualize.index.tolist(), "status": GRN_visualize.loc[:, "prior"]}
)

color_map = ["skyblue"]
for node in GRN_visualize.index.tolist():
    if graph.loc[graph["to"] == node, "status"].values[0] == 1:
        color_map.append("darkgrey")
    else:
        color_map.append("lightgrey")

# %%
G = nx.from_pandas_edgelist(graph, source="from", target="to", create_using=nx.DiGraph())

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, arrowsize=10, node_color=color_map)

if SAVE_FIGURES:
    fig.savefig(FIG_DIR / DATASET / "E2f1_GRN.svg", format="svg", transparent=True, bbox_inches="tight")

# %%

# %%

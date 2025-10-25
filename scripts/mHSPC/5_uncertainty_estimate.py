# %% [markdown]
# # GRN calibration evaluation

# %% [markdown]
# ## Library import

# %%
import copy

from inferelator.postprocessing.model_metrics import RankSummaryPR, RankSummingMetric
import numpy as np
import pandas as pd
import sklearn

from matplotlib import pyplot as plt

import scanpy as sc
import scvi
from regvelo import REGVELOVI

from rgv_tools import DATA_DIR, FIG_DIR

# %% [markdown]
# ## General setting

# %%
scvi.settings.seed = 0

# %% [markdown]
# ## Constants

# %%
DATASET = "mHSPC"

# %%
SAVE_DATA = True
SAVE_FIGURE = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)
if SAVE_FIGURE:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Define functions
#
# We followed the GRN calibration evaluation workflow provided by PMF-GRN, please check https://github.com/nyu-dl/pmf-grn/blob/main/evaluate_calibration.py

# %%
RankSummingMetricCopy = copy.deepcopy(RankSummingMetric)


def get_calibration_score(to_eval, gold_standard, filter_method="overlap", method="auroc"):
    """
    Compute a calibration score comparing predictions to a gold standard.

    Parameters
    ----------
    to_eval : list or DataFrame
        Predictions or scores to evaluate.
    gold_standard : list or DataFrame
        True labels for evaluation.
    filter_method : str, optional
        Method for filtering data before scoring (default is "overlap").
    method : str, optional
        Scoring metric: "auroc" or "auprc" (default is "auroc").

    Returns
    -------
    float
        Area under the ROC or PR curve as the calibration score.
    """
    metrics = RankSummingMetricCopy([to_eval], gold_standard, filter_method)

    if method == "auprc":
        data = RankSummaryPR.calculate_precision_recall(metrics.filtered_data)
        auc = RankSummaryPR.calculate_aupr(data)
    elif method == "auroc":
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(
            metrics.filtered_data["gold_standard"], metrics.filtered_data["combined_confidences"]
        )
        auc = sklearn.metrics.auc(fpr, tpr)
    return auc


# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / "mHSC_ExpressionData.h5ad")

# %%
TF = pd.read_csv(DATA_DIR / DATASET / "raw" / "mouse-tfs.csv")
TF = [i[0].upper() + i[1:].lower() for i in TF["TF"].tolist()]

# %%
TF = np.array(TF)[[i in adata.var_names for i in TF]]

# %%
TF

# %% [markdown]
# ### Load ground truth (ChIP-seq)

# %%
gt = pd.read_csv(DATA_DIR / DATASET / "raw" / "mHSC-ChIP-seq-network.csv")
gt["Gene1"] = [i[0].upper() + i[1:].lower() for i in gt["Gene1"].tolist()]
gt["Gene2"] = [i[0].upper() + i[1:].lower() for i in gt["Gene2"].tolist()]
gt = gt.loc[[i in TF for i in gt["Gene1"]], :]
gt = gt.loc[[i in adata.var_names for i in gt["Gene2"]], :]

p_class = pd.DataFrame(0, index=adata.var_names, columns=TF)

for _, row in gt.iterrows():
    reg = row["Gene1"]
    tar = row["Gene2"]
    if tar in p_class.index and reg in p_class.columns:
        p_class.loc[tar, reg] = 1

# %% [markdown]
# ## Running GRN to calculate caliberate estimate of error

# %%
REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")

for nrun in range(10):
    vae = REGVELOVI(adata, regulators=TF)
    vae.train()

    vae.save(DATA_DIR / DATASET / "processed" / f"hsc_model_run_{nrun}")

# %%
grns = []
for nrun in range(10):
    path = DATA_DIR / DATASET / "processed" / f"hsc_model_run_{nrun}"
    vae = REGVELOVI.load(path, adata)

    w = vae.module.v_encoder.fc1.weight.cpu().detach()
    w = pd.DataFrame(w, index=adata.var_names, columns=adata.var_names)
    w = w.loc[:, TF]
    grns.append(w)

# %%
stacked = np.stack(grns)
median = np.median(stacked, axis=0)
epsilon = np.percentile(np.abs(median)[median != 0], 10)
edge_variance = (np.median(np.abs(stacked - median), axis=0) + epsilon) / (np.abs(median) + epsilon)

# %%
edge_variance = pd.DataFrame(edge_variance, index=p_class.index, columns=p_class.columns)
grn = pd.DataFrame(median, index=p_class.index, columns=p_class.columns)

# %%
percentile_values = np.percentile(edge_variance, np.arange(1, 11) * 10)
auprcs = []
for i in range(2, len(percentile_values)):
    to_eval = copy.deepcopy(np.abs(grn))
    to_eval[edge_variance > percentile_values[i]] = np.nan
    auprcs.append(get_calibration_score(to_eval, p_class, filter_method="overlap", method="auprc"))

# %%
auprcs

# %%
plt.rcParams["svg.fonttype"] = "none"
plt.plot(np.arange(1, len(auprcs) + 1) * 100 / len(auprcs), np.array(auprcs))
plt.xlim(0, 100)
plt.grid()
plt.xlabel("Percentile Cutoff", fontsize=18)
plt.ylabel("AUPRC", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

if SAVE_FIGURE:
    plt.savefig(FIG_DIR / DATASET / "GRN_calibration_auprc.svg", format="svg", transparent=True, bbox_inches="tight")

# %%
auroc = []
for i in range(2, len(percentile_values)):
    to_eval = copy.deepcopy(np.abs(grn))
    to_eval[edge_variance > percentile_values[i]] = np.nan
    auroc.append(get_calibration_score(to_eval, p_class, filter_method="overlap", method="auroc"))

# %%
plt.plot(np.arange(1, len(auroc) + 1) * 100 / len(auroc), auroc)
plt.xlim(0, 100)
plt.grid()
plt.xlabel("Percentile Cutoff", fontsize=18)
plt.ylabel("AUROC", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

if SAVE_FIGURE:
    plt.savefig(FIG_DIR / DATASET / "GRN_calibration_roc.svg", format="svg", transparent=True, bbox_inches="tight")

# %%

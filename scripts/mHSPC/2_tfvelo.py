# %% [markdown]
# # Run TFvelo on mHSPC datasets

# %% [markdown]
# ## Library import

# %%
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
import scanpy as sc
import scipy

import anndata as ad
import scvi

import sys

import TFvelo as TFv

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import (
    set_output,
)

from itertools import product, permutations
from operator import pos

# %% [markdown]
# ## Constants

# %%
DATASET = "mHSPC"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Define functions
#
# We followed the GRN benchmark workflow provided by BEELINE, please check https://github.com/Murali-group/Beeline


# %%
def unsigned(true_edges: pd.DataFrame, pred_edges: pd.DataFrame, type: str = "alledges") -> tuple[float, float, float]:
    """
    Compare true vs predicted edges (unsigned) and compute precision/recall metrics.

    Returns:
        tuple: (eprec, erec, eprec_ratio)
    """
    true_edges_copy = true_edges.copy()
    pred_edges_copy = pred_edges.copy()

    # Drop self-edges and duplicates
    true_edges_copy = true_edges_copy.loc[(true_edges_copy["Gene1"] != true_edges_copy["Gene2"])]
    true_edges_copy.drop_duplicates(keep="first", inplace=True)
    true_edges_copy.reset_index(drop=True, inplace=True)

    pred_edges_copy = pred_edges_copy.loc[(pred_edges_copy["Gene1"] != pred_edges_copy["Gene2"])]
    pred_edges_copy.drop_duplicates(keep="first", inplace=True)
    pred_edges_copy.reset_index(drop=True, inplace=True)

    # Get a list of all possible TF to gene interactions
    unique_nodes = np.unique(true_edges_copy.loc[:, ["Gene1", "Gene2"]])
    possible_edges_all = set(product(set(true_edges_copy.Gene1), set(unique_nodes)))

    # Get a list of all possible interactions
    possible_edges_no_self = set(permutations(unique_nodes, r=2))

    # Find intersection of above lists to ignore self edges
    possible_edges = possible_edges_all.intersection(possible_edges_no_self)

    true_edges_dict = {"|".join(p): 0 for p in possible_edges}

    true_edges_str = true_edges_copy["Gene1"] + "|" + true_edges_copy["Gene2"]
    true_edges_str = true_edges_str[true_edges_str.isin(true_edges_dict)]
    n_edges = len(true_edges_str)

    pred_edges_copy["Edges"] = pred_edges_copy["Gene1"] + "|" + pred_edges_copy["Gene2"]
    pred_edges_copy = pred_edges_copy[pred_edges_copy["Edges"].isin(true_edges_dict)]
    pred_edges_copy_copy = pred_edges_copy.copy()

    if not pred_edges_copy.shape[0] == 0:
        pred_edges_copy.loc[:, "EdgeWeight"] = pred_edges_copy.EdgeWeight.round(6).abs()
        pred_edges_copy.sort_values(by="EdgeWeight", ascending=False, inplace=True)

        maxk = min(pred_edges_copy.shape[0], n_edges)
        edge_weight_topk = pred_edges_copy.iloc[maxk - 1].EdgeWeight

        nnz_min = np.nanmin(pred_edges_copy.EdgeWeight.replace(0, np.nan).values)
        best_val = max(nnz_min, edge_weight_topk)

        newDF = pred_edges_copy.loc[(pred_edges_copy["EdgeWeight"] >= best_val)]
        rank = set(newDF["Gene1"] + "|" + newDF["Gene2"])

        intersectionSet = rank.intersection(true_edges_str)
        eprec = len(intersectionSet) / len(rank)
        erec = len(intersectionSet) / len(true_edges_str)

        random_eprec = n_edges / len(true_edges_dict)
        eprec_ratio = eprec / random_eprec
    else:
        eprec = 1.0
        erec = 1.0
        eprec_ratio = 1.0

    print("EPR: " + str(eprec_ratio))
    return eprec, erec, eprec_ratio


def calculate_auroc(inferred_scores_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> float:
    """
    Calculate AUROC comparing inferred edge scores against ground truth.

    Returns:
        float: AUROC score.
    """
    ground_truth_set = set(zip(ground_truth_df["Gene1"], ground_truth_df["Gene2"]))

    inferred_scores_df["label"] = inferred_scores_df.apply(
        lambda row: (row["Gene1"], row["Gene2"]) in ground_truth_set, axis=1
    ).astype(int)

    y_true = inferred_scores_df["label"]
    y_scores = inferred_scores_df["EdgeWeight"]

    auroc = roc_auc_score(y_true, y_scores)
    return auroc


# %%
def run_TFvelo(adata, TF):
    """TFvelo pipeline."""
    adata.X = adata.X.toarray()
    n_gene = adata.shape[1]
    adata.varm["TFs"] = np.full([n_gene, n_gene], "blank")
    adata.varm["TFs"] = adata.varm["TFs"].astype("U10")

    adata.varm["TFs_id"] = np.full([n_gene, n_gene], -1)
    adata.varm["TFs_times"] = np.full([n_gene, n_gene], 0)
    adata.varm["TFs_correlation"] = np.full([n_gene, n_gene], 0.0)
    adata.varm["knockTF_Log2FC"] = np.full([n_gene, n_gene], 0.0)
    adata.var["n_TFs"] = np.zeros(n_gene, dtype=int)

    gene_names = adata.var_names.tolist()  # all genes as targets
    all_TFs = list(TF)  # select TFs

    for TF_name in all_TFs:
        TF_idx = gene_names.index(TF_name)
        TF_expression = np.ravel(adata[:, TF_name].X)

        for target in gene_names:
            target_idx = gene_names.index(target)
            if target == TF_name:
                continue

            if TF_name in adata.varm["TFs"][target_idx]:
                ii = list(adata.varm["TFs"][target_idx]).index(TF_name)
                adata.varm["TFs_times"][target_idx, ii] += 1
                continue
            target_expression = np.ravel(adata[:, target].X)
            flag = (TF_expression > 0) & (target_expression > 0)  # consider all possible regulation
            if flag.sum() < 2:
                correlation = 0
            else:
                correlation, _ = scipy.stats.spearmanr(target_expression[flag], TF_expression[flag])

            tmp_n_TF = adata.var["n_TFs"][target_idx]
            adata.varm["TFs"][target_idx][tmp_n_TF] = TF_name
            adata.varm["TFs_id"][target_idx][tmp_n_TF] = TF_idx
            adata.varm["TFs_times"][target_idx, tmp_n_TF] = 1
            adata.varm["TFs_correlation"][target_idx, tmp_n_TF] = correlation
            adata.var["n_TFs"][target_idx] += 1

    TFv.tl.recover_dynamics(
        adata,
        n_jobs=64,
        max_iter=20,
        var_names="all",
        WX_method="lsq_linear",
        WX_thres=20,
        n_top_genes=adata.shape[1],
        fit_scaling=True,
        use_raw=0,
        init_weight_method="ones",
        n_time_points=1000,
    )
    n_cells = adata.shape[0]
    expanded_scaling_y = np.expand_dims(np.array(adata.var["fit_scaling_y"]), 0).repeat(n_cells, axis=0)
    adata.layers["velocity"] = adata.layers["velo_hat"] / expanded_scaling_y
    return adata


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
# ## Velocity pipeline

# %%
adata.layers["total"] = adata.X
adata.layers["total_raw"] = adata.layers["total"].copy()

TFv.pp.moments(adata, n_pcs=None, n_neighbors=None)

adata = run_TFvelo(adata, TF)

# %%
grn_estimate = pd.DataFrame(0, index=adata.var_names, columns=adata.var_names)
grn_estimate.loc[:, TF] = adata.varm["fit_weights_final"]
grn_estimate = np.array(grn_estimate)

# %%
grn_estimate = np.abs(grn_estimate)

# %%
grn_estimate = pd.DataFrame(grn_estimate, index=adata.var_names.tolist(), columns=adata.var_names.tolist())

# %%
grn_estimate = grn_estimate.loc[:, TF].copy()

# %%
grn = pd.DataFrame(grn_estimate.stack()).reset_index()
grn.columns = ["Gene2", "Gene1", "EdgeWeight"]

# %%
result = grn[["Gene1", "Gene2", "EdgeWeight"]].sort_values(by="EdgeWeight", ascending=False).reset_index(drop=True)

# %% [markdown]
# ## Load ground truth GRN

# %%
gt = pd.read_csv(DATA_DIR / DATASET / "raw" / "mHSC-ChIP-seq-network.csv")

# %%
gt["Gene1"] = [i[0].upper() + i[1:].lower() for i in gt["Gene1"].tolist()]
gt["Gene2"] = [i[0].upper() + i[1:].lower() for i in gt["Gene2"].tolist()]
gt = gt.loc[[i in TF for i in gt["Gene1"]], :]
gt = gt.loc[[i in adata.var_names for i in gt["Gene2"]], :]
gt

# %% [markdown]
# ## Result

# %%
_, _, EPR_score = unsigned(gt, result)

# %%
AUC_score = calculate_auroc(result, gt)

# %%
result_df = pd.DataFrame({"EPR": EPR_score, "AUC": AUC_score, "Method": ["tfvelo"]})

if SAVE_DATA:
    result_df.to_csv(DATA_DIR / DATASET / "results" / "GRN_benchmark_tfv.csv")

# %%

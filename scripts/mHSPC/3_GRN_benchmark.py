# %% [markdown]
# # Run other GRN inference methods on mHSPC dataset

# %% [markdown]
# ## Library imports

# %%
from itertools import permutations, product

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import celloracle as co
import scanpy as sc
from arboreto.algo import grnboost2

from rgv_tools import DATA_DIR

# %% [markdown]
# ## General setting

# %%
# %matplotlib inline

# %% [markdown]
# ## Constants

# %%
DATASET = "mHSPC"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Functions definations
# We followed the GRN benchmark workflow provided by BEELINE, please check https://github.com/Murali-group/Beeline


# %%
def unsigned(true_edges: pd.DataFrame, pred_edges: pd.DataFrame, type: str = "alledges") -> tuple[float, float, float]:
    """Compare true vs predicted edges (unsigned) and compute precision/recall metrics.

    Returns
    -------
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
    pred_edges_copy.copy()

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
    """Calculate AUROC comparing inferred edge scores against ground truth.

    Returns
    -------
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


# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / "mHSC_ExpressionData.h5ad")

# %%
gt = pd.read_csv(DATA_DIR / DATASET / "raw" / "mHSC-ChIP-seq-network.csv")
TF = pd.read_csv(DATA_DIR / DATASET / "raw" / "mouse-tfs.csv")
TF = [i[0].upper() + i[1:].lower() for i in TF["TF"].tolist()]
TF = np.array(TF)[[i in adata.var_names for i in TF]]

# %%
gt["Gene1"] = [i[0].upper() + i[1:].lower() for i in gt["Gene1"].tolist()]
gt["Gene2"] = [i[0].upper() + i[1:].lower() for i in gt["Gene2"].tolist()]
gt = gt.loc[[i in TF for i in gt["Gene1"]], :]
gt = gt.loc[[i in adata.var_names for i in gt["Gene2"]], :]
gt

# %% [markdown]
# ## Pearson correlation

# %%
grn_estimate = adata.to_df(layer="Ms").corr().values
np.fill_diagonal(grn_estimate, 0)

grn_estimate = np.abs(grn_estimate)
grn_estimate = pd.DataFrame(grn_estimate, index=adata.var_names.tolist(), columns=adata.var_names.tolist())

grn_estimate = grn_estimate.loc[:, TF].copy()

grn = pd.DataFrame(grn_estimate.stack()).reset_index()
grn.columns = ["Gene2", "Gene1", "EdgeWeight"]

result = grn[["Gene1", "Gene2", "EdgeWeight"]].sort_values(by="EdgeWeight", ascending=False).reset_index(drop=True)

# %%
result.shape

# %%
_, _, epr_cor = unsigned(gt, result)

# %%
auc_cor = calculate_auroc(result, gt)
auc_cor

# %% [markdown]
# ## GRNBoost2

# %%
network = grnboost2(expression_data=adata.to_df(layer="Ms"), tf_names=TF.tolist())
grn_estimate = pd.pivot(network, index="target", columns="TF").fillna(0).values.T

# %%
network.columns = ["Gene1", "Gene2", "EdgeWeight"]

# %%
_, _, epr_gbt2 = unsigned(gt, network)

# %%
auc_gbt2 = calculate_auroc(network, gt)
auc_gbt2

# %% [markdown]
# ## Using CellOracle

# %%
base_grn = np.ones((len(TF), adata.n_vars))
base_grn = pd.DataFrame(base_grn, index=TF, columns=adata.var_names)
base_grn["peak_id"] = ["peak_" + i for i in TF]
base_grn["gene_short_name"] = TF
base_grn = base_grn[["peak_id", "gene_short_name"] + adata.var_names.to_list()]

# %%
net = co.Net(gene_expression_matrix=adata.to_df(layer="Ms"), TFinfo_matrix=base_grn, verbose=False)
net.fit_All_genes(bagging_number=100, alpha=1, verbose=False)
net.updateLinkList(verbose=False)

# %%
network = net.linkList[["source", "target", "coef_mean"]]

# %%
network.columns = ["Gene1", "Gene2", "EdgeWeight"]

# %%
network["EdgeWeight"] = np.abs(network["EdgeWeight"])

# %%
_, _, epr_co = unsigned(gt, network)

# %%
auc_co = calculate_auroc(network, gt)
auc_co

# %% [markdown]
# ## Results

# %%
result_df = pd.DataFrame(
    {
        "EPR": [epr_cor, epr_gbt2, epr_co],
        "AUC": [auc_cor, auc_gbt2, auc_co],
        "Method": ["Corr", "GRNBoost2", "CellOracle"],
    }
)

if SAVE_DATA:
    result_df.to_csv(DATA_DIR / DATASET / "results" / "GRN_benchmark.csv")

# %%

# %% [markdown]
# # Calculate velocity and latent time using TFvelo
#
# Notebook benchmarks velocity and latent time inference using TFvelo on dyngen-generated data.

# %% [markdown]
# ## Library imports

# %%
import sys

import TFvelo as TFv

import numpy as np
import pandas as pd
import scipy
import torch
from sklearn.metrics import roc_auc_score

import anndata as ad

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import get_velocity_correlation

sys.path.insert(0, "/lustre/groups/ml01/workspace/yifan.chen/TFvelo/")

# %% [markdown]
# ## Function definitions


# %%
def run_TFvelo(adata):
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
    all_TFs = list(adata.var_names[adata.var["is_tf"]])  # select TFs

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
# ## Constants

# %%
DATASET = "dyngen"

# %%
COMPLEXITY = "complexity_1"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / COMPLEXITY / "results").mkdir(parents=True, exist_ok=True)

# %%
SAVE_DATASETS = True
if SAVE_DATASETS:
    (DATA_DIR / DATASET / COMPLEXITY / "trained_tfvelo").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Velocity pipeline

# %%
velocity_correlation_all = []
velocity_correlation = []
grn_correlation = []

cnt = 0
for filename in (DATA_DIR / DATASET / COMPLEXITY / "processed").iterdir():
    torch.cuda.empty_cache()
    if filename.suffix != ".zarr":
        continue

    simulation_id = int(filename.stem.removeprefix("simulation_"))
    print(f"Run {cnt}, dataset {simulation_id}.")

    adata = ad.io.read_zarr(filename)

    if "spliced" in adata.layers:
        adata.layers["total"] = adata.layers["spliced"].todense() + adata.layers["unspliced"].todense()
    elif "new" in adata.layers:
        adata.layers["total"] = np.array(adata.layers["total"].todense())
    else:
        adata.layers["total"] = adata.X
    adata.layers["total_raw"] = adata.layers["total"].copy()

    TFv.pp.moments(adata, n_pcs=None, n_neighbors=None)

    adata = run_TFvelo(adata)

    # save data
    adata.write_zarr(DATA_DIR / DATASET / COMPLEXITY / "trained_tfvelo" / f"trained_{simulation_id}.zarr")

    velo_corr = get_velocity_correlation(
        ground_truth=adata.layers["true_velocity"], estimated=adata.layers["velocity"], aggregation=None
    )

    velocity_correlation_all.append(velo_corr)

    velo_corr = np.array(velo_corr)

    velocity_correlation.append(np.mean(velo_corr[~np.isnan(velo_corr)]))

    grn_true = adata.uns["true_skeleton"]
    grn_sc_true = adata.uns["true_sc_grn"]

    grn_estimate = pd.DataFrame(0, index=adata.var_names, columns=adata.var_names)
    grn_estimate.loc[:, adata.var_names[adata.var["is_tf"]]] = adata.varm["fit_weights_final"]
    grn_estimate = np.array(grn_estimate)

    grn_auroc = []
    for cell_id in range(adata.n_obs):
        ground_truth = grn_sc_true[:, :, cell_id]

        if ground_truth.sum() > 0:
            ground_truth = ground_truth.T[np.array(grn_true.T) == 1]
            ground_truth[ground_truth != 0] = 1

            estimated = grn_estimate[np.array(grn_true.T) == 1]
            estimated = np.abs(estimated)

            number = min(10000, len(ground_truth))
            estimated, index = torch.topk(torch.tensor(estimated), number)

            grn_auroc.append(roc_auc_score(ground_truth[index], estimated))

    grn_correlation.append(np.mean(grn_auroc))

    cnt += 1

# %%
pd.DataFrame({"velocity": velocity_correlation, "grn": grn_correlation})

# %%
if SAVE_DATA:
    pd.DataFrame({"velocity": velocity_correlation, "grn": grn_correlation}).to_parquet(
        path=DATA_DIR / DATASET / COMPLEXITY / "results" / "tfvelo_correlation.parquet"
    )

# %%
if SAVE_DATA:
    df = pd.DataFrame(velocity_correlation_all).T
    df.columns = df.columns.astype(str)
    df.to_parquet(path=DATA_DIR / DATASET / COMPLEXITY / "results" / "tfvelo_correlation_all.parquet")

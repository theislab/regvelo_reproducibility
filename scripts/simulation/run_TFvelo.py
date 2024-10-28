# %% [markdown]
# # Calculate velocity and latent time using TFvelo

# %% [markdown]
# ## Library imports

# %%
import os
import sys

import TFvelo as TFv
from paths import DATA_DIR, FIG_DIR

import numpy as np

# %%
import scipy

import anndata as ad
import scanpy as sc

# attach your local TFvelo repo directory
sys.path.append("/home/itg/z.xue/VeloBenchmark/TFvelo")

sys.path.append("../..")

# %% [markdown]
# ## General settings

# %%
np.set_printoptions(suppress=True)
SAVE_FIGURES = True
if SAVE_FIGURES:
    os.makedirs(FIG_DIR / "simulation", exist_ok=True)

SAVE_DATASETS = True
if SAVE_DATASETS:
    os.makedirs(DATA_DIR / "simulation", exist_ok=True)

# %%
input_path = DATA_DIR
output_path = DATA_DIR / "simulation"
input_files = os.listdir(input_path)


# %% [markdown]
# ## Function definitions


# %%
def run_TFvelo(input_path, output_path, input_file):
    """TODO."""
    adata = ad.read(os.path.join(input_path, input_file))
    print("Start processing " + os.path.join(input_path, input_file))
    adata.layers["spliced"] = adata.layers["counts_spliced"].copy()
    adata.layers["unspliced"] = adata.layers["counts_unspliced"].copy()

    if "spliced" in adata.layers:
        adata.layers["total"] = adata.layers["spliced"] + adata.layers["unspliced"]
    elif "new" in adata.layers:
        adata.layers["total"] = np.array(adata.layers["total"].todense())
    else:
        adata.layers["total"] = adata.X
    adata.layers["count"] = adata.X.copy()
    adata.layers["total_raw"] = adata.layers["total"].copy()
    n_cells, n_genes = adata.X.shape
    sc.pp.filter_genes(adata, min_cells=int(n_cells / 50))
    sc.pp.filter_cells(adata, min_genes=int(n_genes / 50))
    TFv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000, log=True)  # include the following steps
    adata.X = adata.layers["total"].copy()

    gene_names = []
    for tmp in adata.var_names:
        gene_names.append(tmp.upper())
    adata.var_names = gene_names
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    TFv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    adata.X = adata.X.A
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
# ## Data loading and processing of one instance

# %%
adata = run_TFvelo(input_path, output_path, input_files[0])

# %%
# fit_t and velocity are the computed results
adata

# %%
# save the results
if SAVE_DATASETS:
    adata.write_h5ad(DATA_DIR / "simulation" / "c2f_output.h5ad")

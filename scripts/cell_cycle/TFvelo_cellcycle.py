# %% [markdown]
# ## Run TFvelo

# %%
import pandas as pd
import anndata as ad
import scanpy as sc
import TFvelo as TFv

import numpy as np

import scvelo as scv
import matplotlib

# matplotlib.use('AGG')
import os, sys
import scipy

np.set_printoptions(suppress=True)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# from _calculation import get_gams
sys.path.append("../..")
from paths import DATA_DIR, FIG_DIR

# %% [markdown]
# ## Load the dataset and preprocessing according to input requirements of TFvelo

# %%
adata = sc.read("RegVelo_datasets/cell cycle/adata.h5ad")
adata.X = adata.layers["spliced"].copy()

del adata.layers["ambiguous"]
del adata.layers["matrix"]
del adata.layers["spanning"]

adata.obs.drop(
    [
        "Well_Plate",
        "plate",
        "MeanGreen530",
        "MeanRed585",
        "initial_size_unspliced",
        "initial_size_spliced",
        "initial_size",
    ],
    axis=1,
    inplace=True,
)
adata.var_names = adata.var["name"].values
adata.var.drop(adata.var.columns, axis=1, inplace=True)
adata.obs["pseudo_clusters"] = pd.cut(adata.obs["fucci_time"], bins=5, labels=False).astype(str).astype("category")
adata.obs["pseudo_clusters_equal_size"] = pd.qcut(adata.obs["fucci_time"], q=5, labels=False)
adata.obs["pseudo_clusters_equal_size_num"] = adata.obs["pseudo_clusters_equal_size"].astype(float)
adata.obs["cell_cycle_rad"] = adata.obs["fucci_time"] / adata.obs["fucci_time"].max() * 2 * np.pi
adata.uns["genes_all"] = np.array(adata.var_names)

if "spliced" in adata.layers:
    adata.layers["total"] = adata.layers["spliced"].todense() + adata.layers["unspliced"].todense()
elif "new" in adata.layers:
    adata.layers["total"] = np.array(adata.layers["total"].todense())
else:
    adata.layers["total"] = adata.X
adata.layers["total_raw"] = adata.layers["total"].copy()
n_cells, n_genes = adata.X.shape
sc.pp.filter_genes(adata, min_cells=int(n_cells / 50))
sc.pp.filter_cells(adata, min_genes=int(n_genes / 50))
TFv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000, log=True)  # include the following steps
adata.X = adata.layers["total"].copy()

# %% [markdown]
# ## Running TFvelo with default settings and save the output

# %%
gene_names = []
for tmp in adata.var_names:
    gene_names.append(tmp.upper())
adata.var_names = gene_names
adata.var_names_make_unique()
adata.obs_names_make_unique()

TFv.pp.moments(adata, n_pcs=30)

TFv.pp.get_TFs(adata, databases="ENCODE ChEA")
adata.uns["genes_pp"] = np.array(adata.var_names)
TFv.tl.recover_dynamics(
    adata,
    n_jobs=16,
    max_iter=20,
    var_names="all",
    WX_method="lsq_linear",
    WX_thres=20,
    n_top_genes=2000,
    fit_scaling=True,
    use_raw=0,
    init_weight_method="correlation",
    n_time_points=1000,
)

# %%
losses = adata.varm["loss"].copy()
losses[np.isnan(losses)] = 1e6
adata.var["min_loss"] = losses.min(1)

n_cells = adata.shape[0]
expanded_scaling_y = np.expand_dims(np.array(adata.var["fit_scaling_y"]), 0).repeat(n_cells, axis=0)
adata.layers["velocity"] = adata.layers["velo_hat"] / expanded_scaling_y

adata.write(DATA_DIR / "cell_cycle" / "TFvelo_cellcycle.h5ad")

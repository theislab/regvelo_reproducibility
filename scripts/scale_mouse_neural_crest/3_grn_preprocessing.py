# %% [markdown]
# # Preprocess GRN
#
# Inject pySCENIC learned GRN into RegVelo

# %% [markdown]
# ## Library imports
# %%
import numpy as np

import pandas as pd

import anndata as ad
import scanpy as sc
import scvelo as scv
from velovi import preprocess_data

from rgv_tools import DATA_DIR
from rgv_tools.preprocessing import filter_genes, set_prior_grn

# %% [markdown]
# ## Constants

# %%
DATASET = "mouse_neural_crest"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
TF_all = pd.read_csv("/home/icb/weixu.wang/regulatory_velo/pancreas_dataset/allTFs_mm.txt", header=None)
ldata = ad.io.read_h5ad(DATA_DIR / DATASET / "raw" / "GSE201257_adata_velo_raw.h5ad")

# %% [markdown]
# ## Preprocess for each GRN

# %%
for i in range(1, 5):
    print("loading")
    reg = pd.read_csv(DATA_DIR / DATASET / "processed" / f"regulon_mat_stage_{i}_all_regulons.csv", index_col=0)

    reg.index = reg.index.str.extract(r"(\w+)")[0]
    reg = reg.groupby(reg.index).sum()
    reg[reg != 0] = 1

    TF = np.unique([x.split("(")[0] for x in reg.index.tolist()])
    genes = np.unique(TF.tolist() + reg.columns.tolist())

    GRN = pd.DataFrame(0, index=genes, columns=genes)
    GRN.loc[TF, reg.columns.tolist()] = np.array(reg)

    mask = (GRN.sum(0) != 0) | (GRN.sum(1) != 0)
    GRN = GRN.loc[mask, mask].copy()

    if SAVE_DATA:
        GRN.to_parquet(DATA_DIR / DATASET / "processed" / f"regulon_mat_stage_{i}_processed_all_regulons.parquet")
    print("Done! processed GRN with " + str(reg.shape[0]) + " TF and " + str(reg.shape[1]) + " targets")

# %% [markdown]
# ## Preprocess anndata

# %%
for i in range(1, 5):
    print("loading...")
    adata = scv.read(DATA_DIR / DATASET / "processed" / f"adata_stage{i}_processed.h5ad")
    adata = scv.utils.merge(adata, ldata)
    adata.obsp = None
    del adata.uns["neighbors"]

    print(adata)
    adata.var["TF"] = [i in TF_all.iloc[:, 0].tolist() for i in adata.var_names.tolist()]

    ## select highly variable genes
    scv.pp.filter_genes(adata, min_shared_counts=20)
    scv.pp.normalize_per_cell(adata)

    ## In top 2000 TF is too few, we select all TFs in top 3000
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var["highly_variable"]].copy()

    sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

    ## load GRN
    GRN = pd.read_parquet(DATA_DIR / DATASET / "processed" / f"regulon_mat_stage_{i}_processed_all_regulons.parquet")

    adata = set_prior_grn(adata, GRN.T)
    velocity_genes = preprocess_data(adata.copy()).var_names.tolist()
    TF = adata.var_names[adata.uns["skeleton"].sum(1) != 0]
    var_mask = np.union1d(TF, velocity_genes)

    adata = adata[:, var_mask].copy()
    adata = filter_genes(adata)
    adata = preprocess_data(adata, filter_on_r2=False)

    adata.var["velocity_genes"] = adata.var_names.isin(velocity_genes)
    adata.var["TF"] = adata.var_names.isin(TF)

    print(adata)

    if SAVE_DATA:
        adata.write_h5ad(DATA_DIR / DATASET / "processed" / f"adata_stage{i}_processed_velo_all_regulons.h5ad")

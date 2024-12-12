# %% [markdown]
# # Basic preprocessing of the pancreatic endocrine dataset
#
# Notebook preprocesses the pancreatic endocrine dataset.

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd

import scanpy as sc
import scvelo as scv
from velovi import preprocess_data

from rgv_tools import DATA_DIR
from rgv_tools.preprocessing import set_prior_grn

# %% [markdown]
# ## General settings

# %%
sc.settings.verbosity = 2
scv.settings.verbosity = 3

# %% [markdown]
# ## Constants

# %%
DATASET = "pancreatic_endocrine"

SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
adata = scv.datasets.pancreas()

# %%
TF = pd.read_csv(DATA_DIR / DATASET / "raw" / "allTFs_mm.txt", header=None)
gt_net = pd.read_csv(DATA_DIR / DATASET / "raw" / "skeleton.csv", index_col=0)

# %% [markdown]
# ## Preprocessing

# %%
scv.pp.filter_and_normalize(adata, min_shared_counts=10, n_top_genes=2000)

# %%
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)

# %%
scv.pp.moments(adata, n_pcs=None, n_neighbors=None)
adata

# %%
scv.pl.umap(adata, color="clusters", legend_loc="right")

# %% [markdown]
# ## RegVelo preprocessing

# %%
adata = set_prior_grn(adata, gt_net, keep_dim=True)  ## keep_dim = True due to the the sparse prior GRN

# %%
velocity_genes = preprocess_data(adata.copy()).var_names.tolist()

# %%
tf_grn = adata.var_names[adata.uns["skeleton"].T.sum(0) != 0].tolist()
tfs = list(set(TF.iloc[:, 0].tolist()).intersection(tf_grn))
adata.var["tf"] = adata.var_names.isin(tfs)

# %%
var_mask = np.union1d(adata.var_names[adata.var["tf"]], velocity_genes)
adata = adata[:, var_mask].copy()

# %%
adata = preprocess_data(adata, filter_on_r2=False)

# %%
# Filter the skeleton matrix `W` based on the selected indices
skeleton = adata.uns["skeleton"].loc[adata.var_names.tolist(), adata.var_names.tolist()]

# Update the filtered values in `uns`
adata.uns.update({"skeleton": skeleton, "regulators": adata.var_names.tolist(), "targets": adata.var_names.tolist()})

# %%
## focus on velocity genes to ensure calculation stability of scvelo and veloVI
adata.var["velocity_genes"] = adata.var_names.isin(velocity_genes)

# %% [markdown]
# ## Save dataset

# %%
if SAVE_DATA:
    adata.write_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed.h5ad")

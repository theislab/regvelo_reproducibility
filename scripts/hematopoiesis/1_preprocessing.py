# %% [markdown]
# # Basic preprocessing and analysis of the human hematopoiesis dataset
#
# Notebook for preprocessing human hematopoiesis dataset

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import mplscience

import anndata as ad
import scanpy as sc
import scvelo as scv
from velovi import preprocess_data

from rgv_tools import DATA_DIR, FIG_DIR
from rgv_tools.preprocessing import set_prior_grn

# %% [markdown]
# ## General settings

# %%
sc.settings.verbosity = 2
scv.settings.verbosity = 3

# %%
plt.rcParams["svg.fonttype"] = "none"

# %% [markdown]
# ## Constants

# %%
DATASET = "hematopoiesis_revision"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)

# %%
SAVE_FIGURES = False
if SAVE_FIGURES:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)

FIGURE_FORMAT = "svg"

# %% [markdown]
# ## Data loading

# %%
adata = ad.io.read_h5ad(DATA_DIR / DATASET / "raw" / "hsc_dynamo_adata.h5ad")
adata

# %%
tfs = pd.read_csv(DATA_DIR / DATASET / "raw" / "allTFs_hg38.csv", header=None)
gt_net = pd.read_csv(DATA_DIR / DATASET / "raw" / "skeleton.csv", index_col=0)

# %% [markdown]
# ## Visualization

# %% [markdown]
# ## Preprocessing

# %%
scv.pp.filter_and_normalize(adata, min_shared_counts=10, log=False, n_top_genes=2000)

# %%
sc.pp.neighbors(adata, n_neighbors=50)

# %%
scv.pp.moments(adata, n_pcs=None, n_neighbors=None)
adata

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    sc.pl.scatter(adata, basis="draw_graph_fa", color="cell_type", frameon=False, ax=ax)

    if SAVE_FIGURES:
        fig.savefig(
            FIG_DIR / DATASET / f"intro_figure.{FIGURE_FORMAT}",
            format=FIGURE_FORMAT,
            transparent=True,
            bbox_inches="tight",
        )

# %%
if SAVE_DATA:
    adata.write_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed_full.h5ad")

# %% [markdown]
# ## RegVelo preprocessing

# %%
adata = set_prior_grn(adata, gt_net)

# %%
## We keep the genes that pass the filtering criteria
## min_max_scaling will include 13 new velocity genes, we only consider the velocity genes that is shared before and after scaling
velocity_genes = preprocess_data(adata, min_max_scale=False).var_names.tolist()
keep_genes = preprocess_data(adata.copy()).var_names.tolist()
velocity_genes = set(keep_genes).intersection(velocity_genes)

# %%
tf_grn = adata.var_names[adata.uns["skeleton"].T.sum(0) != 0].tolist()
tf = list(set(tfs.iloc[:, 0].tolist()).intersection(tf_grn))
adata.var["tf"] = adata.var_names.isin(tf)

# %% [markdown]
# Select genes that are either part of the transcription factor (TF) list or `velocity_genes`

# %%
var_mask = np.union1d(adata.var_names[adata.var["tf"]], keep_genes)
adata = adata[:, var_mask].copy()

# %%
adata = preprocess_data(adata, filter_on_r2=False)

# %%
# Filter the skeleton matrix `W` based on the selected indices
skeleton = adata.uns["skeleton"].loc[adata.var_names.tolist(), adata.var_names.tolist()]

# Update the filtered values in `uns`
adata.uns.update({"skeleton": skeleton, "regulators": adata.var_names.tolist(), "targets": adata.var_names.tolist()})

# %%
# focus on velocity genes to ensure calculation stability of scvelo and veloVI
adata.var["velocity_genes"] = adata.var_names.isin(velocity_genes)

# %% [markdown]
# ## Save dataset

# %%
if SAVE_DATA:
    adata.write_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed.h5ad")

# %%

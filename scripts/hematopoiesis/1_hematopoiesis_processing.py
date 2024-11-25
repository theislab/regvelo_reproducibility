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
import seaborn as sns

import scanpy as sc
import scvelo as scv
from velovi import preprocess_data

from rgv_tools import DATA_DIR, FIG_DIR
from rgv_tools.preprocessing import prior_GRN_import

# %% [markdown]
# ## General settings

# %%
plt.rcParams["svg.fonttype"] = "none"
sns.reset_defaults()
sns.reset_orig()
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=14, color_map="viridis")

# %% [markdown]
# ## Constants

# %%
DATASET = "hematopoiesis"
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)

SAVE_FIGURES = False
if SAVE_FIGURES:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "raw" / "hsc_dynamo_adata.h5ad")
TF = pd.read_csv(DATA_DIR / DATASET / "raw" / "allTFs_hg38.csv", header=None)
gt_net = pd.read_csv(DATA_DIR / DATASET / "raw" / "skeleton.csv", index_col=0)

# %% [markdown]
# ## Visualization

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    sc.pl.scatter(adata, basis="draw_graph_fa", color="cell_type", frameon=False, ax=ax)

    if SAVE_FIGURES:
        fig.savefig(FIG_DIR / DATASET / "intro_figure.svg", format="svg", transparent=True, bbox_inches="tight")

# %% [markdown]
# ## Preprocessing

# %%
scv.pp.filter_and_normalize(adata, min_shared_counts=10, n_top_genes=2000)
sc.pp.neighbors(adata, n_neighbors=50)
scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

# %% [markdown]
# ## Load prior gene regulatory graph

# %%
reg_bdata = prior_GRN_import(adata, gt_net)

# %%
velocity_genes = preprocess_data(reg_bdata).var_names.tolist()

# %% [markdown]
# ## RegVelo preprocessing

# %%
TF_GRN = reg_bdata.var_names[reg_bdata.uns["skeleton"].T.sum(0) != 0].tolist()
TF = list(set(TF.iloc[:, 0].tolist()).intersection(TF_GRN))
reg_bdata.var["TF"] = np.isin(reg_bdata.var_names, TF)

# %% [markdown]
# Select genes that are either part of the transcription factor (TF) list or `velocity_genes`

# %%
sg = np.union1d(reg_bdata.var_names[reg_bdata.var["TF"]], velocity_genes)

reg_bdata = reg_bdata[:, sg].copy()

# %%
reg_bdata = preprocess_data(reg_bdata, filter_on_r2=False)

# %%
gene_name = reg_bdata.var_names
full_name = reg_bdata.uns["regulators"]
index = np.isin(full_name, gene_name)
filtered_names = full_name[index]

# Filter the skeleton matrix `W` based on the selected indices
W = reg_bdata.uns["skeleton"][index][:, index]

# Update the filtered values in `uns`
reg_bdata.uns.update({"skeleton": W, "regulators": gene_name.values, "targets": gene_name.values})

# %%
scv.tl.velocity(adata)  ## estimate velocity genes

# %% [markdown]
# ## Save dataset

# %%
if SAVE_DATA:
    reg_bdata.write_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed.h5ad")
    adata.write_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed_full.h5ad")

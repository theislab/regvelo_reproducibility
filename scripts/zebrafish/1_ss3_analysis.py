# %% [markdown]
# # Basic preprocessing and analysis of the zebrafish data
#
# Notebook: preprocesses the zebrafish Smart-seq3 dataset.

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
from rgv_tools.preprocessing import (
    filter_genes_with_upstream_regulators,
    prior_GRN_import,
)

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
DATASET = "zebrafish"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)

# %%
SAVE_FIGURES = True
if SAVE_FIGURES:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "raw" / "adata_zebrafish_preprocessed.h5ad")
TF_list = pd.read_csv(DATA_DIR / DATASET / "raw" / "zebrafish_tfs.csv", index_col=0).iloc[:, 0].tolist()
prior_net = pd.read_csv(DATA_DIR / DATASET / "raw" / "prior_GRN.csv", index_col=0)

# %%
## Only keep necessary list
keep_list = pd.read_csv(DATA_DIR / DATASET / "raw" / "new_tf.csv", sep=";").iloc[:, 0].tolist()

# %%
sc.pp.neighbors(adata, n_neighbors=30)
scv.pp.moments(adata)

# %%
scv.pl.umap(adata, color="cell_type", palette=sc.pl.palettes.vega_20, legend_loc="right")

# %% [markdown]
# ## Visualize results

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(7, 5))
    scv.pl.umap(adata, color="cell_type", ax=ax, palette=sc.pl.palettes.vega_20)

    if SAVE_FIGURES:
        fig.savefig(FIG_DIR / DATASET / "INTRO_figure_all_ct.svg", format="svg", transparent=True, bbox_inches="tight")
    plt.show

# %% [markdown]
# ## Visualize known terminal states

# %%
adata.obs["cell_type2"] = adata.obs["cell_type"].copy()
adata.obs["cell_type2"][
    ~adata.obs["cell_type2"].isin(["NPB_nohox", "mNC_hox34", "Pigment", "mNC_head_mesenchymal", "mNC_arch2"])
] = np.nan

# %%
adata.obs["cell_type2"] = adata.obs["cell_type2"].cat.remove_unused_categories()

# %%
palette = dict(zip(adata.obs["cell_type"].cat.categories, adata.uns["cell_type_colors"]))
subset_palette = {
    name: color for name, color in palette.items() if name in adata.obs["cell_type2"].cat.categories.tolist()
}

# %%
which = "macrostates"
adata.obs[which] = adata.obs["cell_type2"].copy()

state_names = adata.obs[which].cat.categories.tolist()
adata.obs[which] = adata.obs[which].astype(str).astype("category").cat.reorder_categories(["nan"] + state_names)

if which == "macrostates":
    adata.uns[f"{which}_colors"] = ["#dedede"] + list(subset_palette.values())
else:
    adata.uns[f"{which}_colors"] = ["#dedede"] + list(subset_palette.values())
state_names = adata.obs[which].cat.categories.tolist()[1:]


with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(4, 3))
    scv.pl.scatter(
        adata,
        basis="umap",
        c=which,
        add_outline=state_names,
        ax=ax,
        size=60,
    )

    if SAVE_FIGURES:
        fig.savefig(FIG_DIR / DATASET / "INTRO_figure.svg", format="svg", transparent=True, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## Preprocessing

# %%
adata = prior_GRN_import(adata, prior_net).copy()

# %%
velocity_genes = preprocess_data(adata).var_names.tolist()

# %%
adata.var["TF"] = np.isin(adata.var_names, TF_list)

# %%
## velocity_r2 positive genes
sg = np.union1d(list(set(keep_list).intersection(adata.var_names)), velocity_genes)

## Filtering genes, only keep velocity_r2 positive genes and TFs
adata = adata[:, sg].copy()

# %%
adata = filter_genes_with_upstream_regulators(adata)
adata = preprocess_data(adata, filter_on_r2=False)

# %%
scv.tl.velocity(adata)

# %% [markdown]
# ## Save dataset

# %%
if SAVE_DATA:
    adata.write_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed.h5ad")

# %%

# %% [markdown]
# # Cell cycle data preparation
#
# Notebook prepares data for inference tasks.

# %% [markdown]
# ## Library imports

# %%
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import anndata as ad
import scanpy as sc
import scvelo as scv
from anndata import AnnData
from velovi import preprocess_data

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns
from matplotlib.colors import to_hex

from rgv_tools import DATA_DIR, FIG_DIR

# %% [markdown]
# ## Constants

# %%
DATASET = "cell_cycle_rpe1"

# %%
SAVE_DATA = True
SAVE_FIGURES = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)
    (FIG_DIR / DATASET / "comparison").mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Function definitions


# %%
def prepare_data(adata: AnnData) -> None:
    """Update cell cycle data to include only relevant information in the standard format."""
    adata = adata[adata.obs["labeling_time"] != "dmso", :]
    adata.obs["labeling_time"] = adata.obs["labeling_time"].astype(float)
    adata = adata[adata.obs["experiment"] == "Pulse", :].copy()

    adata.layers["unspliced"] = adata.layers["unlabeled_unspliced"] + adata.layers["labeled_unspliced"]
    adata.layers["spliced"] = adata.layers["unlabeled_spliced"] + adata.layers["labeled_spliced"]

    adata.obs["pseudo_clusters"] = (
        pd.cut(adata.obs["cell_cycle_position"], bins=30, labels=False).astype(str).astype("category")
    )

    adata.obs["pseudo_clusters_equal_size"] = pd.qcut(adata.obs["cell_cycle_position"], q=30, labels=False)
    adata.obs["pseudo_clusters_equal_size_num"] = adata.obs["pseudo_clusters_equal_size"].astype(float)

    adata.obs["cell_cycle_rad"] = adata.obs["cell_cycle_position"] / adata.obs["cell_cycle_position"].max() * 2 * np.pi

    cell_cycle_phase = pd.read_csv(
        "/lustre/groups/ml01/workspace/weixu.wang/regvelo_revision/cell_cycle_REF/cell_cycle_phase.csv", index_col=0
    )
    adata.obs["cell_cycle_phase"] = cell_cycle_phase.iloc[:, 0]

    adata = adata[:, adata.var["Symbol"].astype(str) != "nan"].copy()
    adata.var.index = adata.var["Symbol"].astype(str)
    del adata.var

    return adata


# %% [markdown]
# ## Data loading

# %%
adata = ad.io.read_h5ad(DATA_DIR / DATASET / "raw" / "gex_raw.h5ad")
adata

# %% [markdown]
# ## Data processing

# %%
adata = prepare_data(adata=adata)

if SAVE_DATA:
    adata.write(DATA_DIR / "processed" / "adata.h5ad")

adata

# %%
scv.pp.filter_and_normalize(
    adata, min_shared_counts=10, layers_normalize=["X", "new", "total", "unspliced", "spliced"], n_top_genes=2000
)
sc.tl.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
scv.pp.moments(adata, n_pcs=None, n_neighbors=None)
sc.tl.umap(adata)

adata = preprocess_data(adata)
adata

# %%
thre_v = [10, 50, 100, 250, 500, 800]
for thre in thre_v:
    true_skeleton = pd.DataFrame(np.zeros((adata.n_vars, adata.n_vars)), index=adata.var_names, columns=adata.var_names)
    for fname in tqdm((DATA_DIR / "raw" / "tf_list_5k").iterdir()):
        regulator = fname.stem
        targets = pd.read_csv(fname, delimiter="\t")["Target_genes"].tolist()

        score = pd.read_csv(fname, delimiter="\t").iloc[:, 1]

        targets = np.array(targets)[score > thre]

        targets = list(adata.var_names.intersection(targets))

        if len(targets) > 3 and regulator in adata.var_names:
            true_skeleton.loc[regulator, targets] = 1

    adata.varm[f"true_skeleton_{thre}"] = csr_matrix(true_skeleton.values)

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, basis="umap", color="cell_cycle_position", cmap="viridis", title="", ax=ax)

if SAVE_FIGURES:
    fig.savefig(
        FIG_DIR / DATASET / "comparison" / "sceu_cell_cycle_umap.svg",
        format="svg",
        transparent=True,
        bbox_inches="tight",
    )

# %% [markdown]
# ## Calculating cell cycling phase

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, basis="umap", color="cell_cycle_phase", cmap="viridis", title="", ax=ax)

if SAVE_FIGURES:
    fig.savefig(
        FIG_DIR / DATASET / "comparison" / "sceu_cell_cycle_phase_umap.svg",
        format="svg",
        transparent=True,
        bbox_inches="tight",
    )

# %%
if SAVE_DATA:
    adata.write(DATA_DIR / DATASET / "processed" / "adata_processed.h5ad")

# %%

# %%

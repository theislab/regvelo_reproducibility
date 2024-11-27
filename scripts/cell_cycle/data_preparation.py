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

from rgv_tools import DATA_DIR

# %% [markdown]
# ## Constants

# %%
DATASET = "cell_cycle"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Function definitions


# %%
def prepare_data(adata: AnnData) -> None:
    """Update cell cycle data to include only relevant information in the standard format."""
    adata.X = adata.layers["spliced"].copy()

    for layer in ["ambiguous", "matrix", "spanning"]:
        del adata.layers[layer]
    adata.layers["total"] = adata.layers["unspliced"] + adata.layers["spliced"]

    columns_to_drop = [
        "Well_Plate",
        "plate",
        "MeanGreen530",
        "MeanRed585",
        "initial_size_unspliced",
        "initial_size_spliced",
        "initial_size",
    ]
    adata.obs["phase"] = adata.obs["phase"].astype(str).replace({"N/A": np.nan, "S-ph": "S"}).astype("category")
    adata.obs.drop(columns_to_drop, axis=1, inplace=True)

    adata.var["ensum_id"] = adata.var_names
    adata.var_names = adata.var["name"].values.astype(str)
    adata.var_names_make_unique()
    columns_to_drop = [
        "name",
        "biotype",
        "description",
        "Accession",
        "Chromosome",
        "End",
        "Start",
        "Strand",
        "GeneName",
    ]
    adata.var.drop(columns_to_drop, axis=1, inplace=True)


# %% [markdown]
# ## Data loading

# %%
adata = ad.io.read_h5ad(DATA_DIR / DATASET / "raw" / "adata.h5ad")
adata

# %% [markdown]
# ## Data processing

# %%
prepare_data(adata=adata)

true_skeleton = pd.DataFrame(np.zeros((adata.n_vars, adata.n_vars)), index=adata.var_names, columns=adata.var_names)
for fname in tqdm((DATA_DIR / DATASET / "raw" / "tf_list_5k").iterdir()):
    regulator = fname.stem
    targets = pd.read_csv(fname, delimiter="\t")["Target_genes"].tolist()
    true_skeleton.loc[regulator, targets] = 1

adata.varm["true_skeleton"] = csr_matrix(true_skeleton.values)

if SAVE_DATA:
    adata.write(DATA_DIR / DATASET / "processed" / "adata.h5ad")

adata

# %%
scv.pp.filter_and_normalize(adata, min_counts=10, n_top_genes=2000, log=False)
sc.pp.log1p(adata)
sc.tl.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
scv.pp.moments(adata)
sc.tl.umap(adata)

adata = preprocess_data(adata)
adata

# %%
if SAVE_DATA:
    adata.write(DATA_DIR / DATASET / "processed" / "adata_processed.h5ad")

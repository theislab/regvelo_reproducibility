# %% [markdown]
# # Prepare datasets with different neighborhood size
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
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)


# %%
nn_level = [10, 30, 50, 70, 90, 100]

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
adata = ad.io.read_h5ad(DATA_DIR / "raw" / "adata.h5ad")
adata

# %% [markdown]
# ## Data processing

# %%
prepare_data(adata=adata)

if SAVE_DATA:
    adata.write(DATA_DIR / "processed" / "adata.h5ad")

adata

# %%
scv.pp.filter_and_normalize(adata, min_counts=10, n_top_genes=2000, log=False)
sc.pp.log1p(adata)
sc.tl.pca(adata, svd_solver="arpack")

adata_raw = adata.copy()

for level in nn_level:
    ## simulate different level of consistency
    sc.pp.neighbors(adata, n_neighbors=level, n_pcs=30)
    scv.pp.moments(adata, n_neighbors=None, n_pcs=None)

    adata = preprocess_data(adata)

    adata.write(DATA_DIR / "processed" / f"adata_processed_nn{level}.h5ad")
    print(adata)
    adata = adata_raw.copy()

# %%

# %% [markdown]
# # Comparison of velocities among RegVelo, veloVI and scVelo on cell cycle dataset

# %% [markdown]
# ## Library imports

# %%
import os
import sys
import numpy as np
import pandas as pd
from velovi import preprocess_data, VELOVI
import matplotlib.pyplot as plt
import mplscience
import seaborn as sns
from matplotlib.colors import to_hex
import scanpy as sc
import scvelo as scv

sys.path.append("../..")
from paths import DATA_DIR, FIG_DIR

# %%
from regvelovi import REGVELOVI
from typing import Literal
import anndata

# %% [markdown]
# ## Data loading

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

adata

# %% [markdown]
# ## Data preprocessing

# %%
scv.pp.filter_and_normalize(adata, min_counts=10, n_top_genes=2000)

# %%
sc.tl.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
scv.pp.moments(adata)

# %%
sc.tl.umap(adata)

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata, basis="umap", color="fucci_time", cmap="viridis", title="Cell cycling stage", ax=ax)


# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    scv.pl.scatter(adata[adata.obs["phase"] != "N/A"], basis="umap", color="phase", title="Cycling phase", ax=ax)


# %%
adata = preprocess_data(adata)

# %%
adata.write(DATA_DIR / "cell_cycle" / "cell_cycle_processed.h5ad")

# %%
adata

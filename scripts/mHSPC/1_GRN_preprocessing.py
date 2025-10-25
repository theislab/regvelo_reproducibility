# %% [markdown]
# # Preprocess mHSPC datasets

# %% [markdown]
# ## Library imports

# %%
import scanpy as sc
import scvelo as scv
import pandas as pd

from velovi import preprocess_data
from rgv_tools import DATA_DIR, FIG_DIR

# %% [markdown]
# ## General settings

# %%
sc.settings.verbosity = 2
scv.settings.verbosity = 3

# %% [markdown]
# ## Constants

# %%
DATASET = "mHSPC"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
adata_velo = sc.read_loom(DATA_DIR / DATASET / "raw" / "Nestorowa.loom")
adata_p = sc.read_h5ad(DATA_DIR / DATASET / "raw" / "adata_nestorowa.h5ad")

# %%
adata_p.obs

# %% [markdown]
# ## Preprocessing

# %%
adata_velo.obs_names[:5]

# %%
adata_velo.obs_names = [
    i.replace("onefilepercell_SRR3556264Aligned_and_others_KPB9A:", "") for i in adata_velo.obs_names
]
adata_velo.obs_names = [i.replace("Aligned.out.bam", "") for i in adata_velo.obs_names]

# %%
adata_velo = adata_velo[adata_p.obs_names].copy()

# %%
adata_velo.obs_names = adata_p.obs["cell_IDs"]
adata_velo.obs_names = [i.replace("_rep", "") for i in adata_velo.obs_names]

# %%
adata_velo.obs_names

# %%
adata_velo.var_names_make_unique()

# %%
scv.pl.proportions(adata_velo)

# %%
scv.pp.filter_and_normalize(adata_velo, min_shared_counts=20, n_top_genes=3000)

# %%
adata_velo.obs_names = [i.replace("-", "_") for i in adata_velo.obs_names]

# %% [markdown]
# ## Processing Dataset

# %%
adata_subset = adata_velo.copy()
## set annotation
adata_p.obs_names = adata_velo.obs_names
adata_subset.obs = adata_p.obs

print(adata_subset)

## Velocity pipeline
sc.tl.pca(adata_subset)
sc.pp.neighbors(adata_subset, n_neighbors=30, n_pcs=30)
scv.pp.moments(adata_subset)
adata_subset = preprocess_data(adata_subset, filter_on_r2=True)

if SAVE_DATA:
    adata_subset.write_h5ad(DATA_DIR / DATASET / "processed" / "mHSC_ExpressionData.h5ad")

# %%

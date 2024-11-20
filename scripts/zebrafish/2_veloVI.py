# %% [markdown]
# # veloVI-based analyis of zebrafish data
#
# Notebook runs veloVI on zebrafish dataset.

# %% [markdown]
# ## Library imports

# %%
import scanpy as sc
import scvi
from velovi import VELOVI

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import set_output

# %% [markdown]
# ## General settings

# %%
scvi.settings.seed = 0

# %% [markdown]
# ## Constants

# %%
DATASET = "zebrafish"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed.h5ad")

# %% [markdown]
# ## Velocity pipeline

# %%
VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
vae = VELOVI(adata)
vae.train()

# %%
set_output(adata, vae, n_samples=30)

# %% [markdown]
# ## Save dataset

# %%
if SAVE_DATA:
    adata.write_h5ad(DATA_DIR / DATASET / "processed" / "adata_run_velovi.h5ad")

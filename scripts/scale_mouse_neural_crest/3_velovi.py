# %% [markdown]
# # veloVI application on murine neural crest data

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
DATASET = "mouse_neural_crest"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Velocity pipline

# %%
for i in range(1, 5):
    adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / f"adata_stage{i}_processed_velo_all_regulons.h5ad")

    scvi.settings.seed = 0
    VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    vae = VELOVI(adata)
    vae.train()

    set_output(adata, vae, n_samples=30)

    if SAVE_DATA:
        adata.write_h5ad(DATA_DIR / DATASET / "processed" / f"adata_run_stage_{i}_velovi_all_regulons.h5ad")

# %%

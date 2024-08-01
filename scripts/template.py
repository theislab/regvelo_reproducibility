# %% [markdown]
# # A descriptive title
#
# A short description of what this notebook is doing.

# %% [markdown]
# ## Library imports

# %%
import scanpy as sc

from fancypackage import DATA_DIR, FIG_DIR

# %% [markdown]
# ## General settings

# %%
SAVE_FIGURES = False
if SAVE_FIGURES:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_FORMATE = "pdf"

# %% [markdown]
# ## Constants

# %% [markdown]
# ## Function definitions

# %% [markdown]
# ## Data loading

# %%
adata = sc.read(DATA_DIR / "adata.h5ad")
adata

# %% [markdown]
# ## Data preprocessing

# %% [markdown]
# ## More advanced data analysis

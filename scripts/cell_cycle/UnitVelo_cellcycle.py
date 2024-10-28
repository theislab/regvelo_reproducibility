# %% [markdown]
# ## UnitVelo Import libraries

# %%
import os
import sys

from paths import DATA_DIR

import scanpy as sc
import scvelo as scv

# %%
import unitvelo as utv

os.environ["TF_USE_LEGACY_KERAS"] = "True"

sys.path.append("../..")

# %% [markdown]
# ## Load cell cycle dataset, run UnitVelo and save the output in adata

# %%
adata = sc.read(DATA_DIR / "cell_cycle" / "cell_cycle_processed.h5ad")

# %%
velo_config = utv.config.Configuration()
velo_config.R2_ADJUST = True
velo_config.IROOT = None
velo_config.FIT_OPTION = "1"
velo_config.AGENES_R2 = 1
velo_config.GPU = -1

# %%
adata.obs["cluster"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "True"

adata = utv.run_model(adata, label="cluster", config_file=velo_config)
scv.pl.velocity_embedding_stream(adata, color=adata.uns["label"], dpi=100, title="")


# %%
adata

# %%
adata.write(DATA_DIR / "cell_cycle" / "unitvelo_cycle_filteredgene.h5ad")

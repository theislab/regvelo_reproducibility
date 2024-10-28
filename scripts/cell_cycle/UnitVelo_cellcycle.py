# %% [markdown]
# ## UnitVelo Import libraries
#

# %%
import unitvelo as utv
import anndata
import scvelo as scv
import numpy as np
import os
import argparse
import time
import anndata
import numpy as np
import scvelo as scv
import scanpy as sc
import sys
import torch
import os
import anndata as ad
import pandas as pd

os.environ["TF_USE_LEGACY_KERAS"] = "True"

sys.path.append("../..")
from paths import DATA_DIR, FIG_DIR

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

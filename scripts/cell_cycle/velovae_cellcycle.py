# %% [markdown]
# ## Run veloVAE

# %%
import os

os.environ["TF_USE_LEGACY_KERAS"] = "True"
import anndata
import numpy as np
import scvelo as scv
import scanpy as sc
import sys
import torch
import os.path

sys.path.insert(1, "../")
import velovae as vv
from scipy.sparse import csr_matrix
import pandas as pd
import matplotlib.pyplot as plt


# %%
from torch import nn
import numpy as np
import torch
import pandas as pd
import seaborn as sns

import os
from typing import Literal
import anndata as ad
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# %% [markdown]
# ## General setting

# %%
model_path = f"checkpoints/"
figure_path = f"figures/"
data_path = f"data/"

sns.reset_defaults()
sns.reset_orig()
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=20, color_map="viridis")

sys.path.append("../..")
from paths import DATA_DIR, FIG_DIR

# %% [markdown]
# ## load dataset

# %%
adata = sc.read(DATA_DIR / "cell_cycle" / "cell_cycle_processed.h5ad")

# %%
# n_gene = 2000
# vv.preprocess(adata, n_gene)
torch.manual_seed(2022)
np.random.seed(2022)
vae = vv.VAE(adata, tmax=20, dim_z=5, device="cuda:0")

# %%
gene_plot = ["Gng12"]

config = {}
vae.train(adata, config=config, plot=False, gene_plot=gene_plot, embed="umap")
vae.save_model(model_path, "encoder_vae", "decoder_vae")
vae.save_anndata(adata, "vae", data_path, file_name="velovae.h5ad")

# %%
torch.manual_seed(2022)
np.random.seed(2022)
rate_prior = {"alpha": (0.0, 1.0), "beta": (0.0, 0.5), "gamma": (0.0, 0.5)}
full_vb = vv.VAE(adata, tmax=20, dim_z=5, device="cuda:0", full_vb=True, rate_prior=rate_prior)

# %%
full_vb.train(adata, plot=False, gene_plot=gene_plot, embed="umap")
full_vb.save_model(model_path, "encoder_fullvb", "decoder_fullvb")
full_vb.save_anndata(adata, "fullvb", data_path, file_name="fullvb.h5ad")

# %%
sc.tl.umap(adata)
adata.obs["clusters"] = "0"

methods = ["VeloVAE", "FullVB"]
keys = ["vae", "fullvb"]
grid_size = (1, 2)
res, res_type = vv.post_analysis(
    adata, "continuous", methods, keys, compute_metrics=True, raw_count=False, genes=gene_plot, grid_size=(1, 2)
)

# %%
adata

# %%
adata.write(DATA_DIR / "cell_cycle" / "velovae_cycle_filteredgene.h5ad")

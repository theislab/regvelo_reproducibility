# %% [markdown]
# # Calculate velocity and latent time using cell2fate

# %% [markdown]
# ## Library imports

# %%
# Import needed packages
import cell2fate as c2f
import scanpy as sc
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import anndata as ad, scvelo as scv
import contextlib, io
import sys

sys.path.append("../..")
from paths import DATA_DIR, FIG_DIR

# %% [markdown]
# ## General settings

# %%
SAVE_FIGURES = True
if SAVE_FIGURES:
    os.makedirs(FIG_DIR / "simulation", exist_ok=True)

SAVE_DATASETS = True
if SAVE_DATASETS:
    os.makedirs(DATA_DIR / "simulation", exist_ok=True)

# %%
input_path = DATA_DIR
output_path = DATA_DIR / "simulation"
input_files = os.listdir(input_path)


# %% [markdown]
# ## Function definitions


# %%
# Function for train model and get output
def trainc2fmodel(adatafile, input_path, output_path):
    adata = sc.read_h5ad(os.path.join(input_path, adatafile))
    adata.layers["spliced"] = adata.layers["counts_spliced"].A.copy()
    adata.layers["unspliced"] = adata.layers["counts_unspliced"].A.copy()
    adata.X = adata.X.A
    adata.layers["raw_spliced"] = adata.layers["spliced"]
    adata.layers["raw_unspliced"] = adata.layers["unspliced"]
    adata.obs["u_lib_size_raw"] = adata.layers["raw_unspliced"].sum(-1)
    adata.obs["s_lib_size_raw"] = adata.layers["raw_spliced"].sum(-1)
    scv.pp.filter_and_normalize(adata, min_shared_counts=10, n_top_genes=90)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)
    # scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    clusters_to_remove = []
    adata.layers["spliced"] = adata.layers["counts_spliced"].A.copy()
    adata.layers["unspliced"] = adata.layers["counts_unspliced"].A.copy()
    adata = c2f.utils.get_training_data(
        adata, cells_per_cluster=100, cluster_column="leiden", remove_clusters=[], min_shared_counts=10, n_var_genes=90
    )
    c2f.Cell2fate_DynamicalModel.setup_anndata(adata, spliced_label="spliced", unspliced_label="unspliced")
    n_modules = c2f.utils.get_max_modules(adata)
    mod = c2f.Cell2fate_DynamicalModel(adata, n_modules=n_modules)
    mod.train()
    # Compute total velocity
    n_modules = c2f.utils.get_max_modules(adata)
    c2f.Cell2fate_DynamicalModel.setup_anndata(adata, spliced_label="spliced", unspliced_label="unspliced")
    data = mod.export_posterior(adata)
    adata = mod.compute_module_summary_statistics(adata)
    with contextlib.redirect_stdout(io.StringIO()):
        adata.layers["Spliced Mean"] = mod.samples["post_sample_means"]["mu_expression"][..., 1]
        c2f_velocity = (
            torch.tensor(mod.samples["post_sample_means"]["beta_g"])
            * mod.samples["post_sample_means"]["mu_expression"][..., 0]
            - torch.tensor(mod.samples["post_sample_means"]["gamma_g"])
            * mod.samples["post_sample_means"]["mu_expression"][..., 1]
        )
        adata.layers["Velocity"] = c2f_velocity.numpy()
    return adata


# %% [markdown]
# ## Data loading and processing of one instance

# %%
adata = trainc2fmodel(input_files[0], input_path, output_path)

# %%
# obs: Time (hours) and layers: Velocity are your results
adata

# %%
# save the results
if SAVE_DATASETS:
    adata.write_h5ad(DATA_DIR / "simulation" / "c2f_output.h5ad")

# %% [markdown]
# ## Run cell2fate

# %%
import sys

# %%
import cell2fate as c2f
from paths import DATA_DIR

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns

import scanpy as sc
import scvelo as scv

# %%
scaler = MinMaxScaler()

## General setting
sns.reset_defaults()
sns.reset_orig()
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=20, color_map="viridis")

sys.path.append("../..")


# %% [markdown]
# ## Import cell cycle dataset and train the model

# %%
adata = sc.read(DATA_DIR / "cell_cycle" / "cell_cycle_processed.h5ad")


# %%
adata

# %%
clusters_to_remove = []
adata.obs["clusters"] = "0"

adata = c2f.utils.get_training_data(
    adata,
    cells_per_cluster=10**5,
    cluster_column="clusters",
    remove_clusters=clusters_to_remove,
    min_shared_counts=0,
    n_var_genes=2000,
)

# %%
c2f.Cell2fate_DynamicalModel.setup_anndata(adata, spliced_label="spliced_raw", unspliced_label="unspliced_raw")


# %%
n_modules = c2f.utils.get_max_modules(adata)


# %%
mod = c2f.Cell2fate_DynamicalModel(adata, n_modules=n_modules)


# %%
mod.train()


# %%
adata = mod.export_posterior(
    adata, sample_kwargs={"batch_size": None, "num_samples": 30, "return_samples": True, "use_gpu": False}
)


# %%
adata

# %%
adata.write(DATA_DIR / "cell_cycle" / "cell2fate_filteredgene.h5ad")

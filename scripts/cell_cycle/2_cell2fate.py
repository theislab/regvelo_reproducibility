# %% [markdown]
# ## cell2fate benchmark on cell cycle data
#
# Notebook benchmarks velocity, latent time inference, and cross boundary correctness using cell2fate on cell cycle data.
#
# Note that cell2fate requires `anndata==0.8.0` and `scvi-tools==0.16.1`.

# %% [markdown]
# ## Library imports

# %%
import pandas as pd

import anndata as ad
import cell2fate as c2f
import scvelo as scv
from cellrank.kernels import VelocityKernel

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import get_time_correlation

# %% [markdown]
# ## Constants

# %%
DATASET = "cell_cycle"

# %%
STATE_TRANSITIONS = [("G1", "S"), ("S", "G2M")]

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data loading
#
# using original count data to train cell2fate model

# %%
adata_raw = ad.read_h5ad(DATA_DIR / DATASET / "processed" / "adata.h5ad")
genes = ad.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_processed.h5ad").var_names
umap = ad.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_processed.h5ad").obsm["X_umap"].copy()
adata = adata_raw[:, genes].copy()

# %%
adata.obsm["X_umap"] = umap

# %%
adata

# %% [markdown]
# ## Velocity pipeline

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
c2f.Cell2fate_DynamicalModel.setup_anndata(adata, spliced_label="spliced", unspliced_label="unspliced")


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
mod.compute_and_plot_total_velocity(adata, delete=False)

# %%
adata.layers["velocity"] = adata.layers["Velocity"].numpy()
adata.layers["Ms"] = adata.layers["spliced"].copy()

# %%
time_correlation = [get_time_correlation(ground_truth=adata.obs["fucci_time"], estimated=adata.obs["Time (hours)"])]

# %%
scv.tl.velocity_graph(adata, vkey="velocity", n_jobs=1)
scv.tl.velocity_confidence(adata, vkey="velocity")

# %% [markdown]
# ## Cross boundary-correctness

# %%
vk = VelocityKernel(adata, vkey="velocity", xkey="spliced").compute_transition_matrix()

cluster_key = "phase"
rep = "X_pca"

score_df = []
for source, target in STATE_TRANSITIONS:
    cbc = vk.cbc(source=source, target=target, cluster_key=cluster_key, rep=rep)

    score_df.append(
        pd.DataFrame(
            {
                "State transition": [f"{source} - {target}"] * len(cbc),
                "CBC": cbc,
            }
        )
    )
score_df = pd.concat(score_df)

# %% [markdown]
# ## Data saving

# %%
if SAVE_DATA:
    pd.DataFrame({"time": time_correlation}, index=adata.obs_names).to_parquet(
        path=DATA_DIR / DATASET / "results" / "cell2fate_correlation.parquet"
    )
    adata.obs[["velocity_confidence"]].to_parquet(path=DATA_DIR / DATASET / "results" / "cell2fate_confidence.parquet")
    score_df.to_parquet(path=DATA_DIR / DATASET / "results" / "cell2fate_cbc.parquet")

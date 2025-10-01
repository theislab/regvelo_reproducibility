# %% [markdown]
# # veloVI benchmark on cell cycle data
#
# Notebook benchmarks velocity, latent time and GRN inference, and cross boundary correctness using RegVelo on cell cycle data.

# %% [markdown]
# ## Library imports

# %%
import pandas as pd

import anndata as ad
import scanpy as sc
import scvelo as scv
import scvi
from velovi import VELOVI

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import set_output

# %% [markdown]
# ## General settings

# %%
scv.settings.verbosity = 3

# %% [markdown]
# ## Constants

# %%
DATASET = "cell_cycle"

# %%
STATE_TRANSITIONS = [("G1", "S"), ("S", "G2M")]

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / "results").mkdir(parents=True, exist_ok=True)

# %%
nn_levels = [10, 30, 50, 70, 90, 100]


# %% [markdown]
# ## Functions


# %%
def compute_confidence(adata, vkey="velocity"):
    """
    Compute the velocity confidence for a given AnnData object.

    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix (single-cell data) containing layers for velocity analysis.
    vkey : str, optional (default="velocity")
        Key in adata.layers corresponding to the velocity matrix.

    Returns:
    --------
    g_df : pandas.DataFrame
        DataFrame containing the velocity confidence scores for each cell.
    """
    adata.layers[vkey]
    scv.tl.velocity_graph(adata, vkey=vkey, n_jobs=1)
    scv.tl.velocity_confidence(adata, vkey=vkey)

    g_df = pd.DataFrame()
    g_df[f"{vkey}_consistency"] = adata.obs[f"{vkey}_confidence"].to_numpy().ravel()

    return g_df


# %% [markdown]
# ## Velocity pipeline

# %%
for level in nn_levels:
    adata = ad.io.read_h5ad(DATA_DIR / "processed" / f"adata_processed_nn{level}.h5ad")
    sc.pp.neighbors(
        adata, n_neighbors=30, n_pcs=30
    )  # redefine neighborhood graph to ensure graph is the same when calculate velocity consistency

    print(adata)
    scvi.settings.seed = 0
    VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    vae = VELOVI(adata)
    vae.train(max_epochs=1500)

    set_output(adata, vae, n_samples=30)

    t_df = compute_confidence(adata, vkey="fit_t")
    t_df["Dataset"] = "Cell cycle"
    t_df["Method"] = "velovi"

    v_df = compute_confidence(adata, vkey="velocity")
    v_df["Dataset"] = "Cell cycle"
    v_df["Method"] = "velovi"

    t_df.to_parquet(path=DATA_DIR / "results" / f"velovi_confidence_time_{level}.parquet")

    v_df.to_parquet(path=DATA_DIR / "results" / f"velovi_confidence_velocity_{level}.parquet")

# %%
v_df

# %%

# %% [markdown]
# # Using veloVI as baseline for RegVelo identifiability test
#
# Notebook runs preprocessing and using veloVI as baseline.

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd

import scanpy as sc
import scvelo as scv
import scvi
from velovi import preprocess_data, VELOVI

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import (
    get_time_correlation,
    get_velocity_correlation,
    set_output,
)

# %% [markdown]
# ## General settings

# %%
scvi.settings.seed = 0

# %% [markdown]
# ## Constants

# %%
DATASET = "dyngen"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Function definitions


# %%
def update_data(adata) -> None:
    """Update dyngen-simulated data to include only relevant information in the standard format."""
    adata.X = adata.layers["counts_spliced"]

    adata.layers["unspliced"] = adata.layers.pop("counts_unspliced")
    adata.layers["spliced"] = adata.layers.pop("counts_spliced")
    adata.layers["true_velocity"] = adata.layers.pop("rna_velocity")
    adata.layers["true_velocity"] = adata.layers["true_velocity"].toarray()
    adata.layers["unspliced_raw"] = adata.layers["unspliced"].copy()
    adata.layers["spliced_raw"] = adata.layers["spliced"].copy()

    del adata.layers["counts_protein"]
    del adata.layers["logcounts"]

    adata.obs.rename(columns={"sim_time": "true_time"}, inplace=True)
    adata.obs.drop(columns=["step_ix", "simulation_i"], inplace=True)

    adata.var.rename(
        columns={"transcription_rate": "true_alpha", "splicing_rate": "true_beta", "mrna_decay_rate": "true_gamma"},
        inplace=True,
    )
    columns_to_keep = ["true_alpha", "true_beta", "true_gamma", "is_tf"]
    adata.var.drop(columns=adata.var.columns.difference(columns_to_keep), inplace=True)

    slots = list(adata.uns.keys())
    for slot in slots:
        if slot in ["network", "regulatory_network", "skeleton", "regulators", "targets"]:
            adata.uns[f"true_{slot}"] = adata.uns.pop(slot)
        else:
            del adata.uns[slot]

    adata.obsm["true_sc_network"] = adata.obsm.pop("regulatory_network_sc")
    del adata.obsm["dimred"]

    adata.obs_names = adata.obs_names.str.replace("cell", "cell_")


def get_sc_grn(adata):
    """Compute cell-specific GRNs."""
    true_sc_grn = []

    for cell_id in range(adata.n_obs):
        grn = np.zeros([adata.n_vars, adata.n_vars])
        df = adata.uns["true_regulatory_network"][["regulator", "target"]].copy()
        df["value"] = adata.obsm["true_sc_network"][cell_id, :].toarray().squeeze()

        df = pd.pivot(df, index="regulator", columns="target", values="value").fillna(0)
        grn[np.ix_(adata.var_names.get_indexer(df.index), adata.var_names.get_indexer(df.columns))] = df.values
        true_sc_grn.append(grn)
    return np.dstack(true_sc_grn)


# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "raw" / "dataset_dyngen_sim.h5ad")
adata

# %% [markdown]
# ## Preprocessing pipeline

# %%
update_data(adata=adata)
adata.uns["true_sc_grn"] = get_sc_grn(adata=adata)

scv.pp.filter_and_normalize(adata, min_shared_counts=10, log=False)
sc.pp.log1p(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
scv.pp.moments(adata)

adata = preprocess_data(adata, filter_on_r2=True)

mask = pd.Index(adata.uns["true_regulators"]).isin(adata.var_names)
for uns_key in ["network", "skeleton", "sc_grn"]:
    adata.uns[f"true_{uns_key}"] = adata.uns[f"true_{uns_key}"][np.ix_(mask, mask)]

adata.write_zarr(DATA_DIR / DATASET / "processed" / "processed_sim.zarr")

# %% [markdown]
# ## Velocity pipeline

# %%
velocity_correlation = []
time_correlation = []

for i in range(5):
    VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    vae = VELOVI(adata)
    vae.train(max_epochs=1500)

    set_output(adata, vae, n_samples=30)

    velocity_correlation.append(
        get_velocity_correlation(
            ground_truth=adata.layers["true_velocity"], estimated=adata.layers["velocity"], aggregation=np.mean
        )
    )

    ## calculate per gene correlation
    time_corr = [
        get_time_correlation(ground_truth=adata.obs["true_time"], estimated=adata.layers["fit_t"][:, i])
        for i in range(adata.layers["fit_t"].shape[1])
    ]
    time_correlation.append(np.mean(time_corr))

# %%
if SAVE_DATA:
    pd.DataFrame({"velocity": velocity_correlation, "time": time_correlation}).to_parquet(
        path=DATA_DIR / DATASET / "results" / "velovi_correlation.parquet"
    )

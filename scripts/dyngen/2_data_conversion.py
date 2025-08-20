# %% [markdown]
# # Dyngen data preparation
#
# This notebook prepares the dyngen-generated datasets for velocity, latent time, and GRN inference. Here, we only demonstrate how preprocessing was performed and how all algorithms were applied on the scale-1 datasets. For datasets of other scales, the analyses are carried out in exactly the same way.

# %% [markdown]
# ## Library imports

# %%
from tqdm import tqdm

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

import anndata as ad
import scanpy as sc
import scvelo as scv
from anndata import AnnData
from velovi import preprocess_data

from rgv_tools import DATA_DIR

# %% [markdown]
# ## General settings

# %%
scv.settings.verbosity = 3


# %% [markdown]
# ## Function definitions


# %%
def define_uns_elems(adata: AnnData) -> None:
    """Define prior regulation graph."""
    grn = np.zeros([adata.n_vars, adata.n_vars])
    df = adata.uns["regulatory_network"][["regulator", "target", "effect"]].copy()  # extracts regulatory network data

    df = pd.pivot(df, index="regulator", columns="target", values="effect").fillna(
        0
    )  # rows as regulators, columns as targets
    df[df != 0] = 1
    grn[np.ix_(adata.var_names.get_indexer(df.index), adata.var_names.get_indexer(df.columns))] = df.values

    adata.uns["skeleton"] = grn
    adata.uns["regulators"] = adata.var_names
    adata.uns["targets"] = adata.var_names
    adata.uns["network"] = grn


# %%
def update_data(adata: AnnData) -> None:
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

    adata.obs.rename(columns={"sim_time": "true_time"}, inplace=True)  # preserving simulated time as ground truth
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


# %%
def get_sc_grn(adata: AnnData) -> ArrayLike:
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
# ## Constants

# %%
DATASET = "dyngen"

# %%
COMPLEXITY = "complexity_1"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / COMPLEXITY / "processed").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
cnt = 0
for filename in tqdm((DATA_DIR / DATASET / COMPLEXITY / "raw").iterdir()):
    if filename.suffix != ".h5ad":
        continue

    simulation_id = int(filename.stem.removeprefix("dataset_sim"))
    print(f"Run {cnt}, dataset {simulation_id}.")
    adata = ad.io.read_h5ad(filename)

    define_uns_elems(adata=adata)
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

    for key in adata.uns:
        if isinstance(adata.uns[key], pd.Index):
            adata.uns[key] = list(adata.uns[key])  # Convert to list

    adata.write_zarr(DATA_DIR / DATASET / COMPLEXITY / "processed" / f"simulation_{simulation_id}.zarr")
    cnt += 1

# %%

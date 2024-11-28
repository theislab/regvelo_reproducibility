# %% [markdown]
# # veloVI-based analysis of hematopoiesis dataset
#
# Notebook runs the veloVI model on the hematopoiesis dataset.

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import mplscience

import anndata as ad
import cellrank as cr
import scanpy as sc
import scvi
from velovi import VELOVI

from rgv_tools import DATA_DIR, FIG_DIR
from rgv_tools.benchmarking import set_output

# %% [markdown]
# ## General settings

# %%
sc.settings.verbosity = 2

# %%
scvi.settings.seed = 0

# %% [markdown]
# ## Constants

# %%
DATASET = "hematopoiesis"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

# %%
SAVE_FIGURES = True
if SAVE_FIGURES:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)

# %%
TERMINALS_STATES = ["Mon", "Meg", "Bas", "Ery"]

# %% [markdown]
# ## Data loading

# %%
adata = ad.io.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed.h5ad")
adata_full = ad.io.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed_full.h5ad")

# %% [markdown]
# ## Run veloVI

# %%
## prepare model
VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
vae = VELOVI(adata)

# %%
vae.train()

# %%
set_output(adata, vae, n_samples=30)

# %% [markdown]
# ## Evaluate uncertainty

# %%
uncertainty_df, _ = vae.get_directional_uncertainty(n_samples=50)

# %%
adata.obs["veloVI intrinsic uncertainty"] = np.log10(uncertainty_df["directional_cosine_sim_variance"].values + 1e-6)

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    sc.pl.embedding(
        adata,
        color="veloVI intrinsic uncertainty",
        cmap="Greys",
        basis="draw_graph_fa",
        vmin="p1",
        vmax="p99",
        ax=ax,
        frameon=False,
    )

if SAVE_FIGURES:
    fig.savefig(FIG_DIR / DATASET / "vi_uncertainty.svg", format="svg", transparent=True, bbox_inches="tight")

# %%
uncertainty = pd.DataFrame({"uncertainty": uncertainty_df["directional_cosine_sim_variance"].values})

# %% [markdown]
# ## Calculate lineage fate probabilities and identify fate-associated genes

# %%
vk = cr.kernels.VelocityKernel(adata)
vk.compute_transition_matrix()
estimator = cr.estimators.GPCCA(vk)  ## We used vk here due to we want to benchmark on velocity

estimator.compute_macrostates(n_states=5, cluster_key="cell_type")
estimator.set_terminal_states(TERMINALS_STATES)

estimator.compute_fate_probabilities()
estimator.adata = adata_full.copy()
vi_ranking = estimator.compute_lineage_drivers(return_drivers=True, cluster_key="cell_type")

vi_ranking = vi_ranking.loc[:, ["Ery_corr", "Mon_corr", "Ery_pval", "Mon_pval"]]

# %% [markdown]
# ## Save dataset

# %% [markdown]
# Recalculate PCA for downstream CBC computation, as velocity is derived from the moment matrices.

# %%
sc.tl.pca(adata, layer="Ms")

# %% [markdown]
# Save adata with velocity layer

# %%
if SAVE_DATA:
    adata.write_h5ad(DATA_DIR / DATASET / "processed" / "adata_run_velovi.h5ad")

# %% [markdown]
# Save uncertainty and gene ranking results

# %%
if SAVE_DATA:
    uncertainty.to_csv(DATA_DIR / DATASET / "results" / "uncertainty_vi.csv")
    vi_ranking.to_csv(DATA_DIR / DATASET / "results" / "vi_ranking.csv")

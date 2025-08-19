# %% [markdown]
# # CellDancer benchmark on cell cycle data
#
# Use cellDancer inferred cell specific transcription rate and compare with ground truth.
#
# Note that cellDancer requires `anndata == 0.8.0`

# %% [markdown]
# ## Library imports

# %%
import celldancer as cd
import celldancer.cdplt as cdplt
from celldancer.cdplt import colormap

import scanpy as sc
from rgv_tools import DATA_DIR

# %% [markdown]
# ## Constants

# %%
DATASET = "cell_cycle_rpe1"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_processed.h5ad")

# %%
cell_type_u_s = cd.adata_to_df_with_embed(
    adata,
    cell_type_para="cell_cycle_phase",
    save_path=DATA_DIR / DATASET / "processed" / "cell_type_u_s_sample_df_processed.csv",
)

# %%
loss_df, cellDancer_df = cd.velocity(
    cell_type_u_s, permutation_ratio=0.1, norm_u_s=False, norm_cell_distribution=False, n_jobs=8
)

# %%
alpha_matrix = cellDancer_df.pivot(index="cellIndex", columns="gene_name", values="alpha")

# %%
alpha_matrix.index = adata.obs_names

# %% [markdown]
# ## Save data

# %%
if SAVE_DATA:
    DATA_DIR / DATASET / "processed" / "cell_type_u_s_sample_df_processed.csv"
    alpha_matrix.to_csv(DATA_DIR / DATASET / "processed" / "celldancer_alpha_estimate_processed.csv")

# %%

# %% [markdown]
# # Dynamo-based drivers analysis
#
# Notebook uses dynamo's LAP analysis to identify key drivers.

# %%
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import dynamo as dyn
import scanpy as sc
import scvelo as scv
from dynamo.tools.utils import nearest_neighbors

from rgv_tools import DATA_DIR

# %% [markdown]
# ## General settings

# %%
plt.rcParams["svg.fonttype"] = "none"
sns.reset_defaults()
sns.reset_orig()
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=14, color_map="viridis")

# %%
dyn.dynamo_logger.main_silence()

# %% [markdown]
# ## Constants

# %%
DATASET = "hematopoiesis"

SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

# %%
cell_type = ["HSC", "Ery", "Mon"]

# %%
fixed_points = np.array(
    [
        [8.45201833, 9.37697661],
        [14.00630381, 2.53853712],
        [17.30550636, 6.81561775],
        [18.06891717, 11.9840678],
        [14.13613403, 15.22244713],
        [9.72644402, 14.83745969],
    ]
)

# %% [markdown]
# ## Data loading

# %%
adata_labeling = sc.read_h5ad(DATA_DIR / DATASET / "raw" / "hsc_dynamo_adata.h5ad")

# %%
dyn.pl.streamline_plot(adata_labeling, basis="umap", color="cell_type")

HSC_cells = dyn.tl.select_cell(adata_labeling, "cell_type", "HSC")
Meg_cells = dyn.tl.select_cell(adata_labeling, "cell_type", "Meg")
Ery_cells = dyn.tl.select_cell(adata_labeling, "cell_type", "Ery")
Bas_cells = dyn.tl.select_cell(adata_labeling, "cell_type", "Bas")
Mon_cells = dyn.tl.select_cell(adata_labeling, "cell_type", "Mon")
Neu_cells = dyn.tl.select_cell(adata_labeling, "cell_type", "Neu")

# %% [markdown]
# ## Dynamo pipeline

# %%
HSC_cells_indices = nearest_neighbors(fixed_points[0], adata_labeling.obsm["X_umap"])
Meg_cells_indices = nearest_neighbors(fixed_points[1], adata_labeling.obsm["X_umap"])
Ery_cells_indices = nearest_neighbors(fixed_points[2], adata_labeling.obsm["X_umap"])
Bas_cells_indices = nearest_neighbors(fixed_points[3], adata_labeling.obsm["X_umap"])
Mon_cells_indices = nearest_neighbors(fixed_points[4], adata_labeling.obsm["X_umap"])
Neu_cells_indices = nearest_neighbors(fixed_points[5], adata_labeling.obsm["X_umap"])

# %%
plt.scatter(*adata_labeling.obsm["X_umap"].T)
for indices in [
    HSC_cells_indices,
    Meg_cells_indices,
    Ery_cells_indices,
    Bas_cells_indices,
    Mon_cells_indices,
    Neu_cells_indices,
]:
    plt.scatter(*adata_labeling[indices[0]].obsm["X_umap"].T)

# %%
plt.scatter(*adata_labeling.obsm["X_umap"].T)
for indices in [
    HSC_cells_indices,
    Meg_cells_indices,
    Ery_cells_indices,
    Bas_cells_indices,
    Mon_cells_indices,
    Neu_cells_indices,
]:
    plt.scatter(*adata_labeling[indices[0]].obsm["X_umap"].T)
plt.show()

# %%
dyn.tl.neighbors(adata_labeling, basis="umap", result_prefix="umap")

# %%
dyn.tl.cell_velocities(
    adata_labeling,
    enforce=True,
    X=adata_labeling.layers["M_t"],
    V=adata_labeling.layers["velocity_alpha_minus_gamma_s"],
    method="cosine",
    basis="pca",
)
dyn.vf.VectorField(adata_labeling, basis="pca")

# %%
transition_graph = {}
start_cell_indices = [
    HSC_cells_indices,
    Ery_cells_indices,
    Mon_cells_indices,
]
end_cell_indices = start_cell_indices
for i, start in enumerate(start_cell_indices):
    for j, end in enumerate(end_cell_indices):
        if start is not end:
            min_lap_t = True if i == 0 else False
            lap = dyn.pd.least_action(
                adata_labeling,
                [adata_labeling.obs_names[start[0]][0]],
                [adata_labeling.obs_names[end[0]][0]],
                basis="pca",
                adj_key="cosine_transition_matrix",
                min_lap_t=min_lap_t,
                EM_steps=2,
            )
            # The `GeneTrajectory` class can be used to output trajectories for any set of genes of interest
            gtraj = dyn.pd.GeneTrajectory(adata_labeling)
            gtraj.from_pca(lap.X, t=lap.t)
            gtraj.calc_msd()
            ranking = dyn.vf.rank_genes(adata_labeling, "traj_msd")

            print(start, "->", end)
            genes = ranking[:5]["all"].to_list()
            arr = gtraj.select_gene(genes)

            transition_graph[cell_type[i] + "->" + cell_type[j]] = {
                "lap": lap,
                "LAP_pca": adata_labeling.uns["LAP_pca"],
                "ranking": ranking,
                "gtraj": gtraj,
            }

# %%
## evaluate ranking from HSC to Mon and Ery
HSC_Mon_ranking = transition_graph["HSC->Mon"]["ranking"]
HSC_Ery_ranking = transition_graph["HSC->Ery"]["ranking"]

# %% [markdown]
# ## Save dataset

# %%
if SAVE_DATA:
    HSC_Mon_ranking.to_csv(DATA_DIR / DATASET / "results" / "HSC_Mon_ranking.csv")
    HSC_Ery_ranking.to_csv(DATA_DIR / DATASET / "results" / "HSC_Ery_ranking.csv")

# %% [markdown]
# # Pancreatic endocrine cell fate analysis
#
# Notebooks for analyzing pancreatic endocrine cell fate decision

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd
from scipy.stats import ranksums

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

import cellrank as cr
import scanpy as sc
import scvelo as scv

from rgv_tools import DATA_DIR, FIG_DIR

# %% [markdown]
# ## General settings

# %%
plt.rcParams["svg.fonttype"] = "none"
sns.reset_defaults()
sns.reset_orig()
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=14, color_map="viridis")

# %% [markdown]
# ## Constants

# %%
DATASET = "pancreatic_endocrine"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

SAVE_FIGURES = True
if SAVE_FIGURES:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)

# %%
TERMINAL_STATES = ["Alpha", "Beta", "Delta", "Epsilon"]

# %%
VELOCITY_METHODS = ["regvelo", "velovi", "scvelo"]
N_STATES = [7, 10, 10]  # optimal states learned from `3_comparison_TSI`

# %% [markdown]
# ## Data loading

# %% [markdown]
# ### Using CellRank pipeline for fate mapping

# %%
for _idx, (n_state, method) in enumerate(zip(N_STATES, VELOCITY_METHODS)):
    adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / f"adata_run_{method}.h5ad")
    vk = cr.kernels.VelocityKernel(adata).compute_transition_matrix()
    ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
    estimator = cr.estimators.GPCCA(0.8 * vk + 0.2 * ck)

    estimator.compute_macrostates(n_states=n_state, cluster_key="clusters")
    estimator.set_terminal_states(
        list(set(estimator.macrostates.cat.categories.tolist()).intersection(TERMINAL_STATES))
    )
    estimator.compute_fate_probabilities(solver="direct")
    estimator.plot_fate_probabilities(same_plot=False)

    if method == "regvelo":
        fate_prob = estimator.fate_probabilities
        sampleID = adata.obs.index.tolist()
        fate_name = fate_prob.names.tolist()
        fate_prob = pd.DataFrame(fate_prob, index=sampleID, columns=fate_name)

    with mplscience.style_context():
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.set_style(style="whitegrid")
        estimator.plot_fate_probabilities(states=["Alpha"], same_plot=False, title="", ax=ax)

        if SAVE_FIGURES:
            fig.savefig(
                FIG_DIR / DATASET / f"alpha_cell_{method}.svg", format="svg", transparent=True, bbox_inches="tight"
            )

        plt.show()

# %% [markdown]
# ## Identify Epsilon's subpopulstions

# %%
adata.obs["Alpha"] = fate_prob.loc[:, "Alpha"]

# %%
Epsilon = adata[adata.obs["clusters"] == "Epsilon"].copy()
## calculate pca and plot umap
sc.tl.pca(Epsilon)
sc.pp.neighbors(Epsilon)
sc.tl.leiden(Epsilon)
sc.tl.umap(Epsilon)
scv.pl.umap(Epsilon, color="leiden", legend_loc="on data")

# %%
scv.pl.umap(Epsilon, color="Alpha")

# %% [markdown]
# ## Identify differential expressed TF among two populations

# %%
## screening TF, identify the driver
TF_list = adata.var_names[adata.var["tf"]]
pval = []
for i in TF_list:
    x = np.array(Epsilon[Epsilon.obs["leiden"] != "2", i].X.todense()).flatten()
    y = np.array(Epsilon[Epsilon.obs["leiden"] == "2", i].X.todense()).flatten()
    _, res = ranksums(x, y, alternative="greater")
    pval.append(res)

# %%
res = pd.DataFrame({"TF": list(TF_list), "Pval": pval})
res = res.sort_values(by="Pval")
res

# %% [markdown]
# ## Visualize DEG

# %%
cell_states = np.array(Epsilon.obs["leiden"].copy())
cell_states[cell_states == "2"] = "State1"
cell_states[cell_states != "State1"] = "State2"

# %%
Epsilon.obs["cell_states"] = list(cell_states)

# %%
## Visualize gene expression dynamics
with mplscience.style_context():
    markers = ["Pou6f2", "Irx1", "Smarca1", "Arg1", "Hes6", "Neurog3"]
    fig, ax = plt.subplots(figsize=(4, 3))
    # sns.set_style(style="whitegrid")
    sc.pl.dotplot(Epsilon, markers, groupby="cell_states", swap_axes=True, dot_max=0.8, ax=ax)

    if SAVE_FIGURES:
        fig.savefig(
            FIG_DIR / DATASET / "feature_marker_expression.svg", format="svg", transparent=True, bbox_inches="tight"
        )

    plt.show()

# %%

# %%

# %%

# %%

# %%

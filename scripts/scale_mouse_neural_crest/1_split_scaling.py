# %% [markdown]
# # Split mouse neural crest data into different scale

# %% [markdown]
# ## Library imports

# %%
import scvelo as scv
import scanpy as sc
import numpy as np
import anndata as ad

import pandas as pd
import matplotlib.pyplot as plt
import mplscience

from collections import Counter
import pandas as pd

from rgv_tools import DATA_DIR, FIG_DIR

# %% [markdown]
# ## General settings

# %%
sc.settings.verbosity = 2
scv.settings.verbosity = 3

# %%
plt.rcParams["svg.fonttype"] = "none"
scv.settings.set_figure_params("scvelo", dpi=100, transparent=True, fontsize=14, color_map="viridis")

# %% [markdown]
# ## Constants

# %%
DATASET = "mouse_neural_crest"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)

SAVE_FIGURES = True
if SAVE_FIGURES:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)
FIGURE_FORMAT = "svg"

# %%
TERMINAL_STATE = [
    "Melanocytes",
    "enFib",
    "SC",
    "Mesenchyme",
    "Sensory",
    "ChC",
    "SatGlia",
    "Gut_glia",
    "Gut_neuron",
    "Symp",
    "BCC",
]
Location = ["Cranial", "Trunk", "DRG", "Limb", "Enteric", "Sympathoadrenal system", "Incisor"]


# %% [markdown]
# ## Functions definations


# %%
def count_cell_types(cell_type_list, class_list):
    counter = Counter(cell_type_list)

    result = {cls: counter.get(cls, 0) for cls in class_list}

    df = pd.DataFrame({"Cell Type": list(result.keys()), "Count": list(result.values())})

    return df


# %% [markdown]
# ## Data loading

# %%
ldata = ad.io.read_h5ad(DATA_DIR / DATASET / "raw" / "GSE201257_adata_velo_raw.h5ad")
adata = ad.io.read_h5ad(DATA_DIR / DATASET / "raw" / "GSE201257_adata_processed.h5ad")

# %% [markdown]
# ## Annotate all datasets
#
# We followed the annotation procedure provided by original authors, please check https://github.com/LouisFaure/glialfates_paper

# %%
adata.layers["GEX"] = adata.X.copy()

# %%
adata.X = sc.pp.scale(adata.layers["palantir_imp"], max_value=10)

# %%
sc.tl.score_genes(
    adata,
    ["Ret", "Phox2a", "Chrna3", "Sox11"],
    score_name="Gut_neuron",
)
adata.obs.Gut_neuron = adata.obs.Gut_neuron > 1.5
sc.tl.score_genes(adata, ["Prdm12", "Isl2", "Pou4f1", "Six1"], score_name="Sensory")
adata.obs.Sensory = adata.obs.Sensory > 1
sc.tl.score_genes(adata, ["Cartpt", "Prph", "Mapt", "Maoa"], score_name="Symp")
adata.obs.Symp = adata.obs.Symp > 1.5
sc.tl.score_genes(adata, ["Lum", "Dcn", "Fbn1"], score_name="enFib")
adata.obs.enFib = adata.obs.enFib > 2.5
sc.tl.score_genes(adata, ["Th", "Dbh", "Chga", "Chgb", "Slc18a1", "Slc18a2"], score_name="ChC")
adata.obs.ChC = adata.obs.ChC > 2
sc.tl.score_genes(adata, ["Phox2b", "Ctgf", "Nfia", "Tgfb2", "S100b"], score_name="Gut_glia")
adata.obs.Gut_glia = adata.obs.Gut_glia > 2.5
sc.tl.score_genes(adata, ["Sox9", "Wnt1", "Ets1", "Crabp2"], score_name="NCC")
adata.obs.NCC = adata.obs.NCC > 0.6
sc.tl.score_genes(adata, ["Prrx1", "Prrx2", "Pdgfra"], score_name="Mesenchyme")
adata.obs.Mesenchyme = adata.obs.Mesenchyme > 0.4
sc.tl.score_genes(adata, ["Dct", "Mitf", "Pmel", "Tyr"], score_name="Melanocytes")
adata.obs.Melanocytes = adata.obs.Melanocytes > 1
sc.tl.score_genes(adata, ["Fabp7", "Ptn", "Rgcc"], score_name="SatGlia")
adata.obs.SatGlia = adata.obs.SatGlia > 2.6

# %%
sc.tl.score_genes(adata, ["Mpz", "Plp1", "Fbxo7", "Gjc3", "Pmp22", "Dhh", "Mal"], score_name="SC")
adata.obs.SC = adata.obs.SC > 1.35

# %%
sc.tl.score_genes(adata, ["Prss56", "Egr2", "Wif1", "Hey2"], score_name="BCC")
adata.obs.BCC = adata.obs.BCC > 6

# %%
celltypes = [
    "NCC",
    "Symp",
    "ChC",
    "Sensory",
    "Gut_glia",
    "Gut_neuron",
    "Melanocytes",
    "SC",
    "enFib",
    "Mesenchyme",
    "SatGlia",
    "BCC",
]
adata.obs["conflict"] = adata.obs[celltypes].sum(axis=1) > 1

adata.obs.loc[adata.obs[celltypes].sum(axis=1) > 1, celltypes] = False

adata.obs.loc[adata.obs[celltypes].sum(axis=1) == 2, celltypes] = False

annot = adata.obs.loc[:, celltypes].apply(lambda x: np.argwhere(x.values), axis=1)
annot = annot[annot.apply(len) == 1]
annot = annot.apply(lambda x: np.array(celltypes)[x][0][0])

adata.obs["assignments"] = "none"
adata.obs.loc[annot.index, "assignments"] = annot.values
adata.obs["assignments"] = adata.obs["assignments"].astype("category")

# %%
adata.obs.assignments.cat.categories

# %%
sc.pl.umap(adata, color="assignments", groups=celltypes)

# %%
set(TERMINAL_STATE).intersection(np.unique(adata.obs["assignments"]))

# %%
adata.obs["all_states"] = adata.obs["assignments"].astype("str")
adata.obs["all_states"][adata.obs["all_states"] == "none"] = np.nan

# %%
adata.obs["all_states"] = adata.obs["all_states"].astype("category")
state_names = adata.obs["all_states"].cat.categories.tolist()

# %%
adata.obs["all_states"] = (
    adata.obs["all_states"].astype(str).astype("category").cat.reorder_categories(["nan"] + state_names)
)
adata.uns["all_states_colors"] = ["#dedede"] + adata.uns["assignments_colors"][:12]

scv.pl.scatter(
    adata,
    basis="X_umap",
    c="all_states",
    add_outline=state_names,
)

# %% [markdown]
# ## Merge spliced/unspliced readout

# %%
# adata=scv.utils.merge(adata,ldata)
# adata.obsp=None
del adata.uns["neighbors"]

# %%
adata

# %% [markdown]
# ## Split into different scale level

# %%
Location

# %%
adatas = []
for i in range(2, 7):
    adatas.append(adata[[c in Location[: (i + 1)] for c in adata.obs["location"]]].copy())
del adatas[3]  # Merge last two stage

# %%
adatas

# %% [markdown]
# ## For each scale we independently build UMAP
#
# Following preprocessing procedure in glia fate data preprocessing code

# %%
adatas_p = []
for ad_idx in range(len(adatas)):
    ad = adatas[ad_idx]

    st = count_cell_types(ad.obs["assignments"], TERMINAL_STATE)
    ad = ad[[i not in st["Cell Type"][st["Count"] < 30].tolist() for i in ad.obs["assignments"]]].copy()

    ## UMAP embedding
    sc.pp.neighbors(ad, n_neighbors=80, use_rep="X_diff")

    adatas_p.append(ad)

# %%
for idx in range(4):
    ad = adatas_p[idx].copy()
    ad.X = ad.layers["GEX"].copy()

    with mplscience.style_context():
        fig, ax = plt.subplots(figsize=(4, 4))
        # scv.pl.scatter(adata[cells,:], c="assignments", ax=ax,frameon = False,add_outline = set(celltypes).intersection(np.unique(adata.obs["assignments"])),legend_loc = "right")
        scv.pl.scatter(ad, basis="X_umap", c="all_states", add_outline=state_names, legend_loc="right", ax=ax)

        if SAVE_FIGURES:
            fig.savefig(
                FIG_DIR / DATASET / f"scale_{idx+1}_umap.svg",
                format=FIGURE_FORMAT,
                transparent=True,
                bbox_inches="tight",
            )

    if SAVE_DATA:
        ad.write_h5ad(DATA_DIR / DATASET / "processed" / f"adata_stage{idx+1}_processed.h5ad")

# %%

# %% [markdown]
# # Dynamo-based perturbation analysis
#
# Notebooks for predicting TF perturbation effects using dynamo.

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

import dynamo as dyn
import scanpy as sc
from dynamo.preprocessing import Preprocessor

from rgv_tools import DATA_DIR, FIG_DIR
from rgv_tools.perturbation import (
    abundance_test,
    get_list_name,
    Multiple_TFScanning_KO_dyn,
    Multiple_TFScanning_perturbation_dyn,
    split_elements,
    TFScanning_KO_dyn,
    TFScanning_perturbation_dyn,
)

# %% [markdown]
# ## Constants

# %%
DATASET = "zebrafish"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

# %%
SAVE_FIGURES = False
if SAVE_FIGURES:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)


# %%
genes = ["nr2f5", "sox9b", "twist1b", "ets1"]

TERMINAL_STATES = [
    "mNC_head_mesenchymal",
    "mNC_arch2",
    "mNC_hox34",
    "Pigment",
]

# %%
single_ko = ["elk3", "erf", "fli1a", "mitfa", "nr2f5", "rarga", "rxraa", "smarcc1a", "tfec", "nr2f2"]
multiple_ko = ["fli1a_elk3", "mitfa_tfec", "tfec_mitfa_bhlhe40", "fli1a_erf_erfl3", "erf_erfl3"]

# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed.h5ad")

# %%
adata.X = adata.layers["matrix"].copy()

# %% [markdown]
# ## Processing by dynamo

# %%
preprocessor = Preprocessor()
preprocessor.preprocess_adata(adata, recipe="monocle")

# %%
dyn.tl.dynamics(adata)

# %%
dyn.tl.reduceDimension(adata)

# %%
dyn.tl.cell_velocities(adata, basis="pca")

# %%
dyn.vf.VectorField(adata, basis="pca")

# %%
adata_perturb = adata.copy()

# %%
del adata.uns["cell_type_colors"]

# %% [markdown]
# ## Compare predict fate probability changes

# %%
fate_prob = {}
fate_prob_perturb = {}

for g in genes:
    fb, fb_perturb = TFScanning_KO_dyn(adata, 8, "cell_type", TERMINAL_STATES, [g], fate_prob_return=True)
    fate_prob[g] = fb
    fate_prob_perturb[g] = fb_perturb

# %% [markdown]
# ## Visualize dynamo perturbation effects

# %%
for g in genes:
    data = abundance_test(fate_prob[g], fate_prob_perturb[g])
    data = pd.DataFrame(
        {
            "Score": data.iloc[:, 0].tolist(),
            "p-value": data.iloc[:, 1].tolist(),
            "Terminal state": data.index.tolist(),
            "TF": [g] * (data.shape[0]),
        }
    )

    final_df = data.copy()
    final_df["Score"] = 0.5 - final_df["Score"]

    color_label = "cell_type"
    df = pd.DataFrame(final_df["Score"])
    df.columns = ["coefficient"]
    df["Cell type"] = final_df["Terminal state"]
    order = df["Cell type"].tolist()

    palette = dict(zip(adata.obs[color_label].cat.categories, adata_perturb.uns[f"{color_label}_colors"]))
    subset_palette = {name: color for name, color in palette.items() if name in TERMINAL_STATES}

    with mplscience.style_context():
        sns.set_style(style="whitegrid")
        fig, ax = plt.subplots(figsize=(2, 2))
        sns.barplot(
            data=df,
            y="coefficient",
            x="Cell type",
            palette=subset_palette,
            order=order,
            ax=ax,
        )
        ax.tick_params(axis="x", rotation=90)
        plt.title("$\\mathit{" + g + "}$ regulon knock out simulation")

        if SAVE_FIGURES:
            plt.savefig(
                FIG_DIR / DATASET / f"{g}_perturbation_simulation_dynamo.svg",
                format="svg",
                transparent=True,
                bbox_inches="tight",
            )
        # Show the plot
        plt.show()

# %% [markdown]
# ## Perturbation prediction

# %% [markdown]
# ### Single gene knockout

# %%
single_ko = set(single_ko).intersection(adata.var_names)
single_ko = list(single_ko)

# %%
## Dynamo (KO)
ko_dyn = TFScanning_KO_dyn(adata, 8, "cell_type", TERMINAL_STATES, single_ko)

## Dynamo (perturbation)
perturbation_dyn = TFScanning_perturbation_dyn(adata, 8, "cell_type", TERMINAL_STATES, single_ko)

# %%
## Perform KO screening using function based perturbation
coef_KO = pd.DataFrame(np.array(ko_dyn["coefficient"]))
coef_KO.index = ko_dyn["TF"]
coef_KO.columns = get_list_name(ko_dyn["coefficient"][0])

## Perform perturbation screening using gene expression based perturbation
coef_perturb = pd.DataFrame(np.array(perturbation_dyn["coefficient"]))
coef_perturb.index = perturbation_dyn["TF"]
coef_perturb.columns = get_list_name(perturbation_dyn["coefficient"][0])

# %% [markdown]
# ### Multiple gene knock-out prediction

# %%
multiple_ko_list = split_elements(multiple_ko)

# %%
## Dynamo (KO)
ko_dyn = Multiple_TFScanning_KO_dyn(adata, 9, "cell_type", TERMINAL_STATES, multiple_ko_list)

## Dynamo (perturbation)
perturbation_ko = Multiple_TFScanning_perturbation_dyn(adata, 9, "cell_type", TERMINAL_STATES, multiple_ko_list)

# %%
## Perform KO screening using function based perturbation
coef_KO_multiple = pd.DataFrame(np.array(ko_dyn["coefficient"]))
coef_KO_multiple.index = ko_dyn["TF"]
coef_KO_multiple.columns = get_list_name(ko_dyn["coefficient"][0])

## Perform perturbation screening using gene expression based perturbation
coef_perturb_multiple = pd.DataFrame(np.array(perturbation_dyn["coefficient"]))
coef_perturb_multiple.index = perturbation_dyn["TF"]
coef_perturb_multiple.columns = get_list_name(perturbation_dyn["coefficient"][0])

# %% [markdown]
# ## Save dataset

# %%
if SAVE_DATA:
    coef_KO.to_csv(DATA_DIR / DATASET / "results" / "dynamo_KO_single.csv")
    coef_KO_multiple.to_csv(DATA_DIR / DATASET / "results" / "dynamo_KO_multiple.csv")

    coef_perturb.to_csv(DATA_DIR / DATASET / "results" / "dynamo_perturb_single.csv")
    coef_perturb_multiple.to_csv(DATA_DIR / DATASET / "results" / "dynamo_perturb_multiple.csv")

# %%

# %%

# %%

# %%

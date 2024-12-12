# %% [markdown]
# # Identify Neurod2 -> Rfx6 as an important regulatory link for epsilon cells

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

import scanpy as sc
import scvelo as scv
from regvelo import REGVELOVI

from rgv_tools import DATA_DIR, FIG_DIR
from rgv_tools.perturbation import (
    aggregate_model_predictions,
    get_list_name,
    inferred_GRN,
    RegulationScanning,
)

# %% [markdown]
# ## General settings

# %%
# %matplotlib inline

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
    (DATA_DIR / DATASET / "results" / "Neurod2_screening_repeat_runs").mkdir(parents=True, exist_ok=True)

SAVE_FIGURES = True
if SAVE_FIGURES:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)

# %%
TERMINAL_STATES = ["Alpha", "Beta", "Delta", "Epsilon"]

# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed_filtered.h5ad")

# %%
palette = dict(zip(adata.obs["clusters"].cat.categories, adata.uns["clusters_colors"]))

# %%
sc.pl.violin(adata, keys="Neurod2", groupby="clusters", rotation=90)

# %% [markdown]
# ## Build RegVelo's GRN

# %%
GRN_list = []  # Assuming there are 4 metrics

for i in range(5):
    model_name = "rgv_model_" + str(i)
    model = DATA_DIR / DATASET / "processed" / "perturb_repeat_runs" / model_name
    ### load model
    vae = REGVELOVI.load(model, adata)
    GRN_list.append(inferred_GRN(vae, adata, label="clusters", group="all"))

# %%
grn_all = np.mean(np.stack(GRN_list), axis=0)

# %%
predict_weight = grn_all[:, [i == "Neurod2" for i in adata.var.index]]
prior = vae.module.v_encoder.mask_m_raw[:, [i == "Neurod2" for i in adata.var.index]].detach().cpu().numpy()

# %%
predict_weight = pd.DataFrame(predict_weight, index=adata.var.index)
predict_weight.loc[:, "weight"] = predict_weight.iloc[:, 0].abs()
predict_weight.loc[:, "prior"] = prior

# %% [markdown]
# ## Screening from top-10 targets

# %%
genes = predict_weight.sort_values("weight", ascending=False).iloc[:10, :].index.tolist()

# %%
terminal_states = ["Alpha", "Delta", "Beta", "Epsilon"]
for i in range(5):
    print("loading model...")
    model_name = "rgv_model_" + str(i)
    model = DATA_DIR / DATASET / "processed" / "perturb_repeat_runs" / model_name

    coef_name = "coef_" + str(i)
    res_save_coef = DATA_DIR / DATASET / "results" / "Neurod2_screening_repeat_runs" / coef_name

    pval_name = "pval_" + str(i)
    res_save_pval = DATA_DIR / DATASET / "results" / "Neurod2_screening_repeat_runs" / pval_name

    print("inferring perturbation...")
    perturb_screening = RegulationScanning(
        model, adata, 8, "clusters", TERMINAL_STATES, "Neurod2", genes, 0, method="t-statistics"
    )
    coef = pd.DataFrame(np.array(perturb_screening["coefficient"]))
    coef.index = perturb_screening["target"]
    coef.columns = get_list_name(perturb_screening["coefficient"][0])
    pval = pd.DataFrame(np.array(perturb_screening["pvalue"]))
    pval.index = perturb_screening["target"]
    pval.columns = get_list_name(perturb_screening["pvalue"][0])

    coef.to_csv(res_save_coef)
    pval.to_csv(res_save_pval)

# %% [markdown]
# ## Plot perturbation results

# %%
Neurod2_target_perturbation = aggregate_model_predictions(
    DATA_DIR / DATASET / "results" / "Neurod2_screening_repeat_runs", method="t-statistics"
)

# %%
## rank the Epsilon prediction results
with mplscience.style_context():  # Entering the custom style context
    sns.set_style("whitegrid")
    gene_scores = pd.DataFrame(
        {
            "Gene": Neurod2_target_perturbation[0].index.tolist(),
            "Score": Neurod2_target_perturbation[0].loc[:, "Epsilon"],
        }
    )
    gene_scores.loc[:, "weights"] = gene_scores.loc[:, "Score"].abs()
    # gene_scores = gene_scores.sort_values(by='weights', ascending=False).iloc[:10,:]
    gene_scores = gene_scores.sort_values(by="Score", ascending=True).iloc[:10, :]
    # Create the horizontal bar plot using Seaborn
    fig, ax = plt.subplots(figsize=(5, 3))
    g = sns.barplot(x="Score", y="Gene", data=gene_scores, color="grey")

    # Customize plot aesthetics
    g.set_ylabel("Gene", fontsize=14)
    g.set_xlabel("Test statistics", fontsize=14)

    # Customize tick parameters for better readability
    g.tick_params(axis="x", labelsize=14)
    g.tick_params(axis="y", labelsize=14)
    g.set_title("TF-target regulation perturb screening", fontsize=14)

    plt.tight_layout()
    plt.setp(g.get_yticklabels(), fontstyle="italic")

    if SAVE_FIGURES:
        save_path = FIG_DIR / DATASET / "Neurod2_target_screening.svg"
        fig.savefig(save_path, format="svg", transparent=True, bbox_inches="tight")
    plt.show()

# %%

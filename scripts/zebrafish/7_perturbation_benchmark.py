# %% [markdown]
# # Perturbation prediction benchmark
#
# Notebook compares perturbation prediction.

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

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
DATASET = "zebrafish"

# %%
SAVE_FIGURES = True
if SAVE_FIGURES:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Function definitions


# %%
def perturb_prediction(coef, perturbation, gene_list):
    """TODO."""
    gt = perturbation.loc[[i in coef.index for i in perturbation.loc[:, "sgRNA_group"]], :].copy()
    gt = gt.loc[
        [i in ["Pigment_gch2_high", "mNC_hox34", "mNC_arch2", "mNC_head_mesenchymal"] for i in gt.loc[:, "cell_anno"]],
        :,
    ]

    ## zero-center the likelihood of different panel.
    for tf in gene_list:
        gt.loc[[i in tf for i in gt.loc[:, "sgRNA_group"]], "median_likelihood"] = gt.loc[
            [i in tf for i in gt.loc[:, "sgRNA_group"]], "median_likelihood"
        ] - np.mean(gt.loc[[i in tf for i in gt.loc[:, "sgRNA_group"]], "median_likelihood"])

    terminal_states = ["Pigment", "mNC_hox34", "mNC_arch2", "mNC_head_mesenchymal"]
    coef = coef.loc[:, terminal_states]
    coef.columns = ["Pigment_gch2_high", "mNC_hox34", "mNC_arch2", "mNC_head_mesenchymal"]
    pred_effect = []
    TF = []
    for i in range(gt.shape[0]):
        ts = gt.iloc[i, 0]
        tf = gt.iloc[i, 2]
        effect = coef.loc[tf, ts]
        pred_effect.append(effect)
        TF.append(tf)
    pred = pd.DataFrame({"TF": TF, "effect": pred_effect})
    return scipy.stats.spearmanr(pred.iloc[:, 1], gt.iloc[:, 1])[0]


# %% [markdown]
# ## Data loading

# %%
gene_list = ["elk3", "erf", "etv2", "fli1a", "mitfa", "nr2f5", "rarga", "rxraa", "smarcc1a", "tfec", "nr2f2"]
perturbation = pd.read_csv(DATA_DIR / DATASET / "raw" / "df_median_meld_likelihood_new.csv")

# %% [markdown]
# #### predict perturbation

# %%
coef_perturb_co = pd.read_csv(DATA_DIR / DATASET / "results" / "celloracle_perturb_single.csv", index_col=0)
coef_perturb_dyn = pd.read_csv(DATA_DIR / DATASET / "results" / "dynamo_perturb_single.csv", index_col=0)
coef_KO_dyn = pd.read_csv(DATA_DIR / DATASET / "results" / "dynamo_KO_single.csv", index_col=0)

# %%
score_s = []
method = []

score_s.append(perturb_prediction(1 - coef_perturb_co, perturbation, gene_list))
method.append("celloracle")

score_s.append(perturb_prediction(coef_perturb_dyn, perturbation, gene_list))
method.append("Dynamo(perturbation)")

score_s.append(perturb_prediction(1 - coef_KO_dyn, perturbation, gene_list))
method.append("Dynamo(KO)")

# %%
for nrun in range(3):
    coef_perturb_rgv = pd.read_csv(DATA_DIR / DATASET / "results" / f"coef_single_regvelo_{nrun}", index_col=0)
    score_s.append(perturb_prediction(1 - coef_perturb_rgv, perturbation, gene_list))
    method.append("RegVelo")

# %% [markdown]
# ### multiple knock-out

# %%
gene_list = ["fli1a_elk3", "mitfa_tfec", "tfec_mitfa_bhlhe40", "fli1a_erf_erfl3", "erf_erfl3"]

# %% [markdown]
# #### predict perturbation

# %%
coef_perturb_co = pd.read_csv(DATA_DIR / DATASET / "results" / "celloracle_perturb_multiple.csv", index_col=0)
coef_perturb_dyn = pd.read_csv(DATA_DIR / DATASET / "results" / "dynamo_perturb_multiple.csv", index_col=0)
coef_KO_dyn = pd.read_csv(DATA_DIR / DATASET / "results" / "dynamo_KO_multiple.csv", index_col=0)

# %%
score_m = []
method = []

score_m.append(perturb_prediction(1 - coef_perturb_co, perturbation, gene_list))
method.append("celloracle")

score_m.append(perturb_prediction(coef_perturb_dyn, perturbation, gene_list))
method.append("Dynamo(perturbation)")

score_m.append(perturb_prediction(1 - coef_KO_dyn, perturbation, gene_list))
method.append("Dynamo(KO)")

# %%
for nrun in range(3):
    coef_perturb_rgv = pd.read_csv(DATA_DIR / DATASET / "results" / f"coef_multiple_regvelo_{nrun}", index_col=0)
    score_m.append(perturb_prediction(1 - coef_perturb_rgv, perturbation, gene_list))
    method.append("RegVelo")

# %% [markdown]
# ## Plot benchmark results

# %%
dat = pd.DataFrame({"spearman_correlation": score_s, "method": method})
dat2 = pd.DataFrame({"spearman_correlation": score_m, "method": method})
dat["Experimental class"] = "Single TF knock-out"
dat2["Experimental class"] = "Multiple TF knock-out"
df = pd.concat([dat, dat2], axis=0)

# %%
colors = [
    (0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
    (0.9254901960784314, 0.8823529411764706, 0.2),
    (0.33725490196078434, 0.7058823529411765, 0.9137254901960784),
    (0.792156862745098, 0.5686274509803921, 0.3803921568627451),
]

# %%
order = ["RegVelo", "Dynamo (KO)", "Dynamo (perturbation)", "celloracle"]
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(4, 5))

    # Plot the barplot without error bars
    sns.barplot(
        data=df,
        y="Experimental class",
        x="spearman_correlation",
        hue="method",
        hue_order=order,
        palette=colors,
        ax=ax,
        ci=None,
    )

    # Add jittered dots
    sns.stripplot(
        data=df,
        y="Experimental class",
        x="spearman_correlation",
        hue="method",
        hue_order=order,
        dodge=True,
        color="black",
        ax=ax,
        jitter=True,
    )

    # Remove the duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[4 : (4 + len(colors))],
        labels[4 : (4 + len(colors))],
        bbox_to_anchor=(0.5, -0.1),
        loc="upper center",
        ncol=2,
    )

    # Customize labels and other settings
    ax.set(ylabel="", xlabel="Perturbation prediction score")
    ax.set_xlabel(xlabel="Perturbation prediction score", fontsize=13)

    if SAVE_FIGURES:
        plt.savefig(
            FIG_DIR / DATASET / "barplot_joint_knockout_update.svg", format="svg", transparent=True, bbox_inches="tight"
        )
    plt.show()

# %% [markdown]
# ## Comparison

# %%
fc = score_s[3:] / score_s[2]
scipy.stats.ttest_ind(fc, fc * 0, equal_var=False, alternative="greater")

# %%
fc = score_m[3:] / score_m[2]
scipy.stats.ttest_ind(fc, fc * 0, equal_var=False, alternative="greater")

# %% [markdown]
# ## Reason why pick gch2-high pigment cells

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed.h5ad")
sc.pl.umap(adata, color=["gch2", "sox6"])

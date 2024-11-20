# %% [markdown]
# # Ealy driver perturbation prediction
#
# Notebook analyses early drivers including nr2f5, sox9b, twist1b, and ets1.

# %% [markdown]
# ## Library imports

# %%
import pandas as pd
import scipy

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

import cellrank as cr
import scanpy as sc
import scvi
from regvelo import REGVELOVI

from rgv_tools import DATA_DIR, FIG_DIR
from rgv_tools.benchmarking import set_output
from rgv_tools.perturbation import abundance_test, DEG, in_silico_block_simulation
from rgv_tools.plotting import bar_scores

# %% [markdown]
# ## General settings

# %%
scvi.settings.seed = 0

# %% [markdown]
# ## Constants

# %%
DATASET = "zebrafish"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

# %%
SAVE_FIGURES = True
if SAVE_FIGURES:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)

# %%
DRIVERS = ["nr2f5", "sox9b", "twist1b", "ets1"]

# %%
TERMINAL_STATES = [
    "mNC_head_mesenchymal",
    "mNC_arch2",
    "mNC_hox34",
    "Pigment",
]

# %% [markdown]
# ## Data Loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_run_regvelo.h5ad")

# %% [markdown]
# ## CellRank pipeline

# %%
vk = cr.kernels.VelocityKernel(adata)
vk.compute_transition_matrix()
ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()

kernel = 0.8 * vk + 0.2 * ck

# %%
estimator = cr.estimators.GPCCA(kernel)
# evaluate the fate prob on original space
estimator.compute_macrostates(n_states=8, cluster_key="cell_type")
estimator.set_terminal_states(TERMINAL_STATES)
estimator.compute_fate_probabilities()

df = estimator.compute_lineage_drivers(cluster_key="cell_type")

# %% [markdown]
# ## Plotting

# %% [markdown]
# ### Gene expression

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(1, 4, figsize=(20, 3))
    for axis, gene in zip(ax, DRIVERS):
        axis = sc.pl.umap(
            adata, color=[gene], vmin="p1", vmax="p99", frameon=False, legend_fontsize=14, show=False, ax=axis
        )

    if SAVE_FIGURES:
        fig.savefig(FIG_DIR / DATASET / "gene_expression.svg", format="svg", transparent=True, bbox_inches="tight")
    plt.show()

# %%
order = ["3ss", "6-7ss", "10ss", "12-13ss", "17-18ss", "21-22ss"]

with mplscience.style_context():
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 4, figsize=(20, 3), sharey=True)

    for axis, gene in zip(ax, DRIVERS):
        axis = sc.pl.violin(
            adata,
            [gene],
            groupby="stage",
            stripplot=False,  # remove the internal dots
            inner="box",  # adds a boxplot inside violins
            order=order,
            palette=["darkgrey"],
            show=False,
            ax=axis,
        )

    if SAVE_FIGURES:
        fig.savefig(FIG_DIR / DATASET / "gene_expression_time.svg", format="svg", transparent=True, bbox_inches="tight")
    plt.show()

# %%
coord = [[-0.3, 0.35, 0.25], [-3.5, 3.5, 2.5], [-3.5, 3.5, 2.5], [-0.9, 1, 0.7]]

# %%
for i, gene in enumerate(DRIVERS):
    Gep = adata[:, gene].X.A
    res = DEG(Gep, adata.obs["cell_type"].tolist())
    res.index = res.loc[:, "cell_type"]
    res.columns = ["gene", "cell_type", "coefficient", "pvalue"]
    res = res.loc[["mNC_head_mesenchymal", "mNC_arch2", "mNC_hox34", "Pigment", "NPB_nohox"], :].copy()
    bar_scores(
        res,
        adata,
        "cell_type",
        gene,
        figsize=(2, 2),
        title="DEG test",
        min=coord[i][0],
        max=coord[i][1],
        loc=coord[i][2],
    )

    with mplscience.style_context():
        sns.set(style="whitegrid")

        if SAVE_FIGURES:
            plt.savefig(FIG_DIR / DATASET / f"{gene}_DEG.svg", format="svg", transparent=True, bbox_inches="tight")
            plt.show()

# %%
coord = [[-0.25, 0.5, 0.4], [-0.25, 0.5, 0.4], [-0.4, 0.5, 0.4], [-0.25, 0.5, 0.4]]

# %%
for i, gene in enumerate(DRIVERS):
    Gep = adata[:, gene].X.A.reshape(-1)
    score = []
    pvalue = []
    for i in range(adata.obsm["lineages_fwd"].shape[1]):
        score.append(scipy.stats.pearsonr(pd.DataFrame(adata.obsm["lineages_fwd"]).iloc[:, i], Gep)[0])
        pvalue.append(scipy.stats.pearsonr(pd.DataFrame(adata.obsm["lineages_fwd"]).iloc[:, i], Gep)[1])

    test_result = pd.DataFrame({"coefficient": score, "pvalue": pvalue})
    test_result.index = adata.obsm["lineages_fwd"].names.tolist()
    test_result = test_result.loc[["mNC_head_mesenchymal", "mNC_arch2", "mNC_hox34", "Pigment"], :].copy()
    bar_scores(test_result, adata, "cell_type", gene, figsize=(2, 2), min=coord[i][0], max=coord[i][1], loc=coord[i][2])

    with mplscience.style_context():
        sns.set(style="whitegrid")
        if SAVE_FIGURES:
            plt.savefig(FIG_DIR / DATASET / f"{gene}_cor.svg", format="svg", transparent=True, bbox_inches="tight")
        plt.show()

# %% [markdown]
# ### Driver ranking

# %%
for ts in adata.obsm["lineages_fwd"].names.tolist():
    for gene in DRIVERS:
        sns.histplot(df.loc[:, f"{ts}_corr"], color="skyblue", binwidth=0.05)
        # Add a vertical line at x=0.5
        plt.axvline(x=df.loc[gene, f"{ts}_corr"], color="red", linestyle="--")
        # Add labels and title
        plt.xlabel("Correlation")
        plt.ylabel("Frequency")
        plt.title(ts)
        # Show plot

        if SAVE_FIGURES:
            plt.savefig(FIG_DIR / DATASET / f"{gene}_{ts}.svg", format="svg", transparent=True, bbox_inches="tight")
        plt.show()
        # Close the plot to free up memory
        plt.close()

# %% [markdown]
# ## Applying RegVelo for perturbation prediction

# %%
model = DATA_DIR / DATASET / "processed" / "rgv_model"
vae = REGVELOVI.load(model, adata)
set_output(adata, vae, n_samples=30, batch_size=adata.n_obs)

# %%
vk = cr.kernels.VelocityKernel(adata)
vk.compute_transition_matrix()
ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
kernel = 0.8 * vk + 0.2 * ck

estimator = cr.estimators.GPCCA(kernel)
## evaluate the fate prob on original space
estimator.compute_macrostates(n_states=8, cluster_key="cell_type")
estimator.set_terminal_states(TERMINAL_STATES)
estimator.compute_fate_probabilities()
estimator.plot_fate_probabilities(same_plot=False)

# %%
fate_prob_perturb = []

cand_list = ["ets1", "nr2f2", "nr2f5", "sox9b", "twist1a", "twist1b", "sox10", "mitfa", "tfec", "tfap2b"]

for TF in cand_list:
    adata_target_perturb, reg_vae_perturb = in_silico_block_simulation(model, adata, TF)

    n_states = 8
    vk = cr.kernels.VelocityKernel(adata_target_perturb)
    vk.compute_transition_matrix()
    ck = cr.kernels.ConnectivityKernel(adata_target_perturb).compute_transition_matrix()
    kernel = 0.8 * vk + 0.2 * ck

    estimator = cr.estimators.GPCCA(kernel)
    ## evaluate the fate prob on original space
    estimator.compute_macrostates(n_states=n_states, cluster_key="cell_type")
    estimator.set_terminal_states(TERMINAL_STATES)
    estimator.compute_fate_probabilities()
    ## visualize coefficient
    cond1_df = pd.DataFrame(
        adata_target_perturb.obsm["lineages_fwd"], columns=adata_target_perturb.obsm["lineages_fwd"].names.tolist()
    )

    fate_prob_perturb.append(cond1_df)

# %%
cond2_df = pd.DataFrame(adata.obsm["lineages_fwd"], columns=adata.obsm["lineages_fwd"].names.tolist())
df = []
for i in range(len(fate_prob_perturb)):
    data = abundance_test(cond2_df, fate_prob_perturb[i])
    data = pd.DataFrame(
        {
            "Score": data.iloc[:, 0].tolist(),
            "p-value": data.iloc[:, 1].tolist(),
            "Terminal state": data.index.tolist(),
            "TF": [cand_list[i]] * (data.shape[0]),
        }
    )
    df.append(data)

df = pd.concat(df)

df["Score"] = 0.5 - df["Score"]

# %%
# Create a DataFrame for easier plotting
with mplscience.style_context():
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 3))
    # sns.barplot(x='Terminal state', y='AUROC',data=data, hue = "Method",palette=pal,ax = ax)
    color_label = "cell_type"
    palette = dict(zip(adata.obs[color_label].cat.categories, adata.uns[f"{color_label}_colors"]))
    subset_palette = {name: color for name, color in palette.items() if name in cond2_df.columns.tolist()}

    sns.barplot(x="TF", y="Score", hue="Terminal state", data=df, ax=ax, palette=palette, dodge=True)

    # Add vertical lines to separate groups
    for i in range(len(df["TF"].unique()) - 1):
        plt.axvline(x=i + 0.5, color="gray", linestyle="--")

    # Label settings
    plt.ylabel("Perturbation coefficient", fontsize=14)
    plt.xlabel("TF", fontsize=14)
    plt.xticks(fontsize=14)  # Increase font size of x-axis tick labels
    plt.yticks(fontsize=14)  # Increase font size of y-axis tick labels

    # Customize the legend
    plt.legend(loc="lower center", fontsize=14, bbox_to_anchor=(0.5, -0.6), ncol=3)

    if SAVE_FIGURES:
        plt.savefig(
            FIG_DIR / DATASET / "driver_perturbation_simulation_all.svg",
            format="svg",
            transparent=True,
            bbox_inches="tight",
        )
    # Show the plot
    plt.show()

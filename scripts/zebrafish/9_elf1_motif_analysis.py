# %% [markdown]
# # elf1 perturbation simulation

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

import cellrank as cr
import scanpy as sc
import scvelo as scv
from regvelo import REGVELOVI

from rgv_tools import DATA_DIR, FIG_DIR
from rgv_tools.benchmarking import set_output
from rgv_tools.perturbation import (
    abundance_test,
    get_list_name,
    in_silico_block_simulation,
    inferred_GRN,
    RegulationScanning,
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

# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_run_regvelo.h5ad")

# %% [markdown]
# ## elf1 perturbation simulation

# %%
model = DATA_DIR / DATASET / "processed" / "rgv_hard_model_all"
vae = REGVELOVI.load(model, adata)
set_output(adata, vae)

# %%
terminal_states = [
    "mNC_head_mesenchymal",
    "mNC_arch2",
    "mNC_hox34",
    "Pigment",
]
vk = cr.kernels.VelocityKernel(adata)
vk.compute_transition_matrix()
ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
g_raw = cr.estimators.GPCCA(vk)
## evaluate the fate prob on original space
g_raw.compute_macrostates(n_states=7, cluster_key="cell_type")
g_raw.set_terminal_states(terminal_states)
g_raw.compute_fate_probabilities()
g_raw.plot_fate_probabilities(same_plot=False)

# %%
## Elf1
adata_target_perturb, reg_vae_perturb = in_silico_block_simulation(model, adata, "elf1")

n_states = 7
vk = cr.kernels.VelocityKernel(adata_target_perturb)
vk.compute_transition_matrix()
ck = cr.kernels.ConnectivityKernel(adata_target_perturb).compute_transition_matrix()
kernel = 0.8 * vk + 0.2 * ck

g = cr.estimators.GPCCA(kernel)
## evaluate the fate prob on original space
g.compute_macrostates(n_states=n_states, cluster_key="cell_type")
g.set_terminal_states(terminal_states)
g.compute_fate_probabilities()
## visualize coefficient
cond1_df = pd.DataFrame(
    adata_target_perturb.obsm["lineages_fwd"], columns=adata_target_perturb.obsm["lineages_fwd"].names.tolist()
)
cond2_df = pd.DataFrame(adata.obsm["lineages_fwd"], columns=adata.obsm["lineages_fwd"].names.tolist())

## abundance test
data = abundance_test(cond2_df, cond1_df)
data = pd.DataFrame(
    {
        "Score": data.iloc[:, 0].tolist(),
        "p-value": data.iloc[:, 1].tolist(),
        "Terminal state": data.index.tolist(),
        "TF": ["elf1"] * (data.shape[0]),
    }
)

# %%
final_df = data.copy()
final_df["Score"] = 0.5 - final_df["Score"]

# %%
color_label = "cell_type"
df = pd.DataFrame(final_df["Score"])
df.columns = ["coefficient"]
df["Cell type"] = final_df["Terminal state"]
order = df["Cell type"].tolist()

palette = dict(zip(adata.obs[color_label].cat.categories, adata.uns[f"{color_label}_colors"]))
subset_palette = {name: color for name, color in palette.items() if name in final_df.loc[:, "Terminal state"].tolist()}

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
    ax.set(ylim=(-0.05, 0.05))
    ax.tick_params(axis="x", rotation=90)
    plt.title("$\\mathit{" + "elf1" + "}$ regulon knock out simulation")

    if SAVE_FIGURES:
        plt.savefig(
            FIG_DIR / DATASET / "elf1_perturbation_simulation.svg", format="svg", transparent=True, bbox_inches="tight"
        )
    # Show the plot
    plt.show()

# %% [markdown]
# ## GRN computation

# %%
GRN = inferred_GRN(vae, adata, label="cell_type", group="all", data_frame=True)

# %% [markdown]
# ### elf1 target screening

# %%
targets = GRN.loc[:, "elf1"]
targets = np.array(targets.index.tolist())[np.array(targets) != 0]

# %%
terminal_states = [
    "mNC_head_mesenchymal",
    "mNC_arch2",
    "mNC_hox34",
    "Pigment",
]

print("inferring perturbation...")
perturb_screening = RegulationScanning(model, adata, 7, "cell_type", terminal_states, "elf1", targets, 0)
coef = pd.DataFrame(np.array(perturb_screening["coefficient"]))
coef.index = perturb_screening["target"]
coef.columns = get_list_name(perturb_screening["coefficient"][0])

# %%
coef.sort_values("Pigment", ascending=False).iloc[:15,]

# %%
Pigment = coef.sort_values(by="Pigment", ascending=False)[:15]["Pigment"]

# %%
df = pd.DataFrame({"Gene": Pigment.index.tolist(), "Score": np.array(Pigment)})

# Sort DataFrame by -log10(p-value) for ordered plotting
df = df.sort_values(by="Score", ascending=False)

# Highlight specific genes
# Set up the plot
with mplscience.style_context():
    sns.set_style(style="white")
    fig, ax = plt.subplots(figsize=(3, 6))
    sns.scatterplot(data=df, x="Score", y="Gene", palette="purple", s=200, legend=False)

    for _, row in df.iterrows():
        plt.hlines(row["Gene"], xmin=0.5, xmax=row["Score"], colors="grey", linestyles="-", alpha=0.5)

    # Customize plot appearance
    plt.xlabel("Score")
    plt.ylabel("")
    plt.title("Pigment")

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_color("black")
    plt.gca().spines["bottom"].set_color("black")
    # Show plot

    if SAVE_FIGURES:
        fig.savefig(FIG_DIR / DATASET / "elf1_driver.svg", format="svg", transparent=True, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## Activity of positive regulated genes

# %%
elf1_g = np.array(GRN.index.tolist())[np.array(GRN.loc[:, "elf1"]) != 0]
fli1a_g = np.array(GRN.index.tolist())[np.array(GRN.loc[:, "fli1a"]) != 0]

# %%
score = adata[:, elf1_g].layers["Ms"].mean(1) - adata[:, fli1a_g].layers["Ms"].mean(1)
score = scipy.stats.zscore(np.array(score))

# %%
sns.scatterplot(x=score, y=-adata.obs["latent_time"])
max_abs_x = max(abs(np.min(score)), abs(np.max(score)))
plt.xlim(-max_abs_x, max_abs_x)

# Display the plot
plt.axvline(0, color="grey", linestyle="--")  # Optional: add a vertical line at x=0 for clarity

# %%
adata.obsm["X_togglesiwtch"] = np.column_stack((score, -adata.obs["latent_time"]))

# %%
sc.pl.embedding(adata, basis="togglesiwtch", color="cell_type", palette=sc.pl.palettes.vega_20, legend_loc="on data")

# %%
adata.obs["ToggleState"] = [i if i in ["mNC_head_mesenchymal", "Pigment"] else np.nan for i in adata.obs["macrostates"]]
adata.obs["ToggleState"] = adata.obs["ToggleState"].astype("category")

# %%
which = "ToggleState"
# adata.obs[which] = adata.obs["cell_type2"].copy()

state_names = adata.obs[which].cat.categories.tolist()
adata.obs[which] = adata.obs[which].astype(str).astype("category").cat.reorder_categories(["nan"] + state_names)

if which == "ToggleState":
    adata.uns[f"{which}_colors"] = ["#dedede"] + list(subset_palette.values())
else:
    adata.uns[f"{which}_colors"] = ["#dedede"] + list(subset_palette.values())
state_names = adata.obs[which].cat.categories.tolist()[1:]


with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(4, 3))
    scv.pl.scatter(adata, basis="togglesiwtch", c=which, add_outline=state_names, ax=ax, size=60)

    if SAVE_FIGURES:
        fig.savefig(FIG_DIR / DATASET / "fli1a_elf1.svg", format="svg", transparent=True, bbox_inches="tight")

# %%
if SAVE_DATA:
    adata.write_h5ad(DATA_DIR / DATASET / "results" / "elf1_screening.csv")

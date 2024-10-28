# %% [markdown]
# # E2f1 perturbation on pancreatic endocrine dataset

# %% [markdown]
# ## Library imports

import os

# %%
from paths import DATA_DIR, FIG_DIR
from regvelovi import REGVELOVI

import networkx as nx
import numpy as np
import pandas as pd

# %%
# %%
import scipy

# %%
## predicting using AUROC
from sklearn import metrics

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

import cellrank as cr
import scanpy as sc
import scvelo as scv
import torch
from scvelo.preprocessing.moments import get_moments

# %% [markdown]
# ## General settings

# %%
plt.rcParams["svg.fonttype"] = "none"

sns.reset_defaults()
sns.reset_orig()
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=14, color_map="viridis")

# %%
SAVE_FIGURES = True
if SAVE_FIGURES:
    os.makedirs(FIG_DIR / "pancreatic_endocrine", exist_ok=True)

SAVE_DATASETS = True
if SAVE_DATASETS:
    os.makedirs(DATA_DIR / "pancreatic_endocrine", exist_ok=True)


# %% [markdown]
# ## Function defination


# %%
def min_max_scaling(data):
    """Apply min-max scaling to a numpy array or pandas Series.

    Parameters
    ----------
    data (np.ndarray or pd.Series): The input data to be scaled.

    Returns
    -------
    np.ndarray or pd.Series: Scaled data with values between 0 and 1.
    """
    min_val = np.min(data)
    max_val = np.max(data)

    scaled_data = (data - min_val) / (max_val - min_val)

    return scaled_data


def _in_silico_block_simulation(model, adata, gene, regulation_block=True, target_block=True, effects=0, cutoff=1e-3):
    """TODO."""
    reg_vae_perturb = REGVELOVI.load(model, adata)
    perturb_GRN = reg_vae_perturb.module.v_encoder.fc1.weight.detach().clone()

    if regulation_block:
        perturb_GRN[
            (perturb_GRN[:, [i == gene for i in adata.var.index]].abs() > cutoff).cpu().numpy().reshape(-1),
            [i == gene for i in adata.var.index],
        ] = effects
    if target_block:
        perturb_GRN[
            [i == gene for i in adata_target.var.index],
            (perturb_GRN[[i == gene for i in adata_target.var.index], :].abs() > 1e-3).cpu().numpy().reshape(-1),
        ] = effects

    reg_vae_perturb.module.v_encoder.fc1.weight.data = perturb_GRN
    adata_target_perturb = add_regvelo_outputs_to_adata(adata, reg_vae_perturb)

    return adata_target_perturb, reg_vae_perturb


def add_regvelo_outputs_to_adata(adata_raw, vae, filter=False):
    """TODO."""
    latent_time = vae.get_latent_time(n_samples=30, batch_size=adata_raw.shape[0])
    velocities = vae.get_velocity(n_samples=30, batch_size=adata_raw.shape[0])

    t = latent_time
    scaling = 20 / t.max(0)
    adata = adata_raw[:, vae.module.target_index].copy()

    adata.layers["velocity"] = velocities / scaling
    adata.layers["latent_time_regvelo"] = latent_time

    adata.layers["fit_t"] = latent_time.values * scaling[np.newaxis, :]
    adata.var["fit_scaling"] = 1.0

    return adata


# %%
## calculate cosine similarity
def normalize(vector):
    """TODO."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def cosine_similarity(vec1, vec2):
    """TODO."""
    vec1_normalized = normalize(vec1)
    vec2_normalized = normalize(vec2)
    return np.dot(vec1_normalized, vec2_normalized)


def cosine_dist(X_mean, Y_mean):
    """TODO."""
    kl_div = []
    for i in range(X_mean.shape[1]):
        mu_x = X_mean[:, i]
        mu_y = Y_mean[:, i]

        kl = 1 - cosine_similarity(mu_x, mu_y)
        kl_div.append(kl)

    return np.array(kl_div)


def normalize_rows(array):
    """TODO."""
    # Calculate the L2 norm for each row
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1
    # Normalize each row by its L2 norm
    normalized_array = array / norms
    return normalized_array


def kl_divergence(X_mean, X_var, Y_mean, Y_var):
    """TODO."""
    kl_div = []
    for i in range(X_mean.shape[1]):
        mu_x = X_mean[:, i]
        sigma_x = X_var[:, i]
        mu_y = Y_mean[:, i]
        sigma_y = Y_var[:, i]

        kl = np.mean(
            np.log(sigma_y + 1e-6)
            - np.log(sigma_x + 1e-6)
            + (1e-6 + sigma_x**2 + (mu_x - mu_y) ** 2) / (1e-6 + 2 * sigma_y**2)
            - 0.5
        )
        kl_div.append(kl)

    return np.array(kl_div)


# %%
def GRN_Jacobian(reg_vae, Ms):
    """TODO."""
    reg_vae.module.v_encoder.fc1.weight.detach()
    reg_vae.module.v_encoder.fc1.bias.detach()
    reg_vae.module.v_encoder.alpha_unconstr_max.detach()
    ## calculate the jacobian matrix respect to each cell
    Jaco_m = []
    for i in range(Ms.shape[0]):
        s = Ms[i, :]
        ## calculate sigmoid probability
        # alpha_unconstr = torch.matmul(net,torch.tensor(s[reg_vae.module.v_encoder.regulator_index]))
        # alpha_unconstr = alpha_unconstr + bias
        # alpha_unconstr = reg_vae.module.v_encoder.fc1(torch.tensor(s[reg_vae.module.v_encoder.regulator_index]).to("cuda:0")).detach()
        # coef = (F.sigmoid(alpha_unconstr))
        # alpha_max = torch.clamp(F.softplus(max_rate),0,50)
        # Jaco_m.append(torch.matmul(torch.diag(coef), net))
        Jaco_m.append(
            reg_vae.module.v_encoder.GRN_Jacobian(
                torch.tensor(s[reg_vae.module.v_encoder.regulator_index]).to("cuda:0")
            ).detach()
        )
    Jaco_m = torch.stack(Jaco_m, 2)
    return Jaco_m


# %%
def calculate_aurocs(targets, thresholds):
    """TODO."""
    auroc_scores_1 = []
    auroc_scores_2 = []

    for threshold in thresholds:
        # Convert continuous ground truth to binary based on threshold
        chip_targets = np.array(chip_target.index.tolist())[chip_target.iloc[:, 0] > threshold].tolist()
        targets.loc[:, "gt"] = 0
        targets.loc[:, "gt"][targets.index.isin(chip_targets)] = 1

        # Calculate AUROC scores
        # auroc_1 = roc_auc_score(targets.loc[:,"gt"], targets.loc[:,"prior"])
        # auroc_2 = roc_auc_score(targets.loc[:,"gt"], targets.loc[:,"weight"])  # Example of second AUROC (could be another model)
        fpr, tpr, thresholds = metrics.roc_curve(targets.loc[:, "gt"], targets.loc[:, "prior"])
        auroc_1 = metrics.auc(fpr, tpr)
        fpr, tpr, thresholds = metrics.roc_curve(targets.loc[:, "gt"], targets.loc[:, "weight"])
        auroc_2 = metrics.auc(fpr, tpr)

        auroc_scores_1.append(auroc_1)
        auroc_scores_2.append(auroc_2)

    return auroc_scores_1, auroc_scores_2


# %% [markdown]
# ## Data loading

# %%
model_path = DATA_DIR / "pancreatic_endocrine" / "rgv_model9"
data_path = DATA_DIR / "pancreatic_endocrine" / "reg_bdata.h5ad"

# %%
reg_bdata = sc.read_h5ad(data_path)
TF = pd.read_csv("RegVelo_datasets/pancreatic endocrine/allTFs_mm.txt", header=None)

# %%
W = reg_bdata.uns["skeleton"].copy()
W = torch.tensor(np.array(W)).int()

# %%
reg_vae = REGVELOVI.load(model_path, reg_bdata)
adata_target = add_regvelo_outputs_to_adata(reg_bdata, reg_vae)
scv.tl.velocity_graph(adata_target)

# %%
scv.tl.score_genes_cell_cycle(adata_target)

# %%
adata_target_raw = adata_target.copy()
adata_target = adata_target[adata_target.obs["phase"] != "G1"].copy()

# %% [markdown]
# ## correlate pearson correlation between gene expression and s score and g2m score

# %%
score = min_max_scaling(adata_target.obs["S_score"]) - min_max_scaling(adata_target.obs["G2M_score"])
gene_ranking = []
for i in range(adata_target.shape[1]):
    gene_ranking.append(scipy.stats.pearsonr(adata_target.X.A[:, i], score)[0])
rank_df = pd.DataFrame({"Ranking": gene_ranking})
rank_df.index = adata_target.var_names.tolist()

# %%
rank_df.loc[list(set(TF.iloc[:, 0]).intersection(adata_target.var_names.to_list())), :].sort_values(
    by="Ranking", ascending=False
)

# %%
genes_to_plot = rank_df.loc[list(set(TF.iloc[:, 0]).intersection(adata_target.var_names.to_list())), :].sort_values(
    by="Ranking", ascending=False
)

# %%
## plot the heatmap and show the phase pivot genes
adata_plot = adata_target.copy()
adata_plot.obs["S_score_vs_G2M_score"] = 1 - pd.to_numeric(score)  # Ensure pseudotime is numeric
adata_plot = adata_plot[adata_plot.obs.sort_values("S_score_vs_G2M_score").index]

# %%
sc.pp.neighbors(adata_plot)
ck = cr.kernels.ConnectivityKernel(adata_plot).compute_transition_matrix()
g = cr.estimators.GPCCA(ck)
## evaluate the fate prob on original space
g.compute_macrostates(n_states=7, cluster_key="clusters")
g.set_terminal_states(["Beta"])
g.compute_fate_probabilities()

# %%
## using cellrank to plot figure
model = cr.models.GAM(adata_plot)

# %%
if SAVE_FIGURES:
    with mplscience.style_context():
        cr.pl.heatmap(
            adata_plot,
            model,
            genes=genes_to_plot[:50].index.tolist(),
            show_fate_probabilities=False,
            show_all_genes=True,
            time_key="S_score_vs_G2M_score",
            keep_gene_order=True,
            figsize=(8, 15),
            save=FIG_DIR / "pancreatic_endocrine" / "E2f1_perturbation" / "heatmap_plot.svg",
        )

# %%
adata_target = adata_target_raw.copy()

# %% [markdown]
# ## perform E2f1 perturbation simulation

# %%
adata_target_perturb, reg_vae_perturb = _in_silico_block_simulation(
    model_path, reg_bdata, "E2f1", target_block=False, cutoff=0
)

# %%
vec = cosine_dist(adata_target.layers["velocity"].T, adata_target_perturb.layers["velocity"].T)
adata_target_perturb.obs["perturbation_effect"] = vec

# %%
exp = get_moments(adata_target, normalize_rows(adata_target.layers["velocity"]), second_order=False).astype(
    np.float64, copy=False
)
var = get_moments(adata_target, normalize_rows(adata_target.layers["velocity"]), second_order=True).astype(
    np.float64, copy=False
)
exp_perturb = get_moments(
    adata_target_perturb, normalize_rows(adata_target_perturb.layers["velocity"]), second_order=False
).astype(np.float64, copy=False)
var_perturb = get_moments(
    adata_target_perturb, normalize_rows(adata_target_perturb.layers["velocity"]), second_order=True
).astype(np.float64, copy=False)
vec = kl_divergence(exp.T, var.T, exp_perturb.T, var_perturb.T)
adata_target_perturb.obs["KL_divergence"] = np.log2(vec + 1)

vec = ((exp - exp_perturb) ** 2).sum(1)
adata_target_perturb.obs["first moment"] = np.log2(vec + 1)

vec = np.abs(np.log2(var_perturb.sum(1) + 1) - np.log2(var.sum(1) + 1))
# vec[~np.isnan(vec)] = min_max_scaling(vec[~np.isnan(vec)])
adata_target_perturb.obs["second moment"] = vec

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    sc.pl.umap(
        adata_target_perturb,
        color="KL_divergence",
        cmap="viridis",
        title="KL divergence",
        ax=ax,
        frameon=False,
    )

if SAVE_FIGURES:
    fig.savefig(
        FIG_DIR / "pancreatic_endocrine" / "E2f1_perturbation" / "KL_divergence.svg",
        format="svg",
        transparent=True,
        bbox_inches="tight",
    )

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    sc.pl.umap(
        adata_target_perturb,
        color="first moment",
        cmap="viridis",
        title="First moment difference",
        ax=ax,
        frameon=False,
    )

if SAVE_FIGURES:
    fig.savefig(
        FIG_DIR / "pancreatic_endocrine" / "E2f1_perturbation" / "first_moment.svg",
        format="svg",
        transparent=True,
        bbox_inches="tight",
    )

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    sc.pl.umap(
        adata_target_perturb,
        color="second moment",
        cmap="viridis",
        title="Second moment difference",
        ax=ax,
        frameon=False,
    )

if SAVE_FIGURES:
    fig.savefig(
        FIG_DIR / "pancreatic_endocrine" / "E2f1_perturbation" / "second_moment.svg",
        format="svg",
        transparent=True,
        bbox_inches="tight",
    )

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    sc.pl.umap(
        adata_target_perturb,
        color="perturbation_effect",
        cmap="viridis",
        title="Perturbation effect",
        ax=ax,
        frameon=False,
    )

if SAVE_FIGURES:
    fig.savefig(
        FIG_DIR / "pancreatic_endocrine" / "E2f1_perturbation" / "perturbation_effect.svg",
        format="svg",
        transparent=True,
        bbox_inches="tight",
    )

# %%
adata_target_perturb.obs["phase"] = adata_target.obs["phase"].copy()
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    sc.pl.umap(
        adata_target,
        color="phase",
        ax=ax,
        frameon=False,
    )

if SAVE_FIGURES:
    fig.savefig(
        FIG_DIR / "pancreatic_endocrine" / "E2f1_perturbation" / "phase_label.svg",
        format="svg",
        transparent=True,
        bbox_inches="tight",
    )

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 4))
    sc.pl.umap(adata_target, color="E2f1", vmin="p1", vmax="p99", frameon=False, ax=ax)

if SAVE_FIGURES:
    fig.savefig(
        FIG_DIR / "pancreatic_endocrine" / "E2f1_perturbation" / "E2f1_express.svg",
        format="svg",
        transparent=True,
        bbox_inches="tight",
    )

# %%
PS = (
    adata_target_perturb.obs["perturbation_effect"][
        (adata_target_perturb.obs["clusters"] == "Ductal") & (adata_target.obs["phase"] == "S")
    ].tolist()
    + adata_target_perturb.obs["perturbation_effect"][adata_target_perturb.obs["clusters"] == "Ngn3 high EP"].tolist()
    + adata_target_perturb.obs["perturbation_effect"][adata_target_perturb.obs["clusters"] == "Ngn3 low EP"].tolist()
)


x = (
    [2] * np.sum((adata_target_perturb.obs["clusters"] == "Ductal") & (adata_target.obs["phase"] == "S"))
    + [1] * np.sum(adata_target.obs["clusters"] == "Ngn3 high EP")
    + [0] * np.sum(adata_target_perturb.obs["clusters"] == "Ngn3 low EP")
)
# KL = scipy.stats.zscore(KL)
PS = np.array(PS)
difference = [np.mean(PS[np.array(x) == 2]), np.mean(PS[np.array(x) == 0]), np.mean(PS[np.array(x) == 1])]
difference1 = difference / np.sqrt(np.var(difference))
# scipy.stats.pearsonr(np.array(x),np.array(KL))
difference1

# %%
GEP = (
    adata_target_perturb[
        (adata_target_perturb.obs["clusters"] == "Ductal") & (adata_target.obs["phase"] == "S"), "E2f1"
    ]
    .X.A.reshape(-1)
    .tolist()
    + adata_target_perturb[adata_target_perturb.obs["clusters"] == "Ngn3 high EP", "E2f1"].X.A.reshape(-1).tolist()
    + adata_target_perturb[adata_target_perturb.obs["clusters"] == "Ngn3 low EP", "E2f1"].X.A.reshape(-1).tolist()
)


x = (
    [2] * np.sum((adata_target_perturb.obs["clusters"] == "Ductal") & (adata_target.obs["phase"] == "S"))
    + [1] * np.sum(adata_target_perturb.obs["clusters"] == "Ngn3 high EP")
    + [0] * np.sum(adata_target_perturb.obs["clusters"] == "Ngn3 low EP")
)
# KL = scipy.stats.zscore(KL)
GEP = np.array(GEP)
difference = [np.mean(GEP[np.array(x) == 2]), np.mean(GEP[np.array(x) == 0]), np.mean(GEP[np.array(x) == 1])]
difference2 = difference / np.sqrt(np.var(difference))
# scipy.stats.pearsonr(np.array(x),np.array(KL))
difference2

# %%
## Visualize the effects through the barplot
sns.set_style("ticks")
figsize = (2, 2)
df = pd.DataFrame(difference2.tolist() + difference1.tolist())
df.columns = ["Scaled value"]
df["Cell type"] = ["Ductal+S phase", "Ngn3 low EP", "Ngn3 high EP"] * 2
df["Group"] = ["Gene expression"] * 3 + ["KL divergence"] * 3
palette = {"Gene expression": "#555555", "KL divergence": "#ffc0cb"}
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=df,
        x="Scaled value",
        y="Cell type",
        hue="Group",
        ax=ax,
        palette=palette,
    )
    # ax.set(ylim=(-2.1,2.1))
    plt.legend(loc="upper center", bbox_to_anchor=(0.1, -0.3), ncol=2)
    plt.show()

    if SAVE_FIGURES:
        fig.savefig(
            FIG_DIR / "pancreatic_endocrine" / "E2f1_perturbation" / "perturb_express_compare_e2f1.svg",
            format="svg",
            transparent=True,
            bbox_inches="tight",
        )

# %%
## repeat models
for nrun in range(0, 5):
    print("training model...")
    REGVELOVI.setup_anndata(reg_bdata, spliced_layer="Ms", unspliced_layer="Mu")
    reg_vae = REGVELOVI(reg_bdata, W=W.T, regulators=TF.iloc[:, 0].tolist())
    # reg_vae.module.v_encoder.grn.data = reg_vae.module.v_encoder.grn.data * W.T
    reg_vae.train()

    print("save model...")
    model_name = "rgv_model_" + str(nrun)
    model = DATA_DIR / "pancreatic_endocrine" / "cell_cycle_repeat_runs" / model_name
    reg_vae.save(model)

# %%
reg_bdata.obs["phase"] = adata_target.obs["phase"].copy()

# %%
## aggregate GRN
model = DATA_DIR / "pancreatic_endocrine" / "cell_cycle_repeat_runs" / "rgv_model_0"
### load model
reg_vae = REGVELOVI.load(model, reg_bdata)
# grn1 = reg_vae.module.v_encoder.fc1.weight.data.cpu().detach().clone()
grn1 = (
    reg_vae.module.v_encoder.GRN_Jacobian(
        torch.tensor(
            reg_bdata.layers["Ms"][(reg_bdata.obs["clusters"] == "Ductal") & (reg_bdata.obs["phase"] == "S"), :]
        ).to("cuda:0")
    )
    .cpu()
    .detach()
    .clone()
)


model = DATA_DIR / "pancreatic_endocrine" / "cell_cycle_repeat_runs" / "rgv_model_1"
### load model
reg_vae = REGVELOVI.load(model, reg_bdata)
# grn2 = reg_vae.module.v_encoder.fc1.weight.data.cpu().detach().clone()
grn2 = (
    reg_vae.module.v_encoder.GRN_Jacobian(
        torch.tensor(
            reg_bdata.layers["Ms"][(reg_bdata.obs["clusters"] == "Ductal") & (reg_bdata.obs["phase"] == "S"), :]
        ).to("cuda:0")
    )
    .cpu()
    .detach()
    .clone()
)

model = DATA_DIR / "pancreatic_endocrine" / "cell_cycle_repeat_runs" / "rgv_model_2"
### load model
reg_vae = REGVELOVI.load(model, reg_bdata)
# grn3 = reg_vae.module.v_encoder.fc1.weight.data.cpu().detach().clone()
grn3 = (
    reg_vae.module.v_encoder.GRN_Jacobian(
        torch.tensor(
            reg_bdata.layers["Ms"][(reg_bdata.obs["clusters"] == "Ductal") & (reg_bdata.obs["phase"] == "S"), :]
        ).to("cuda:0")
    )
    .cpu()
    .detach()
    .clone()
)

model = DATA_DIR / "pancreatic_endocrine" / "cell_cycle_repeat_runs" / "rgv_model_3"
### load model
reg_vae = REGVELOVI.load(model, reg_bdata)
# grn4 = reg_vae.module.v_encoder.fc1.weight.data.cpu().detach().clone()
grn4 = (
    reg_vae.module.v_encoder.GRN_Jacobian(
        torch.tensor(
            reg_bdata.layers["Ms"][(reg_bdata.obs["clusters"] == "Ductal") & (reg_bdata.obs["phase"] == "S"), :]
        ).to("cuda:0")
    )
    .cpu()
    .detach()
    .clone()
)

model = DATA_DIR / "pancreatic_endocrine" / "cell_cycle_repeat_runs" / "rgv_model_4"
### load model
reg_vae = REGVELOVI.load(model, reg_bdata)
# grn5 = reg_vae.module.v_encoder.fc1.weight.data.cpu().detach().clone()
grn5 = (
    reg_vae.module.v_encoder.GRN_Jacobian(
        torch.tensor(
            reg_bdata.layers["Ms"][(reg_bdata.obs["clusters"] == "Ductal") & (reg_bdata.obs["phase"] == "S"), :]
        ).to("cuda:0")
    )
    .cpu()
    .detach()
    .clone()
)

# %%
stacked_tensors = torch.stack((grn1, grn2, grn3, grn4, grn5))

# Compute the median across corresponding entries
grn_median = torch.mean(stacked_tensors, dim=0)

# %%
GRN = grn_median.clone()

# %%
targets = GRN[:, [i == "E2f1" for i in adata_target.var.index]].detach().cpu().numpy()
prior = reg_vae.module.v_encoder.mask_m_raw[:, [i == "E2f1" for i in adata_target.var.index]].detach().cpu().numpy()

# %%
targets = pd.DataFrame(targets, index=adata_target.var.index)

# %%
targets.loc[:, "weight"] = targets.iloc[:, 0].abs()
targets.loc[:, "prior"] = prior

# %%
### predicting E2f1 ChIP-seq dataset
chip_target = pd.read_csv(
    "/home/icb/weixu.wang/regulatory_velo/pancreas_dataset/figures/E2f1_function/E2f1.1.tsv", index_col=0, sep="\t"
)

# %%
chip_targets = np.array(chip_target.index.tolist())[chip_target.iloc[:, 0] > 200].tolist()

# %%
targets.loc[:, "gt"] = 0
targets.loc[:, "gt"][targets.index.isin(chip_targets)] = 1


fpr, tpr, thresholds = metrics.roc_curve(targets.loc[:, "gt"], targets.loc[:, "prior"])
metrics.auc(fpr, tpr)

# %%
fpr, tpr, thresholds = metrics.roc_curve(targets.loc[:, "gt"], targets.loc[:, "weight"])
metrics.auc(fpr, tpr)

# %%
## select different quantile value, and look at the change of AUROC
# Function to calculate AUROC scores given a threshold
# Sample continuous ground truth and predicted scores
np.random.seed(42)  # For reproducibility

# Define thresholds using quantiles
num_thresholds = 100
quantile_99 = np.quantile(chip_target.iloc[:, 0], 0.99)
thresholds = np.linspace(0, quantile_99, num_thresholds)

# Calculate AUROC scores for different thresholds
auroc_scores_1, auroc_scores_2 = calculate_aurocs(targets, thresholds)


plt.rcParams["svg.fonttype"] = "none"
with mplscience.style_context():
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(4, 2))
    data = pd.DataFrame(
        {
            "AUROC": auroc_scores_1 + auroc_scores_2,
            "GRN": ["prior network"] * len(auroc_scores_1) + ["RegVelo inferred network"] * len(auroc_scores_2),
        }
    )

    # Plot the boxplot
    sns.boxplot(y="GRN", x="AUROC", data=data)

    plt.xlabel("GRN")
    plt.ylabel("AUROC Score")
    plt.title("E2f1 targets prediction")

    if SAVE_FIGURES:
        fig.savefig(
            FIG_DIR / "pancreatic_endocrine" / "E2f1_perturbation" / "GRN_boxplot.svg",
            format="svg",
            transparent=True,
            bbox_inches="tight",
        )

# %%
scipy.stats.wilcoxon(auroc_scores_1, auroc_scores_2)

# %%
GRN_visualize = targets.sort_values("weight", ascending=False).iloc[:20, :]

# %%
df = pd.DataFrame({"from": ["E2f1"] * 20, "to": GRN_visualize.index.tolist(), "status": GRN_visualize.loc[:, "prior"]})

# %%
# df.loc[df['to'] == node, 'status'].values[0]

# %%
color_map = ["skyblue"]
for node in GRN_visualize.index.tolist():
    if df.loc[df["to"] == node, "status"].values[0] == 1:
        color_map.append("darkgrey")
    else:
        color_map.append("lightgrey")

# %%
G = nx.from_pandas_edgelist(df, source="from", target="to", create_using=nx.DiGraph())

# %%
with mplscience.style_context():
    fig, ax = plt.subplots(figsize=(6, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, arrowsize=10, node_color=color_map)

if SAVE_FIGURES:
    fig.savefig(
        FIG_DIR / "pancreatic_endocrine" / "E2f1_perturbation" / "E2f1_GRN.svg",
        format="svg",
        transparent=True,
        bbox_inches="tight",
    )

# %%

# %% [markdown]
# # Simulate Toy GRN to benchmark velocity, latent time and GRN inference performance

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import torchsde
from torch.distributions.multivariate_normal import MultivariateNormal

import seaborn as sns

import celloracle as co
import scanpy as sc
import scvelo as scv
from anndata import AnnData
from arboreto.algo import grnboost2
from regvelo import REGVELOVI
from velovi import preprocess_data, VELOVI

from rgv_tools import DATA_DIR, FIG_DIR
from rgv_tools.datasets import VelocityEncoder
from rgv_tools.datasets._simulate import draw_poisson

# %% [markdown]
# ## General settings

# %%
sns.reset_defaults()
sns.reset_orig()

# %%
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=20, color_map="viridis")

# %%
SAVE_FIGURES = True
if SAVE_FIGURES:
    (FIG_DIR / "simulation" / "toy_GRN").mkdir(parents=True, exist_ok=True)

SAVE_DATASETS = True
if SAVE_DATASETS:
    (DATA_DIR / "simulation" / "toy_GRN").mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Function defination


# %%
def sign_concordance(GRN, ref_GRN):
    """TODO."""
    sign_GRN = np.sign(GRN)[GRN != 0]
    sign_ref_GRN = np.sign(ref_GRN)[GRN != 0]
    score = sum(sign_GRN == sign_ref_GRN)
    return score


# %%
def calculate_power_matrix(A, B):
    """TODO."""
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Both matrices must have the same dimensions")

    rows = len(A)
    cols = len(A[0])

    C = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            C[i][j] = A[i][j] ** B[i][j]

    return np.array(C)


# %%
def add_regvelo_outputs_to_adata(adata_raw, vae, filter=False):
    """TODO."""
    latent_time = vae.get_latent_time(n_samples=30, batch_size=adata_raw.shape[0])
    velocities = vae.get_velocity(n_samples=30, batch_size=adata_raw.shape[0])

    t = latent_time
    scaling = 20 / t.max(0)
    adata = adata_raw[:, vae.module.target_index].copy()

    adata.layers["velocity"] = velocities
    adata.layers["latent_time_regvelo"] = latent_time

    adata.layers["fit_t"] = latent_time.values * np.array(scaling)[np.newaxis, :]
    adata.var["fit_scaling"] = 1.0
    adata.var["fit_beta_regvelo"] = (
        torch.clip(torch.nn.functional.softplus(vae.module.v_encoder.beta_mean_unconstr), 0, 50).cpu().detach().numpy()
        / scaling
    )
    adata.var["fit_gamma_regvelo"] = (
        torch.clip(torch.nn.functional.softplus(vae.module.v_encoder.gamma_mean_unconstr), 0, 50).cpu().detach().numpy()
        / scaling
    )

    return adata


# %%
def add_velovi_outputs_to_adata(adata, vae):
    """TODO."""
    latent_time = vae.get_latent_time(n_samples=30)
    velocities = vae.get_velocity(n_samples=30)

    t = latent_time
    scaling = 20 / t.max(0)

    adata.layers["velocity"] = velocities / scaling
    adata.layers["latent_time_velovi"] = latent_time

    adata.var["fit_alpha_velovi"] = vae.get_rates()["alpha"] / scaling
    adata.var["fit_beta_velovi"] = vae.get_rates()["beta"] / scaling
    adata.var["fit_gamma_velovi"] = vae.get_rates()["gamma"] / scaling
    adata.var["fit_t_"] = (
        torch.nn.functional.softplus(vae.module.switch_time_unconstr).detach().cpu().numpy()
    ) * scaling
    adata.layers["fit_t"] = latent_time.values * np.array(scaling)[np.newaxis, :]
    adata.var["fit_scaling"] = 1.0


# %% [markdown]
# ## Initialize all values

# %%
velo_scv = []
velo_velovi = []
velo_rgv = []
corr_t_rgv = []
corr_t_scv = []
corr_t_velovi = []
corr_t_emperical = []
corr_t_dpt = []
beta_list = []
gamma_list = []
rgv_GRN = []
cor_GRN = []
gb_GRN = []
co_GRN = []

# %% [markdown]
# ## Run simulation for 100 times

# %%
for sim_idx in range(100):
    print(sim_idx)
    torch.cuda.empty_cache()
    torch.manual_seed(sim_idx)

    ## simulate alpha beta and gamma
    n_vars = 6
    mu = torch.tensor([5, 0.5, 0.125], dtype=torch.float32).log()
    R = torch.tensor([[1.0, 0.2, 0.2], [0.2, 1.0, 0.8], [0.2, 0.8, 1.0]], dtype=torch.float32)
    C = torch.tensor([0.4, 0.4, 0.4], dtype=torch.float32)[:, None]
    cov = C * C.T * R
    distribution = MultivariateNormal(loc=mu, covariance_matrix=cov)
    alpha, beta, gamma = distribution.sample(sample_shape=torch.Size([n_vars])).exp().T

    mean_alpha = alpha.mean()
    coef_m = torch.tensor(
        [
            [0, 1, -mean_alpha, 2, 2],
            [1, 0, -mean_alpha, 2, 2],
            [0, 2, mean_alpha, 2, 4],
            [0, 3, mean_alpha, 2, 4],
            [2, 3, -mean_alpha, 2, 2],
            [3, 2, -mean_alpha, 2, 2],
            [1, 4, mean_alpha, 2, 4],
            [1, 5, mean_alpha, 2, 4],
            [4, 5, -mean_alpha, 2, 2],
            [5, 4, -mean_alpha, 2, 2],
        ]
    )

    n_regulators = 6
    n_targets = 6
    K = torch.zeros([n_targets, n_regulators], dtype=torch.float32)
    n = torch.zeros([n_targets, n_regulators], dtype=torch.float32)
    h = torch.zeros([n_targets, n_regulators], dtype=torch.float32)

    K[coef_m[:, 1].int(), coef_m[:, 0].int()] = coef_m[:, 2]
    n[coef_m[:, 1].int(), coef_m[:, 0].int()] = coef_m[:, 3]
    h[coef_m[:, 1].int(), coef_m[:, 0].int()] = coef_m[:, 4]

    t = draw_poisson(1500, seed=sim_idx)

    alpha_b = torch.zeros((6,), dtype=torch.float32)
    sde = VelocityEncoder(K=K, n=n, h=h, alpha_b=alpha_b, beta=beta, gamma=gamma)

    ## set up G batches, Each G represent a module (a target gene centerred regulon)
    ## infer the observe gene expression through ODE solver based on x0, t, and velocity_encoder
    y0 = torch.tensor([1.0, 0, 1.0, 0, 1.0, 0] + torch.zeros(6).abs().tolist()).reshape(1, -1)
    ys = torchsde.sdeint(sde, y0, t, method="euler")

    pre_u = ys[:, 0, :6]
    pre_s = ys[:, 0, 6:]
    pre_u = torch.clip(pre_u, 0)
    pre_s = torch.clip(pre_s, 0)

    pre_s = pd.DataFrame(pre_s.numpy())
    pre_u = pd.DataFrame(pre_u.numpy())
    gt_velo = np.array(pre_u) * beta.numpy() - np.array(pre_s) * gamma.numpy()
    adata = AnnData(np.array(pre_s))

    ## Preprocessing
    adata.layers["Ms"] = np.array(pre_s)
    adata.layers["Mu"] = np.array(pre_u)
    adata.layers["spliced"] = np.array(pre_s)
    adata.layers["unspliced"] = np.array(pre_u)
    adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
    adata.var_names = [f"Gene_{i:d}" for i in range(adata.n_vars)]

    adata.obs["time"] = t.numpy()
    reg_bdata = adata.copy()
    reg_bdata.uns["regulators"] = adata.var.index.values
    reg_bdata.uns["targets"] = adata.var.index.values
    reg_bdata.uns["skeleton"] = np.ones((len(adata.var.index), len(adata.var.index)))
    reg_bdata.uns["network"] = np.ones((len(adata.var.index), len(adata.var.index)))

    ## veloVI & RegVelo min-max scaling
    reg_bdata = preprocess_data(reg_bdata, filter_on_r2=False)

    ## delete self-regulation
    W = reg_bdata.uns["skeleton"].copy()
    W = torch.tensor(np.array(W)).int()

    REGVELOVI.setup_anndata(reg_bdata, spliced_layer="Ms", unspliced_layer="Mu")
    reg_vae = REGVELOVI(reg_bdata, W=W, lam2=1, soft_constraint=False, simple_dynamics=True)
    reg_vae.train()

    adata_target = add_regvelo_outputs_to_adata(reg_bdata, reg_vae)
    pre_t = adata_target.layers["latent_time_regvelo"].mean(1)
    pre_t = t.max() * ((pre_t - np.min(pre_t)) / (np.max(pre_t) - np.min(pre_t)))
    adata_target.obs["latent_time_regvelo"] = pre_t

    ## print regvelo performance
    corr_t_rgv.append(scipy.stats.spearmanr(pre_t, adata.obs["time"])[0])
    print("RegVelo: " + str(scipy.stats.spearmanr(pre_t, adata.obs["time"])[0]))

    ## compare velocity correlation
    pre_velo = np.array(pre_u) * np.array(adata_target.var["fit_beta_regvelo"]) - np.array(pre_s) * np.array(
        adata_target.var["fit_gamma_regvelo"]
    )
    corr_rgv = []
    for i in range(6):
        corr_rgv.append(scipy.stats.pearsonr(pre_velo[:, i], gt_velo[:, i])[0])

    ## print regvelo performance
    velo_rgv.append(np.mean(corr_rgv))
    print("RegVelo: " + str(np.mean(corr_rgv)))

    ## Run scVelo
    sc.pp.neighbors(reg_bdata)
    scv.tl.recover_dynamics(reg_bdata, fit_scaling=False, var_names=adata.var_names, n_jobs=1)
    reg_bdata.var["fit_scaling"] = 1.0
    scv.tl.velocity(reg_bdata, mode="dynamical", min_likelihood=-np.inf, min_r2=None)
    pre_t = reg_bdata.layers["fit_t"].mean(1)
    pre_t = t.max() * ((pre_t - np.min(pre_t)) / (np.max(pre_t) - np.min(pre_t)))

    ## print regvelo performance
    corr_t_scv.append(scipy.stats.spearmanr(pre_t, adata.obs["time"])[0])
    print("scVelo: " + str(scipy.stats.spearmanr(pre_t, adata.obs["time"])[0]))

    ## compare velocity correlation
    pre_velo = np.array(pre_u) * np.array(reg_bdata.var["fit_beta"]) - np.array(pre_s) * np.array(
        reg_bdata.var["fit_gamma"]
    )
    corr_scv = []
    for i in range(6):
        corr_scv.append(scipy.stats.pearsonr(pre_velo[:, i], gt_velo[:, i])[0])

    ## print regvelo performance
    velo_scv.append(np.mean(corr_scv))
    print("scVelo: " + str(np.mean(corr_scv)))

    ## import veloVI
    VELOVI.setup_anndata(adata_target, spliced_layer="Ms", unspliced_layer="Mu")
    vae = VELOVI(adata_target)
    vae.train()
    add_velovi_outputs_to_adata(adata_target, vae)
    pre_t = adata_target.layers["fit_t"].mean(1)
    pre_t = 200 * ((pre_t - np.min(pre_t)) / (np.max(pre_t) - np.min(pre_t)))

    corr_t_velovi.append(scipy.stats.spearmanr(pre_t, adata.obs["time"])[0])
    print("veloVI: " + str(scipy.stats.spearmanr(pre_t, adata.obs["time"])))

    ## compare velocity
    pre_velo = np.array(pre_u) * np.array(adata_target.var["fit_beta_velovi"]) - np.array(pre_s) * np.array(
        adata_target.var["fit_gamma_velovi"]
    )
    corr = []
    for i in range(6):
        corr.append(scipy.stats.pearsonr(pre_velo[:, i], gt_velo[:, i])[0])

    ## print regvelo performance
    velo_velovi.append(np.mean(corr))
    print("veloVI: " + str(np.mean(corr)))

    adata_target.obs["latent_time_velovi"] = pre_t
    ### calculate diffusion pseudotime
    adata_target.uns["iroot"] = np.flatnonzero(adata_target.obs["time"] == 0)[0]

    sc.pp.neighbors(adata_target)
    sc.tl.diffmap(adata_target)
    sc.tl.dpt(adata_target)

    adata_target.obs["emperical_time"] = adata_target.layers["Mu"].mean(1)
    corr_t_emperical.append(scipy.stats.spearmanr(adata_target.obs["emperical_time"], adata.obs["time"])[0])
    corr_t_dpt.append(scipy.stats.spearmanr(adata_target.obs["dpt_pseudotime"], adata.obs["time"])[0])

    #### Benchmark GRN performance
    GRN = (
        reg_vae.module.v_encoder.GRN_Jacobian(torch.tensor(np.array(pre_s)).mean(0).to("cuda:0"))
        .cpu()
        .detach()
        .cpu()
        .numpy()
    )
    pre = GRN[np.where(~np.eye(GRN.shape[0], dtype=bool))]
    label = K[np.where(~np.eye(K.shape[0], dtype=bool))]
    label[label != 0] = 1
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(label, pre, pos_label=1)
    rgv_GRN.append(sklearn.metrics.auc(fpr, tpr))

    # calculate correlation
    C = np.abs(np.array(pd.DataFrame(adata.layers["spliced"]).corr()))
    pre2 = C[np.where(~np.eye(C.shape[0], dtype=bool))]
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(label, pre2, pos_label=1)
    cor_GRN.append(sklearn.metrics.auc(fpr, tpr))

    # GRNBoost2
    ex_matrix = adata.to_df("spliced")
    tf = ex_matrix.columns.tolist()
    network = grnboost2(expression_data=ex_matrix, tf_names=tf)
    table = np.array(network)

    # Get unique TFs and targets
    unique_tfs = np.unique(table[:, 0])
    unique_targets = np.unique(table[:, 1])

    # Create a new NumPy array to store the rearranged data
    GRN = np.zeros((len(unique_targets), len(unique_tfs)))

    # Fill in the new array with importance values
    for row in table:
        tf_index = np.where(unique_tfs == row[0])[0][0]
        target_index = np.where(unique_targets == row[1])[0][0]
        GRN[target_index, tf_index] = row[2]

    pre = GRN[np.where(~np.eye(GRN.shape[0], dtype=bool))]
    label = K[np.where(~np.eye(K.shape[0], dtype=bool))]
    label[label != 0] = 1
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(label, pre, pos_label=1)
    gb_GRN.append(sklearn.metrics.auc(fpr, tpr))

    ## Run celloracle
    base_GRN_sim = np.array(W)
    base_GRN_sim[base_GRN_sim != 0] = 1
    base_GRN_sim = pd.DataFrame(base_GRN_sim, columns=["Gene0", "Gene1", "Gene2", "Gene3", "Gene4", "Gene5"])
    base_GRN_sim.loc[:, "peak_id"] = [(f"Peak_{i}") for i in ["0", "1", "2", "3", "4", "5"]].copy()
    base_GRN_sim.loc[:, "gene_short_name"] = ["Gene0", "Gene1", "Gene2", "Gene3", "Gene4", "Gene5"]
    base_GRN_sim = base_GRN_sim.loc[
        :, ["peak_id", "gene_short_name", "Gene0", "Gene1", "Gene2", "Gene3", "Gene4", "Gene5"]
    ]

    adata.var.index = ["Gene0", "Gene1", "Gene2", "Gene3", "Gene4", "Gene5"]
    net = co.Net(
        gene_expression_matrix=adata.to_df(),  # Input gene expression matrix as data frame
        TFinfo_matrix=base_GRN_sim,  # Input base GRN
        verbose=True,
    )

    net.fit_All_genes(bagging_number=100, alpha=1, verbose=True)
    net.updateLinkList(verbose=True)
    inference_result = net.linkList.copy()

    GRN_table = inference_result.iloc[:, :3].copy()
    table = np.array(GRN_table)

    # Get unique TFs and targets
    unique_tfs = np.unique(table[:, 0])
    unique_targets = np.unique(table[:, 1])

    # Create a new NumPy array to store the rearranged data
    GRN = np.zeros((len(unique_targets), len(unique_tfs)))

    # Fill in the new array with importance values
    for row in table:
        tf_index = np.where(unique_tfs == row[0])[0][0]
        target_index = np.where(unique_targets == row[1])[0][0]
        GRN[target_index, tf_index] = row[2]
    pre = np.abs(GRN)[np.where(~np.eye(GRN.shape[0], dtype=bool))]
    label = K[np.where(~np.eye(K.shape[0], dtype=bool))]
    label[label != 0] = 1
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(label, pre, pos_label=1)
    co_GRN.append(sklearn.metrics.auc(fpr, tpr))

    print(
        "AUC: "
        + "RegVelo: "
        + str(rgv_GRN[len(rgv_GRN) - 1])
        + " Cor: "
        + str(cor_GRN[len(cor_GRN) - 1])
        + " GRNBoost2: "
        + str(gb_GRN[len(gb_GRN) - 1])
        + " CellOracle: "
        + str(co_GRN[len(co_GRN) - 1])
    )
    print(
        "Velocity: "
        + "RegVelo: "
        + str(velo_rgv[len(velo_rgv) - 1])
        + " veloVI: "
        + str(velo_velovi[len(velo_velovi) - 1])
        + " scVelo: "
        + str(velo_scv[len(velo_scv) - 1])
    )

# %%
len(rgv_GRN)

# %%
dat = pd.DataFrame(
    {
        "GRN": rgv_GRN + cor_GRN + gb_GRN + co_GRN,
        "Model": ["RegVelo"] * 100 + ["Correlation"] * 100 + ["GRNBoost2"] * 100 + ["CellOracle"] * 100,
    }
)
if SAVE_DATASETS:
    dat.to_csv(DATA_DIR / "simulation" / "toy_GRN" / "GRN_benchmark_result.csv")

# %%
## boxplot to show latent time correlation on each gene
dat = pd.DataFrame(
    {
        "Time": corr_t_rgv + corr_t_velovi + corr_t_scv + corr_t_dpt,
        "Model": ["RegVelo"] * 100 + ["veloVI"] * 100 + ["scVelo"] * 100 + ["Diffusion pseudotime"] * 100,
    }
)
if SAVE_DATASETS:
    dat.to_csv(DATA_DIR / "simulation" / "toy_GRN" / "latent_time_benchmark_result.csv")

# %%
dat = pd.DataFrame({"RegVelo": velo_rgv, "scVelo": velo_scv, "veloVI": velo_velovi})
if SAVE_DATASETS:
    dat.to_csv(DATA_DIR / "simulation" / "toy_GRN" / "velocity_benchmark.csv")

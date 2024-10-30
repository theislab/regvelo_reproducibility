# %% [markdown]
# # Perform benchmark on dyngen simulated datasets

# %% [markdown]
# ## Library imports

# %%
import math
import os
import random
import sys
from typing import Literal

import velovae as vv
from distributed import Client, LocalCluster

import numpy as np
import pandas as pd
import scipy
import sklearn

## define function
import torch
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon

import mplscience
import seaborn as sns
from matplotlib import pyplot as plt

import anndata
import scanpy as sc
import scvelo as scv
import unitvelo as utv
from arboreto.algo import grnboost2
from regvelo import REGVELOVI
from velovi import preprocess_data, VELOVI

from rgv_tools import DATA_DIR, FIG_DIR

# %% [markdown]
# ## General settings

# %%
sys.path.append(os.getcwd() + "/RegVelo_datasets/VeloVAE")

# %%
sns.reset_defaults()
sns.reset_orig()
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=14, color_map="viridis")

# %%
plt.rcParams["svg.fonttype"] = "none"

# %%
SAVE_FIGURES = True
if SAVE_FIGURES:
    (FIG_DIR / "simulation" / "dyngen_results").mkdir(parents=True, exist_ok=True)

SAVE_DATASETS = True
if SAVE_DATASETS:
    (DATA_DIR / "simulation" / "dyngen_results").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "simulation" / "dyngen_results" / "copy_file").mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Function definitions


# %%
def csgn_groundtruth(adata):
    """TODO."""
    csgn_array = adata.obsm["regulatory_network_sc"].toarray()
    csgn_tensor = torch.zeros([len(adata.uns["regulators"]), len(adata.uns["targets"]), csgn_array.shape[0]])

    for k in range(csgn_array.shape[0]):
        ## generate a 3D tensor to indicate the ground truth network for each cell
        grnboost_m = np.zeros((len(adata.uns["regulators"]), len(adata.uns["targets"])))
        grnboost_m = pd.DataFrame(grnboost_m, index=adata.uns["regulators"], columns=adata.uns["targets"])
        for i in range(adata.uns["regulatory_network"].shape[0]):
            # ind = (adata.uns["regulatory_network"]["regulator"] == j) & (adata.uns["regulatory_network"]["target"] == i)
            regulator = adata.uns["regulatory_network"].iloc[i]["regulator"]
            target = adata.uns["regulatory_network"].iloc[i]["target"]
            grnboost_m.loc[regulator, target] = csgn_array[k, i]
        tensor = torch.tensor(np.array(grnboost_m))
        csgn_tensor[:, :, k] = tensor

    return csgn_tensor


# %%
def csgn_benchmark(GRN, csgn):
    """TODO."""
    csgn[csgn != 0] = 1
    if len(GRN.shape) > 2:
        print("Input is cell type specific GRN...")
        score = []
        for i in range(csgn.shape[2]):
            W = csgn[:, :, i]
            W[W != 0] = 1
            # auprc = sklearn.metrics.average_precision_score(W.T.ravel(), np.abs(GRN[:,:,i].numpy().ravel()))
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(
                y_true=W.T.ravel(), y_score=GRN[:, :, i].numpy().ravel(), pos_label=1
            )  # positive class is 1; negative class is 0
            auroc = sklearn.metrics.auc(fpr, tpr)
            score.append(auroc)
    else:
        print("Input is global GRN...")
        score = []
        for i in range(csgn.shape[2]):
            W = csgn[:, :, i]
            W[W != 0] = 1
            # auprc = sklearn.metrics.average_precision_score(W.T.ravel(), np.abs(GRN.numpy().ravel()))
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(
                y_true=W.T.ravel(), y_score=GRN.numpy().ravel(), pos_label=1
            )  # positive class is 1; negative class is 0
            auroc = sklearn.metrics.auc(fpr, tpr)
            score.append(auroc)
    return score


# %%
def csgn_benchmark2(GRN, W, csgn):
    """TODO."""
    csgn[csgn != 0] = 1
    if len(GRN.shape) > 2:
        print("Input is cell type specific GRN...")
        score = []
        for i in range(csgn.shape[2]):
            net = csgn[:, :, i]
            # auprc = sklearn.metrics.average_precision_score(W.T.ravel(), np.abs(GRN[:,:,i].numpy().ravel()))
            pre = GRN[:, :, i][np.array(W.T) == 1]
            gt = net.T[np.array(W.T) == 1]
            gt[gt != 0] = 1

            number = min(10000, len(gt))
            pre, index = torch.topk(pre, number)
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(
                y_true=gt[index], y_score=pre, pos_label=1
            )  # positive class is 1; negative class is 0
            auroc = sklearn.metrics.auc(fpr, tpr)
            score.append(auroc)
    else:
        print("Input is global GRN...")
        score = []
        for i in range(csgn.shape[2]):
            net = csgn[:, :, i]
            pre = GRN[np.array(W.T) == 1]
            gt = net.T[np.array(W.T) == 1]
            gt[gt != 0] = 1

            number = min(10000, len(gt))
            pre, index = torch.topk(pre, number)
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(
                y_true=gt[index], y_score=pre, pos_label=1
            )  # positive class is 1; negative class is 0
            auroc = sklearn.metrics.auc(fpr, tpr)
            score.append(auroc)
    return score


# %%
def sanity_check(
    adata,
    network_mode: Literal["GENIE3", "full_ODE"] = "GENIE3",
) -> anndata.AnnData:
    """TODO."""
    if network_mode == "GENIE3":
        reg_index = [i in adata.var.index.values for i in adata.uns["regulators"]]
        tar_index = [i in adata.var.index.values for i in adata.uns["targets"]]
        adata.uns["regulators"] = adata.uns["regulators"][reg_index]
        adata.uns["targets"] = adata.uns["targets"][tar_index]
        W = adata.uns["skeleton"]
        W = W[reg_index, :]
        W = W[:, tar_index]
        adata.uns["skeleton"] = W
        W = adata.uns["network"]
        W = W[reg_index, :]
        W = W[:, tar_index]
        adata.uns["network"] = W
        regulators = adata.uns["regulators"][adata.uns["skeleton"].sum(1) > 0]
        targets = adata.uns["targets"][adata.uns["skeleton"].sum(0) > 0]
        adata = adata[:, np.unique(regulators.tolist() + targets.tolist())].copy()
        ## to make sure consistency
        regulator_index = [i in regulators for i in adata.var.index.values]
        target_index = [i in targets for i in adata.var.index.values]
        regulators = adata.var.index.values[regulator_index]
        targets = adata.var.index.values[target_index]
        print("num regulators: " + str(len(regulators)))
        print("num targets: " + str(len(targets)))
        W = pd.DataFrame(adata.uns["skeleton"], index=adata.uns["regulators"], columns=adata.uns["targets"])
        W = W.loc[regulators, targets]
        adata.uns["skeleton"] = W
        W = pd.DataFrame(adata.uns["network"], index=adata.uns["regulators"], columns=adata.uns["targets"])
        W = W.loc[regulators, targets]
        adata.uns["network"] = W
        adata.uns["regulators"] = regulators
        adata.uns["targets"] = targets
    if network_mode == "full_ODE":
        ## filter the gene first
        csgn = adata.uns["csgn"]
        gene_name = adata.var.index.tolist()
        full_name = adata.uns["regulators"]
        index = [i in gene_name for i in full_name]
        full_name = full_name[index]
        adata = adata[:, full_name].copy()
        W = adata.uns["skeleton"]
        W = W[index, :]
        W = W[:, index]
        adata.uns["skeleton"] = W
        W = adata.uns["network"]
        W = W[index, :]
        W = W[:, index]
        csgn = csgn[index, :, :]
        csgn = csgn[:, index, :]
        adata.uns["network"] = W
        ###
        W = adata.uns["skeleton"]
        gene_name = adata.var.index.tolist()
        indicator = W.sum(0) > 0  ## every gene would need to have a upstream regulators
        regulators = [gene for gene, boolean in zip(gene_name, indicator) if boolean]
        targets = [gene for gene, boolean in zip(gene_name, indicator) if boolean]
        print("num regulators: " + str(len(regulators)))
        print("num targets: " + str(len(targets)))
        W = adata.uns["skeleton"]
        W = W[indicator, :]
        W = W[:, indicator]
        adata.uns["skeleton"] = W
        W = adata.uns["network"]
        W = W[indicator, :]
        W = W[:, indicator]
        adata.uns["network"] = W
        csgn = csgn[indicator, :, :]
        csgn = csgn[:, indicator, :]
        adata.uns["csgn"] = csgn
        adata.uns["regulators"] = regulators
        adata.uns["targets"] = targets
        W = pd.DataFrame(adata.uns["skeleton"], index=adata.uns["regulators"], columns=adata.uns["targets"])
        W = W.loc[regulators, targets]
        adata.uns["skeleton"] = W
        W = pd.DataFrame(adata.uns["network"], index=adata.uns["regulators"], columns=adata.uns["targets"])
        W = W.loc[regulators, targets]
        adata.uns["network"] = W
        adata = adata[:, indicator].copy()
        adata.obsm["knn"] = adata.uns["neighbors"]["indices"].copy()
    return adata


# %%
def add_velovi_outputs_to_adata(adata, vae):
    """TODO."""
    latent_time = vae.get_latent_time(n_samples=25)
    velocities = vae.get_velocity(n_samples=25, velo_statistic="mean")
    t = latent_time
    scaling = 20 / t.max(0)
    adata.layers["velocity"] = velocities / scaling
    adata.layers["latent_time_velovi"] = latent_time
    adata.var["fit_alpha"] = vae.get_rates()["alpha"] / scaling
    adata.var["fit_beta"] = vae.get_rates()["beta"] / scaling
    adata.var["fit_gamma"] = vae.get_rates()["gamma"] / scaling
    adata.var["fit_t_"] = (
        torch.nn.functional.softplus(vae.module.switch_time_unconstr).detach().cpu().numpy()
    ) * scaling
    adata.layers["fit_t_velovi"] = latent_time.values * scaling[np.newaxis, :]
    adata.var["fit_scaling"] = 1.0


# %%
def add_noise_graph(W, noise_level=0.2):
    """TODO."""
    W_c = 1 - W
    edge = torch.nonzero(W)
    ## drop edge
    num_edge = edge.shape[0]
    selected_numbers = random.sample(range(edge.shape[0]), math.ceil((1 - noise_level) * num_edge))
    edge = edge[selected_numbers, :]
    #
    edge_c = torch.nonzero(W_c)
    ## select noise edge
    selected_numbers = random.sample(range(edge_c.shape[0]), math.ceil((noise_level) * num_edge))
    edge_c = edge_c[selected_numbers, :]
    ### generate final edge
    edge = torch.cat([edge, edge_c], 0)
    ## generate disturbed graph
    binary_tensor = torch.zeros(W.shape)
    binary_tensor[edge[:, 0], edge[:, 1]] = 1
    return binary_tensor


# %%
def add_regvelo_outputs_to_adata(adata_raw, vae):
    """TODO."""
    latent_time = vae.get_latent_time(n_samples=25)
    velocities = vae.get_velocity(n_samples=25)

    t = latent_time
    scaling = 20 / t.max(0)
    adata = adata_raw[:, vae.module.target_index].copy()

    adata.layers["velocity"] = velocities / scaling
    # adata.layers["velocity"] = velocities
    adata.layers["latent_time_regvelo"] = latent_time

    adata.layers["fit_t"] = latent_time.values * scaling[np.newaxis, :]
    # adata.layers["fit_t"] = latent_time.values
    adata.var["fit_scaling"] = 1.0

    # adata.obs["latent_time"] = vae.compute_shared_time(adata.layers["fit_t"])
    return adata


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
def get_sign_ratio(vector1, vector2):
    """TODO."""
    if len(vector1) != len(vector2):
        raise ValueError("Both vectors must have the same length.")
    same_sign_count = 0
    total_count = 0
    for sign1, sign2 in zip(vector1, vector2):
        if sign1 != 0 and sign2 != 0:
            if sign1 == sign2:
                same_sign_count += 1
            total_count += 1
    if total_count == 0:
        return 0.0
    return same_sign_count / total_count


# %%
def get_significance(pvalue):
    """TODO."""
    if pvalue < 0.001:
        return "***"
    elif pvalue < 0.01:
        return "**"
    elif pvalue < 0.1:
        return "*"
    else:
        return "n.s."


# %%
def add_significance2(ax, bottom: int, top: int, significance: str, level: int = 0, **kwargs):
    """TODO."""
    bracket_level = kwargs.pop("bracket_level", 1)
    bracket_height = kwargs.pop("bracket_height", 0.02)
    text_height = kwargs.pop("text_height", 0.01)

    left, right = ax.get_xlim()
    x_axis_range = right - left

    bracket_level = (x_axis_range * 0.07 * level) + right * bracket_level
    bracket_height = bracket_level - (x_axis_range * bracket_height)

    ax.plot([bracket_height, bracket_level, bracket_level, bracket_height], [bottom, bottom, top, top], **kwargs)

    ax.text(
        bracket_level + (x_axis_range * text_height),
        (bottom + top) * 0.5,
        significance,
        va="center",
        ha="left",
        c="k",
        rotation=90,
    )


# %% [markdown]
# ## Data loading

# %%
adatas = [file for file in (DATA_DIR / "dyngen").iterdir() if file.endswith(".h5ad")]

# %% [markdown]
# ## Running Benchmark

# %%
gene_time_corr_all = []
gene_velo_corr_all = []
AUC_GRN_result = []
AUC_GRN_result_all = []

# %%
for adata_name in adatas:
    address = os.getcwd() + "/RegVelo_datasets/dyngen_simulation/" + adata_name

    adata = sc.read_h5ad(address)
    adata_raw = adata.copy()
    csgn = csgn_groundtruth(adata)
    adata.uns["csgn"] = csgn

    adata.X = adata.X.copy()
    adata.layers["spliced"] = adata.layers["counts_spliced"].copy()
    adata.layers["unspliced"] = adata.layers["counts_unspliced"].copy()

    scv.pp.filter_and_normalize(adata, min_shared_counts=10)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
    scv.pp.moments(adata)
    # scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

    adata.X = np.log1p(adata.X.copy())

    sc.tl.leiden(adata)
    adata_raw.obs["cluster"] = adata.obs["leiden"].copy()
    adata_raw.obsm["X_pca"] = adata.obsm["X_pca"].copy()
    adata_raw.layers["spliced"] = adata_raw.layers["counts_spliced"].copy()
    adata_raw.layers["unspliced"] = adata_raw.layers["counts_unspliced"].copy()

    adata = preprocess_data(adata, filter_on_r2=True)
    adata = sanity_check(adata, network_mode="full_ODE")
    adata.uns["Ms"] = adata.layers["Ms"]
    adata.uns["Mu"] = adata.layers["Mu"]

    ## save raw data
    adata_raw.var["highly_variable"] = [adata_raw.var.index[i] in adata.var.index for i in range(adata_raw.shape[1])]
    adata_raw = adata_raw[:, adata_raw.var["highly_variable"]]
    save_address = "data_file_" + adata_name
    save_address = DATA_DIR / "simulation" / "dyngen_results" / "copy_file" / save_address
    adata_raw.write(save_address)

    ## Run scVelo model (dynamical)
    scv.tl.recover_dynamics(adata, fit_scaling=False, var_names=adata.var_names, n_jobs=1)
    adata.var["fit_scaling"] = 1.0
    adata.layers["fit_t_dynamical"] = adata.layers["fit_t"].copy()
    scv.tl.velocity(adata, mode="dynamical", min_likelihood=-np.inf, min_r2=None)
    scv.tl.latent_time(adata, min_likelihood=None)

    velocity_gt = adata.layers["rna_velocity"]
    velocity = adata.layers["velocity"]
    dim = velocity.shape[1]

    corr = []
    for i in range(velocity.shape[1]):
        corr.append(
            scipy.stats.pearsonr(np.array(velocity_gt.todense()[:, i]).ravel(), np.array(velocity[:, i]).ravel())
        )

    corr = np.array(corr)[:, 0]
    dynamical_corr = corr

    ## Run scVelo model (stochastic)
    scv.tl.velocity(adata, mode="stochastic")
    velocity_gt = adata.layers["rna_velocity"]
    velocity = adata.layers["velocity"]
    dim = velocity.shape[1]

    corr = []
    for i in range(velocity.shape[1]):
        corr.append(
            scipy.stats.pearsonr(np.array(velocity_gt.todense()[:, i]).ravel(), np.array(velocity[:, i]).ravel())
        )

    corr = np.array(corr)[:, 0]
    stochastic_corr = corr

    ## Run scVelo model (deterministic)
    scv.tl.velocity(adata, mode="deterministic")
    velocity_gt = adata.layers["rna_velocity"]
    velocity = adata.layers["velocity"]
    dim = velocity.shape[1]

    corr = []
    for i in range(velocity.shape[1]):
        corr.append(
            scipy.stats.pearsonr(np.array(velocity_gt.todense()[:, i]).ravel(), np.array(velocity[:, i]).ravel())
        )

    corr = np.array(corr)[:, 0]
    deterministic_corr = corr

    ### fit VeloVI
    torch.cuda.empty_cache()
    VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    vae = VELOVI(adata)
    vae.train()
    add_velovi_outputs_to_adata(adata, vae)

    velocity_gt = adata.layers["rna_velocity"]
    velocity = adata.layers["velocity"]
    corr = []
    for i in range(velocity.shape[1]):
        corr.append(
            scipy.stats.pearsonr(np.array(velocity_gt.todense()[:, i]).ravel(), np.array(velocity[:, i]).ravel())
        )
        # corr.append(get_sign_ratio(np.sign(np.array(velocity_gt.todense()[:,i]).ravel()), np.sign(np.array(velocity[:,i]).ravel())))

    corr = np.array(corr)[:, 0]
    velovi_corr = corr

    ### Run RegVelo
    W = adata.uns["skeleton"].copy()
    GRN_gt = adata.uns["skeleton"].copy()
    W = torch.tensor(np.array(W)).int()
    W = torch.ones(W.shape)

    """
    intersection = list(set(adata.uns["regulators"]).intersection(adata.uns["targets"]))
    for i in intersection:
        index1 = [j == i for j in adata.uns["regulators"]]
        index2 = [j == i for j in adata.uns["targets"]]
        W[index1,index2] = 0
    """

    torch.cuda.empty_cache()
    REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    reg_vae = REGVELOVI(adata, W=W, t_max=20)
    reg_vae.train()

    adata_target = add_regvelo_outputs_to_adata(adata, reg_vae)

    velocity_gt = adata_target.layers["rna_velocity"]
    velocity = adata_target.layers["velocity"]
    corr = []
    for i in range(velocity.shape[1]):
        corr.append(
            scipy.stats.pearsonr(np.array(velocity_gt.todense()[:, i]).ravel(), np.array(velocity[:, i]).ravel())
        )
        # corr.append(get_sign_ratio(np.sign(np.array(velocity_gt.todense()[:,i]).ravel()), np.sign(np.array(velocity[:,i]).ravel())))

    corr = np.array(corr)[:, 0]
    regvelo_corr = corr

    Jaco_m = GRN_Jacobian(reg_vae, adata.layers["Ms"])

    # calculate unitvelo
    velo = utv.config.Configuration()
    velo.GPU = -1
    velo.FIT_OPTION = "1"
    adata_utv = utv.run_model(str(save_address), "cluster", config_file=velo)

    dim = adata.shape[1]
    velocity_gt = adata_utv.layers["rna_velocity"]
    velocity = adata_utv.layers["velocity"]

    corr = []
    for i in range(velocity.shape[1]):
        corr.append(
            scipy.stats.pearsonr(np.array(velocity_gt.todense()[:, i]).ravel(), np.array(velocity[:, i]).ravel())
        )

    corr = np.array(corr)[:, 0]
    unitvelo_corr = corr

    ## calculate veloVAE
    adata_vv = adata_utv.copy()
    vae = vv.VAE(adata_vv, tmax=20, dim_z=10, device="cuda:0")

    config = {
        # You can change any hyperparameters here!
    }
    vae.train(adata_vv, config=config, plot=False, embed="dimred")

    file_name = adata_name + "_sim_vae.h5ad"
    vae.save_anndata(adata_vv, "vae", "GRN_benchmark", file_name=file_name)

    ## calculate veloVAE(VAE)
    velocity_gt = adata_vv.layers["rna_velocity"]
    velocity = adata_vv.layers["vae_velocity"]

    corr = []
    for i in range(velocity.shape[1]):
        corr.append(
            scipy.stats.pearsonr(np.array(velocity_gt.todense()[:, i]).ravel(), np.array(velocity[:, i]).ravel())
        )

    vae_corr = np.array(corr)[:, 0]

    try:
        # Perform some computation in f(a)
        rate_prior = {"alpha": (0, 1.0), "beta": (0, 0.5), "gamma": (0, 0.5)}
        full_vb = vv.VAE(adata_vv, tmax=20, dim_z=10, device="cuda:0", full_vb=True, rate_prior=rate_prior)

        config = {
            # You can change any hyperparameters here!
        }
        full_vb.train(adata_vv, config=config, plot=False, embed="dimred")

        file_name = adata_name + "_sim_fullvb.h5ad"
        full_vb.save_anndata(adata_vv, "fullvb", "GRN_benchmark", file_name=file_name)

        velocity_gt = adata_vv.layers["rna_velocity"]
        velocity = adata_vv.layers["fullvb_velocity"]

        corr = []
        for i in range(velocity.shape[1]):
            corr.append(
                scipy.stats.pearsonr(np.array(velocity_gt.todense()[:, i]).ravel(), np.array(velocity[:, i]).ravel())
            )

        fullvb_corr = np.array(corr)[:, 0]
        # If no error is raised, break the loop and return the result
    except:
        # If an error is raised, increment a and try again, and need to recompute double knock-out reults
        fullvb_corr = [np.nan] * len(vae_corr)
        raise

    # Done velocity!
    # calculate correlation
    regulator_index = [i in adata.uns["regulators"] for i in adata.var.index.values]
    target_index = [i in adata.uns["targets"] for i in adata.var.index.values]

    corr_m = 1 - cdist(adata.layers["Ms"].T, adata.layers["Ms"].T, metric="correlation")
    corr_m = torch.tensor(corr_m)
    corr_m = corr_m[target_index,]
    corr_m = corr_m[:, regulator_index]
    corr_m = corr_m.float()

    GRN = torch.mean(Jaco_m, 2)

    GRN_weight = reg_vae.module.v_encoder.fc1.weight.detach()

    ### Run GRNBoost2 to benchmark the GRN inference performance
    GEP = pd.DataFrame(adata.layers["Ms"], columns=adata.var.index.values)
    local_cluster = LocalCluster()
    client = Client(local_cluster)
    network = grnboost2(expression_data=GEP, tf_names=adata.uns["regulators"], client_or_address=client)

    client.close()
    local_cluster.close()

    ind = [i in adata.uns["targets"] for i in network["target"]]
    network = network[ind]

    grnboost_m = np.zeros((len(adata.uns["targets"]), len(adata.uns["regulators"])))
    grnboost_m = pd.DataFrame(grnboost_m, index=adata.uns["targets"], columns=adata.uns["regulators"])
    for i in adata.uns["targets"]:
        for j in adata.uns["regulators"]:
            ind = (network["TF"] == j) & (network["target"] == i)
            if sum(ind) > 0:
                pdd = network[ind]
                grnboost_m.loc[i, j] = pdd["importance"].values

    Jaco_m = Jaco_m.cpu().detach()
    GRN = GRN.cpu().detach()
    GRN_weight = GRN_weight.cpu().detach()

    score = csgn_benchmark2(torch.abs(Jaco_m), GRN_gt, adata.uns["csgn"])
    score2 = csgn_benchmark2(torch.abs(GRN), GRN_gt, adata.uns["csgn"])
    score3 = csgn_benchmark2(torch.abs(GRN_weight), GRN_gt, adata.uns["csgn"])
    score4 = csgn_benchmark2(torch.abs(corr_m), GRN_gt, adata.uns["csgn"])
    score5 = csgn_benchmark2(torch.tensor(np.array(grnboost_m)), GRN_gt, adata.uns["csgn"])

    score_all = csgn_benchmark2(torch.abs(Jaco_m), W, adata.uns["csgn"])
    score2_all = csgn_benchmark2(torch.abs(GRN), W, adata.uns["csgn"])
    score3_all = csgn_benchmark2(torch.abs(GRN_weight), W, adata.uns["csgn"])
    score4_all = csgn_benchmark2(torch.abs(corr_m), W, adata.uns["csgn"])
    score5_all = csgn_benchmark2(torch.tensor(np.array(grnboost_m)), W, adata.uns["csgn"])

    ### Visualize the Violin Plots
    AUROC = [np.mean(score), np.mean(score2), np.mean(score3), np.mean(score4), np.mean(score5)]
    AUROC_all = [np.mean(score_all), np.mean(score2_all), np.mean(score3_all), np.mean(score4_all), np.mean(score5_all)]
    AUC_GRN_result.append(AUROC)
    AUC_GRN_result_all.append(AUROC_all)
    print(AUROC)
    print(AUROC_all)

    fit_t_dynamical = adata.layers["fit_t_dynamical"]
    fit_t_velovi = adata.layers["fit_t_velovi"]
    fit_t_regvelo = adata_target.layers["fit_t"]

    ## gene specific latent time correlation with ground truth
    velocity = adata_target.layers["velocity"]
    corr = []
    for i in range(velocity.shape[1]):
        corr.append(scipy.stats.spearmanr(fit_t_dynamical[:, i], adata_target.obs["sim_time"]))

    corr = np.array(corr)[:, 0]
    corr_latent_time_dynamical = corr

    velocity = adata_target.layers["velocity"]
    corr = []
    for i in range(velocity.shape[1]):
        corr.append(scipy.stats.spearmanr(fit_t_velovi[:, i], adata_target.obs["sim_time"]))

    corr = np.array(corr)[:, 0]
    corr_latent_time_velovi = corr

    velocity = adata_target.layers["velocity"]
    corr = []
    for i in range(velocity.shape[1]):
        corr.append(scipy.stats.spearmanr(fit_t_regvelo[:, i], adata_target.obs["sim_time"]))

    corr = np.array(corr)[:, 0]
    corr_latent_time_regvelo = corr

    ## outputs

    gene_time_corr_res = [
        np.mean(corr_latent_time_dynamical),
        np.mean(corr_latent_time_velovi),
        np.mean(corr_latent_time_regvelo),
    ]

    ## calculate velocity correlation
    gene_velo_corr_res = [
        np.mean(dynamical_corr),
        np.mean(deterministic_corr),
        np.mean(stochastic_corr),
        np.mean(velovi_corr),
        np.mean(regvelo_corr),
        np.mean(unitvelo_corr),
        np.mean(fullvb_corr),
        np.mean(vae_corr),
    ]

    gene_time_corr_all.append(gene_time_corr_res)
    gene_velo_corr_all.append(gene_velo_corr_res)

    df = pd.DataFrame(gene_velo_corr_all)
    df.columns = [
        "dynamical",
        "deterministic",
        "stochastic",
        "velovi",
        "regvelo",
        "UniTVelo",
        "VeloVAE(fullvb)",
        "VeloVAE(vae)",
    ]

    if SAVE_DATASETS:
        df.to_csv(DATA_DIR / "simulation" / "dyngen_results" / "gene_velo_corr_all_final_res.csv")

    df = pd.DataFrame(gene_time_corr_all)
    df.columns = ["dynamical", "velovi", "regvelo"]

    if SAVE_DATASETS:
        df.to_csv(DATA_DIR / "simulation" / "dyngen_results" / "gene_time_corr_all_final_res.csv")

    df = pd.DataFrame(AUC_GRN_result)
    df.columns = ["Jaco", "GRN_mean", "GRN_weight", "corr_m", "GRNBoost2"]

    if SAVE_DATASETS:
        df.to_csv(DATA_DIR / "simulation" / "dyngen_results" / "AUROC_res_all_final_res.csv")

    df = pd.DataFrame(AUC_GRN_result_all)
    df.columns = ["Jaco", "GRN_mean", "GRN_weight", "corr_m", "GRNBoost2"]

    if SAVE_DATASETS:
        df.to_csv(DATA_DIR / "simulation" / "dyngen_results" / "AUROC_res_all_full_final_res.csv")

    print("Done " + adata_name + "!")

# %% [markdown]
# ## Benchmark analysis

# %% [markdown]
# ### Velocity benchmark

# %%
TFvelo = pd.read_csv("RegVelo_datasets/dyngen_benchmark/TFvelo_res.csv", index_col=0)
cell2fate = pd.read_csv("RegVelo_datasets/dyngen_benchmark/c2f_velo_cor.csv", index_col=0)

# %%
full_gene_velo = pd.read_csv(
    DATA_DIR / "simulation" / "dyngen_results" / "gene_velo_corr_all_final_res.csv", index_col=0
)

# %%
full_gene_velo.columns = [
    "scVelo",
    "scVelo(deterministic)",
    "scVelo(stochastic)",
    "veloVI",
    "RegVelo",
    "UniTVelo",
    "VeloVAE(vae)",
    "VeloVAE(fullvb)",
]

# %%
result_df = full_gene_velo.T
result_df.loc[:, "index"] = result_df.index
new_df = pd.melt(result_df, id_vars=["index"], value_name="Performance", var_name="Method")
new_df = new_df.iloc[:, [0, 2]].copy()
new_df.columns = ["Method", "Performance"]

# %%
new_df = new_df.append(pd.DataFrame({"Method": "TFvelo", "Performance": TFvelo["velocity_corr"]}))
new_df = new_df.append(pd.DataFrame({"Method": "cell2fate", "Performance": cell2fate.iloc[:, 0]}))

# %%
new_df["Method"] = pd.Categorical(
    new_df["Method"],
    categories=[
        "RegVelo",
        "veloVI",
        "scVelo",
        "scVelo(stochastic)",
        "scVelo(deterministic)",
        "UniTVelo",
        "VeloVAE(vae)",
        "VeloVAE(fullvb)",
        "TFvelo",
        "cell2fate",
    ],
)

# %%
new_df.loc[:, "Performance"] = (new_df.loc[:, "Performance"] + 1) / 2

# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(3, 5))

    # sns.violinplot(data=new_df, x="Method", y="Performance", palette="colorblind", ax=ax)
    colors = sns.color_palette("colorblind", n_colors=3)
    colors = colors + ["lightgrey"] * 7
    sns.violinplot(data=new_df, y="Method", x="Performance", ax=ax, palette=colors, linewidth=0.8)

    ttest_res = wilcoxon(
        new_df.iloc[:, 1][new_df.iloc[:, 0] == "RegVelo"].values,
        new_df.iloc[:, 1][new_df.iloc[:, 0] == "veloVI"].values,
        alternative="greater",
    )
    significance = get_significance(ttest_res.pvalue)
    add_significance2(
        ax=ax,
        bottom=0,
        top=1,
        significance=significance,
        lw=1,
        bracket_level=1.05,
        c="k",
        level=0,
    )

    x_min, x_max = ax.get_xlim()
    ax.set(ylabel="", xlabel="Velocity correlation")
    ax.set_xlim([x_min, x_max + 0.02])

    if SAVE_FIGURES:
        fig.savefig(
            FIG_DIR / "simulation" / "dyngen_results" / "velocity_benchmark.svg",
            format="svg",
            transparent=True,
            bbox_inches="tight",
        )
    plt.show()

# %% [markdown]
# ### Latent time benchmark

# %%
df = pd.read_csv(DATA_DIR / "simulation" / "dyngen_results" / "gene_time_corr_all_final_res.csv", index_col=0)

# %%
result = df.T
result.index = ["scVelo", "veloVI", "RegVelo"]
result.loc[:, "index"] = result.index
new_df = pd.melt(result, id_vars=["index"], value_name="Performance", var_name="Method")
new_df = new_df.iloc[:, [0, 2]].copy()
new_df.columns = ["Method", "Performance"]
new_df["Method"] = pd.Categorical(new_df["Method"], categories=["RegVelo", "veloVI", "scVelo"])

# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(3, 2))
    # pal = {"RegVelo":"#f3e1eb","veloVI":"#b5bbe3","scVelo":"#0fcfc0"}
    sns.violinplot(data=new_df, y="Method", x="Performance", ax=ax)

    ttest_res = wilcoxon(
        new_df.iloc[:, 1][new_df.iloc[:, 0] == "RegVelo"].values,
        new_df.iloc[:, 1][new_df.iloc[:, 0] == "veloVI"].values,
        alternative="greater",
    )
    significance = get_significance(ttest_res.pvalue)
    add_significance2(
        ax=ax,
        bottom=0,
        top=1,
        significance=significance,
        lw=1,
        bracket_level=1.05,
        c="k",
        level=0,
    )

    y_min, y_max = ax.get_ylim()
    ax.set(ylabel="", xlabel="latent time correlation")
    ax.set_ylim([y_min, y_max + 0.02])
    if SAVE_FIGURES:
        fig.savefig(
            FIG_DIR / "simulation" / "dyngen_results" / "latent_time_benchmark.svg",
            format="svg",
            transparent=True,
            bbox_inches="tight",
        )
    plt.show()

# %% [markdown]
# ### GRN inference benchmark

# %%
df_co = pd.read_csv("RegVelo_datasets/dyngen_benchmark/AUROC_res_all_celloracle.csv", index_col=0)
df_spliceJAC = pd.read_csv("RegVelo_datasets/dyngen_benchmark/AUROC_res_all_spliceAJ.csv", index_col=0)

# %%
df = pd.read_csv(DATA_DIR / "simulation" / "dyngen_results" / "AUROC_res_all_full_final_res.csv", index_col=0)

# %%
performance = (
    df.iloc[:, 0].tolist()
    + df.iloc[:, 3].tolist()
    + df.iloc[:, 4].tolist()
    + df_co.iloc[:, 0].tolist()
    + df_spliceJAC.iloc[:, 0].tolist()
    + TFvelo.iloc[:, 0].tolist()
)
method = (
    ["RegVelo"] * len(df.iloc[:, 0].tolist())
    + ["Cor"] * len(df.iloc[:, 0].tolist())
    + ["GRNBoost2"] * len(df.iloc[:, 0].tolist())
    + ["CellOracle"] * len(df_co.iloc[:, 0].tolist())
    + ["spliceJAC"] * len(df_spliceJAC.iloc[:, 0].tolist())
    + ["TFvelo"] * TFvelo.shape[0]
)

# %%
new_df = pd.DataFrame({"Method": method, "Performance": performance})
new_df = new_df.loc[~np.isnan(new_df.iloc[:, 1]), :].copy()

# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(3, 3))

    # sns.violinplot(data=result, x="Method", y="Performance", palette="colorblind",order = ["RegVelo","GRNBoost2","CellOracle","Cor","spliceJAC"], ax=ax)
    sns.violinplot(
        data=new_df,
        y="Method",
        x="Performance",
        color="lightpink",
        order=["RegVelo", "GRNBoost2", "CellOracle", "Cor", "spliceJAC", "TFvelo"],
        ax=ax,
    )

    ttest_res = wilcoxon(
        new_df.iloc[:, 1][new_df.iloc[:, 0] == "RegVelo"].values,
        new_df.iloc[:, 1][new_df.iloc[:, 0] == "GRNBoost2"].values,
        alternative="greater",
    )
    significance = get_significance(ttest_res.pvalue)
    add_significance2(
        ax=ax,
        bottom=0,
        top=1,
        significance=significance,
        lw=1,
        bracket_level=1.05,
        c="k",
        level=0,
    )

    x_min, x_max = ax.get_xlim()
    ax.set(ylabel="", xlabel="AUROC")
    ax.set_xlim([x_min, x_max + 0.02])
    if SAVE_FIGURES:
        fig.savefig(
            FIG_DIR / "simulation" / "dyngen_results" / "GRN_benchmark.svg",
            format="svg",
            transparent=True,
            bbox_inches="tight",
        )
    plt.show()

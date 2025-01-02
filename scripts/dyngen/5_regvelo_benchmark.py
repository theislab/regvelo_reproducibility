# %% [markdown]
# # Prior GRN sensitivity evaluation
#
# Notebook run RegVelo with perturbed prior GRN with different noise level

# %% [markdown]
# ## Library imports

# %%
import math
import random

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

import anndata as ad
import scvi
from regvelo import REGVELOVI

from rgv_tools import DATA_DIR
from rgv_tools.benchmarking import (
    get_time_correlation,
    get_velocity_correlation,
    set_output,
)

# %% [markdown]
# ## General settings

# %%
scvi.settings.seed = 0

# %% [markdown]
# ## Constants

# %%
DATASET = "prior_benchmark"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)


# %%
NOISE_LEVEL = [0.2, 0.4, 0.6, 0.8, 1]

# %% [markdown]
# ## Function definitions


# %%
def csgn_benchmark(GRN, csgn):
    """GRN benchmark."""
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
def add_noise_graph(W, noise_level=0.2):
    """Adding noise graph."""
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
def GRN_Jacobian(reg_vae, Ms):
    """Infer GRN."""
    reg_vae.module.v_encoder.fc1.weight.detach()
    reg_vae.module.v_encoder.fc1.bias.detach()
    reg_vae.module.v_encoder.alpha_unconstr_max.detach()
    ## calculate the jacobian matrix respect to each cell
    Jaco_m = []
    for i in range(Ms.shape[0]):
        s = Ms[i, :]

        Jaco_m.append(
            reg_vae.module.v_encoder.GRN_Jacobian(
                torch.tensor(s[reg_vae.module.v_encoder.regulator_index]).to("cuda:0")
            ).detach()
        )
    Jaco_m = torch.stack(Jaco_m, 2)
    return Jaco_m


# %% [markdown]
# ## Data loading

# %%
adata = ad.io.read_zarr(DATA_DIR / DATASET / "processed" / "processed_sim.zarr")
adata

# %% [markdown]
# ## Perturbed GRN

# %%
w_perturb = []

for noise in NOISE_LEVEL:
    W = []
    for i in range(5):
        print("Noise level: " + str(noise))
        print(str(i + 1) + "th run...")
        W.append(add_noise_graph(W=torch.tensor(np.array(adata.uns["true_skeleton"].copy())).int(), noise_level=noise))
    w_perturb.append(W)

# %% [markdown]
# ## Velocity pipeline

# %%
REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")

for i in range(len(NOISE_LEVEL)):
    for nrun in range(5):
        vae = REGVELOVI(adata, W=w_perturb[i][nrun].T)
        vae.train()

        DATA_NAME = str(NOISE_LEVEL[i]) + "_noise_rep" + str(nrun)
        vae.save(DATA_DIR / DATASET / "processed" / DATA_NAME)

# %% [markdown]
# ## Identifiability analysis

# %% [markdown]
# ### GRN identifiability

# %%
grn_pair_corr = []
for i in NOISE_LEVEL:
    estimate_all = []
    for j in range(5):
        model_name = str(i) + "_noise_rep" + str(j)
        path = DATA_DIR / DATASET / "processed" / model_name
        vae = REGVELOVI.load(path, adata)
        estimate_all.append(vae.module.v_encoder.fc1.weight.detach().clone().cpu().numpy().reshape(-1))

    cor, _ = scipy.stats.spearmanr(np.column_stack(estimate_all), axis=0)
    cor = cor[np.triu_indices(cor.shape[0], k=1)]
    grn_pair_corr.append(cor.tolist())

# %% [markdown]
# ### Latent time

# %%
cor_all_time = []
for i in NOISE_LEVEL:
    estimate_all = []
    for j in range(5):
        model_name = str(i) + "_noise_rep" + str(j)
        path = DATA_DIR / DATASET / "processed" / model_name
        vae = REGVELOVI.load(path, adata)

        set_output(adata, vae, n_samples=30, batch_size=adata.n_obs)
        estimate_all.append(adata.layers["fit_t"].reshape(-1))

    cor, _ = scipy.stats.spearmanr(np.column_stack(estimate_all), axis=0)
    cor = cor[np.triu_indices(cor.shape[0], k=1)]
    cor_all_time.append(cor.tolist())

# %% [markdown]
# ### Velocity

# %%
cor_all_velo = []
for i in NOISE_LEVEL:
    estimate_all = []
    for j in range(5):
        model_name = str(i) + "_noise_rep" + str(j)
        path = DATA_DIR / DATASET / "processed" / model_name
        vae = REGVELOVI.load(path, adata)

        set_output(adata, vae, n_samples=30, batch_size=adata.n_obs)
        estimate_all.append(adata.layers["velocity"].reshape(-1))

    cor, _ = scipy.stats.spearmanr(np.column_stack(estimate_all), axis=0)
    cor = cor[np.triu_indices(cor.shape[0], k=1)]
    cor_all_velo.append(cor.tolist())

# %%
if SAVE_DATA:
    pd.DataFrame({"velocity": cor_all_velo, "time": cor_all_time, "grn": grn_pair_corr}).to_parquet(
        path=DATA_DIR / DATASET / "results" / "identifiability_test.parquet"
    )

# %% [markdown]
# ## Prediction accuracy

# %% [markdown]
# #### GRN

# %%
auc_all = []
for i in NOISE_LEVEL:
    cor = []
    for j in range(5):
        model_name = str(i) + "_noise_rep" + str(j)
        path = DATA_DIR / DATASET / "processed" / model_name
        vae = REGVELOVI.load(path, adata)

        Jaco_m = GRN_Jacobian(vae, adata.layers["Ms"])
        Jaco_m = Jaco_m.cpu().detach()
        score = csgn_benchmark(torch.abs(Jaco_m), adata.uns["true_sc_grn"])
        cor.append(np.mean(score))

    auc_all.append(cor)

# %%
auc_all_prior = []
for i in NOISE_LEVEL:
    cor = []
    for j in range(5):
        model_name = str(i) + "_noise_rep" + str(j)
        path = DATA_DIR / DATASET / "processed" / model_name
        vae = REGVELOVI.load(path, adata)

        weight = vae.module.v_encoder.mask_m_raw.detach().clone().cpu()
        score = csgn_benchmark(weight, adata.uns["true_sc_grn"])
        cor.append(np.mean(score))
    auc_all_prior.append(cor)

# %% [markdown]
# #### Velocity

# %%
cor_velo_all = []
for i in NOISE_LEVEL:
    cor = []
    for j in range(5):
        model_name = str(i) + "_noise_rep" + str(j)
        path = DATA_DIR / DATASET / "processed" / model_name
        vae = REGVELOVI.load(path, adata)

        set_output(adata, vae, n_samples=30, batch_size=adata.n_obs)
        cor.append(
            get_velocity_correlation(
                ground_truth=adata.layers["true_velocity"], estimated=adata.layers["velocity"], aggregation=np.mean
            )
        )

    cor_velo_all.append(cor)

# %% [markdown]
# #### Latent time

# %%
cor_time_all = []
for i in NOISE_LEVEL:
    cor = []
    for j in range(5):
        model_name = str(i) + "_noise_rep" + str(j)
        path = DATA_DIR / DATASET / "processed" / model_name
        vae = REGVELOVI.load(path, adata)

        set_output(adata, vae, n_samples=30, batch_size=adata.n_obs)
        time_corr = [
            get_time_correlation(ground_truth=adata.obs["true_time"], estimated=adata.layers["fit_t"][:, i])
            for i in range(adata.layers["fit_t"].shape[1])
        ]
        cor.append(np.mean(time_corr))

    cor_time_all.append(cor)

# %%
if SAVE_DATA:
    pd.DataFrame(
        {"velocity": cor_velo_all, "time": cor_time_all, "grn": auc_all, "grn_prior": auc_all_prior}
    ).to_parquet(path=DATA_DIR / DATASET / "results" / "regvelo_benchmark.parquet")

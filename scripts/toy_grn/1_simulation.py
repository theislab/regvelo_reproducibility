# %% [markdown]
# # Simulation of toy GRN and cellular dynamics
#
# Notebooks simulate a toy GRN to benchmark velocity, latent time and GRN inference

# %% [markdown]
# ## Library imports

# %%
from tqdm import tqdm

import torch
import torchsde

import anndata as ad
from anndata import AnnData

from rgv_tools import DATA_DIR
from rgv_tools.datasets import VelocityEncoder
from rgv_tools.datasets._simulate import get_sde_parameters

# %% [markdown]
# ## General settings

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / "toy_grn" / "raw").mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Constants

# %%
N_SIMULATIONS = 100

# %% [markdown]
# ## Function definitions


# %%
def uns_merge(uns_list):
    """Define merge strategie for `.uns` when concatenating AnnData objects."""
    return dict(zip(map(str, range(len(uns_list))), uns_list))


# %% [markdown]
# ## Data generation

# %%
adatas = []

# %%
for sim_idx in tqdm(range(N_SIMULATIONS)):
    torch.cuda.empty_cache()
    torch.manual_seed(sim_idx)

    K, n, h, beta, gamma, t = get_sde_parameters(n_obs=1500, n_vars=6, seed=sim_idx)
    alpha_b = torch.zeros((6,), dtype=torch.float32)
    sde = VelocityEncoder(K=K, n=n, h=h, alpha_b=alpha_b, beta=beta, gamma=gamma)

    ## set up G batches, Each G represent a module (a target gene centerred regulon)
    ## infer the observe gene expression through ODE solver based on x0, t, and velocity_encoder
    y0 = torch.tensor([1.0, 0, 1.0, 0, 1.0, 0] + torch.zeros(6).abs().tolist()).reshape(1, -1)
    ys = torchsde.sdeint(sde, y0, t, method="euler")

    unspliced = torch.clip(ys[:, 0, :6], 0).numpy()
    spliced = torch.clip(ys[:, 0, 6:], 0).numpy()

    adata = AnnData(spliced)
    adata.obs_names = "cell_" + adata.obs_names + f"-simulation_{sim_idx}"
    adata.var_names = "gene_" + adata.var_names

    adata.layers["unspliced"] = unspliced
    adata.layers["Mu"] = unspliced

    adata.layers["spliced"] = spliced
    adata.layers["Ms"] = spliced

    beta = beta.numpy()
    gamma = gamma.numpy()
    adata.layers["true_velocity"] = unspliced * beta - spliced * gamma

    adata.uns = {
        "true_alpha_b": alpha_b.numpy(),
        "true_beta": beta,
        "true_gamma": gamma,
        "true_K": K.numpy(),
        "true_n": n.numpy(),
        "true_h": h.numpy(),
    }

    adata.obs["true_time"] = t

    adatas.append(adata)
    del adata

# %%
adata = ad.concat(adatas, label="dataset", uns_merge=uns_merge)
adata

# %% [markdown]
# ## Data saving

# %%
if SAVE_DATA:
    adata.write_zarr(DATA_DIR / "toy_grn" / "raw" / "adata.zarr")

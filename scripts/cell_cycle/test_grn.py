# %% [markdown]
# # Regulation is important for velocity inference of RegVelo

# %% [markdown]
# ## Library imports

# %%
import pandas as pd
import torch
from scipy.stats import ttest_ind

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

import anndata as ad
import scvi
from cellrank.kernels import VelocityKernel
from regvelo import REGVELOVI

from rgv_tools import DATA_DIR, FIG_DIR
from rgv_tools.benchmarking import get_time_correlation, set_output
from rgv_tools.plotting import add_significance, get_significance

# %% [markdown]
# ## General settings

# %%
scvi.settings.dl_pin_memory_gpu_training = False

# %% [markdown]
# ## Constants

# %%
DATASET = "cell_cycle"

# %%
significance_palette = {"n.s.": "#dedede", "*": "#90BAAD", "**": "#A1E5AB", "***": "#ADF6B1"}

# %%
STATE_TRANSITIONS = [("G1", "S"), ("S", "G2M")]

# %%
SAVE_DATA = True
SAVE_FIGURES = True

if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)
if SAVE_FIGURES:
    (FIG_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
adata = ad.io.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_processed.h5ad")
adata

# %% [markdown]
# ## Model loading

# %%
vae = REGVELOVI.load(DATA_DIR / DATASET / "regvelo_model", adata)

# %% [markdown]
# ## Fix the learned GRN (30 times)

# %%
REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")

# %%
W = torch.ones((adata.n_vars, adata.n_vars), dtype=int)

score_v0 = []
score_t0 = []
for _nrun in range(30):
    vae_r = REGVELOVI(adata, W=W.T * 0, soft_constraint=False)
    vae_r.module.v_encoder.fc1.weight.data = vae.module.v_encoder.fc1.weight.data.detach().cpu().clone()
    vae_r.module.v_encoder.fc1.bias.data = vae.module.v_encoder.fc1.bias.data.detach().cpu().clone()
    vae_r.train()

    set_output(adata, vae_r, n_samples=30, batch_size=adata.n_obs)

    ## calculate CBC
    vk = VelocityKernel(adata).compute_transition_matrix()

    cluster_key = "phase"
    rep = "X_pca"

    score_df = []
    for source, target in STATE_TRANSITIONS:
        cbc = vk.cbc(source=source, target=target, cluster_key=cluster_key, rep=rep)

        score_df.append(
            pd.DataFrame(
                {
                    "State transition": [f"{source} - {target}"] * len(cbc),
                    "CBC": cbc,
                }
            )
        )
    score_df = pd.concat(score_df)

    score_v0.append(score_df["CBC"].mean())
    print(score_v0[len(score_v0) - 1])

    ## calculate latent time correlation
    score_t0.append(
        get_time_correlation(ground_truth=adata.obs["fucci_time"], estimated=adata.layers["fit_t"].mean(axis=1))
    )
    print(score_t0[len(score_t0) - 1])

# %% [markdown]
# ## Randomly shuffled GRN

# %%
score_v1 = []
dfs = []
score_t1 = []
for _nrun in range(30):
    original_tensor = vae.module.v_encoder.fc1.weight.data.detach().cpu().clone()
    original_shape = original_tensor.shape
    # Flatten the tensor
    flattened_tensor = original_tensor.flatten()
    # Generate a random permutation of indices
    permutation = torch.randperm(flattened_tensor.size(0))
    # Shuffle the flattened tensor using the permutation
    shuffled_flattened_tensor = flattened_tensor[permutation]

    permutation = torch.randperm(flattened_tensor.size(0))
    shuffled_flattened_tensor = shuffled_flattened_tensor[permutation]
    # Reshape the shuffled tensor back to the original shape
    shuffled_tensor = shuffled_flattened_tensor.reshape(original_shape)

    vae_r = REGVELOVI(adata, W=W.T * 0, soft_constraint=False)
    vae_r.module.v_encoder.fc1.weight.data = shuffled_tensor
    vae_r.module.v_encoder.fc1.bias.data = vae.module.v_encoder.fc1.bias.data.detach().cpu().clone()
    vae_r.train()

    set_output(adata, vae_r, n_samples=30, batch_size=adata.n_obs)

    ## calculate CBC
    vk = VelocityKernel(adata).compute_transition_matrix()

    cluster_key = "phase"
    rep = "X_pca"

    score_df = []
    for source, target in STATE_TRANSITIONS:
        cbc = vk.cbc(source=source, target=target, cluster_key=cluster_key, rep=rep)

        score_df.append(
            pd.DataFrame(
                {
                    "State transition": [f"{source} - {target}"] * len(cbc),
                    "CBC": cbc,
                }
            )
        )
    score_df = pd.concat(score_df)

    score_v1.append(score_df["CBC"].mean())
    print(score_v1[len(score_v1) - 1])

    ## calculate latent time correlation
    score_t1.append(
        get_time_correlation(ground_truth=adata.obs["fucci_time"], estimated=adata.layers["fit_t"].mean(axis=1))
    )
    print(score_t1[len(score_t1) - 1])

# %% [markdown]
# ## Remove GRN

# %%
score_v2 = []
score_t2 = []
for _nrun in range(30):
    vae_r = REGVELOVI(adata, W=W.T * 0, soft_constraint=False)
    vae_r.module.v_encoder.fc1.weight.data = vae.module.v_encoder.fc1.weight.data.detach().cpu().clone() * 0
    vae_r.module.v_encoder.fc1.bias.data = vae.module.v_encoder.fc1.bias.data.detach().cpu().clone()
    vae_r.train()

    set_output(adata, vae_r, n_samples=30, batch_size=adata.n_obs)

    ## calculate CBC
    vk = VelocityKernel(adata).compute_transition_matrix()

    cluster_key = "phase"
    rep = "X_pca"

    score_df = []
    for source, target in STATE_TRANSITIONS:
        cbc = vk.cbc(source=source, target=target, cluster_key=cluster_key, rep=rep)

        score_df.append(
            pd.DataFrame(
                {
                    "State transition": [f"{source} - {target}"] * len(cbc),
                    "CBC": cbc,
                }
            )
        )
    score_df = pd.concat(score_df)

    score_v2.append(score_df["CBC"].mean())
    print(score_v2[len(score_v2) - 1])

    ## calculate latent time correlation
    score_t2.append(
        get_time_correlation(ground_truth=adata.obs["fucci_time"], estimated=adata.layers["fit_t"].mean(axis=1))
    )
    print(score_t2[len(score_t2) - 1])

# %%
df = pd.DataFrame(
    {"CBC score": score_v0 + score_v1 + score_v2, "Model": ["Original"] * 30 + ["Random"] * 30 + ["No regulation"] * 30}
)

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(3, 4))

    sns.violinplot(data=df, x="Model", y="CBC score", ax=ax)

    ttest_res = ttest_ind(score_v0, score_v1, equal_var=False, alternative="greater")
    significance = get_significance(ttest_res.pvalue)
    add_significance(
        ax=ax,
        left=0,
        right=1,
        significance=significance,
        lw=1,
        bracket_level=1.05,
        c="k",
        level=0,
    )

    ttest_res = ttest_ind(score_v0, score_v2, equal_var=False, alternative="greater")
    significance = get_significance(ttest_res.pvalue)
    add_significance(ax=ax, left=0, right=2, significance=significance, lw=1, c="k", level=2, bracket_level=0.95)

    y_min, y_max = ax.get_ylim()
    ax.set_ylim([y_min, y_max + 0.02])

    if SAVE_FIGURES:
        fig.savefig(FIG_DIR / DATASET / "CBC_score_GRN.svg", format="svg", transparent=True, bbox_inches="tight")

# %%
df = pd.DataFrame(
    {
        "Latent time correlation": score_t0 + score_t1 + score_t2,
        "Model": ["Original"] * 30 + ["Random"] * 30 + ["No regulation"] * 30,
    }
)

with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(3, 4))

    sns.violinplot(data=df, x="Model", y="Latent time correlation", ax=ax)

    ttest_res = ttest_ind(score_t0, score_t1, equal_var=False, alternative="greater")
    significance = get_significance(ttest_res.pvalue)
    add_significance(
        ax=ax,
        left=0,
        right=1,
        significance=significance,
        lw=1,
        bracket_level=1.05,
        c="k",
        level=0,
    )

    ttest_res = ttest_ind(score_t0, score_t2, equal_var=False, alternative="greater")
    significance = get_significance(ttest_res.pvalue)
    add_significance(ax=ax, left=0, right=2, significance=significance, lw=1, c="k", level=2, bracket_level=0.95)

    y_min, y_max = ax.get_ylim()
    ax.set_ylim([y_min, y_max + 0.02])

    if SAVE_FIGURES:
        fig.savefig(FIG_DIR / DATASET / "Time_GRN.svg", format="svg", transparent=True, bbox_inches="tight")

# %%

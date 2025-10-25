# %% [markdown]
# # Perform ablation study to the regvelo
#
# Select four different GRN to perform ablation including:
#  - prior GRN (binary matrix)
#  - SCENIC+ learned GRN
#  - RegVelo learned weight but shuffled GRN
#  - RegVelo learned GRN

# %% [markdown]
# ## Library imports


# %%
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

import scanpy as sc
import scvi
from regvelo import REGVELOVI

from rgv_tools import DATA_DIR, FIG_DIR
from rgv_tools.perturbation import get_list_name, TFScanning

# %% [markdown]
# ## General settings

# %%
# %matplotlib inline

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
SAVE_FIGURES = False
if SAVE_FIGURES:
    (FIG_DIR).mkdir(parents=True, exist_ok=True)

# %%
genes = ["nr2f5", "sox9b", "twist1b", "ets1"]

# %%
TERMINAL_STATES = [
    "mNC_head_mesenchymal",
    "mNC_arch2",
    "mNC_hox34",
    "Pigment",
]

# %%
MODEL = DATA_DIR / DATASET / "processed" / "rgv_model"


# %% [markdown]
# ## Define function

# %%
def shuffle_binary_grn(GRN: pd.DataFrame, TFs: list) -> pd.DataFrame:
    """
    
    Shuffle a binary GRN matrix by randomizing:
      - Only the TF (row) labels among themselves.
      - All column (target gene) labels.
    The GRN matrix values remain unchanged.

    Parameters
    ----------
    GRN : pd.DataFrame
        Square gene regulatory network matrix (regulators × targets)
    TFs : list
        List of transcription factor gene names (subset of GRN.index)
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        GRN with TF row labels shuffled and all column labels shuffled.

    """
    new_row_labels = list(GRN.index)
    new_col_labels = list(GRN.columns)

    shuffled_TF_labels = list(np.random.permutation(TFs))
    for old, new in zip(TFs, shuffled_TF_labels):
        idx = new_row_labels.index(old)
        new_row_labels[idx] = new

    shuffled_col_labels = list(np.random.permutation(new_col_labels))

    shuffled_GRN = GRN.copy()
    shuffled_GRN.index = new_row_labels
    shuffled_GRN.columns = shuffled_col_labels

    return shuffled_GRN


# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed.h5ad")

# %% [markdown]
# ## Trained regvelo model

# %%
# Prepare skeleton
W = adata.uns["skeleton"].copy()
W = torch.tensor(np.array(W)).int()

# Prepare TF
TF = adata.var_names[adata.var["TF"]]

# Prepare model
REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
vae = REGVELOVI(adata, W=W.T, regulators=TF, soft_constraint=False)

# %%
vae.train()

# %%
vae.save(MODEL)

# %% [markdown]
# ## Prepare GRN

# %% [markdown]
# ### Prepare binary GRN (prior GRN)

# %%
W = adata.uns["skeleton"].copy()
binary = torch.tensor(np.array(W)).int()

# %% [markdown]
# ### Prepare SCENIC+ GRN (weighted prior GRN)

# %%
net = pd.read_csv(DATA_DIR / DATASET / "raw" / "eRegulon_metadata_all.csv", index_col=0)

# %%
grn = pd.DataFrame({"source": net["TF"], "target": net["Gene"], "weight": net["TF2G_rho"]})

# %%
matrix = grn.pivot_table(index="source", columns="target", values="weight", fill_value=0)
# Ensure all genes (TFs and targets) appear as both rows and columns
genes = sorted(set(grn["source"]) | set(grn["target"]))
matrix = matrix.reindex(index=genes, columns=genes, fill_value=0)

# %%
matrix = matrix.loc[W.index, W.columns].copy()
np.fill_diagonal(matrix.values, 0)
scenic_grn = torch.tensor(np.array(matrix)).int()

# %%
matrix[W == 0] = 0

# %% [markdown]
# ### Prepare random shuffled GRN

# %%
vae = REGVELOVI.load(MODEL, adata)

# %%
W = pd.DataFrame(vae.module.v_encoder.fc1.weight.data.cpu().numpy(), index=adata.var_names, columns=adata.var_names)

# %%
randomized = W.copy()
mask = W != 0
weights = np.array(W)[mask]
np.random.shuffle(weights)

randomized.values[mask] = weights

# %% [markdown]
# ## Trained Model (fixed prior GRN)

# %%
scvi.settings.seed = 0

# %%
vae_b = REGVELOVI(adata, W=binary.T * 0, regulators=TF, soft_constraint=False)
vae_b.module.v_encoder.fc1.weight.data = torch.tensor(binary.T, dtype=torch.float32)
vae_b.train()

# %%
## Randomized GRN
for nrun in range(10):
    filename_br = "rgv_model_b_random" + f"_{nrun+1}"
    MODEL_B_R = DATA_DIR / DATASET / "processed" / filename_br

    # Prepare skeleton
    W = adata.uns["skeleton"].copy()
    W_r = shuffle_binary_grn(W, W.index[W.sum(1) != 0])
    W_r = W_r.loc[W.index, W.columns]

    binary_r = torch.tensor(np.array(W_r)).int()

    # Prepare model
    vae_br = REGVELOVI(adata, W=binary_r.T * 0, regulators=TF, soft_constraint=False)
    vae_br.module.v_encoder.fc1.weight.data = torch.tensor(binary_r.T, dtype=torch.float32)
    vae_br.train()

    vae_br.save(MODEL_B_R)

# %% [markdown]
# ## Trained Model (fixed SCENIC+ weighted prior GRN)

# %%
scvi.settings.seed = 0

# %%
vae_s = REGVELOVI(adata, W=scenic_grn.T * 0, regulators=TF, soft_constraint=False)
vae_s.module.v_encoder.fc1.weight.data = torch.tensor(scenic_grn.T, dtype=torch.float32)
vae_s.train()

# %% [markdown]
# ## Trained Model (shuffled GRN)

# %%
scvi.settings.seed = 0

# %%
randomized = torch.tensor(np.array(randomized))

# %%
vae_r = REGVELOVI(adata, W=randomized * 0, regulators=TF, soft_constraint=False)
vae_r.module.v_encoder.fc1.weight.data = torch.tensor(randomized, dtype=torch.float32)
vae_r.train()

# %%
MODEL_B = DATA_DIR / DATASET / "processed" / "rgv_model_b"
MODEL_S = DATA_DIR / DATASET / "processed" / "rgv_model_s"
MODEL_R = DATA_DIR / DATASET / "processed" / "rgv_model_r"

# %%
vae_b.save(MODEL_B)
vae_s.save(MODEL_S)
vae_r.save(MODEL_R)

# %% [markdown]
# ## Driver ranking

# %%
perturb_screening = TFScanning(MODEL, adata, 7, "cell_type", TERMINAL_STATES, TF, 0)

# %%
perturb_screening_b = TFScanning(MODEL_B, adata, 7, "cell_type", TERMINAL_STATES, TF, 0)

# %%
perturb_screening_s = TFScanning(MODEL_S, adata, 7, "cell_type", TERMINAL_STATES, TF, 0)

# %%
perturb_screening_r = TFScanning(MODEL_R, adata, 7, "cell_type", TERMINAL_STATES, TF, 0)

# %%
coef_name = "coef_raw"
coef_save = DATA_DIR / DATASET / "results" / coef_name

coef = pd.DataFrame(np.array(perturb_screening["coefficient"]))
coef.index = perturb_screening["TF"]
coef.columns = get_list_name(perturb_screening["coefficient"][0])

pval = pd.DataFrame(np.array(perturb_screening["pvalue"]))
pval.index = perturb_screening["TF"]
pval.columns = get_list_name(perturb_screening["pvalue"][0])

rows_with_nan = coef.isna().any(axis=1)
# Set all values in those rows to NaN
coef.loc[rows_with_nan, :] = np.nan
pval.loc[rows_with_nan, :] = np.nan

coef.to_csv(coef_save)

# %%
coef_name = "coef_binary"
coef_save_b = DATA_DIR / DATASET / "results" / coef_name

coef = pd.DataFrame(np.array(perturb_screening_b["coefficient"]))
coef.index = perturb_screening_b["TF"]
coef.columns = get_list_name(perturb_screening_b["coefficient"][0])

pval = pd.DataFrame(np.array(perturb_screening_b["pvalue"]))
pval.index = perturb_screening_b["TF"]
pval.columns = get_list_name(perturb_screening_b["pvalue"][0])

rows_with_nan = coef.isna().any(axis=1)
# Set all values in those rows to NaN
coef.loc[rows_with_nan, :] = np.nan
pval.loc[rows_with_nan, :] = np.nan

coef.to_csv(coef_save_b)

# %%
coef_name = "coef_scenic"
coef_save_s = DATA_DIR / DATASET / "results" / coef_name

coef = pd.DataFrame(np.array(perturb_screening_s["coefficient"]))
coef.index = perturb_screening_s["TF"]
coef.columns = get_list_name(perturb_screening_s["coefficient"][0])

pval = pd.DataFrame(np.array(perturb_screening_s["pvalue"]))
pval.index = perturb_screening_s["TF"]
pval.columns = get_list_name(perturb_screening_s["pvalue"][0])

rows_with_nan = coef.isna().any(axis=1)
# Set all values in those rows to NaN
coef.loc[rows_with_nan, :] = np.nan
pval.loc[rows_with_nan, :] = np.nan

coef.to_csv(coef_save_s)

# %%
coef_name = "coef_random"
coef_save_r = DATA_DIR / DATASET / "results" / coef_name

coef = pd.DataFrame(np.array(perturb_screening_r["coefficient"]))
coef.index = perturb_screening_r["TF"]
coef.columns = get_list_name(perturb_screening_r["coefficient"][0])

pval = pd.DataFrame(np.array(perturb_screening_r["pvalue"]))
pval.index = perturb_screening_r["TF"]
pval.columns = get_list_name(perturb_screening_r["pvalue"][0])

rows_with_nan = coef.isna().any(axis=1)
# Set all values in those rows to NaN
coef.loc[rows_with_nan, :] = np.nan
pval.loc[rows_with_nan, :] = np.nan

coef.to_csv(coef_save_r)

# %% [markdown]
# ## Repeatively to run regvelo with another nine times

# %%
## Randomized GRN
for nrun in range(9):
    filename = "rgv_model" + f"_{nrun+1}"
    filename_r = "rgv_model_r" + f"_{nrun+1}"

    MODEL = DATA_DIR / DATASET / "processed" / filename
    MODEL_R = DATA_DIR / DATASET / "processed" / filename_r

    # Prepare skeleton
    W = adata.uns["skeleton"].copy()
    W = torch.tensor(np.array(W)).int()

    # Prepare TF
    TF = adata.var_names[adata.var["TF"]]

    # Prepare model
    REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    vae = REGVELOVI(adata, W=W.T, regulators=TF, soft_constraint=False)
    vae.train()

    ### randomize GRN
    W = pd.DataFrame(vae.module.v_encoder.fc1.weight.data.cpu().numpy(), index=adata.var_names, columns=adata.var_names)
    randomized = W.copy()
    mask = W != 0
    weights = np.array(W)[mask]
    np.random.shuffle(weights)

    randomized.values[mask] = weights
    randomized = torch.tensor(np.array(randomized))

    vae_r = REGVELOVI(adata, W=randomized * 0, regulators=TF, soft_constraint=False)
    vae_r.module.v_encoder.fc1.weight.data = torch.tensor(randomized, dtype=torch.float32)
    vae_r.train()

    vae.save(MODEL)
    vae_r.save(MODEL_R)

# %%
perturb_screening_list = {}
perturb_screening_r_list = {}

for nrun in range(9):
    filename = "rgv_model" + f"_{nrun+1}"
    filename_r = "rgv_model_r" + f"_{nrun+1}"

    MODEL = DATA_DIR / DATASET / "processed" / filename
    MODEL_R = DATA_DIR / DATASET / "processed" / filename_r

    perturb_screening = TFScanning(MODEL, adata, 7, "cell_type", TERMINAL_STATES, TF, 0)
    perturb_screening_r = TFScanning(MODEL_R, adata, 7, "cell_type", TERMINAL_STATES, TF, 0)

    ## pertub coef
    coef_name = "coef_raw" + f"_{nrun+1}"
    coef_save = DATA_DIR / "results" / coef_name

    coef = pd.DataFrame(np.array(perturb_screening["coefficient"]))
    coef.index = perturb_screening["TF"]
    coef.columns = get_list_name(perturb_screening["coefficient"][0])

    rows_with_nan = coef.isna().any(axis=1)
    # Set all values in those rows to NaN
    coef.loc[rows_with_nan, :] = np.nan
    coef.to_csv(coef_save)
    perturb_screening_list[f"{nrun+1}"] = coef

    ## pertub coef_random
    coef_name = "coef_random" + f"_{nrun+1}"
    coef_save = DATA_DIR / "results" / coef_name

    coef = pd.DataFrame(np.array(perturb_screening_r["coefficient"]))
    coef.index = perturb_screening_r["TF"]
    coef.columns = get_list_name(perturb_screening_r["coefficient"][0])

    rows_with_nan = coef.isna().any(axis=1)
    # Set all values in those rows to NaN
    coef.loc[rows_with_nan, :] = np.nan
    coef.to_csv(coef_save)
    perturb_screening_r_list[f"{nrun+1}"] = coef

# %%
perturb_screening_b_r_list = {}

for nrun in range(10):
    filename_br = "rgv_model_b_random" + f"_{nrun+1}"
    MODEL_B_R = DATA_DIR / DATASET / "processed" / filename_br

    perturb_screening_b_r = TFScanning(MODEL_B_R, adata, 7, "cell_type", TERMINAL_STATES, TF, 0)

    ## pertub coef
    coef_name = "coef_binary_random_update" + f"_{nrun+1}"
    coef_save = DATA_DIR / "results" / coef_name

    coef = pd.DataFrame(np.array(perturb_screening_b_r["coefficient"]))
    coef.index = perturb_screening_b_r["TF"]
    coef.columns = get_list_name(perturb_screening_b_r["coefficient"][0])

    rows_with_nan = coef.isna().any(axis=1)
    # Set all values in those rows to NaN
    coef.loc[rows_with_nan, :] = np.nan
    coef.to_csv(coef_save)
    perturb_screening_b_r_list[f"{nrun+1}"] = coef

# %% [markdown]
# ## Results

# %%
coef_name = "coef_raw"
coef_save = DATA_DIR / DATASET / "results" / coef_name

coef_name = "coef_binary"
coef_save_b = DATA_DIR / DATASET / "results" / coef_name

coef_name = "coef_scenic"
coef_save_s = DATA_DIR / DATASET / "results" / coef_name

coef_name = "coef_random"
coef_save_r = DATA_DIR / DATASET / "results" / coef_name

driver_head_mes = ["nr2f5", "nr2f2", "sox9b", "twist1a", "twist1b"]
driver_pigment = ["sox10", "mitfa", "tfec", "bhlhe40", "tfap2b", "tfap2a"]

# %%
score_head_mes = []

coef = pd.read_csv(coef_save, index_col=0)
label_rgv = [1 if i in driver_head_mes else 0 for i in coef.index.tolist()]
score_head_mes.append(roc_auc_score(label_rgv, coef["mNC_head_mesenchymal"]))

for nrun in range(9):
    coef_name = "coef_raw" + f"_{nrun+1}"
    coef = pd.read_csv(DATA_DIR / "results" / coef_name, index_col=0)
    label_rgv = [1 if i in driver_head_mes else 0 for i in coef.index.tolist()]
    score_head_mes.append(roc_auc_score(label_rgv, coef["mNC_head_mesenchymal"]))

coef = pd.read_csv(coef_save_b, index_col=0)
label_rgv = [1 if i in driver_head_mes else 0 for i in coef.index.tolist()]
score_head_mes.append(roc_auc_score(label_rgv, coef["mNC_head_mesenchymal"]))

for nrun in range(10):
    coef_name = "coef_binary_random_update" + f"_{nrun+1}"
    coef = pd.read_csv(DATA_DIR / "results" / coef_name, index_col=0)
    label_rgv = [1 if i in driver_head_mes else 0 for i in coef.index.tolist()]
    score_head_mes.append(roc_auc_score(label_rgv, coef["mNC_head_mesenchymal"]))

coef = pd.read_csv(coef_save_s, index_col=0)
label_rgv = [1 if i in driver_head_mes else 0 for i in coef.index.tolist()]
score_head_mes.append(roc_auc_score(label_rgv, coef["mNC_head_mesenchymal"]))

coef = pd.read_csv(coef_save_r, index_col=0)
label_rgv = [1 if i in driver_head_mes else 0 for i in coef.index.tolist()]
score_head_mes.append(roc_auc_score(label_rgv, coef["mNC_head_mesenchymal"]))

for nrun in range(9):
    coef_name = "coef_random" + f"_{nrun+1}"
    coef = pd.read_csv(DATA_DIR / "results" / coef_name, index_col=0)
    label_rgv = [1 if i in driver_head_mes else 0 for i in coef.index.tolist()]
    score_head_mes.append(roc_auc_score(label_rgv, coef["mNC_head_mesenchymal"]))

# %%
score_pigment = []

coef = pd.read_csv(coef_save, index_col=0)
label_rgv = [1 if i in driver_pigment else 0 for i in coef.index.tolist()]
score_pigment.append(roc_auc_score(label_rgv, coef["Pigment"]))

for nrun in range(9):
    coef_name = "coef_raw" + f"_{nrun+1}"
    coef = pd.read_csv(DATA_DIR / "results" / coef_name, index_col=0)
    label_rgv = [1 if i in driver_pigment else 0 for i in coef.index.tolist()]
    score_pigment.append(roc_auc_score(label_rgv, coef["Pigment"]))

coef = pd.read_csv(coef_save_b, index_col=0)
label_rgv = [1 if i in driver_pigment else 0 for i in coef.index.tolist()]
score_pigment.append(roc_auc_score(label_rgv, coef["Pigment"]))

for nrun in range(10):
    coef_name = "coef_binary_random_update" + f"_{nrun+1}"
    coef = pd.read_csv(DATA_DIR / "results" / coef_name, index_col=0)
    label_rgv = [1 if i in driver_pigment else 0 for i in coef.index.tolist()]
    score_pigment.append(roc_auc_score(label_rgv, coef["Pigment"]))

coef = pd.read_csv(coef_save_s, index_col=0)
label_rgv = [1 if i in driver_pigment else 0 for i in coef.index.tolist()]
score_pigment.append(roc_auc_score(label_rgv, coef["Pigment"]))

coef = pd.read_csv(coef_save_r, index_col=0)
label_rgv = [1 if i in driver_pigment else 0 for i in coef.index.tolist()]
score_pigment.append(roc_auc_score(label_rgv, coef["Pigment"]))

for nrun in range(9):
    coef_name = "coef_random" + f"_{nrun+1}"
    coef = pd.read_csv(DATA_DIR / "results" / coef_name, index_col=0)
    label_rgv = [1 if i in driver_pigment else 0 for i in coef.index.tolist()]
    score_pigment.append(roc_auc_score(label_rgv, coef["Pigment"]))

# %%
score = (np.array(score_head_mes) + np.array(score_pigment)) / 2

# %%
df = pd.DataFrame(
    {
        "AUROC": score,
        "method": ["RegVelo (PS)"] * 10
        + ["Prior GRN (Binary)"]
        + ["Prior GRN (Binary, random)"] * 10
        + ["Prior GRN (SCENIC+ weight)"]
        + ["RegVelo (Randomized)"] * 10,
    }
)

# %%
with mplscience.style_context():
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots(figsize=(5, 3))
    order = [
        "Prior GRN (Binary)",
        "Prior GRN (Binary, random)",
        "Prior GRN (SCENIC+ weight)",
        "RegVelo (PS)",
        "RegVelo (Randomized)",
    ]
    # Plot the barplot without error bars
    sns.barplot(data=df, x="method", y="AUROC", hue="method", order=order, ax=ax, ci=None)

    # Add jittered dots
    sns.stripplot(
        data=df, x="method", y="AUROC", hue="method", order=order, dodge=False, color="black", ax=ax, jitter=True
    )

    # Remove the duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[3:6], labels[3:6], bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=2)

    # Customize labels and other settings
    ax.set(ylabel="", xlabel="AUROC")
    ax.set_xlabel(xlabel="AUROC", fontsize=13)

    if SAVE_FIGURES:
        plt.savefig(
            FIG_DIR / DATASET / "AUROC_ranking_results.svg", format="svg", transparent=True, bbox_inches="tight"
        )
    plt.show()

# %%
df

# %%

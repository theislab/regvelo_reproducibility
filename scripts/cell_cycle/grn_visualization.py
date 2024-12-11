# %% [markdown]
# # Visualize GRN

# %%
import networkx as nx
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import scvelo as scv
import scvi
from regvelo import REGVELOVI

from rgv_tools import DATA_DIR, FIG_DIR

# %% [markdown]
# ## General settings

# %%
scvi.settings.dl_pin_memory_gpu_training = False

# %%
plt.rcParams["svg.fonttype"] = "none"

# %%
sns.reset_defaults()
sns.reset_orig()
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=14, color_map="viridis")

# %%
SAVE_FIGURES = True
if SAVE_FIGURES:
    (FIG_DIR / "cell_cycle").mkdir(parents=True, exist_ok=True)

SAVE_DATASETS = False
if SAVE_DATASETS:
    (DATA_DIR / "cell_cycle").mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Function definitions


# %%
def add_regvelo_outputs_to_adata(adata_raw, vae, n_samples=30):
    """TODO."""
    latent_time = vae.get_latent_time(n_samples=n_samples, time_statistic="mean", batch_size=adata_raw.shape[0])
    velocities = vae.get_velocity(n_samples=n_samples, velo_statistic="mean", batch_size=adata_raw.shape[0])

    t = latent_time
    scaling = 20 / t.max(0)
    adata = adata_raw[:, vae.module.target_index].copy()

    adata.layers["velocity"] = velocities / scaling
    adata.layers["latent_time_regvelo"] = latent_time

    adata.layers["fit_t"] = latent_time.values * scaling[np.newaxis, :]
    adata.var["fit_scaling"] = 1.0

    return adata


# %% [markdown]
# ## Load datasets

# %%
adata = sc.read_h5ad(DATA_DIR / "cell_cycle" / "cell_cycle_processed.h5ad")

# %%
reg_bdata = adata.copy()
reg_bdata.uns["regulators"] = adata.var.index.values
reg_bdata.uns["targets"] = adata.var.index.values
reg_bdata.uns["skeleton"] = np.ones((len(adata.var.index), len(adata.var.index)))
reg_bdata

# %%
TF = pd.read_csv("RegVelo_datasets/cell cycle/allTFs_hg38.txt", header=None)
TF = reg_bdata.var_names[reg_bdata.var_names.isin(TF.iloc[:, 0])]

# %% [markdown]
# ## Identify most activate TF

# %%
reg_vae = REGVELOVI.load(DATA_DIR / "cell_cycle" / "model_1", reg_bdata)
GRN = (
    reg_vae.module.v_encoder.GRN_Jacobian(torch.tensor(reg_bdata.layers["Ms"].mean(0)).to("cuda:0"))
    .detach()
    .cpu()
    .numpy()
)
pd.DataFrame(
    (GRN[:, reg_bdata.var.index.isin(TF)]).mean(0), index=reg_bdata.var.index[reg_bdata.var.index.isin(TF)]
).sort_values(0, ascending=False).iloc[:10, :]

# %% [markdown]
# ## Visualize GRN

# %%
targets = GRN[:, [i == "TGIF1" for i in reg_bdata.var.index]].reshape(-1)
prior = 0

# %%
targets = pd.DataFrame(targets, index=reg_bdata.var.index)

# %%
targets.loc[:, "weight"] = targets.iloc[:, 0].abs()
targets.loc[:, "prior"] = prior

# %%
GRN_visualize_tgif1 = targets.sort_values("weight", ascending=False).iloc[:50, :]

# %%
targets = GRN[:, [i == "ETV1" for i in reg_bdata.var.index]].reshape(-1)
prior = 0

# %%
targets = pd.DataFrame(targets, index=reg_bdata.var.index)

# %%
targets.loc[:, "weight"] = targets.iloc[:, 0].abs()
targets.loc[:, "prior"] = prior

# %%
GRN_visualize_etv1 = targets.sort_values("weight", ascending=False).iloc[:50, :]

# %%
df1 = pd.DataFrame(
    {"from": ["TGIF1"] * 50, "to": GRN_visualize_tgif1.index.tolist(), "status": GRN_visualize_tgif1.loc[:, "prior"]}
)

df2 = pd.DataFrame(
    {"from": ["ETV1"] * 50, "to": GRN_visualize_etv1.index.tolist(), "status": GRN_visualize_etv1.loc[:, "prior"]}
)

# %%
df = pd.concat([df1, df2], axis=0)

# %%
list1 = ["TGIF1"] + GRN_visualize_tgif1.index.tolist()
list2 = ["ETV1"] + GRN_visualize_etv1.index.tolist()

# Define communities
community1 = list(set(list1) - set(list2))  # Nodes only in list1
community2 = list(set(list2) - set(list1))  # Nodes only in list2
community3 = list(set(list1).intersection(list2))  # Nodes shared by both
communities = [frozenset(set(community1)), frozenset(set(community2)), frozenset(set(community3))]

# %%
G = nx.from_pandas_edgelist(df, source="from", target="to", create_using=nx.DiGraph())

# %%
if SAVE_FIGURES:
    fig, ax = plt.subplots(figsize=(6, 4))
    supergraph = nx.cycle_graph(len(communities))
    superpos = nx.spring_layout(G, scale=3)

    # Use the "supernode" positions as the center of each node cluster
    centers = list(superpos.values())
    pos = {}
    for center, comm in zip(centers, communities):
        pos.update(nx.spring_layout(nx.subgraph(G, comm), center=center))

    # Nodes colored by cluster
    node_list = ["TGIF1", "ETV1", "BUB1", "TOP2A", "RRM2", "WDHD1", "TFDP1"]
    for nodes, clr in zip(communities, ("tab:blue", "tab:blue", "tab:green")):
        nx.draw_networkx_nodes(G, pos=pos, nodelist=nodes, node_color=clr, node_size=100)
        nx.draw_networkx_labels(G, pos=pos, labels={node: node for node in node_list}, font_size=14, font_color="black")
    nx.draw_networkx_edges(G, pos=pos, edge_color="lightgrey")

    plt.tight_layout()

    fig.savefig(FIG_DIR / "cell_cycle" / "O2SC_GRN.svg", format="svg", transparent=True, bbox_inches="tight")
    plt.show()

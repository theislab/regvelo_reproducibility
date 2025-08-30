# %% [markdown]
# # Reannotation of pigment cells
#
# In our smart-seq3, although we know our pigment cells mainly express gch2 gene, we still need to annotate the actual gch2+ pigment cells to make sure our evaluated cell state align to the cell state in perturb-seq. Therefore, we could integrate the pigment cell population in perturb-seq and smart-seq3, and transfer perturb-seq label on smart-seq3 dataset.

# %% [markdown]
# ## Library imports
#
#
# import faiss
#
# import numpy as np
# import pandas as pd
#
# import matplotlib.pyplot as plt
# import seaborn as sns

# %%
import scanpy as sc
import scanpy.external as sce
import scvelo as scv

from rgv_tools import DATA_DIR

# %% [markdown]
# ## General setting

# %%
plt.rcParams["svg.fonttype"] = "none"
sns.reset_defaults()
sns.reset_orig()
scv.settings.set_figure_params("scvelo", dpi_save=400, dpi=80, transparent=True, fontsize=14, color_map="viridis")

# %% [markdown]
# ## Constants

# %%
DATASET = "zebrafish"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data loading
#
# Note: The perturb-seq datasets will be released after the paper is officially online. Users can access them via `regvelo.datasets.zebrafish_perturb`.

# %%
perturb_seq = sc.read_h5ad(DATA_DIR / DATASET / "raw" / "perturbseq_all.h5ad")
ss3 = sc.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed.h5ad")

# %%
sc.pl.scatter(ss3, basis="umap", color="cell_type")

# %%
sc.pl.scatter(perturb_seq, basis="phate", color="cell_anno")

# %% [markdown]
# ## Integrate ss3 and perturb seq dataset

# %%
ss3_p = ss3[ss3.obs["cell_type"] == "Pigment"]

# %%
ref_p = perturb_seq[perturb_seq.obs["sgRNA_group"] == "control"].copy()
ref_p = ref_p[ref_p.obs["cell_anno"].isin(["Pigment", "Pigment_gch2_high"])].copy()

# %%
sc.pp.scale(ref_p)
sc.pp.scale(ss3_p)

# %%
gene = list(set(ss3_p.var_names).intersection(ref_p.var_names))

merged_adata = ref_p[:, gene].concatenate(ss3_p[:, gene], batch_key="batch")

# %%
sc.tl.pca(merged_adata)

# %%
sce.pp.harmony_integrate(merged_adata, "batch")

# %%
sc.pp.neighbors(merged_adata, n_pcs=30, use_rep="X_pca_harmony")

# %%
sc.tl.umap(merged_adata)
sc.pl.umap(
    merged_adata,
    color="batch",
    # Setting a smaller point size to get prevent overlap
)

# %%
sc.pl.umap(
    merged_adata,
    color=["cell_type", "cell_anno"],
    # Setting a smaller point size to get prevent overlap
)

# %%
X = merged_adata.obsm["X_pca_harmony"].astype("float32")

# build FAISS KNN index
n_cells = X.shape[0]
k = 10

index = faiss.IndexFlatL2(X.shape[1])
index.add(X)
distances, neighbors = index.search(X, k)

# project neighbors cell type label
sgRNA_labels = merged_adata.obs["cell_anno"].values
neighbor_labels = sgRNA_labels[neighbors]

# %%
mutant_mask = (neighbor_labels != "control").astype(int)
mutant_neighbor_counts = mutant_mask.sum(axis=1)

# %%
score = (np.array(neighbor_labels[merged_adata.obs["cell_type"].isin(["Pigment"])]) == "Pigment_gch2_high").astype(int)
score = score.sum(axis=1) / 10

# %%
annotation = pd.DataFrame(merged_adata.obs["cell_type"][merged_adata.obs["cell_type"].isin(["Pigment"])])

# %%
annotation.loc[score >= 0.7, "pigment_annotation"] = "Pigment_gch2"

# %% [markdown]
# ## Save data

# %%
if SAVE_DATA:
    annotation.to_csv(DATA_DIR / DATASET / "processed" / "annotation.csv")

# %%

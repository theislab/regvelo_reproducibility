# %% [markdown]
# # Construct pySCENIC GRN
#
# We used pySCENIC to construct a gene regulatory network (GRN) for each scale dataset. For simplicity of demonstration, we present the preprocessing steps for scale-2; the preprocessing for the other scales is identical.

# %% [markdown]
# ## Library imports

import glob

# %%
# import dependencies

import loompy as lp

import pandas as pd

import anndata as ad
import scanpy as sc

from rgv_tools import DATA_DIR

# %% [markdown]
# ## General settings

# %%
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)

# %% [markdown]
# ## Constants

# %%
DATASET = "mouse_neural_crest"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / DATASET / "processed" / "scenic").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
adata_raw = ad.io.read_h5ad(DATA_DIR / DATASET / "raw" / "GSE201257_adata_QC_filtered.h5ad")
adata = ad.io.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_stage2_processed.h5ad")
adata = adata_raw[adata.obs_names]

# %%
adata

# %%
sc.pp.filter_genes(adata, min_cells=5)

# %%
sc.pp.normalize_total(adata, target_sum=1e3)
sc.pp.log1p(adata)

# %%
adata = sc.AnnData(adata.X, obs=adata.obs, var=adata.var)
adata.var["Gene"] = adata.var_names
adata.obs["CellID"] = adata.obs_names

# %%
if SAVE_DATA:
    adata.write_loom(DATA_DIR / DATASET / "processed" / "scenic" / "adata_stage_2_check.loom")

# %%
adata.X = adata.X.toarray().copy()

# %% [markdown]
# ## SCENIC step

# %%
f_loom_path_scenic = DATA_DIR / DATASET / "processed" / "scenic" / "adata_stage_2_check.loom"
f_tfs = "allTFs_mm.txt"
adj_path = DATA_DIR / DATASET / "processed" / "scenic" / "adj_stage_2.csv"

# %%
# !pyscenic grn {f_loom_path_scenic} {f_tfs} -o {adj_path} --num_workers 30

# %%
# ranking databases
f_db_glob = "cisTarget_databases/*feather"  ## download feather file according instructions of pySCENIC
f_db_names = " ".join(glob.glob(f_db_glob))

# motif databases
f_motif_path = (
    "cisTarget_databases/motifs-v9-nr.mgi-m0.001-o0.0.tbl"  ## download motif file according instructions of pySCENIC
)

regulon_path = DATA_DIR / DATASET / "processed" / "scenic" / "stage_2_all_regulons.csv"

# %%
# !pyscenic ctx adj_path \
#     {f_db_names} \
#     --annotations_fname {f_motif_path} \
#     --expression_mtx_fname {f_loom_path_scenic} \
#     --output {regulon_path} \
#     --all_modules \
#     --num_workers 30

# %%
f_pyscenic_output = DATA_DIR / DATASET / "processed" / "scenic" / "pyscenic_output_stage_2_all_regulons.loom"

# %%
# !pyscenic aucell \
#     {f_loom_path_scenic} \
#     {regulon_path} \
#     --output {f_pyscenic_output} \
#     --num_workers 2

# %%
lf = lp.connect(f_pyscenic_output, mode="r+", validate=False)
exprMat = pd.DataFrame(lf[:, :], index=lf.ra.Gene, columns=lf.ca.CellID)
auc_mtx = pd.DataFrame(lf.ca.RegulonsAUC, index=lf.ca.CellID)
regulons = lf.ra.Regulons

# %%
res = pd.concat([pd.Series(r.tolist(), index=regulons.dtype.names) for r in regulons], axis=1)

# %%
res.columns = lf.row_attrs["var_names"]

# %% [markdown]
# ## Save data

# %%
if SAVE_DATA:
    res.to_csv(DATA_DIR / DATASET / "processed" / "regulon_mat_stage_2_all_regulons.csv")

# %%

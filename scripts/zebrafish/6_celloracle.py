# %% [markdown]
# # CellOracle-based perturbation prediction
#
# Notebook predicts TF perturbation effects with CellOracle.

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd

import celloracle as co
import scanpy as sc
import scvelo as scv
from celloracle.applications import Gradient_calculator, Oracle_development_module
from scvelo import logging as logg

from rgv_tools import DATA_DIR

# %% [markdown]
# ## Function definitions


# %%
def split_elements(character_list):
    """TODO."""
    result_list = []
    for element in character_list:
        if "_" in element:
            parts = element.split("_")
            result_list.append(parts)
        else:
            result_list.append([element])
    return result_list


# %%
def combine_elements(split_list):
    """TODO."""
    result_list = []
    for parts in split_list:
        combined_element = "_".join(parts)
        result_list.append(combined_element)
    return result_list


# %%
def pipeline(gene_for_KO):
    """TODO."""
    # 1. Simulate KO
    oracle.simulate_shift(perturb_condition={gene_for_KO: 0}, ignore_warning=True, n_propagation=3)
    oracle.estimate_transition_prob(n_neighbors=n_neighbors, knn_random=True, sampled_fraction=1)
    oracle.calculate_embedding_shift(sigma_corr=0.05)

    ## calculate overall score
    dev = Oracle_development_module()
    # Load development flow
    dev.load_differentiation_reference_data(gradient_object=gradient)
    # Load simulation result
    dev.load_perturb_simulation_data(oracle_object=oracle, n_neighbors=n_neighbors)
    # Calculate inner product
    dev.calculate_inner_product()
    dev.calculate_digitized_ip(n_bins=10)

    # Save results in a hdf5 file.
    ps_max = dev.get_negative_PS_p_value(return_ps_sum=True, plot=False)[1]

    # Do simulation for all conditions.
    Lineage = []
    PS_score = []
    for lineage_name, cell_idx in index_dictionary.items():
        dev = Oracle_development_module()
        # Load development flow
        dev.load_differentiation_reference_data(gradient_object=gradient)
        # Load simulation result
        dev.load_perturb_simulation_data(
            oracle_object=oracle, cell_idx_use=cell_idx, name=lineage_name, n_neighbors=n_neighbors
        )
        # Calculate inner product
        dev.calculate_inner_product()
        dev.calculate_digitized_ip(n_bins=10)

        # Save results in a hdf5 file.
        ps = dev.get_negative_PS_p_value(return_ps_sum=True, plot=False)[1]
        ps = ps / ps_max
        Lineage.append(lineage_name)
        PS_score.append(ps)

    df = pd.DataFrame({"Lineage": Lineage, "PS_score": PS_score})
    df.index = Lineage

    return df


# %%
def pipeline2(gene_for_KO):
    """TODO."""
    oracle.simulate_shift(perturb_condition={key: 0 for key in gene_for_KO}, ignore_warning=True, n_propagation=3)
    oracle.estimate_transition_prob(n_neighbors=n_neighbors, knn_random=True, sampled_fraction=1)
    oracle.calculate_embedding_shift(sigma_corr=0.05)

    ## calculate overall score
    dev = Oracle_development_module()
    # Load development flow
    dev.load_differentiation_reference_data(gradient_object=gradient)
    # Load simulation result
    dev.load_perturb_simulation_data(oracle_object=oracle, n_neighbors=n_neighbors)
    # Calculate inner product
    dev.calculate_inner_product()
    dev.calculate_digitized_ip(n_bins=10)

    # Save results in a hdf5 file.
    ps_max = dev.get_negative_PS_p_value(return_ps_sum=True, plot=False)[1]

    # Do simulation for all conditions.
    Lineage = []
    PS_score = []
    for lineage_name, cell_idx in index_dictionary.items():
        dev = Oracle_development_module()
        # Load development flow
        dev.load_differentiation_reference_data(gradient_object=gradient)
        # Load simulation result
        dev.load_perturb_simulation_data(
            oracle_object=oracle, cell_idx_use=cell_idx, name=lineage_name, n_neighbors=n_neighbors
        )
        # Calculate inner product
        dev.calculate_inner_product()
        dev.calculate_digitized_ip(n_bins=10)

        # Save results in a hdf5 file.
        ps = dev.get_negative_PS_p_value(return_ps_sum=True, plot=False)[1]
        ps = ps / ps_max
        Lineage.append(lineage_name)
        PS_score.append(ps)

    df = pd.DataFrame({"Lineage": Lineage, "PS_score": PS_score})
    df.index = Lineage

    return df


# %%
def TFScanning_perturbation(adata, n_states, cluster_label, terminal_states, TF):
    """TODO."""
    coef = []
    for tf in TF:
        ## TODO: mask using dynamo
        ## each time knock-out a TF
        df = pipeline(tf)
        coef.append(df.loc[:, "PS_score"])
        logg.info("Done " + tf)
    d = {"TF": TF, "coefficient": coef}
    # df = pd.DataFrame(data=d)
    return d


# %%
def Multiple_TFScanning_perturbation(data, n_states, cluster_label, terminal_states, TF_pair):
    """TODO."""
    coef = []
    for tf in TF_pair:
        ## TODO: mask using dynamo
        ## each time knock-out a TF
        df = pipeline2(tf)
        coef.append(df.loc[:, "PS_score"])
        logg.info("Done " + combine_elements([tf])[0])
    d = {"TF": combine_elements(TF_pair), "coefficient": coef}
    # df = pd.DataFrame(data=d)
    return d


# %%
def get_list_name(lst):
    """TODO."""
    names = []
    for name, _ in lst.items():
        names.append(name)
    return names


# %% [markdown]
# ## Constants

# %%
DATASET = "zebrafish"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "raw" / "adata_zebrafish_preprocessed.h5ad")

# %%
df = pd.read_csv(DATA_DIR / DATASET / "raw" / "eRegulon_metadata_all.csv", index_col=0)

# %% [markdown]
# ## Data processing

# %%
df.loc[:, "Target"] = [f"{a}*{b}" for a, b in zip(df.loc[:, "Region"].tolist(), df.loc[:, "Gene"].tolist())]
df = df.loc[:, ["TF", "Gene", "Target"]]

regulators = df["TF"].unique()
targets = df["Target"].unique()

# Create an empty binary matrix
binary_matrix = pd.DataFrame(0, columns=regulators, index=targets)

# Fill the binary matrix based on the relationships in the CSV file
for _, row in df.iterrows():
    binary_matrix.at[row["Target"], row["TF"]] = 1

original_list = binary_matrix.index.tolist()
peak = [item.split("*")[0] for item in original_list]
target = [item.split("*")[1] for item in original_list]

binary_matrix.loc[:, "peak_id"] = peak
binary_matrix.loc[:, "gene_short_name"] = target

binary_matrix = binary_matrix[
    ["peak_id", "gene_short_name"] + [col for col in binary_matrix if col not in ["peak_id", "gene_short_name"]]
]
binary_matrix = binary_matrix.reset_index(drop=True)

# %%
scv.pp.moments(adata, n_pcs=50, n_neighbors=30)

# %% [markdown]
# ## CellOracle pipeline

# %%
adata.X = adata.layers["matrix"].copy()
oracle = co.Oracle()
oracle.import_anndata_as_raw_count(adata=adata, cluster_column_name="cell_type", embedding_name="X_umap")

oracle.import_TF_data(TF_info_matrix=binary_matrix)

# %%
oracle.perform_PCA()

# %%
n_comps = np.where(np.diff(np.diff(np.cumsum(oracle.pca.explained_variance_ratio_)) > 0.002))[0][0]

# %%
n_comps = min(n_comps, 50)

# %%
n_cell = oracle.adata.shape[0]
print(f"cell number is :{n_cell}")

k = int(0.025 * n_cell)
print(f"Auto-selected k is :{k}")

# %%
oracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k * 8, b_maxl=k * 4, n_jobs=4)

# %%
links = oracle.get_links(cluster_name_for_GRN_unit="cell_type", alpha=10, verbose_level=10)

# %%
links.filter_links(p=0.001, weight="coef_abs", threshold_number=2000)
links.get_network_score()

# %%
links.filter_links()
oracle.get_cluster_specific_TFdict_from_Links(links_object=links)
oracle.fit_GRN_for_simulation(alpha=10, use_cluster_specific_TFdict=True)

# %% [markdown]
# ### Reference vector field definition

# %%
## calculate pseudotime
scv.tl.recover_dynamics(adata, var_names=adata.var_names, n_jobs=4)
scv.tl.velocity(adata, mode="dynamical")
scv.tl.latent_time(adata, min_likelihood=None)

# %%
## use the velocity latent time inferred by scVelo to create gradient field
n_grid = 40
min_mass = 1.5
oracle.adata.obs["Pseudotime"] = adata.obs["latent_time"].copy()
gradient = Gradient_calculator(oracle_object=oracle, pseudotime_key="Pseudotime")
gradient.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=30)
gradient.calculate_mass_filter(min_mass=min_mass, plot=True)
gradient.transfer_data_into_grid(args={"method": "polynomial", "n_poly": 3}, plot=True)
gradient.calculate_gradient()

# %%
scale_dev = 40
gradient.visualize_results(scale=scale_dev, s=5)

# %%
cell_idx_pigment = np.where(oracle.adata.obs["cell_type"].isin(["Pigment"]))[0]

cell_idx_hox34 = np.where(oracle.adata.obs["cell_type"].isin(["mNC_hox34"]))[0]

cell_idx_arch2 = np.where(oracle.adata.obs["cell_type"].isin(["mNC_arch2"]))[0]

cell_idx_mesenchymal = np.where(oracle.adata.obs["cell_type"].isin(["mNC_head_mesenchymal"]))[0]

# %%
index_dictionary = {
    "Pigment": cell_idx_pigment,
    "mNC_hox34": cell_idx_hox34,
    "mNC_arch2": cell_idx_arch2,
    "mNC_head_mesenchymal": cell_idx_mesenchymal,
}

# %% [markdown]
# ## Perturbation prediction

# %% [markdown]
# #### single knock-out

# %%
n_propagation = 3
n_neighbors = 30

gene_list = ["elk3", "erf", "fli1a", "mitfa", "nr2f5", "rarga", "rxraa", "smarcc1a", "tfec", "nr2f2"]

terminal_states = [
    "mNC_head_mesenchymal",
    "mNC_arch2",
    "mNC_hox34",
    "Pigment",
]

# %%
gene_list = set(gene_list).intersection(adata.var_names)
gene_list = list(gene_list)

# %%
## celloracle perturbation
d = TFScanning_perturbation(adata, 8, "cell_type", terminal_states, gene_list)

# %%
coef = pd.DataFrame(np.array(d["coefficient"]))
coef.index = d["TF"]
coef.columns = get_list_name(d["coefficient"][0])

# %% [markdown]
# #### double knock-out

# %%
multiple_ko = ["fli1a_elk3", "mitfa_tfec", "tfec_mitfa_bhlhe40", "fli1a_erf_erfl3", "erf_erfl3"]
multiple_ko_list = split_elements(multiple_ko)

# %%
d = Multiple_TFScanning_perturbation(adata, 8, "cell_type", terminal_states, multiple_ko_list)

# %%
coef_multiple = pd.DataFrame(np.array(d["coefficient"]))
coef_multiple.index = d["TF"]
coef_multiple.columns = get_list_name(d["coefficient"][0])

# %%
if SAVE_DATA:
    coef.to_csv(DATA_DIR / DATASET / "results" / "celloracle_perturb_single.csv")
    coef_multiple.to_csv(DATA_DIR / DATASET / "results" / "celloracle_perturb_multiple.csv")

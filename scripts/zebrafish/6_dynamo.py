# %% [markdown]
# # Dynamo-based perturbation analysis
#
# Notebooks predicts TF perturbation effects using dynamo.

# %% [markdown]
# ## Library imports

# %%
import numpy as np
import pandas as pd
from scipy.stats import ranksums
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

import cellrank as cr
import dynamo as dyn
import scanpy as sc
from dynamo.preprocessing import Preprocessor
from scvelo import logging as logg

from rgv_tools import DATA_DIR, FIG_DIR

# %% [markdown]
# ## Constants

# %%
DATASET = "zebrafish"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

# %%
SAVE_FIGURES = False
if SAVE_FIGURES:
    (FIG_DIR / DATASET).mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Function definitions


# %%
def p_adjust_bh(p):
    """TODO."""
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]


# %%
def get_list_name(lst):
    """TODO."""
    names = []
    for name, _ in lst.items():
        names.append(name)
    return names


# %%
def abundance_test(prob_raw, prob_pert, correction=True):
    """TODO."""
    y = [1] * prob_raw.shape[0] + [0] * prob_pert.shape[0]
    X = pd.concat([prob_raw, prob_pert])

    table = []
    for i in range(prob_raw.shape[1]):
        pred = np.array(X.iloc[:, i])

        if np.sum(pred) == 0:
            pval = np.nan
            score = np.nan
        else:
            pval = ranksums(pred[np.array(y) == 1], pred[np.array(y) == 0])[1]
            score = roc_auc_score(y, pred)
        # result = list(scipy.stats.spearmanr(pred,y))
        result = [score, pval]
        # result[0] = np.array( (np.mean(pred[np.array(y)==1])) / (np.mean(pred[np.array(y)==0])))
        table.append(np.expand_dims(np.array(result), 0))

    table = np.concatenate(table, 0)
    table = pd.DataFrame(table)
    table.index = prob_raw.columns
    table.columns = ["coefficient", "p-value"]
    ## Running FDR addjust
    table.loc[:, "FDR adjusted p-value"] = p_adjust_bh(table.loc[:, "p-value"].tolist())

    return table


# %%
def TFScanning_KO(adata, n_states, cluster_label, terminal_states, TF, fate_prob_return=False):
    """TODO."""
    vk = cr.kernels.VelocityKernel(adata, attr="obsm", xkey="X_pca", vkey="velocity_pca")
    vk.compute_transition_matrix()
    ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
    g = cr.estimators.GPCCA(0.8 * vk + 0.2 * ck)
    ## evaluate the fate prob on original space
    g.compute_macrostates(n_states=n_states, n_cells=10, cluster_key=cluster_label)
    ## set a high number of states, and merge some of them and rename
    if terminal_states is None:
        g.predict_terminal_states()
        terminal_states = g.terminal_states.cat.categories.tolist()
    g.set_terminal_states(terminal_states)
    # g2 = g2.rename_terminal_states({"Mono_1": "Mono","DCs_1":"DC"})
    g.compute_fate_probabilities(solver="direct")
    fate_prob = g.fate_probabilities
    sampleID = adata.obs.index.tolist()
    fate_name = fate_prob.names.tolist()
    fate_prob = pd.DataFrame(fate_prob, index=sampleID, columns=fate_name)
    ## create dictionary
    terminal_id = terminal_states.copy()
    terminal_type = terminal_states.copy()
    for i in terminal_states:
        for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            terminal_id.append(i + "_" + str(j))
            terminal_type.append(i)

    fate_prob_original = fate_prob.copy()
    coef = []
    pvalue = []
    for tf in TF:
        ## TODO: mask using dynamo
        adata_target = adata.copy()
        dyn.pd.KO(adata_target, tf, store_vf_ko=True)
        ## perturb the regulations
        vk = cr.kernels.VelocityKernel(adata_target, attr="obsm", xkey="X_pca", vkey="velocity_pca_KO")
        vk.compute_transition_matrix()
        ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
        g2 = cr.estimators.GPCCA(0.8 * vk + 0.2 * ck)
        ## evaluate the fate prob on original space
        g2.compute_macrostates(n_states=n_states, n_cells=10, cluster_key=cluster_label)
        ## intersection the states
        terminal_states_perturb = g2.macrostates.cat.categories.tolist()
        # terminal_states_perturb = list(set(terminal_states_perturb).intersection(terminal_id))
        terminal_states_perturb = list(set(terminal_states_perturb).intersection(terminal_states))
        g2.set_terminal_states(terminal_states_perturb)
        g2.compute_fate_probabilities(solver="direct")
        fb = g2.fate_probabilities
        sampleID = adata.obs.index.tolist()
        fate_name = fb.names.tolist()
        fb = pd.DataFrame(fb, index=sampleID, columns=fate_name)
        fate_prob2 = pd.DataFrame(columns=terminal_states, index=sampleID)
        for i in terminal_states_perturb:
            # fate_prob2.loc[:,i] = fb.loc[:,[j == i for j in terminal_states_perturb_names]].sum(1).tolist()
            fate_prob2.loc[:, i] = fb.loc[:, i]
        fate_prob2 = fate_prob2.fillna(0)
        arr = np.array(fate_prob2.sum(0))
        arr[arr != 0] = 1
        fate_prob = fate_prob * arr

        fate_prob2.index = [i + "_perturb" for i in fate_prob2.index]
        test_result = abundance_test(fate_prob, fate_prob2)
        coef.append(test_result.loc[:, "coefficient"])
        pvalue.append(test_result.loc[:, "FDR adjusted p-value"])
        logg.info("Done " + tf)
        fate_prob = fate_prob_original.copy()
    d = {"TF": TF, "coefficient": coef, "pvalue": pvalue}
    # df = pd.DataFrame(data=d)
    if fate_prob_return:
        return fate_prob, fate_prob2
    else:
        return d


# %%
def pipeline(adata, gene_for_KO):
    """TODO."""
    dyn.pd.perturbation(adata, gene_for_KO, [-1000], emb_basis="umap")
    effect = np.diag(adata.layers["j_delta_x_perturbation"].toarray().dot(adata.layers["velocity_S"].toarray().T))
    effect = effect / np.max(np.abs(effect))
    adata.obs["effect"] = effect
    Pigment_score = np.mean(adata.obs["effect"][adata.obs["cell_type"] == "Pigment"])
    mNC_hox34_score = np.mean(adata.obs["effect"][adata.obs["cell_type"] == "mNC_hox34"])
    mNC_arch2_score = np.mean(adata.obs["effect"][adata.obs["cell_type"] == "mNC_arch2"])
    mNC_head_mesenchymal_score = np.mean(adata.obs["effect"][adata.obs["cell_type"] == "mNC_head_mesenchymal"])
    score = [Pigment_score, mNC_hox34_score, mNC_arch2_score, mNC_head_mesenchymal_score]
    celltype = ["Pigment", "mNC_hox34", "mNC_arch2", "mNC_head_mesenchymal"]
    df = pd.DataFrame({"PS_score": score, "celltype": celltype})
    df["KO"] = gene_for_KO
    df.index = celltype
    return df


# %%
def TFScanning_perturbation(adata, n_states, cluster_label, terminal_states, TF):
    """TODO."""
    coef = []
    for tf in TF:
        ## TODO: mask using dynamo
        ## each time knock-out a TF
        df = pipeline(adata, tf)
        coef.append(df.loc[:, "PS_score"])
        logg.info("Done " + tf)
    d = {"TF": TF, "coefficient": coef}
    # df = pd.DataFrame(data=d)
    return d


# %% [markdown]
# #### Function for multiple knock-out


# %%
## perform multiple knock-out simulation benchmark
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
def Multiple_TFScanning_KO(adata, n_states, cluster_label, terminal_states, TF_pair):
    """TODO."""
    vk = cr.kernels.VelocityKernel(adata, attr="obsm", xkey="X_pca", vkey="velocity_pca")
    vk.compute_transition_matrix()
    ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
    g = cr.estimators.GPCCA(0.8 * vk + 0.2 * ck)
    ## evaluate the fate prob on original space
    g.compute_macrostates(n_states=n_states, n_cells=10, cluster_key=cluster_label)
    ## set a high number of states, and merge some of them and rename
    if terminal_states is None:
        g.predict_terminal_states()
        terminal_states = g.terminal_states.cat.categories.tolist()
    g.set_terminal_states(terminal_states)
    g.compute_fate_probabilities(solver="direct")
    fate_prob = g.fate_probabilities
    sampleID = adata.obs.index.tolist()
    fate_name = fate_prob.names.tolist()
    fate_prob = pd.DataFrame(fate_prob, index=sampleID, columns=fate_name)
    ## create dictionary
    terminal_id = terminal_states.copy()
    terminal_type = terminal_states.copy()
    for i in terminal_states:
        for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            terminal_id.append(i + "_" + str(j))
            terminal_type.append(i)

    fate_prob_original = fate_prob.copy()

    coef = []
    pvalue = []
    for tf_pair in TF_pair:
        adata_target = adata.copy()
        dyn.pd.KO(adata_target, tf_pair, store_vf_ko=True)

        vk = cr.kernels.VelocityKernel(adata_target, attr="obsm", xkey="X_pca", vkey="velocity_pca_KO")
        vk.compute_transition_matrix()
        ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
        g2 = cr.estimators.GPCCA(0.8 * vk + 0.2 * ck)
        ## evaluate the fate prob on original space
        g2.compute_macrostates(n_states=n_states, n_cells=10, cluster_key=cluster_label)
        ## intersection the states

        terminal_states_perturb = g2.macrostates.cat.categories.tolist()
        # terminal_states_perturb = list(set(terminal_states_perturb).intersection(terminal_id))
        terminal_states_perturb = list(set(terminal_states_perturb).intersection(terminal_states))

        g2.set_terminal_states(terminal_states_perturb)
        g2.compute_fate_probabilities(solver="direct")
        fb = g2.fate_probabilities
        sampleID = adata.obs.index.tolist()
        fate_name = fb.names.tolist()
        fb = pd.DataFrame(fb, index=sampleID, columns=fate_name)
        fate_prob2 = pd.DataFrame(columns=terminal_states, index=sampleID)

        for i in terminal_states_perturb:
            # fate_prob2.loc[:,i] = fb.loc[:,[j == i for j in terminal_states_perturb_names]].sum(1).tolist()
            fate_prob2.loc[:, i] = fb.loc[:, i]

        fate_prob2 = fate_prob2.fillna(0)
        arr = np.array(fate_prob2.sum(0))
        arr[arr != 0] = 1
        fate_prob = fate_prob * arr

        fate_prob2.index = [i + "_perturb" for i in fate_prob2.index]
        test_result = abundance_test(fate_prob, fate_prob2)
        coef.append(test_result.loc[:, "coefficient"])
        pvalue.append(test_result.loc[:, "FDR adjusted p-value"])
        print(tf_pair)
        logg.info("Done " + combine_elements([tf_pair])[0])
        fate_prob = fate_prob_original.copy()
    d = {"TF": combine_elements(TF_pair), "coefficient": coef, "pvalue": pvalue}
    # df = pd.DataFrame(data=d)
    return d


# %%
def pipeline_double(adata, gene_for_KO):
    """TODO."""
    dyn.pd.perturbation(adata, gene_for_KO, [-1000] * len(gene_for_KO), emb_basis="umap")
    effect = np.diag(adata.layers["j_delta_x_perturbation"].toarray().dot(adata.layers["velocity_S"].toarray().T))
    effect = effect / np.max(np.abs(effect))
    adata.obs["effect"] = effect

    Pigment_score = np.mean(adata.obs["effect"][adata.obs["cell_type"] == "Pigment"])
    mNC_hox34_score = np.mean(adata.obs["effect"][adata.obs["cell_type"] == "mNC_hox34"])
    mNC_arch2_score = np.mean(adata.obs["effect"][adata.obs["cell_type"] == "mNC_arch2"])
    mNC_head_mesenchymal_score = np.mean(adata.obs["effect"][adata.obs["cell_type"] == "mNC_head_mesenchymal"])

    score = [Pigment_score, mNC_hox34_score, mNC_arch2_score, mNC_head_mesenchymal_score]
    celltype = ["Pigment", "mNC_hox34", "mNC_arch2", "mNC_head_mesenchymal"]
    df = pd.DataFrame({"PS_score": score, "celltype": celltype})
    df["KO"] = combine_elements([gene_for_KO])[0]
    df.index = celltype

    return df


# %%
def Multiple_TFScanning_perturbation(adata, n_states, cluster_label, terminal_states, TF_pair):
    """TODO."""
    coef = []
    for tf in TF_pair:
        ## TODO: mask using dynamo
        ## each time knock-out a TF
        df = pipeline_double(adata, tf)
        coef.append(df.loc[:, "PS_score"])
        logg.info("Done " + combine_elements([tf])[0])
    d = {"TF": combine_elements(TF_pair), "coefficient": coef}
    # df = pd.DataFrame(data=d)
    return d


# %% [markdown]
# ## Data Loading

# %%
adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / "adata_preprocessed.h5ad")

# %%
adata.X = adata.layers["matrix"].copy()

# %%
genes = ["nr2f5", "sox9b", "twist1b", "ets1"]

# %% [markdown]
# ## Processing by dynamo

# %%
preprocessor = Preprocessor()
preprocessor.preprocess_adata(adata, recipe="monocle")

# %%
dyn.tl.dynamics(adata)

# %%
dyn.tl.reduceDimension(adata)

# %%
dyn.tl.cell_velocities(adata, basis="pca")

# %%
dyn.vf.VectorField(adata, basis="pca")

# %%
adata_perturb = adata.copy()

# %%
del adata.uns["cell_type_colors"]

# %% [markdown]
# ## Compare predict fate probability changes

# %%
terminal_states = [
    "mNC_head_mesenchymal",
    "mNC_arch2",
    "mNC_hox34",
    "Pigment",
]

fate_prob = {}
fate_prob_perturb = {}

for g in genes:
    fb, fb_perturb = TFScanning_KO(adata, 8, "cell_type", terminal_states, [g], fate_prob_return=True)
    fate_prob[g] = fb
    fate_prob_perturb[g] = fb_perturb

# %% [markdown]
# ## Visualize dynamo perturbation effects

# %%
for g in genes:
    data = abundance_test(fate_prob[g], fate_prob_perturb[g])
    data = pd.DataFrame(
        {
            "Score": data.iloc[:, 0].tolist(),
            "p-value": data.iloc[:, 1].tolist(),
            "Terminal state": data.index.tolist(),
            "TF": [g] * (data.shape[0]),
        }
    )

    final_df = data.copy()
    final_df["Score"] = 0.5 - final_df["Score"]

    color_label = "cell_type"
    df = pd.DataFrame(final_df["Score"])
    df.columns = ["coefficient"]
    df["Cell type"] = final_df["Terminal state"]
    order = df["Cell type"].tolist()

    palette = dict(zip(adata.obs[color_label].cat.categories, adata_perturb.uns[f"{color_label}_colors"]))
    subset_palette = {name: color for name, color in palette.items() if name in terminal_states}

    with mplscience.style_context():
        sns.set_style(style="whitegrid")
        fig, ax = plt.subplots(figsize=(2, 2))
        sns.barplot(
            data=df,
            y="coefficient",
            x="Cell type",
            palette=subset_palette,
            order=order,
            ax=ax,
        )
        ax.tick_params(axis="x", rotation=90)
        plt.title("$\\mathit{" + g + "}$ regulon knock out simulation")

        if SAVE_FIGURES:
            plt.savefig(
                FIG_DIR / DATASET / f"{g}_perturbation_simulation_dynamo.svg",
                format="svg",
                transparent=True,
                bbox_inches="tight",
            )
        # Show the plot
        plt.show()

# %% [markdown]
# ## Perturbation benchmark

# %%
gene_list = ["elk3", "erf", "etv2", "fli1a", "mitfa", "nr2f5", "rarga", "rxraa", "smarcc1a", "tfec", "nr2f2"]

# %%
gene_list = set(gene_list).intersection(adata.var_names)
gene_list = list(gene_list)

# %%
## Dynamo (KO)
d = TFScanning_KO(adata, 8, "cell_type", terminal_states, gene_list)

## Dynamo (perturbation)
d2 = TFScanning_perturbation(adata, 8, "cell_type", terminal_states, gene_list)

# %%
## Perform KO screening using function based perturbation
coef_KO = pd.DataFrame(np.array(d["coefficient"]))
coef_KO.index = d["TF"]
coef_KO.columns = get_list_name(d["coefficient"][0])

## Perform perturbation screening using gene expression based perturbation
coef_perturb = pd.DataFrame(np.array(d2["coefficient"]))
coef_perturb.index = d2["TF"]
coef_perturb.columns = get_list_name(d2["coefficient"][0])

# %% [markdown]
# ## Multiple gene knock-out prediction

# %%
multiple_ko = ["fli1a_elk3", "mitfa_tfec", "tfec_mitfa_bhlhe40", "fli1a_erf_erfl3", "erf_erfl3"]
multiple_ko_list = split_elements(multiple_ko)

# %%
## Dynamo (KO)
d = Multiple_TFScanning_KO(adata, 9, "cell_type", terminal_states, multiple_ko_list)

## Dynamo (perturbation)
d2 = Multiple_TFScanning_perturbation(adata, 9, "cell_type", terminal_states, multiple_ko_list)

# %%
## Perform KO screening using function based perturbation
coef_KO_multiple = pd.DataFrame(np.array(d["coefficient"]))
coef_KO_multiple.index = d["TF"]
coef_KO_multiple.columns = get_list_name(d["coefficient"][0])

## Perform perturbation screening using gene expression based perturbation
coef_perturb_multiple = pd.DataFrame(np.array(d2["coefficient"]))
coef_perturb_multiple.index = d2["TF"]
coef_perturb_multiple.columns = get_list_name(d2["coefficient"][0])

# %% [markdown]
# ## Save dataset

# %%
if SAVE_DATA:
    coef_KO.to_csv(DATA_DIR / DATASET / "results" / "dynamo_KO_single.csv")
    coef_KO_multiple.to_csv(DATA_DIR / DATASET / "results" / "dynamo_KO_multiple.csv")

    coef_perturb.to_csv(DATA_DIR / DATASET / "results" / "dynamo_perturb_single.csv")
    coef_perturb_multiple.to_csv(DATA_DIR / DATASET / "results" / "dynamo_perturb_multiple.csv")

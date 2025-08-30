from typing import List

import numpy as np
import pandas as pd
from scipy.stats import ranksums, ttest_rel
from sklearn.metrics import roc_auc_score

import cellrank as cr
from anndata import AnnData
import dynamo as dyn
from scvelo import logging as logg

from ._ptools import combine_elements


###########################
# Function: p_adjust_bh
def p_adjust_bh(p: List[float]) -> np.ndarray:
    """Perform Benjamini-Hochberg correction for multiple hypothesis testing.

    Parameters
    ----------
    p : List[float]
        List of p-values.

    Returns
    -------
    np.ndarray
        Adjusted p-values.
    """
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]


###########################
# Function: abundance_test
def abundance_test(prob_raw: pd.DataFrame, prob_pert: pd.DataFrame, method: str = "likelihood") -> pd.DataFrame:
    """Perform an abundance test between two probability datasets.

    Parameters
    ----------
    prob_raw : pd.DataFrame
        Raw probabilities dataset.
    prob_pert : pd.DataFrame
        Perturbed probabilities dataset.
    method : str, optional (default="likelihood")
        Method to calculate scores: "likelihood" or "t-statistics".

    Returns
    -------
    pd.DataFrame
        Dataframe with coefficients, p-values, and FDR adjusted p-values.
    """
    y = [1] * prob_raw.shape[0] + [0] * prob_pert.shape[0]
    X = pd.concat([prob_raw, prob_pert], axis=0)

    table = []
    for i in range(prob_raw.shape[1]):
        pred = np.array(X.iloc[:, i])
        if np.sum(pred) == 0:
            score, pval = np.nan, np.nan
        else:
            pval = ranksums(pred[np.array(y) == 0], pred[np.array(y) == 1])[1]
            if method == "t-statistics":
                score = np.mean(pred[np.array(y) == 0]) - np.mean(pred[np.array(y) == 1])
                score = score / np.sqrt(np.std(pred[np.array(y) == 1]) ** 2 + np.std(pred[np.array(y) == 0]) ** 2)
            elif method == "likelihood":
                score = roc_auc_score(y, pred)
            else:
                raise NotImplementedError("Supported methods are 't-statistics' and 'likelihood'.")

        table.append(np.expand_dims(np.array([score, pval]), 0))

    table = np.concatenate(table, axis=0)
    table = pd.DataFrame(table, index=prob_raw.columns, columns=["coefficient", "p-value"])
    table["FDR adjusted p-value"] = p_adjust_bh(table["p-value"].tolist())
    return table


###########################
## markov_density_simulation
def markov_density_simulation(
    adata: "AnnData",
    T: np.ndarray,
    start_indices,
    terminal_indices,
    terminal_states,
    n_steps: int = 100,
    n_simulations: int = 500,
    method: str = "stepwise",
    seed: int = 0,
):
    """Simulate transitions on a velocity-derived Markov transition matrix.

    Parameters
    ----------
    T : np.ndarray
        Transition matrix of shape (n_cells, n_cells).
    start_indices : array-like
        Indices of starting cells.
    terminal_indices : array-like
        Indices of terminal (absorbing) cells.
    n_steps : int, optional
        Maximum number of steps per simulation (default: 100).
    n_simulations : int, optional
        Number of simulations per starting cell (default: 200).
    method : {'stepwise', 'one-step'}, optional
        Simulation method to use:
        - 'stepwise': simulate trajectories step by step.
        - 'one-step': sample directly from T^n.
    seed : int, optional
        Random seed for reproducibility (default: 0).

    Returns
    -------
    arrivals : pd.Series
        Total number of simulations that ended at each terminal cell.
    arrivals_dens : pd.Series
        Fraction of simulations that ended at each terminal cell.
    """
    np.random.seed(seed)

    T = np.asarray(T)
    start_indices = np.asarray(start_indices)
    terminal_indices = np.asarray(terminal_indices)
    terminal_set = set(terminal_indices)
    n_cells = T.shape[0]

    arrivals_array = np.zeros(n_cells, dtype=int)

    if method == "stepwise":
        row_sums = T.sum(axis=1)
        cum_T = np.cumsum(T, axis=1)

        for start in start_indices:
            for _ in range(n_simulations):
                current = start
                for _ in range(n_steps):
                    if row_sums[current] == 0:
                        break  # dead end
                    r = np.random.rand()
                    next_state = np.searchsorted(cum_T[current], r * row_sums[current])
                    current = next_state
                    if current in terminal_set:
                        arrivals_array[current] += 1
                        break

    elif method == "one-step":
        T_end = np.linalg.matrix_power(T, n_steps)
        for start in start_indices:
            x0 = np.zeros(n_cells)
            x0[start] = 1
            x_end = x0 @ T_end  # final distribution
            if x_end.sum() > 0:
                samples = np.random.choice(n_cells, size=n_simulations, p=x_end)
                for s in samples:
                    if s in terminal_set:
                        arrivals_array[s] += 1
            else:
                raise ValueError(f"Invalid probability distribution: x_end sums to 0 for start index {start}")
    else:
        raise ValueError("method must be 'stepwise' or 'one-step'")

    total_simulations = n_simulations * len(start_indices)
    visits = pd.Series({tid: arrivals_array[tid] for tid in terminal_indices}, dtype=int)
    visits_dens = pd.Series({tid: arrivals_array[tid] / total_simulations for tid in terminal_indices})

    adata.obs["visits"] = np.nan
    adata.obs["visits"].iloc[terminal_indices] = visits

    dens_cum = []
    for ts in terminal_states:
        ts_cells = np.where(adata.obs["term_states_fwd"] == ts)[0]
        density = visits_dens.loc[ts_cells].sum()
        dens_cum.append(density)

    return visits, visits_dens


###########################
# density_likelihood_dyn
def density_likelihood_dyn(adata, tf, start_indices, terminal_states, n_simulations=500):
    ## compute transition matrix
    vk = cr.kernels.VelocityKernel(adata, attr="obsm", xkey="X_pca", vkey="velocity_pca")
    vk.compute_transition_matrix()

    adata_target = adata.copy()
    dyn.pd.KO(adata_target, tf, store_vf_ko=True)
    ## perturb the regulations
    vk_p = cr.kernels.VelocityKernel(adata_target, attr="obsm", xkey="X_pca", vkey="velocity_pca_KO")
    vk_p.compute_transition_matrix()

    vkt = vk.transition_matrix.A
    vkt_p = vk_p.transition_matrix.A

    terminal_indices = np.where(adata.obs["term_states_fwd"].isin(terminal_states))[0]
    arrivals, _ = markov_density_simulation(
        adata, vkt, start_indices, terminal_indices, terminal_states, n_simulations=n_simulations
    )
    arrivals_p, _ = markov_density_simulation(
        adata_target, vkt_p, start_indices, terminal_indices, terminal_states, n_simulations=n_simulations
    )

    cont_sim = []
    pert_sim = []
    for ts in terminal_states:
        terminal_indices = np.where(adata.obs["term_states_fwd"].isin([ts]))[0]
        # arrivals = simulate_markov_chain_from_velocity_graph(adata, vkt,start_indices, terminal_indices, n_steps=100,n_simulations = n_simulations,seed = 0)
        # arrivals_p = simulate_markov_chain_from_velocity_graph(adata_perturb, vkt_p,start_indices, terminal_indices, n_steps=100,n_simulations = n_simulations,seed = 0)
        terminal_indices_sub = np.where(adata.obs["term_states_fwd"].isin([ts]))[0]
        arrivals = adata.obs["visits"].iloc[terminal_indices_sub]
        arrivals_p = adata_target.obs["visits"].iloc[terminal_indices_sub]

        cont_sim.append(arrivals)
        pert_sim.append(arrivals_p)

    # y = [0] * len(arrivals) + [1] * len(arrivals_p)
    dl_score = []
    dl_sig = []
    for i in range(len(terminal_states)):
        cont_sim[i].tolist() + pert_sim[i].tolist()
        # pred = cont_sim[i].tolist() + pert_sim[i].tolist()
        _, p_value = ttest_rel(cont_sim[i], pert_sim[i])
        dl_score.append(np.mean(pert_sim[i]) - np.mean(cont_sim[i]))
        dl_sig.append(p_value)

    return dl_score, dl_sig, cont_sim, pert_sim


###########################
# Function: TFScanning
def TFScanning_KO_dyn(adata, n_states, cluster_label, terminal_states, TF, fate_prob_return=False):
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


def pipeline(adata, gene_for_KO):
    """TODO."""
    dyn.pd.perturbation(adata, gene_for_KO, [-1000], emb_basis="pca")
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


def TFScanning_perturbation_dyn(adata, n_states, cluster_label, terminal_states, TF):
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


def Multiple_TFScanning_KO_dyn(adata, n_states, cluster_label, terminal_states, TF_pair):
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


def Multiple_TFScanning_perturbation_dyn(adata, n_states, cluster_label, terminal_states, TF_pair):
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

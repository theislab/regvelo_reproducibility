import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from scipy.stats import ranksums, ttest_ind, ttest_rel
from sklearn.metrics import roc_auc_score

import cellrank as cr
from anndata import AnnData
from regvelo import REGVELOVI
from scvelo import logging as logg


###########################
def markov_density_simulation(
    adata : "AnnData",
    T : np.ndarray, 
    start_indices, 
    terminal_indices, 
    terminal_states,
    n_steps : int = 100, 
    n_simulations : int = 500, 
    method: str = "stepwise", 
    seed : int = 0,
):

    """
    Simulate transitions on a velocity-derived Markov transition matrix.

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

    adata.obs[f"visits"] = np.nan
    adata.obs[f"visits"].iloc[terminal_indices] = visits

    dens_cum = []
    for ts in terminal_states:
        ts_cells = np.where(adata.obs["term_states_fwd"] == ts)[0]
        density = visits_dens.loc[ts_cells].sum()
        dens_cum.append(density)
    
    return visits, visits_dens

def delta_to_probability(delta_hits, k=0.005):
    """
    Convert a difference score (delta_hits) into a probability using a logistic function.

    Parameters
    ----------
    delta_hits : float or array-like
        Input value(s) representing the change or difference score.
    k : float, optional (default: 0.005)
        Scaling factor controlling the steepness of the logistic function.

    Returns
    -------
    float or np.ndarray
        Probability value(s) in the range (0, 1).
    """
    return 1 / (1 + np.exp(-k * delta_hits))


def smooth_score(adata, key="sim_pop_fc", n_neighbors=10):
    """
    Perform neighbor-based smoothing of cell scores using a nearest-neighbor graph.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with `.obs` and `.obsm`.
    key : str, optional (default: "sim_pop_fc")
        Column name in `adata.obs` containing values to be smoothed.
    n_neighbors : int, optional (default: 10)
        Number of neighbors to use for smoothing.

    Updates
    -------
    adata.obs[key + "_smooth"] : pd.Series
        Smoothed version of the input score.
    """
    values = adata.obs[key]
    valid_cells = values[~values.isna()].index  # only smooth valid entries

    from sklearn.neighbors import NearestNeighbors
    embedding = adata.obsm['X_pca']  # assumes PCA embedding exists
    valid_idx = adata.obs_names.get_indexer(valid_cells)

    # Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(embedding)
    distances, indices = nbrs.kneighbors(embedding[valid_idx])

    neighbor_indices = indices[:, 1:]  # exclude self (first neighbor)

    # Smooth values by averaging neighbors
    neighbor_values = values.values[neighbor_indices]
    mean_neighbor_values = np.nanmean(neighbor_values, axis=1)
    
    # Store smoothed values in adata
    adata.obs[key+"_smooth"] = np.nan
    adata.obs.loc[valid_cells, key+"_smooth"] = mean_neighbor_values

def density_likelihood(adata,adata_perturb,start_indices,terminal_states,n_simulations = 500):
    """
    Compare density of arrivals in terminal states between control and perturbed systems.

    Parameters
    ----------
    adata : AnnData
        Reference dataset used to compute transition probabilities.
    adata_perturb : AnnData
        Perturbed dataset for comparison.
    start_indices : list or array
        Indices of starting cells.
    terminal_states : list
        Labels of terminal states in adata.obs["term_states_fwd"].
    n_simulations : int, optional (default: 500)
        Number of Markov chain simulations to run per condition.

    Returns
    -------
    dl_score : list of float
        Mean difference in terminal state arrivals between perturbed and control simulations.
    dl_sig : list of float
        p-values from paired t-tests between control and perturbation.
    cont_sim : list of pd.Series
        Arrival distributions for control simulations.
    pert_sim : list of pd.Series
        Arrival distributions for perturbed simulations.
    """
    ## compute transition matrix
    vk = cr.kernels.VelocityKernel(adata)
    vk.compute_transition_matrix()
    
    vk_p = cr.kernels.VelocityKernel(adata_perturb)
    vk_p.compute_transition_matrix()
    
    vkt = vk.transition_matrix.A
    vkt_p = vk_p.transition_matrix.A
    
    terminal_indices = np.where(adata.obs["term_states_fwd"].isin(terminal_states))[0]
    arrivals,_ = markov_density_simulation(adata, vkt,start_indices, terminal_indices,terminal_states,n_simulations = n_simulations)
    arrivals_p,_ = markov_density_simulation(adata_perturb, vkt_p,start_indices, terminal_indices,terminal_states,n_simulations = n_simulations)
    
    cont_sim = []
    pert_sim = []
    for ts in terminal_states:
        #terminal_indices = np.where(adata.obs["term_states_fwd"].isin([ts]))[0]
        #arrivals = simulate_markov_chain_from_velocity_graph(adata, vkt,start_indices, terminal_indices, n_steps=100,n_simulations = n_simulations,seed = 0)
        #arrivals_p = simulate_markov_chain_from_velocity_graph(adata_perturb, vkt_p,start_indices, terminal_indices, n_steps=100,n_simulations = n_simulations,seed = 0)
        #arrivals,_ = markov_density_simulation(adata, vkt,start_indices, terminal_indices)
        #arrivals_p,_ = markov_density_simulation(adata_perturb, vkt_p,start_indices, terminal_indices)
        terminal_indices_sub = np.where(adata.obs["term_states_fwd"].isin([ts]))[0]
        arrivals = adata.obs["visits"].iloc[terminal_indices_sub]
        arrivals_p = adata_perturb.obs["visits"].iloc[terminal_indices_sub]
        
        cont_sim.append(arrivals)
        pert_sim.append(arrivals_p)
    
    y = [0] * len(arrivals) + [1] * len(arrivals_p)
    dl_score = []
    dl_sig = []
    for i in range(len(terminal_states)):
        pred = cont_sim[i].tolist() + pert_sim[i].tolist()
        #pred = cont_sim[i].tolist() + pert_sim[i].tolist()
        #_, p_value = ttest_rel(cont_sim[i], pert_sim[i])
        _, p_value = ttest_rel(cont_sim[i], pert_sim[i])
        dl_score.append((np.mean(pert_sim[i])) - (np.mean(cont_sim[i])))
        dl_sig.append(p_value)
    
    return dl_score,dl_sig,cont_sim,pert_sim


# Function: in_silico_block_simulation
def in_silico_block_simulation(
    model: str, adata: AnnData, gene: str, effects: float = 0, cutoff: float = 0
) -> (AnnData, REGVELOVI):
    """Simulate in silico perturbations by altering a gene regulatory network (GRN).

    Parameters
    ----------
    model : str
        Path to the model.
    adata : AnnData
        Annotated data matrix.
    gene : str
        Target gene for perturbation.
    effects : float, optional (default=0)
        Perturbation magnitude.
    cutoff : float, optional (default=0)
        Threshold for gene perturbation effects.

    Returns
    -------
    tuple
        Perturbed AnnData object and loaded model.
    """
    vae_perturb = REGVELOVI.load(model, adata)
    perturb_GRN = vae_perturb.module.v_encoder.fc1.weight.detach().clone()
    perturb_GRN[:,[i in gene for i in adata.var.index]] = effects

    vae_perturb.module.v_encoder.fc1.weight.data = perturb_GRN
    adata_perturb = vae_perturb.add_regvelo_outputs_to_adata(adata=adata)

    return adata_perturb, vae_perturb


###########################
# Function: threshold_top_k
def threshold_top_k(tensor: torch.Tensor, k: int) -> torch.Tensor:
    """Generate a mask indicating the top k positive and negative values in a tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor to threshold.
    k : int
        Number of top values to select.

    Returns
    -------
    torch.Tensor
        A mask tensor indicating the top k values.
    """
    flattened_tensor = tensor.flatten()
    _, top_k_pos_indices = torch.topk(flattened_tensor, k)
    _, top_k_neg_indices = torch.topk(-flattened_tensor, k)

    mask = torch.zeros_like(flattened_tensor, dtype=torch.float)
    mask[top_k_pos_indices] = 1
    mask[top_k_neg_indices] = 1
    return mask.view(tensor.shape)


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
                score = ttest_ind(pred[np.array(y) == 0], pred[np.array(y) == 1])[0]
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
# Function: get_list_name
def get_list_name(lst: Dict[str, object]) -> List[str]:
    """Retrieve the list of keys from a dictionary.

    Parameters
    ----------
    lst : Dict[str, object]
        Input dictionary.

    Returns
    -------
    List[str]
        List of dictionary keys.
    """
    return list(lst.keys())


###########################
# Function: has_zero_column_sum
def has_zero_column_sum(matrix: np.ndarray) -> bool:
    """Check if any column in a 2D matrix has a sum of zero.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix.

    Returns
    -------
    bool
        True if any column sum is zero, False otherwise.
    """
    return np.any(matrix.sum(axis=0) == 0)


###########################
# Function: split_elements
def split_elements(character_list: List[str]) -> List[List[str]]:
    """Splits each string in a list of strings by the '_' character.

    Parameters
    ----------
    character_list : List[str]
        List of strings to split.

    Returns
    -------
    List[List[str]]
        List of split string parts.
    """
    return [element.split("_") if "_" in element else [element] for element in character_list]


###########################
# Function: combine_elements
def combine_elements(split_list: List[List[str]]) -> List[str]:
    """Combines lists of strings into single strings joined by the '_' character.

    Parameters
    ----------
    split_list : List[List[str]]
        List of lists of strings to combine.

    Returns
    -------
    List[str]
        List of combined strings.
    """
    return ["_".join(parts) for parts in split_list]


###########################
# Function: TFScanning
def TFScanning(
    model_path: str,
    adata: AnnData,
    n_states: int,
    cluster_label: str,
    terminal_states: Optional[List[str]],
    TF: List[str],
    effect: float = 1e-3,
    method: str = "likelihood",
) -> Dict[str, Union[List[str], List[pd.Series]]]:
    """Perform transcription factor scanning and perturbation analysis on a gene regulatory network.

    Parameters
    ----------
    model_path : str
        Path to the model file.
    adata : AnnData
        Annotated data matrix.
    n_states : int
        Number of states in the model.
    cluster_label : str
        Label identifying clusters in the data.
    terminal_states : Optional[List[str]]
        List of terminal state labels.
    TF : List[str]
        List of transcription factors.
    effect : float, default=1e-3
        Effect size parameter to adjust the sensitivity of the scanning.

    Returns
    -------
    Dict[str, Union[List[str], List[pd.Series]]]
        A dictionary with keys corresponding to results of the scanning.
    """
    reg_vae = REGVELOVI.load(model_path, adata)
    adata = reg_vae.add_regvelo_outputs_to_adata(adata=adata)
    raw_GRN = reg_vae.module.v_encoder.fc1.weight.detach().clone()

    ## build kernel matrix
    vk = cr.kernels.VelocityKernel(adata).compute_transition_matrix()
    ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
    combined_kernel = 0.8 * vk + 0.2 * ck
    g = cr.estimators.GPCCA(combined_kernel)

    ## evaluate the fate prob on original space
    g.compute_macrostates(n_states=n_states, n_cells=30, cluster_key=cluster_label)

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

    ## save cell fate probability
    fate_prob_original = fate_prob.copy()
    ## update n_states
    n_states = len(g.macrostates.cat.categories.tolist())

    coef = []
    pvalue = []
    for tf in TF:
        ## perturb the TF
        perturb_GRN = raw_GRN.clone()
        vec = perturb_GRN[:, [i in tf for i in adata.var.index.tolist()]].clone()
        vec[vec.abs() > effect] = 0
        perturb_GRN[:, [i in tf for i in adata.var.index.tolist()]] = vec
        reg_vae_perturb = REGVELOVI.load(model_path, adata)
        reg_vae_perturb.module.v_encoder.fc1.weight.data = perturb_GRN
        adata_target = reg_vae_perturb.add_regvelo_outputs_to_adata(adata=adata)

        ## build new kernel matrix
        vk = cr.kernels.VelocityKernel(adata_target)
        vk.compute_transition_matrix()
        ck = cr.kernels.ConnectivityKernel(adata_target).compute_transition_matrix()
        combined_kernel = 0.8 * vk + 0.2 * ck
        g2 = cr.estimators.GPCCA(combined_kernel)

        ## evaluate the fate prob on original space
        n_states_perturb = n_states
        while True:
            try:
                # Perform some computation in f(a)
                g2.compute_macrostates(n_states=n_states_perturb, n_cells=30, cluster_key=cluster_label)

                break
            except:
                # If an error is raised, increase `n_state` and try again
                n_states_perturb += 1
                vk = cr.kernels.VelocityKernel(adata)
                vk.compute_transition_matrix()
                ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
                g = cr.estimators.GPCCA(0.8 * vk + 0.2 * ck)
                ## evaluate the fate prob on original space
                g.compute_macrostates(n_states=n_states_perturb, n_cells=30, cluster_key=cluster_label)
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
                raise

        ## intersection the states
        terminal_states_perturb = g2.macrostates.cat.categories.tolist()
        terminal_states_perturb = list(set(terminal_states_perturb).intersection(terminal_states))

        g2.set_terminal_states(terminal_states_perturb)
        g2.compute_fate_probabilities(solver="direct")
        fb = g2.fate_probabilities
        sampleID = adata.obs.index.tolist()
        fate_name = fb.names.tolist()
        fb = pd.DataFrame(fb, index=sampleID, columns=fate_name)
        fate_prob2 = pd.DataFrame(columns=terminal_states, index=sampleID)

        for i in terminal_states_perturb:
            fate_prob2.loc[:, i] = fb.loc[:, i]

        fate_prob2 = fate_prob2.fillna(0)
        arr = np.array(fate_prob2.sum(0))
        arr[arr != 0] = 1
        fate_prob = fate_prob * arr

        fate_prob2.index = [i + "_perturb" for i in fate_prob2.index]
        test_result = abundance_test(fate_prob, fate_prob2, method)
        coef.append(test_result.loc[:, "coefficient"])
        pvalue.append(test_result.loc[:, "FDR adjusted p-value"])
        logg.info("Done " + tf)
        fate_prob = fate_prob_original.copy()

    d = {"TF": TF, "coefficient": coef, "pvalue": pvalue}
    return d


##########################
# Function: Multiple_TFScanning
def Multiple_TFScanning(
    model_path: str,
    adata: AnnData,
    n_states: int,
    cluster_label: str,
    terminal_states: Optional[List[str]],
    TF_pair: List[List[str]],
    effect: float = 1e-3,
    method: str = "likelihood",
) -> Dict[str, Union[List[str], List[float]]]:
    """Performs multiple transcription factor (TF) scanning with perturbation analysis.

    Parameters
    ----------
    model_path : str
        Path to the saved model.
    adata : AnnData
        Annotated data matrix.
    n_states : int
        Number of states for macrostates computation.
    cluster_label : str
        Key for clustering labels in the `adata` object.
    terminal_states : Optional[List[str]]
        List of terminal state names or None to predict them.
    TF_pair : List[List[str]]
        List of TF pairs to evaluate.
    effect : float
        Effect size parameter to adjust the sensitivity of the scanning.

    Returns
    -------
    Dict[str, Union[List[str], List[float]]]
        Dictionary containing TF names, coefficients, and p-values.
    """
    reg_vae = REGVELOVI.load(model_path, adata)
    adata = reg_vae.add_regvelo_outputs_to_adata(adata=adata)

    vk = cr.kernels.VelocityKernel(adata)
    vk.compute_transition_matrix()
    ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
    g = cr.estimators.GPCCA(0.8 * vk + 0.2 * ck)

    g.compute_macrostates(n_states=n_states, n_cells=30, cluster_key=cluster_label)
    if terminal_states is None:
        g.predict_terminal_states()
        terminal_states = g.terminal_states.cat.categories.tolist()
    g.set_terminal_states(terminal_states)

    g.compute_fate_probabilities(solver="direct")
    fate_prob = g.fate_probabilities
    sampleID = adata.obs.index.tolist()
    fate_name = fate_prob.names.tolist()
    fate_prob = pd.DataFrame(fate_prob, index=sampleID, columns=fate_name)
    fate_prob_original = fate_prob.copy()

    n_states = len(g.macrostates.cat.categories.tolist())
    coef = []
    pvalue = []
    for tf_pair in TF_pair:
        perturb_GRN = reg_vae.module.v_encoder.fc1.weight.detach().clone()
        vec = perturb_GRN[:, [i in tf_pair for i in adata.var.index.tolist()]].clone()
        vec[vec.abs() > effect] = 0
        perturb_GRN[:, [i in tf_pair for i in adata.var.index.tolist()]] = vec

        reg_vae_perturb = REGVELOVI.load(model_path, adata)
        reg_vae_perturb.module.v_encoder.fc1.weight.data = perturb_GRN
        adata_target = reg_vae_perturb.add_regvelo_outputs_to_adata(adata=adata)

        vk = cr.kernels.VelocityKernel(adata_target)
        vk.compute_transition_matrix()
        ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
        g2 = cr.estimators.GPCCA(0.8 * vk + 0.2 * ck)
        g2.compute_macrostates(n_states=n_states, n_cells=30, cluster_key=cluster_label)

        terminal_states_perturb = g2.macrostates.cat.categories.tolist()
        terminal_states_perturb = list(set(terminal_states_perturb).intersection(terminal_states))

        g2.set_terminal_states(terminal_states_perturb)
        g2.compute_fate_probabilities(solver="direct")
        fb = g2.fate_probabilities
        sampleID = adata.obs.index.tolist()
        fate_name = fb.names.tolist()
        fb = pd.DataFrame(fb, index=sampleID, columns=fate_name)

        fate_prob2 = pd.DataFrame(columns=terminal_states, index=sampleID)

        for i in terminal_states_perturb:
            fate_prob2.loc[:, i] = fb.loc[:, i]

        fate_prob2 = fate_prob2.fillna(0)
        arr = np.array(fate_prob2.sum(0))
        arr[arr != 0] = 1
        fate_prob = fate_prob * arr

        fate_prob2.index = [i + "_perturb" for i in fate_prob2.index]
        test_result = abundance_test(fate_prob, fate_prob2, method=method)
        coef.append(test_result.loc[:, "coefficient"])
        pvalue.append(test_result.loc[:, "FDR adjusted p-value"])
        logg.info(f"Done {combine_elements([tf_pair])[0]}")
        fate_prob = fate_prob_original.copy()

    d = {"TF": combine_elements(TF_pair), "coefficient": coef, "pvalue": pvalue}
    return d


##########################
# Function: aggregate_model_predictions
def aggregate_model_predictions(path, method="likelihood", n_aggregation=5):
    """Aggregate prediction results of multiple regression model runs.

    Parameters
    ----------
    - path (str): The directory containing CSV files with prediction results.
    - method (str): The aggregation method, either "t-statistics" (median) or "likelihood" (mean).
    - n_aggregation (int): Number of model runs to aggregate per group.

    Returns
    -------
    - List[pd.DataFrame]: A list of aggregated prediction results, one per aggregation group.

    Raises
    ------
    - NotImplementedError: If the specified method is not supported.
    """
    # Identify and sort prediction result files
    prediction_files = [f for f in os.listdir(path) if f.startswith("coef_")]
    prediction_files = sorted(prediction_files, key=lambda x: int(x.split("_")[1].split(".")[0]))

    # Group files for aggregation
    grouped_runs = [prediction_files[i : i + n_aggregation] for i in range(0, len(prediction_files), n_aggregation)]

    aggregated_predictions = []
    for group in grouped_runs:
        group_predictions = []
        for run_file in group:
            # Load prediction results
            prediction_data = pd.read_csv(os.path.join(path, run_file), index_col=0)
            group_predictions.append(prediction_data)

        # Perform aggregation based on the specified method
        if method == "t-statistics":
            aggregated_result = pd.concat(group_predictions).groupby(level=0).median()
        elif method == "likelihood":
            aggregated_result = pd.concat(group_predictions).groupby(level=0).mean()
        else:
            raise NotImplementedError("Supported methods are 't-statistics' and 'likelihood'.")

        aggregated_predictions.append(aggregated_result)

    return aggregated_predictions

import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from scipy.stats import ranksums, ttest_ind
from sklearn.metrics import roc_auc_score

import cellrank as cr
from anndata import AnnData
from regvelo import REGVELOVI
from scvelo import logging as logg


###########################
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
    perturb_GRN[
        (perturb_GRN[:, [i == gene for i in adata.var.index]].abs() > cutoff).cpu().numpy().reshape(-1),
        [i == gene for i in adata.var.index],
    ] = effects

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

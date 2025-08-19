from typing import Callable

import numpy as np
import pandas as pd
import scipy
from numpy.typing import ArrayLike
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

def pearsonr(x: ArrayLike, y: ArrayLike, axis: int = 0) -> ArrayLike:
    """Compute Pearson correlation between axes of two arrays.

    Parameters
    ----------
    x
        Input array.
    y
        Input array.
    axis
        Axis along which Pearson correlation is computed.

    Returns
    -------
    Axis-wise Pearson correlations.
    """
    centered_x = x - np.mean(x, axis=axis, keepdims=True)
    centered_y = y - np.mean(y, axis=axis, keepdims=True)

    r_num = np.add.reduce(centered_x * centered_y, axis=axis)
    r_den = np.sqrt((centered_x * centered_x).sum(axis=axis) * (centered_y * centered_y).sum(axis=axis))

    return r_num / r_den


def compute_average_correlations(matrices, method="p"):
    """Compute average correlations between pairs of matrices.

    Parameters
    ----------
    matrices
        List of NumPy arrays, where each matrix should have the same shape.
    method
        Correlation method to use:
        - `"p"`: Pearson correlation.
        - `"sp"`: Spearman correlation.
        Defaults to `"p"`.

    Returns
    -------
    list of float
        A list of average correlation values for each pair of matrices.

    Notes
    -----
    - Matrices must have the same shape for valid comparison.
    - Transposed columns of matrices are used to compute pairwise correlations.
    """
    n = len(matrices)
    assert all(mat.shape == matrices[0].shape for mat in matrices), "All matrices must have the same shape."

    correlations_list = []

    # Iterate through each pair of matrices
    for i in range(n):
        for j in range(i + 1, n):  # Avoid duplicate pairs
            mat1, mat2 = matrices[i], matrices[j]

            # Calculate average correlation for paired columns
            if method == "p":
                col_correlations = [
                    pearsonr(col1, col2)[0]  # Pearson correlation coefficient
                    for col1, col2 in zip(mat1.T, mat2.T)  # Transpose for column access
                ]
            if method == "sp":
                col_correlations = [
                    spearmanr(col1, col2)[0]  # Pearson correlation coefficient
                    for col1, col2 in zip(mat1.T, mat2.T)  # Transpose for column access
                ]

            avg_corr = np.mean(col_correlations)
            correlations_list.append(avg_corr)

    return correlations_list


def get_velocity_correlation(
    ground_truth: ArrayLike, estimated: ArrayLike, aggregation: Callable | None, axis: int = 0
) -> ArrayLike | float:
    """Compute Pearson correlation between ground truth and estimated values.

    Parameters
    ----------
    ground_truth
        Array of ground truth value.
    estimated
        Array of estimated values.
    aggregation
        If `None`, the function returns every pairwise correlation between ground truth and the estimate. If it is a
        function, the correlations are aggregated accordningly.
    axis
        Axis along which ground truth and estimate is compared.

    Returns
    -------
    Axis-wise Pearson correlations potentially aggregated.
    """
    correlation = pearsonr(ground_truth, estimated, axis=axis)

    if aggregation is None:
        return correlation
    elif callable(aggregation):
        return aggregation(correlation)


def get_time_correlation(ground_truth: ArrayLike, estimated: ArrayLike) -> float:
    """Compute Spearman correlation between ground truth and estimated values.

    Parameters
    ----------
    ground_truth
        Array of ground truth value.
    estimated
        Array of estimated values.

    Returns
    -------
    Spearman correlation.
    """
    return scipy.stats.spearmanr(ground_truth, estimated)[0]


def get_grn_auroc(ground_truth: ArrayLike, estimated: ArrayLike) -> float:
    """Compute AUROC for .

    Parameters
    ----------
    ground_truth
        Array of ground truth value.
    estimated
        Array of estimated values.

    Returns
    -------
    AUROC score.
    """
    mask = np.where(~np.eye(ground_truth.shape[0], dtype=bool))
    ground_truth = ground_truth[mask].astype(bool).astype(float)
    ground_truth[ground_truth != 0] = 1

    return roc_auc_score(ground_truth, estimated[mask])


def get_grn_auroc_cc(ground_truth: ArrayLike, estimated: ArrayLike) -> float:
    """Compute AUROC for cell cycling data.

    Parameters
    ----------
    ground_truth
        Array of ground truth value.
    estimated
        Array of estimated values.

    Returns
    -------
    AUROC score.
    """
    np.fill_diagonal(ground_truth, 0)
    np.fill_diagonal(estimated, 0)

    regulator = ground_truth.sum(1) != 0
    ground_truth = ground_truth[regulator, :]
    estimated = estimated[regulator, :]

    auc = []
    for i in range(ground_truth.shape[0]):
        auc.append(roc_auc_score(ground_truth[i, :], estimated[i, :]))

    return auc


def perturb_prediction(coef, perturbation, gene_list):
    """TODO."""
    gt = perturbation.loc[[i in coef.index for i in perturbation.loc[:, "sgRNA_group"]], :].copy()
    gt = gt.loc[
        [i in ["Pigment_gch2_high", "mNC_hox34", "mNC_arch2", "mNC_head_mesenchymal"] for i in gt.loc[:, "cell_anno"]],
        :,
    ]

    ## zero-center the likelihood of different panel.
    for tf in gene_list:
        gt.loc[[i in tf for i in gt.loc[:, "sgRNA_group"]], "median_likelihood"] = gt.loc[
            [i in tf for i in gt.loc[:, "sgRNA_group"]], "median_likelihood"
        ] - np.mean(gt.loc[[i in tf for i in gt.loc[:, "sgRNA_group"]], "median_likelihood"])

    terminal_states = ["Pigment", "mNC_hox34", "mNC_arch2", "mNC_head_mesenchymal"]
    coef = coef.loc[:, terminal_states]
    coef.columns = ["Pigment_gch2_high", "mNC_hox34", "mNC_arch2", "mNC_head_mesenchymal"]
    pred_effect = []
    TF = []
    for i in range(gt.shape[0]):
        ts = gt.iloc[i, 0]
        tf = gt.iloc[i, 2]
        effect = coef.loc[tf, ts]
        pred_effect.append(effect)
        TF.append(tf)
    pred = pd.DataFrame({"TF": TF, "effect": pred_effect})
    return scipy.stats.spearmanr(pred.iloc[:, 1], gt.iloc[:, 1])[0]

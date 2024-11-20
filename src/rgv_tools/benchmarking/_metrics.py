from typing import Callable

import numpy as np
import scipy
from numpy.typing import ArrayLike
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score


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

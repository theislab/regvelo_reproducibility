from typing import Callable

import numpy as np
import scipy
from numpy.typing import ArrayLike


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

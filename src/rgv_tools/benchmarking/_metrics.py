from typing import Callable

import numpy as np
import scipy
from numpy.typing import ArrayLike


def pearsonr(x: ArrayLike, y: ArrayLike, axis: int = 0) -> ArrayLike:
    centered_x = x - np.mean(x, axis=axis, keepdims=True)
    centered_y = y - np.mean(y, axis=axis, keepdims=True)

    r_num = np.add.reduce(centered_x * centered_y, axis=axis)
    r_den = np.sqrt((centered_x * centered_x).sum(axis=axis) * (centered_y * centered_y).sum(axis=axis))

    return r_num / r_den


def get_velocity_correlation(ground_truth: ArrayLike, estimated: ArrayLike, aggregation: Callable | None) -> ArrayLike:
    correlation = pearsonr(ground_truth, estimated)

    if aggregation is None:
        return correlation
    elif callable(aggregation):
        return aggregation(correlation)


def get_time_correlation(ground_truth: ArrayLike, estimated: ArrayLike) -> ArrayLike:
    return scipy.stats.spearmanr(ground_truth, estimated)[0]

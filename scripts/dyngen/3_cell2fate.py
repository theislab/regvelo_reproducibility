# %% [markdown]
# # cell2fate benchmark on dyngen data
#
# Notebook benchmarks velocity and latent time inference using cell2fate on dyngen-generated data.
#
# Note that cell2fate requires `anndata==0.8.0` and `scvi-tools==0.16.1`.

# %% [markdown]
# ## Library imports

# %%
import contextlib
import io

import numpy as np
import pandas as pd
import scipy
import torch

import anndata as ad
import cell2fate as c2f
import scanpy as sc

from pathlib import Path

from typing import Callable, Union
from numpy.typing import ArrayLike

# %%
DATA_DIR = Path("/lustre/groups/ml01/workspace/yifan.chen/regvelo_reproducibility/data")

# %% [markdown]
# ## Function definitions


# %%
# Function for train model and get output
def train_c2f_model(adata):
    """cell2fate pipeline."""
    c2f.Cell2fate_DynamicalModel.setup_anndata(adata, spliced_label="spliced_raw", unspliced_label="unspliced_raw")
    n_modules = c2f.utils.get_max_modules(adata)
    mod = c2f.Cell2fate_DynamicalModel(adata, n_modules=n_modules)
    mod.train()

    adata = mod.export_posterior(
        adata, sample_kwargs={"batch_size": None, "num_samples": 30, "return_samples": True, "use_gpu": False}
    )
    adata = mod.compute_module_summary_statistics(adata)
    with contextlib.redirect_stdout(io.StringIO()):
        adata.layers["Spliced Mean"] = mod.samples["post_sample_means"]["mu_expression"][..., 1]
        c2f_velocity = (
            torch.tensor(mod.samples["post_sample_means"]["beta_g"])
            * mod.samples["post_sample_means"]["mu_expression"][..., 0]
            - torch.tensor(mod.samples["post_sample_means"]["gamma_g"])
            * mod.samples["post_sample_means"]["mu_expression"][..., 1]
        )
        adata.layers["velocity"] = c2f_velocity.numpy()

    adata.layers["Ms"] = adata.layers["spliced"].copy()

    return adata


# %%
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


# %%
def get_velocity_correlation(
    ground_truth: ArrayLike, estimated: ArrayLike, aggregation: Union[Callable, None], axis: int = 0
) -> Union[ArrayLike, float]:
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


# %% [markdown]
# ## Constants

# %%
DATASET = "dyngen"

# %%
COMPLEXITY = "complexity_1"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / COMPLEXITY / "results").mkdir(parents=True, exist_ok=True)

# %%
SAVE_DATASETS = True
if SAVE_DATASETS:
    (DATA_DIR / DATASET / COMPLEXITY / "trained_cell2fate").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Velocity pipeline

# %%
import os

velocity_correlation = []

cnt = 0
for filename in (DATA_DIR / DATASET / COMPLEXITY / "processed").iterdir():
    torch.cuda.empty_cache()
    if filename.suffix != ".zarr":
        continue

    simulation_id = int(filename.stem.removeprefix("simulation_"))
    print(f"Run {cnt}, dataset {simulation_id}.")

    adata = ad.read_zarr(filename)

    ## cell2fate needs cluster information
    sc.tl.leiden(adata)

    adata = c2f.utils.get_training_data(
        adata,
        cells_per_cluster=10**5,
        cluster_column="leiden",
        remove_clusters=[],
    )

    adata = train_c2f_model(adata)

    # save data
    adata.write_zarr(DATA_DIR / DATASET / COMPLEXITY / "trained_cell2fate" / f"trained_{simulation_id}.zarr")

    velocity_correlation.append(
        get_velocity_correlation(
            ground_truth=adata.layers["true_velocity"], estimated=adata.layers["velocity"], aggregation=np.mean
        )
    )
    cnt += 1

# %%
if SAVE_DATA:
    pd.DataFrame({"velocity": velocity_correlation}).to_parquet(
        path=DATA_DIR / DATASET / COMPLEXITY / "results" / "cell2fate_correlation.parquet"
    )

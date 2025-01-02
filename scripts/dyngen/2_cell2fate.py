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

from rgv_tools import DATA_DIR

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
def get_velocity_correlation(ground_truth, estimated, aggregation, axis: int = 0):
    """Compute Pearson correlation between ground truth and estimated values."""
    # Ensure inputs are numpy arrays for easier manipulation
    ground_truth = np.asarray(ground_truth)
    estimated = np.asarray(estimated)

    # Compute correlation along the specified axis
    correlations = []
    for i in range(ground_truth.shape[0]):
        corr, _ = scipy.stats.pearsonr(ground_truth[i], estimated[i])
        correlations.append(corr)

    correlations = np.array(correlations)

    if aggregation is None:
        return correlations
    elif callable(aggregation):
        return aggregation(correlations)
    else:
        raise ValueError("Aggregation must be callable or None.")


# %% [markdown]
# ## Constants

# %%
DATASET = "dyngen"

# %%
SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "results").mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Velocity pipeline

# %%
velocity_correlation = []

for filename in (DATA_DIR / DATASET / "processed").iterdir():
    torch.cuda.empty_cache()
    if filename.suffix != ".zarr":
        continue

    adata = ad.read_zarr(filename)

    ## cell2fate need cluster information
    sc.tl.leiden(adata)

    adata = c2f.utils.get_training_data(
        adata,
        cells_per_cluster=10**5,
        cluster_column="leiden",
        remove_clusters=[],
        min_shared_counts=10,
        n_var_genes=90,
    )

    adata = train_c2f_model(adata)

    velocity_correlation.append(
        get_velocity_correlation(
            ground_truth=adata.layers["true_velocity"], estimated=adata.layers["velocity"], aggregation=np.mean
        )
    )

# %%
if SAVE_DATA:
    pd.DataFrame({"velocity": velocity_correlation}).to_parquet(
        path=DATA_DIR / DATASET / "results" / "cell2fate_correlation.parquet"
    )

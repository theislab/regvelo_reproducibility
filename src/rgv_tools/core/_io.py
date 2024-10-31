from pathlib import Path

import dask.array as da
import zarr

from anndata import AnnData
from anndata.experimental import read_elem


def read_as_dask(store: Path | str, layers: str | list[str]) -> AnnData:
    """Read AnnData with `X` and layers read with dask.

    Parameters
    ----------
    store
        Store or path to directory in file system or name of zip file.
    layers
        Layers to include in the AnnData.

    Returns
    -------
    AnnData backed with dask.
    """
    group = zarr.open(store=store)

    adata = AnnData(
        obs=read_elem(group["obs"]),
        var=read_elem(group["var"]),
        uns=read_elem(group["uns"]),
        obsm=read_elem(group["obsm"]),
        varm=read_elem(group["varm"]),
    )

    adata.X = da.from_zarr(group["X"])

    if isinstance(layers, str):
        layers = [layers]
    for layer in layers:
        adata.layers[layer] = da.from_zarr(group[f"layers/{layer}"])

    return adata

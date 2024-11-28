import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from anndata import AnnData


def set_prior_grn(adata: AnnData, gt_net: pd.DataFrame, keep_dim: bool = False) -> None:
    """Constructs a gene regulatory network (GRN) based on ground-truth interactions and gene expression data.

    Parameters
    ----------
    adata
        An annotated data matrix where `adata.X` contains gene expression data, and `adata.var` has gene identifiers.
    gt_net
        A DataFrame representing the ground-truth regulatory network with regulators as columns and targets as rows.
    keep_dim
        A boolean variable represeting if keep the output adata has the same dimensions.

    Returns
    -------
    None. Modifies `AnnData` object to include the GRN information, with network-related metadata stored in `uns`.
    """
    regulator_mask = adata.var_names.isin(gt_net.columns)
    regulators = adata.var_names[regulator_mask]

    target_mask = adata.var_names.isin(gt_net.index)
    targets = adata.var_names[target_mask]

    if keep_dim:
        skeleton = pd.DataFrame(0, index=adata.var_names, columns=adata.var_names, dtype=float)
        skeleton.loc[targets, regulators] = gt_net.loc[targets, regulators]

        gt_net = skeleton.copy()

    # Compute correlation matrix for genes
    gex = adata.layers["Ms"]
    correlation = 1 - cdist(gex.T, gex.T, metric="correlation")
    correlation = correlation[np.ix_(target_mask, regulator_mask)]
    correlation[np.isnan(correlation)] = 0

    # Filter ground-truth network and combine with correlation matrix
    grn = gt_net.loc[targets, regulators] * correlation

    # Threshold and clean the network
    grn = (grn.abs() >= 0.01).astype(int)
    np.fill_diagonal(grn.values, 0)  # Remove self-loops

    if keep_dim:
        skeleton = pd.DataFrame(0, index=targets, columns=regulators, dtype=float)
        skeleton.loc[grn.columns, grn.index] = grn.T
    else:
        grn = grn.loc[grn.sum(axis=1) > 0, grn.sum(axis=0) > 0]

        # Prepare a matrix with all unique genes from the final network
        genes = grn.index.union(grn.columns).unique()
        skeleton = pd.DataFrame(0, index=genes, columns=genes, dtype=float)
        skeleton.loc[grn.columns, grn.index] = grn.T

    # Subset the original data to genes in the network and set final properties
    adata = adata[:, skeleton.index].copy()
    skeleton = skeleton.loc[adata.var_names, adata.var_names]

    adata.uns["regulators"] = adata.var_names.to_numpy()
    adata.uns["targets"] = adata.var_names.to_numpy()
    adata.uns["skeleton"] = skeleton
    adata.uns["network"] = np.ones((adata.n_vars, adata.n_vars))


def filter_genes(adata: AnnData) -> AnnData:
    """Filter genes in an AnnData object to ensure each gene has upstream regulators.

    The function iteratively refines the skeleton matrix to maintain only genes with regulatory connections. Only used
    by `soft_constraint=False` RegVelo model.

    Parameters
    ----------
    adata
        Annotated data object (AnnData) containing gene expression data, a skeleton matrix of regulatory interactions,
        and a list of regulators.

    Returns
    -------
    adata
        Updated AnnData object with filtered genes and a refined skeleton matrix where all genes have at least one
        upstream regulator.
    """
    # Initial filtering based on regulators
    var_mask = adata.var_names.isin(adata.uns["regulators"])

    # Filter genes based on `full_names`
    adata = adata[:, var_mask].copy()

    # Update skeleton matrix
    skeleton = adata.uns["skeleton"].values
    skeleton = skeleton[np.ix_(var_mask, var_mask)]
    adata.uns["skeleton"] = skeleton

    # Iterative refinement
    while adata.uns["skeleton"].sum(0).min() == 0:
        # Update filtering based on skeleton
        skeleton = adata.uns["skeleton"]
        mask = skeleton.sum(0) > 0

        regulators = adata.var_names[mask].tolist()
        print(f"Number of genes: {len(regulators)}")

        # Filter skeleton and update `adata`
        skeleton = skeleton[np.ix_(mask, mask)]
        adata.uns["skeleton"] = skeleton

        # Update adata with filtered genes
        adata = adata[:, mask].copy()
        adata.uns["regulators"] = regulators
        adata.uns["targets"] = regulators

        # Re-index skeleton with updated gene names
        skeleton_df = pd.DataFrame(
            adata.uns["skeleton"],
            index=adata.uns["regulators"],
            columns=adata.uns["targets"],
        )
        adata.uns["skeleton"] = skeleton_df

    return adata

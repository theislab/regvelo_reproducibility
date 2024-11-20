import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

from anndata import AnnData


def prior_GRN_import(adata: AnnData, gt_net: pd.DataFrame) -> AnnData:
    """Constructs a gene regulatory network (GRN) based on ground-truth interactions and gene expression data in `adata`.

    Parameters
    ----------
    - adata: AnnData
        An annotated data matrix where `adata.X` contains gene expression data,
        and `adata.var` has gene identifiers.
    - gt_net: pd.DataFrame
        A DataFrame representing the ground-truth regulatory network with regulators as columns
        and targets as rows.

    Returns
    -------
    - AnnData: A modified `AnnData` object containing the GRN information,
               with network-related metadata stored in `uns`.
    """
    # Filter indices based on the ground-truth network
    regulator_index = [gene in gt_net.columns for gene in adata.var.index]
    target_index = [gene in gt_net.index for gene in adata.var.index]

    # Compute correlation matrix for genes
    corr_m = 1 - cdist(adata.X.todense().T, adata.X.todense().T, metric="correlation")
    corr_m = torch.tensor(corr_m).float()
    corr_m = corr_m[target_index][:, regulator_index]
    corr_m[torch.isnan(corr_m)] = 0  # Replace NaNs with zero

    # Filter ground-truth network and combine with correlation matrix
    filtered_gt_net = gt_net.loc[adata.var.index[target_index], adata.var.index[regulator_index]]
    GRN_final = filtered_gt_net * pd.DataFrame(corr_m, index=filtered_gt_net.index, columns=filtered_gt_net.columns)

    # Threshold and clean the network
    GRN_final = (GRN_final.abs() >= 0.01).astype(int)
    np.fill_diagonal(GRN_final.values, 0)  # Remove self-loops
    GRN_final = GRN_final.loc[GRN_final.sum(axis=1) > 0, GRN_final.sum(axis=0) > 0]

    # Prepare a matrix with all unique genes from the final network
    genes = np.unique(GRN_final.index.to_list() + GRN_final.columns.to_list())
    W = pd.DataFrame(0, index=genes, columns=genes, dtype=float)
    W.loc[GRN_final.columns, GRN_final.index] = GRN_final.T

    # Subset the original data to genes in the network and set final properties
    reg_bdata = adata[:, W.index].copy()
    W = torch.tensor(W.loc[reg_bdata.var.index, reg_bdata.var.index].values)

    reg_bdata.uns["regulators"] = reg_bdata.var.index.to_numpy()
    reg_bdata.uns["targets"] = reg_bdata.var.index.to_numpy()
    reg_bdata.uns["skeleton"] = W.numpy()
    reg_bdata.uns["network"] = np.ones((len(reg_bdata.var.index), len(reg_bdata.var.index)))

    return reg_bdata


def filter_genes_with_upstream_regulators(adata: AnnData) -> AnnData:
    """Filter genes in an AnnData object to ensure each gene has upstream regulators. The function iteratively refines the skeleton matrix to maintain only genes with regulatory connections. Merely used by `soft_constraint = False` regvelo model.

    Parameters
    ----------
    adata
        Annotated data object (AnnData) containing gene expression data,
        a skeleton matrix of regulatory interactions, and a list of regulators.

    Returns
    -------
    adata
        Updated AnnData object with filtered genes and a refined skeleton matrix
        where all genes have at least one upstream regulator.
    """
    # Initial filtering based on regulators
    gene_names = adata.var.index.tolist()
    full_names = adata.uns["regulators"]
    indices = [name in gene_names for name in full_names]

    # Filter genes based on `full_names`
    filtered_genes = [name for name, keep in zip(full_names, indices) if keep]
    adata = adata[:, filtered_genes].copy()

    # Update skeleton matrix
    skeleton = adata.uns["skeleton"]
    skeleton = skeleton[np.ix_(indices, indices)]
    adata.uns["skeleton"] = skeleton

    # Iterative refinement
    while adata.uns["skeleton"].sum(0).min() == 0:
        # Update filtering based on skeleton
        skeleton = np.array(adata.uns["skeleton"])
        regulators_indicator = skeleton.sum(0) > 0

        regulators = [gene for gene, keep in zip(adata.var.index.tolist(), regulators_indicator) if keep]
        print(f"Number of genes: {len(regulators)}")

        # Filter skeleton and update `adata`
        skeleton = skeleton[np.ix_(regulators_indicator, regulators_indicator)]
        adata.uns["skeleton"] = skeleton

        # Update adata with filtered genes
        adata = adata[:, regulators_indicator].copy()
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

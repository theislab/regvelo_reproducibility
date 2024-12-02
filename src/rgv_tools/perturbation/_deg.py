from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import ranksums


def DEG(gene_expression: Union[np.ndarray, pd.DataFrame], cell_labels: Union[np.ndarray, pd.Series]) -> pd.DataFrame:
    """Perform Differential Expression Analysis (DEG) for genes across cell types.

    Parameters
    ----------
    gene_expression : Union[np.ndarray, pd.DataFrame]
        A 2D array or DataFrame containing gene expression values.
        - Rows correspond to individual cells.
        - Columns correspond to genes.

    cell_labels : Union[np.ndarray, pd.Series]
        A 1D array or Series containing the cell type labels for each cell.
        The length must match the number of rows in `gene_expression`.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the results of the DEG analysis with columns:
        - 'gene': Gene name or index.
        - 'cell_type': Cell type for which the analysis was performed.
        - 'log2FC': Log2 fold change of expression for the gene in the cell type vs. others.
        - 'p_value': P-value from the Wilcoxon rank-sum test for differential expression.
    """
    # Convert gene_expression to a DataFrame if it is a NumPy array
    if isinstance(gene_expression, np.ndarray):
        df = pd.DataFrame(gene_expression)
    else:
        df = gene_expression.copy()

    # Add cell labels to the DataFrame
    df["cell_type"] = cell_labels

    # Get the unique cell types and gene names
    cell_types = np.unique(cell_labels)
    genes = df.columns[:-1]  # All columns except the last one (cell_type)

    # Initialize a list to store the results
    results = []

    for cell_type in cell_types:
        # Filter data for the current cell type and the rest
        current_group = df[df["cell_type"] == cell_type].drop(columns=["cell_type"])
        other_group = df[df["cell_type"] != cell_type].drop(columns=["cell_type"])

        for gene in genes:
            # Calculate mean expression for the current cell type and the others
            mean_current = current_group[gene].mean()
            mean_others = other_group[gene].mean()

            # Prevent division by zero in fold change calculation
            if mean_others == 0:
                mean_others = np.finfo(float).eps

            # Calculate log2 fold change
            log2_fold_change = np.log2(mean_current / mean_others)

            # Perform the Wilcoxon rank-sum test
            stat, p_value = ranksums(current_group[gene], other_group[gene])

            # Append the results
            results.append({"gene": gene, "cell_type": cell_type, "log2FC": log2_fold_change, "p_value": p_value})

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    return results_df

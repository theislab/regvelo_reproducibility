from typing import Tuple

import pandas as pd

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns

from anndata import AnnData


def bar_scores(
    test_result,
    adata: AnnData,
    color_label: str,
    gene_name: str,
    figsize: Tuple[float, float] = (4, 3),
    title="correlation with cell fate",
    min: float = 0,
    max: float = 1,
    loc: float = 0.8,
) -> None:
    """Create a bar plot to visualize the correlation coefficients of a gene with cell types.

    Parameters
    ----------
    test_result : pd.DataFrame
        Contains the results of statistical tests with columns:
        - 'coefficient': Correlation coefficients for each cell type.
        - 'pvalue': P-values for the statistical tests.
        The index should represent cell types.

    adata : AnnData
        AnnData object containing cell type annotations in `adata.obs` and color mappings in `adata.uns`.

    color_label : str
        The column in `adata.obs` that contains cell type labels.

    gene_name : str
        The gene name for which the correlation coefficients are calculated.

    figsize : Tuple[float, float], optional (default=(4, 3))
        The size of the resulting plot.

    min : float, optional (default=0)
        Minimum value for the x-axis.

    max : float, optional (default=1)
        Maximum value for the x-axis.

    loc : float, optional (default=0.8)
        Horizontal location for placing significance annotations on the plot.

    Returns
    -------
    None
        The function creates and displays a bar plot.
    """
    # Data preparation
    df = pd.DataFrame(test_result.loc[:, "coefficient"])
    df["Cell type"] = test_result.index
    df["pvalue"] = test_result["pvalue"]

    # Determine significance levels
    def significance_stars(p: float) -> str:
        """Convert p-values to significance stars."""
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.1:
            return "*"
        else:
            return ""

    df["stars"] = df["pvalue"].apply(significance_stars)
    order = test_result.index.tolist()

    # Create a color palette for the barplot
    palette = dict(zip(adata.obs[color_label].cat.categories, adata.uns[f"{color_label}_colors"]))
    subset_palette = {name: color for name, color in palette.items() if name in test_result.index.tolist()}

    # Plotting
    with mplscience.style_context():
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=figsize)

        sns.barplot(
            data=df,
            x="coefficient",
            y="Cell type",
            palette=subset_palette,
            order=order,
            ax=ax,
        )

        # Add significance annotations
        for i, cell_type in enumerate(test_result.index.tolist()):
            p_val = test_result.loc[cell_type, "pvalue"]
            if p_val < 0.001:
                level = "***"
            elif p_val < 0.01:
                level = "**"
            elif p_val < 0.05:
                level = "*"
            else:
                level = "ns"

            plt.text(loc, i, level, ha="center", va="center", color="red", fontsize=14)

        # Final adjustments
        ax.tick_params(axis="x", rotation=90)
        plt.xlim(min, max)
        plt.title(f"$\\mathit{{{gene_name}}}$ {title}")

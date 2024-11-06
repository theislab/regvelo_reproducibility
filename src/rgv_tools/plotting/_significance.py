from matplotlib.axis import Axis


# Taken from https://yoseflab.github.io/velovi_reproducibility/estimation_comparison/sceu.html
def get_significance(pvalue: float) -> str:
    """Return representation of p-value.

    Parameters
    ----------
    pvalue
        P-value informing significance.

    Returns
    -------
    String representation of p-value.
    """
    if pvalue < 0.001:
        return "***"
    elif pvalue < 0.01:
        return "**"
    elif pvalue < 0.1:
        return "*"
    else:
        return "n.s."


# Adapted from https://yoseflab.github.io/velovi_reproducibility/estimation_comparison/sceu.html
def add_significance(
    ax: Axis, left: int, right: int, significance: str, level: int = 0, orientation: str = "horizontal", **kwargs
) -> None:
    """Add significance bracket to Matplotlib Axis plot.

    Parameters
    ----------
    ax
        Matplotlib axis.
    left
        Position of left (top) subplot.
    right
        Position of right (bottom) subplot.
    siginifcance
        Significance ID to plot.
    level
        Level of bracket.
    orientation
        Orientation of bracket.
    **kwargs
        Keyword arguments passed to Axis.plot and to specify bracket level (`bracket_level`) and height
        (`bracket_height`), and distance between text and bracket (`text_height`).

    Returns
    -------
    None. Only updates Axis object.
    """
    bracket_level = kwargs.pop("bracket_level", 1)
    bracket_height = kwargs.pop("bracket_height", 0.02)
    text_height = kwargs.pop("text_height", 0.01)

    if orientation == "horizontal":
        bottom, top = ax.get_ylim()
    else:
        bottom, top = ax.get_xlim()
    axis_range = top - bottom

    bracket_level = (axis_range * 0.07 * level) + top * bracket_level
    bracket_height = bracket_level - (axis_range * bracket_height)

    if orientation == "horizontal":
        ax.plot([left, left, right, right], [bracket_height, bracket_level, bracket_level, bracket_height], **kwargs)
        ax.text(
            (left + right) * 0.5,
            bracket_level + (axis_range * text_height),
            significance,
            ha="center",
            va="bottom",
            c="k",
        )
    else:
        ax.plot([bracket_height, bracket_level, bracket_level, bracket_height], [left, left, right, right], **kwargs)
        ax.text(
            bracket_level + (axis_range * text_height),
            (left + right) * 0.5,
            significance,
            va="center",
            ha="left",
            c="k",
            rotation=-90,
        )

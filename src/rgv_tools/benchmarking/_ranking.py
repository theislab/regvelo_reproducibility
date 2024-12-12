from itertools import chain, product
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import auc

import matplotlib.pyplot as plt
import mplscience
import seaborn as sns


# Taken from https://github.com/theislab/cellrank2_reproducibility/blob/main/notebooks/labeling_kernel/method_comparison.ipynb
def get_var_ranks(
    var_names: List[str], drivers: pd.DataFrame, macrostate: str, var_type: str, model: str, threshold: int = 100
):
    """Get ranking of a set of variables towards a given macrostate."""
    _df = drivers.loc[
        var_names, [f"{macrostate}_corr", f"{macrostate}_pval", f"Corr. rank - {macrostate}"]
    ].sort_values(by=[f"Corr. rank - {macrostate}"])

    _df["Type"] = var_type
    _df["Algorithm"] = model

    print(
        f"{var_type} towards {macrostate} for {model} in top {threshold}: "
        f"{(_df[f'Corr. rank - {macrostate}'] <= threshold).sum()} (out of {_df.shape[0]})"
    )

    return _df


## Taken from https://github.com/theislab/cellrank2_reproducibility/blob/main/notebooks/labeling_kernel/method_comparison.ipynb
def get_optimal_auc(n_vars):
    """Compute AUC if given all genes are ranked first."""
    return n_vars * (n_vars + 1) / 2 + (2000 - n_vars) * n_vars


## Taken from https://github.com/theislab/cellrank2_reproducibility/blob/main/notebooks/labeling_kernel/method_comparison.ipynb
def get_gene_ranks(TERMINAL_STATES, DATA_DIR, DATASET):
    """Loads gene ranking of each method."""
    gene_ranks = {}
    for terminal_state in TERMINAL_STATES:
        # If Dynamo is included: [[f"Corr. rank - {terminal_state}", "Algorithm", "Run"]].fillna(0)
        gene_ranks[terminal_state] = (
            pd.concat(
                [
                    pd.read_csv(DATA_DIR / DATASET / "results" / f"gene_ranks_{terminal_state}-rgvelo.csv"),
                    pd.read_csv(DATA_DIR / DATASET / "results" / f"gene_ranks_{terminal_state}-scVelo.csv"),
                    pd.read_csv(DATA_DIR / DATASET / "results" / f"gene_ranks_{terminal_state}-veloVI.csv"),
                ]
            )
            .rename(columns={"Unnamed: 0": "Gene"})
            .drop_duplicates(subset=["Gene", "Algorithm"])[["Gene", f"Corr. rank - {terminal_state}", "Algorithm"]]
        )

        # gene_ranks[terminal_state].replace({"EM Model": "scVelo"}, inplace=True)
        gene_ranks[terminal_state].replace(
            {"RegVelo", "scVelo", "veloVI"},
            inplace=True,
        )

        # Random rank assignment
        np.random.seed(0)
        var_names = (
            gene_ranks[terminal_state].loc[gene_ranks[terminal_state]["Algorithm"] == "RegVelo", "Gene"].unique()
        )
        random_ranking = pd.DataFrame(
            {
                "Gene": var_names,
                f"Corr. rank - {terminal_state}": np.random.choice(np.arange(2000), size=len(var_names), replace=False),
                "Algorithm": "Random assignment",
            }
        )

        # Optimal gene ranking
        optimal_ranking = pd.DataFrame(
            {
                "Gene": var_names,
                f"Corr. rank - {terminal_state}": np.arange(len(var_names)),
                "Algorithm": "Optimal assignment",
            }
        )
        gene_ranks[terminal_state] = pd.concat([gene_ranks[terminal_state], random_ranking, optimal_ranking])
    return gene_ranks


## Taken from https://github.com/theislab/cellrank2_reproducibility/blob/main/notebooks/labeling_kernel/method_comparison.ipynb
def get_rank_threshold(gene_ranks, n_methods, TERMINAL_STATES):
    """Computes number of genes ranked below a given threshold for each method."""
    rank_threshold = np.arange(0, 2000)
    dfs = {}

    for terminal_state in TERMINAL_STATES:
        col_name = f"Corr. rank - {terminal_state}"

        if "Run" in gene_ranks[terminal_state].columns:
            dfs[terminal_state] = pd.DataFrame(
                gene_ranks[terminal_state]
                .groupby(["Algorithm", "Run"])
                .apply(lambda x: (x[col_name].values < rank_threshold.reshape(-1, 1)).sum(axis=1))  # noqa: B023
                .to_dict()
            )
            dfs[terminal_state] = pd.melt(dfs[terminal_state]).rename(
                {"variable_0": "Algorithm", "variable_1": "Run", "value": "Rank CDF"}, axis=1
            )
            dfs[terminal_state]["Rank threshold"] = np.concatenate([rank_threshold] * n_methods[terminal_state])
        else:
            dfs[terminal_state] = pd.DataFrame(
                gene_ranks[terminal_state]
                .groupby(["Algorithm"])
                .apply(lambda x: (x[col_name].values < rank_threshold.reshape(-1, 1)).sum(axis=1))  # noqa: B023
                .to_dict()
            )
            dfs[terminal_state] = pd.melt(dfs[terminal_state]).rename(
                {"variable": "Algorithm", "value": "Rank CDF"}, axis=1
            )
            dfs[terminal_state]["Rank threshold"] = np.concatenate([rank_threshold] * n_methods[terminal_state])
    return dfs


## Taken from https://github.com/theislab/cellrank2_reproducibility/blob/main/notebooks/labeling_kernel/method_comparison.ipynb
def plot_gene_ranking(
    rank_threshold, methods, TERMINAL_STATES, path, format, fname: str = "", palette: Optional[Dict[str, str]] = None
):
    """Plots number of ranked genes below a specified threshold."""
    _n_methods = list(map(len, methods.values()))
    _argmax_n_methods = np.argmax(_n_methods)
    _methods = list(methods.values())[_argmax_n_methods]
    _n_methods = _n_methods[_argmax_n_methods]

    if palette is None:
        palette = dict(zip(_methods, sns.color_palette("colorblind").as_hex()[:_n_methods]))
        palette["Optimal assignment"] = "#000000"

    with mplscience.style_context():
        sns.set_style(style="whitegrid")

        fig, ax = plt.subplots(figsize=(6 * len(TERMINAL_STATES), 4), ncols=len(TERMINAL_STATES))

        for ax_id, terminal_state in enumerate(TERMINAL_STATES):
            _df = rank_threshold[terminal_state]
            _df["line_style"] = "-"
            _df.loc[_df["Algorithm"] == "Optimal assignment", "line_style"] = "--"

            sns.lineplot(
                data=_df,
                x="Rank threshold",
                y="Rank CDF",
                hue="Algorithm",
                style="Algorithm",
                dashes={
                    "Optimal assignment": (2, 2),
                    "Random assignment": "",
                    "RegVelo": "",
                    "veloVI": "",
                    "scVelo": "",
                },
                palette=palette,
                ax=ax[ax_id],
            )
            ax[ax_id].set_title(terminal_state)
            if ax_id == 0:
                handles, labels = ax[ax_id].get_legend_handles_labels()
                # handles[3].set_linestyle("--")
            ax[ax_id].get_legend().remove()

        handles = [handles[0], handles[1], handles[2], handles[3], handles[4]]
        labels = [labels[0], labels[1], labels[2], labels[3], labels[4]]
        fig.legend(handles=handles[:6], labels=labels[:6], loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout()
        plt.show()
        fig.savefig(path, format=format)


## Taken from https://github.com/theislab/cellrank2_reproducibility/blob/main/notebooks/labeling_kernel/method_comparison.ipynb
def get_aucs(gene_ranking_dfs, optimal_aucs, methods, TERMINAL_STATES):
    """Computes area under the ranking threshold curve."""
    all_methods = list(set(chain(*methods.values())))

    # Absolute AUC
    auc_df = pd.DataFrame(index=all_methods, columns=TERMINAL_STATES, dtype=float)

    # Given AUC w.r.t. optimal AUC
    auc_rel_df = pd.DataFrame(index=all_methods, columns=TERMINAL_STATES, dtype=float)

    rank_threshold = np.arange(0, 2000)
    aucs_ = {terminal_state: {} for terminal_state in TERMINAL_STATES}
    for method, terminal_state in product(all_methods, TERMINAL_STATES):
        _df = gene_ranking_dfs[terminal_state]
        if (method == "Dynamo") and _df["Algorithm"].isin([method]).any():
            aucs_[terminal_state][method] = [
                auc(x=rank_threshold, y=_df.loc[(_df["Algorithm"] == method) & (_df["Run"] == run), "Rank CDF"].values)
                for run in _df["Run"].unique()
            ]
            auc_df.loc[method, terminal_state] = np.mean(aucs_[terminal_state][method])
            auc_rel_df.loc[method, terminal_state] = (
                auc_df.loc[method, terminal_state] / optimal_aucs[terminal_state].loc[method]
            )
        elif method == "Dynamo":
            pass
        else:
            aucs_[terminal_state][method] = auc(
                x=rank_threshold, y=_df.loc[_df["Algorithm"] == method, "Rank CDF"].values
            )
            auc_df.loc[method, terminal_state] = aucs_[terminal_state][method]
            auc_rel_df.loc[method, terminal_state] = (
                auc_df.loc[method, terminal_state] / optimal_aucs[terminal_state].loc[method]
            )
    return auc_df, auc_rel_df

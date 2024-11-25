from pathlib import Path

ROOT = Path(__file__).parents[3].resolve()


DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

METHOD_PALETTE = {
    "regvelo": "#0173b2",
    "velovi": "#de8f05",
    "scvelo": "#029e73",
    "dpt": "#949494",
    "correlation": "#949494",
    "grnboost2": "#949494",
    "celloracle": "#949494",
}

METHOD_PALETTE_DRIVER = {"RegVelo(PS)": "#0173b2", "RegVelo(CR)": "#d55e00", "dynamo(LAP)": "#cc78bc"}

METHOD_PALETTE_RANKING = {
    "RegVelo": "#0173b2",
    "veloVI": "#de8f05",
    "scVelo": "#029e73",
    "dynamo": "#c282B5",
    "Random assignment": "#949494",
    "Optimal assignment": "#000000",
}

METHOD_PALETTE_TSI = {
    "RegVelo": "#0173b2",
    "veloVI": "#de8f05",
    "scVelo": "#029e73",
}

METHOD_PALETTE_PERTURBATION = {
    "RegVelo": "#0173b2",
    "Dynamo (KO)": "#ede343",
    "Dynamo (perturbation)": "#56b4e9",
    "celloracle": "#ca9162",
}

SIGNIFICANCE_PALETTE = {"n.s.": "#dedede", "*": "#90BAAD", "**": "#A1E5AB", "***": "#ADF6B1"}

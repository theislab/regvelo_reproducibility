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

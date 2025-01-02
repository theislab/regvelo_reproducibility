# RegVelo's reproducibility repository

This repository contains the jupyter notebooks to reproduce results shown in [RegVelo: gene-regulatory-informed dynamics of single cells](https://doi.org/10.1101/2024.12.11.627935).
All datasets are freely available via a [FigShare project](https://figshare.com/account/home#/projects/226860).

## Installation

To run the analyses notebooks locally, clone and install the repository as follows:

```bash
conda create -n regvelo-py310 python=3.10 --yes && conda activate regvelo-py310

git clone https://github.com/theislab/regvelo_reproducibility.git
cd regvelo_reproducibility
pip install -e .
python -m ipykernel install --user --name regvelo-py310 --display-name "regvelo-py310"
```

## Remarks

-   Results related to the in vivo Perturb-seq and multiome data analysis can be found [here](https://github.com/zhiyhu/neural-crest-scmultiomics)

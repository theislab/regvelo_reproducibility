from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

import cellrank as cr
from anndata import AnnData
from regvelo import REGVELOVI
from scvelo import logging as logg

from ._ptools import abundance_test


def inferred_GRN(
    vae,
    adata: AnnData,
    label: str,
    group: Union[str, List[str]],
    cell_specific_grn: bool = False,
    data_frame: bool = False,
) -> Union[np.ndarray, pd.DataFrame]:
    """Infer the Gene Regulatory Network (GRN) using a trained VAE model.

    Parameters
    ----------
    vae
        Trained Variational Autoencoder (VAE) model with a `v_encoder` that supports GRN inference.
    adata
        Annotated data matrix containing cell-specific information and gene expression layers.
    label
        Column name in `adata.obs` specifying cell type or grouping labels.
    group
        Group or groups of cells to analyze. Use `"all"` for a global GRN or specify one or more groups.
    cell_specific_grn
        If `True`, compute cell-specific GRN using individual cell data. Defaults to `False`.
    data_frame
        If `True` and `cell_specific_grn` is `False`, return the GRN as a Pandas DataFrame. Defaults to `False`.

    Returns
    -------
    np.ndarray or pd.DataFrame
        The inferred GRN as a NumPy array or Pandas DataFrame. For cell-specific GRN, always returns a NumPy array.
    """
    if cell_specific_grn is not True:
        # Retrieve unique cell types or groups from the specified label
        cell_types = np.unique(adata.obs[label])

        if group == "all":
            print("Computing global GRN...")
        else:
            # Subset the data to include only specified groups, raising an error for invalid groups
            if all(elem in cell_types for elem in group):
                adata = adata[adata.obs[label].isin(group)]
            else:
                raise TypeError(f"The group label contains elements not present in `adata.obs[{label}]`.")

        # Compute the GRN using the VAE's encoder and global mean gene expression
        GRN = (
            vae.module.v_encoder.GRN_Jacobian(torch.tensor(adata.layers["Ms"].mean(0)).to("cuda:0"))
            .detach()
            .cpu()
            .numpy()
        )
    else:
        # Compute the cell-specific GRN using the VAE's encoder for each cell
        GRN = vae.module.v_encoder.GRN_Jacobian2(torch.tensor(adata.layers["Ms"]).to("cuda:0")).detach().cpu().numpy()

    # Normalize GRN by the mean absolute non-zero values
    GRN = GRN / np.mean(np.abs(GRN)[GRN != 0])

    # Convert to a DataFrame if requested and not cell-specific
    if cell_specific_grn is not True and data_frame:
        GRN = pd.DataFrame(
            GRN,
            index=adata.var.index.tolist(),
            columns=adata.var.index.tolist(),
        )

    return GRN


def in_silico_block_regulation_simulation(
    model: str, adata: AnnData, regulator: str, target: str, n_samples: int = 50, effects: float = 0
) -> AnnData:
    """Simulate in-silico blocking of a specific regulation between a regulator and a target gene.

    Parameters
    ----------
    model
        Path to the pretrained REGVELOVI model.
    adata
        Annotated data matrix containing gene expression and cell-specific information.
    regulator
        Name of the regulator gene whose effect on the target gene will be blocked.
    target
        Name of the target gene affected by the regulator.
    n_samples
        Number of samples to generate during the simulation. Default is 50.
    effects
        Effect size to simulate the blocked regulation. Defaults to 0 (complete block).

    Returns
    -------
    AnnData
        Updated AnnData object with perturbation results added as new outputs.
    """
    reg_vae_perturb = REGVELOVI.load(model, adata)
    perturb_GRN = reg_vae_perturb.module.v_encoder.fc1.weight.detach().clone()
    perturb_GRN[[i == target for i in adata.var.index], [i == regulator for i in adata.var.index]] = effects

    reg_vae_perturb.module.v_encoder.fc1.weight.data = perturb_GRN
    adata_target_perturb = reg_vae_perturb.add_regvelo_outputs_to_adata(adata=adata, n_samples=n_samples)

    return adata_target_perturb


def RegulationScanning(
    model_path: str,
    adata: AnnData,
    n_states: int,
    cluster_label: str,
    terminal_states: Optional[List[str]],
    TF: List[str],
    target: List[str],
    effect: float = 1e-3,
) -> Dict[str, Union[List[str], List[pd.Series]]]:
    """Perform transcription factor scanning and perturbation analysis on a gene regulatory network.

    Parameters
    ----------
    model_path : str
        Path to the model file.
    adata : AnnData
        Annotated data matrix.
    n_states : int
        Number of states in the model.
    cluster_label : str
        Label identifying clusters in the data.
    terminal_states : Optional[List[str]]
        List of terminal state labels.
    TF : List[str]
        List of transcription factors.
    effect : float, default=1e-3
        Effect size parameter to adjust the sensitivity of the scanning.

    Returns
    -------
    Dict[str, Union[List[str], List[pd.Series]]]
        A dictionary with keys corresponding to results of the scanning.
    """
    reg_vae = REGVELOVI.load(model_path, adata)
    adata = reg_vae.add_regvelo_outputs_to_adata(adata=adata, n_samples=50)

    ## curated all targets of specific TF
    # built kernel matrix
    vk = cr.kernels.VelocityKernel(adata).compute_transition_matrix()
    ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
    combined_kernel = 0.8 * vk + 0.2 * ck
    g2 = cr.estimators.GPCCA(combined_kernel)
    ## evaluate the fate prob on original space
    g2.compute_macrostates(n_states=n_states, n_cells=30, cluster_key=cluster_label)

    ## predict cell fate probabilities
    if terminal_states is None:
        g2.predict_terminal_states()
        terminal_states = g2.terminal_states.cat.categories.tolist()
    g2.set_terminal_states(terminal_states)
    g2.compute_fate_probabilities(solver="direct")
    fate_prob = g2.fate_probabilities
    sampleID = adata.obs.index.tolist()
    fate_name = fate_prob.names.tolist()
    fate_prob = pd.DataFrame(fate_prob, index=sampleID, columns=fate_name)
    fate_prob_original = fate_prob.copy()
    ##################

    n_states = len(g2.macrostates.cat.categories.tolist())
    coef = []
    pvalue = []
    for gene in target:
        adata_target = in_silico_block_regulation_simulation(model_path, adata, TF, gene, effects=effect)
        ## perturb the regulations
        vk = cr.kernels.VelocityKernel(adata_target)
        vk.compute_transition_matrix()
        ck = cr.kernels.ConnectivityKernel(adata_target).compute_transition_matrix()
        combined_kernel = 0.8 * vk + 0.2 * ck
        g2 = cr.estimators.GPCCA(combined_kernel)
        ## evaluate the fate prob on original space
        n_states_perturb = n_states
        while True:
            try:
                # Perform some computation in f(a)
                g2.compute_macrostates(n_states=n_states_perturb, n_cells=30, cluster_key=cluster_label)

                break
            except:
                n_states_perturb += 1
                vk = cr.kernels.VelocityKernel(adata)
                vk.compute_transition_matrix()
                ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
                g = cr.estimators.GPCCA(0.8 * vk + 0.2 * ck)
                ## evaluate the fate prob on original space
                g.compute_macrostates(n_states=n_states_perturb, n_cells=30, cluster_key=cluster_label)
                ## set a high number of states, and merge some of them and rename
                if terminal_states is None:
                    g.predict_terminal_states()
                    terminal_states = g.terminal_states.cat.categories.tolist()
                g.set_terminal_states(terminal_states)
                g.compute_fate_probabilities(solver="direct")
                fate_prob = g.fate_probabilities
                sampleID = adata.obs.index.tolist()
                fate_name = fate_prob.names.tolist()
                fate_prob = pd.DataFrame(fate_prob, index=sampleID, columns=fate_name)
                raise

        ## intersection the states
        terminal_states_perturb = g2.macrostates.cat.categories.tolist()
        terminal_states_perturb = list(set(terminal_states_perturb).intersection(terminal_states))

        g2.set_terminal_states(terminal_states_perturb)
        g2.compute_fate_probabilities(solver="direct")
        fb = g2.fate_probabilities
        sampleID = adata.obs.index.tolist()
        fate_name = fb.names.tolist()
        fb = pd.DataFrame(fb, index=sampleID, columns=fate_name)
        fate_prob2 = pd.DataFrame(columns=terminal_states, index=sampleID)

        for i in terminal_states_perturb:
            fate_prob2.loc[:, i] = fb.loc[:, i]

        fate_prob2 = fate_prob2.fillna(0)
        arr = np.array(fate_prob2.sum(0))
        arr[arr != 0] = 1
        fate_prob = fate_prob * arr

        fate_prob2.index = [i + "_perturb" for i in fate_prob2.index]
        test_result = abundance_test(fate_prob, fate_prob2)
        coef.append(test_result.loc[:, "coefficient"])
        pvalue.append(test_result.loc[:, "FDR adjusted p-value"])
        logg.info("Done " + gene)
        fate_prob = fate_prob_original.copy()

    d = {"target": target, "coefficient": coef, "pvalue": pvalue}
    return d

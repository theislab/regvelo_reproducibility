import logging as logg  # For logging operations

import pandas as pd  # For DataFrame manipulation

from celloracle.applications import Oracle_development_module

from ._ptools import combine_elements


def pipeline_single(oracle, gradient, gene_for_KO, index_dictionary, n_neighbors):
    """TODO."""
    # 1. Simulate KO
    oracle.simulate_shift(perturb_condition={gene_for_KO: 0}, ignore_warning=True, n_propagation=3)
    oracle.estimate_transition_prob(n_neighbors=n_neighbors, knn_random=True, sampled_fraction=1)
    oracle.calculate_embedding_shift(sigma_corr=0.05)

    ## calculate overall score
    dev = Oracle_development_module()
    # Load development flow
    dev.load_differentiation_reference_data(gradient_object=gradient)
    # Load simulation result
    dev.load_perturb_simulation_data(oracle_object=oracle, n_neighbors=n_neighbors)
    # Calculate inner product
    dev.calculate_inner_product()
    dev.calculate_digitized_ip(n_bins=10)

    # Save results in a hdf5 file.
    ps_max = dev.get_negative_PS_p_value(return_ps_sum=True, plot=False)[1]

    # Do simulation for all conditions.
    Lineage = []
    PS_score = []
    for lineage_name, cell_idx in index_dictionary.items():
        dev = Oracle_development_module()
        # Load development flow
        dev.load_differentiation_reference_data(gradient_object=gradient)
        # Load simulation result
        dev.load_perturb_simulation_data(
            oracle_object=oracle, cell_idx_use=cell_idx, name=lineage_name, n_neighbors=n_neighbors
        )
        # Calculate inner product
        dev.calculate_inner_product()
        dev.calculate_digitized_ip(n_bins=10)

        # Save results in a hdf5 file.
        ps = dev.get_negative_PS_p_value(return_ps_sum=True, plot=False)[1]
        ps = ps / ps_max
        Lineage.append(lineage_name)
        PS_score.append(ps)

    df = pd.DataFrame({"Lineage": Lineage, "PS_score": PS_score})
    df.index = Lineage

    return df


def pipeline_multiple(oracle, gradient, gene_for_KO, index_dictionary, n_neighbors):
    """TODO."""
    oracle.simulate_shift(perturb_condition={key: 0 for key in gene_for_KO}, ignore_warning=True, n_propagation=3)
    oracle.estimate_transition_prob(n_neighbors=n_neighbors, knn_random=True, sampled_fraction=1)
    oracle.calculate_embedding_shift(sigma_corr=0.05)

    ## calculate overall score
    dev = Oracle_development_module()
    # Load development flow
    dev.load_differentiation_reference_data(gradient_object=gradient)
    # Load simulation result
    dev.load_perturb_simulation_data(oracle_object=oracle, n_neighbors=n_neighbors)
    # Calculate inner product
    dev.calculate_inner_product()
    dev.calculate_digitized_ip(n_bins=10)

    # Save results in a hdf5 file.
    ps_max = dev.get_negative_PS_p_value(return_ps_sum=True, plot=False)[1]

    # Do simulation for all conditions.
    Lineage = []
    PS_score = []
    for lineage_name, cell_idx in index_dictionary.items():
        dev = Oracle_development_module()
        # Load development flow
        dev.load_differentiation_reference_data(gradient_object=gradient)
        # Load simulation result
        dev.load_perturb_simulation_data(
            oracle_object=oracle, cell_idx_use=cell_idx, name=lineage_name, n_neighbors=n_neighbors
        )
        # Calculate inner product
        dev.calculate_inner_product()
        dev.calculate_digitized_ip(n_bins=10)

        # Save results in a hdf5 file.
        ps = dev.get_negative_PS_p_value(return_ps_sum=True, plot=False)[1]
        ps = ps / ps_max
        Lineage.append(lineage_name)
        PS_score.append(ps)

    df = pd.DataFrame({"Lineage": Lineage, "PS_score": PS_score})
    df.index = Lineage

    return df


def TFScanning_perturbation_co(
    adata, n_states, cluster_label, terminal_states, TF, oracle, gradient, index_dictionary, n_neighbors
):
    """TODO."""
    coef = []
    for tf in TF:
        ## TODO: mask using dynamo
        ## each time knock-out a TF
        df = pipeline_single(oracle, gradient, tf, index_dictionary, n_neighbors)
        coef.append(df.loc[:, "PS_score"])
        logg.info("Done " + tf)
    d = {"TF": TF, "coefficient": coef}
    # df = pd.DataFrame(data=d)
    return d


def Multiple_TFScanning_perturbation_co(
    data, n_states, cluster_label, terminal_states, TF_pair, oracle, gradient, index_dictionary, n_neighbors
):
    """TODO."""
    coef = []
    for tf in TF_pair:
        ## TODO: mask using dynamo
        ## each time knock-out a TF
        df = pipeline_multiple(oracle, gradient, tf, index_dictionary, n_neighbors)
        coef.append(df.loc[:, "PS_score"])
        logg.info("Done " + combine_elements([tf])[0])
    d = {"TF": combine_elements(TF_pair), "coefficient": coef}
    # df = pd.DataFrame(data=d)
    return d

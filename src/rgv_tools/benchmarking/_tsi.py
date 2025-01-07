from typing import Any, List, Set

from anndata import AnnData


def generate_sequence(k: int, n: int) -> List[int]:
    """Generates a sequence of numbers from 1 to k. If the length of the sequence is less than n, the remaining positions are filled with the value k.

    Parameters
    ----------
        k: The last value to appear in the initial sequence.
        n: The target length of the sequence.

    Returns
    -------
        List[int]: A list of integers from 1 to k, padded with k to length n if necessary.
    """
    sequence = list(range(1, k + 1))

    # If the length of the sequence is already >= n, trim it to n
    if len(sequence) >= n:
        return sequence[:n]

    # Fill the rest of the sequence with the number k
    sequence.extend([k] * (n - len(sequence)))

    return sequence


def plot_tsi(
    adata: AnnData,
    kernel: Any,
    threshold: float,
    terminal_states: Set[str],
    cluster_key: str,
    max_states: int = 12,
) -> List[int]:
    """Calculate the number of unique terminal states for each macrostate count.

    Parameters
    ----------
    adata
        Annotated data matrix (e.g., from single-cell experiments).
    kernel
        Computational kernel used to compute macrostates and predict terminal states.
    threshold
        Stability threshold for predicting terminal states.
    terminal_states
        Set of known terminal state names to match against.
    cluster_key
        Key in `adata.obs` that identifies cluster assignments for cells.
    max_states
        Maximum number of macrostates to consider. Default is 12.

    Returns
    -------
    list of int
        A list where each entry represents the count of unique terminal states found
        at each macrostate count from 1 to `max_states`.
    """
    # Create a mapping of state identifiers to their corresponding types
    all_states = list(set(adata.obs[cluster_key].tolist()))
    all_id = all_states.copy()
    all_type = all_states.copy()
    for state in all_states:
        for i in range(1, max_states + 1):
            all_id.append(f"{state}_{i}")
            all_type.append(state)
    all_dict = dict(zip(all_id, all_type))

    pre_value = []

    for num_macro in range(1, max_states):
        try:
            # Compute macrostates and predict terminal states
            kernel.compute_macrostates(n_states=num_macro, cluster_key=cluster_key)
            kernel.predict_terminal_states(stability_threshold=threshold)

            # Map terminal states to their types using `all_dict`
            pre_terminal = kernel.terminal_states.cat.categories.tolist()
            subset_dict = {key: all_dict[key] for key in pre_terminal}
            pre_terminal_names = list(set(subset_dict.values()))

            # Count overlap with known terminal states
            pre_value.append(len(set(pre_terminal_names).intersection(terminal_states)))

        except:  # noqa
            # Log error and repeat the last valid value or use 0 if empty
            pre_value.append(pre_value[-1] if pre_value else 0)

    return pre_value


def get_tsi_score(
    adata: AnnData,
    points: List[float],
    cluster_key: str,
    terminal_states: Set[str],
    kernel: Any,
    max_states: int = 12,
) -> List[float]:
    """Calculate the Terminal State Integration (TSI) score for given thresholds.

    Parameters
    ----------
    adata
        Annotated data matrix (e.g., from single-cell experiments).
    points
        List of threshold values to evaluate for stability of terminal states.
    cluster_key
        Key in `adata.obs` to access cluster assignments for cells.
    terminal_states
        Set of known terminal state names to match against.
    kernel
        Computational kernel used to compute macrostates and predict terminal states.
    max_states
        Maximum number of macrostates to consider. Default is 12.

    Returns
    -------
    list of float
        A list of TSI scores, one for each threshold in `points`. Each score represents
        the normalized area under the staircase function compared to the goal sequence.
    """
    # Define the goal sequence and calculate its area
    x_values = range(max_states)
    y_values = [0] + generate_sequence(len(terminal_states), max_states - 1)
    area_gs = sum((x_values[i + 1] - x_values[i]) * y_values[i] for i in range(len(x_values) - 1))

    tsi_score = []

    for threshold in points:
        # Compute the staircase function for the current threshold
        pre_value = plot_tsi(adata, kernel, threshold, terminal_states, cluster_key, max_states)
        y_values = [0] + pre_value

        # Calculate the area under the staircase function
        area_velo = sum((x_values[i + 1] - x_values[i]) * y_values[i] for i in range(len(x_values) - 1))

        # Compute the normalized TSI score and append to results
        tsi_score.append(area_velo / area_gs if area_gs else 0)

    return tsi_score

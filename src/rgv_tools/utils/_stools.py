from typing import Dict


def reverse_cluster(cluster: str) -> str:
    """Reverses the order of parts in a cluster string.

    Parameters
    ----------
    - cluster: The string to be reversed, expected to have parts separated by ' - '.

    Returns
    -------
    - A new string with parts in reverse order.
    """
    parts = cluster.split(" - ")
    reversed_parts = " - ".join(parts[::-1])
    return reversed_parts


def reverse_key(key: str) -> str:
    """Reverses the order of parts in a key string.

    Parameters
    ----------
    - key: The string to be reversed, expected to have parts separated by ' - '.

    Returns
    -------
    - A new string with parts in reverse order.
    """
    parts = key.split(" - ")
    reversed_parts = " - ".join(parts[::-1])
    return reversed_parts


def reverse_cluster_dict(cluster_dict: Dict[str, any]) -> Dict[str, any]:
    """Reverses the keys of a dictionary where keys are strings with parts separated by ' - '.

    Parameters
    ----------
    - cluster_dict: A dictionary where keys are strings to be reversed.

    Returns
    -------
    - A new dictionary with reversed keys.
    """
    reversed_dict = {}
    for key, value in cluster_dict.items():
        reversed_key = reverse_cluster(key)
        reversed_dict[reversed_key] = value

    return reversed_dict

"""
Functions for analyzing and manipulating network connectivity.
"""

import numpy as np
from collections import deque
from typing import List, Dict, Any, Set


def find_connected_components(network: Dict[str, np.ndarray]) -> List[List[int]]:
    """
    Find all connected components in the network using breadth-first search.

    Parameters
    ----------
    network : dict
        Network dictionary containing 'throat.conns' with shape (num_throats, 2)
        defining connections between pores.

    Returns
    -------
    components : list of lists
        Each inner list contains the pore indices belonging to one connected component.

    Examples
    --------
    >>> components = find_connected_components(network)
    >>> print(f"Found {len(components)} components")
    >>> print(f"Largest component has {len(max(components, key=len))} pores")
    """
    conns = network['throat.conns']
    num_pores = len(network['pore.all'])

    # Build adjacency list
    adjacency = [[] for _ in range(num_pores)]
    for pore_i, pore_j in conns:
        adjacency[pore_i].append(pore_j)
        adjacency[pore_j].append(pore_i)

    # BFS to find all connected components
    visited = np.zeros(num_pores, dtype=bool)
    components = []

    for start_pore in range(num_pores):
        if visited[start_pore]:
            continue

        # Start new component with BFS
        component = []
        queue = deque([start_pore])
        visited[start_pore] = True

        while queue:
            pore = queue.popleft()
            component.append(pore)

            for neighbor in adjacency[pore]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        components.append(component)

    return components


def trim_pores(network: Dict[str, np.ndarray], pores_to_remove: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Remove specified pores and their connected throats from the network.

    This function:
    1. Removes all pore properties for the specified pores
    2. Removes all throats connected to these pores
    3. Reindexes remaining pores to be contiguous (0, 1, 2, ...)
    4. Updates throat.conns to reflect the new pore indices

    Parameters
    ----------
    network : dict
        Network dictionary with pore.* and throat.* properties.
    pores_to_remove : array_like
        Indices of pores to remove from the network.

    Returns
    -------
    trimmed_network : dict
        New network dictionary with specified pores and their throats removed.

    Examples
    --------
    >>> # Remove pores [5, 10, 15]
    >>> trimmed = trim_pores(network, [5, 10, 15])
    >>> print(f"Original: {len(network['pore.all'])} pores")
    >>> print(f"Trimmed: {len(trimmed['pore.all'])} pores")
    """
    pores_to_remove = np.asarray(pores_to_remove)
    num_pores = len(network['pore.all'])

    # Create mask for pores to keep
    pores_to_keep_mask = np.ones(num_pores, dtype=bool)
    pores_to_keep_mask[pores_to_remove] = False
    pores_to_keep = np.where(pores_to_keep_mask)[0]

    # Create mapping from old pore index to new pore index
    old_to_new = np.full(num_pores, -1, dtype=np.int32)
    old_to_new[pores_to_keep] = np.arange(len(pores_to_keep), dtype=np.int32)

    # Find throats to keep (both pores must be kept)
    conns = network['throat.conns']
    throat_keep_mask = pores_to_keep_mask[conns[:, 0]] & pores_to_keep_mask[conns[:, 1]]
    throats_to_keep = np.where(throat_keep_mask)[0]

    # Build new network
    trimmed_network = {}

    # Copy and trim pore properties
    for key in network.keys():
        if key.startswith('pore.'):
            trimmed_network[key] = network[key][pores_to_keep_mask]

    # Copy and trim throat properties
    for key in network.keys():
        if key.startswith('throat.'):
            if key == 'throat.conns':
                # Update throat.conns with new pore indices
                old_conns = network[key][throat_keep_mask]
                new_conns = old_to_new[old_conns]
                trimmed_network[key] = new_conns
            else:
                trimmed_network[key] = network[key][throat_keep_mask]

    return trimmed_network


def remove_disconnected_components(network: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Remove all pores and throats that are not part of the largest connected component.

    This function:
    1. Finds all connected components in the network
    2. Identifies the largest component by number of pores
    3. Removes all pores not in the largest component
    4. Returns a trimmed network containing only the largest component

    Parameters
    ----------
    network : dict
        Network dictionary with pore.* and throat.* properties.

    Returns
    -------
    trimmed_network : dict
        Network containing only the largest connected component.

    Examples
    --------
    >>> trimmed = remove_disconnected_components(network)
    >>> # Verify it's fully connected
    >>> components = find_connected_components(trimmed)
    >>> assert len(components) == 1, "Result should have exactly one component"

    Notes
    -----
    If the network is already fully connected, this function returns a copy
    with no pores removed.
    """
    # Find all connected components
    components = find_connected_components(network)

    # Find the largest component
    largest_component = max(components, key=len)
    largest_component_set = set(largest_component)

    # Find pores to remove (all pores not in largest component)
    num_pores = len(network['pore.all'])
    pores_to_remove = [i for i in range(num_pores) if i not in largest_component_set]

    # If no pores to remove, return a copy of the network
    if len(pores_to_remove) == 0:
        return {key: value.copy() for key, value in network.items()}

    # Trim the network
    trimmed_network = trim_pores(network, np.array(pores_to_remove))

    return trimmed_network

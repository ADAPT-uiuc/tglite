import os
import numpy as np

from . import _c
from ._core import TError


def get_num_cpus(default=16) -> int:
    cpus = os.cpu_count()
    return default if cpus is None else cpus


def check_edges_times(edges: np.ndarray, times: np.ndarray):
    if edges.shape[0] != times.shape[0]:
        raise TError("edge list and timestamps must have same leading dimension")
    if edges.shape[1] != 2:
        raise TError("edge list must have only 2 columns")
    if edges.dtype != np.int32:
        raise TError("currently only supports int32 node/edge ids")
    if times.dtype != np.float32:
        raise TError("currently only supports float32 timestamps")


def check_num_nodes(edges: np.ndarray, num_nodes: int = None) -> int:
    """Returns the number of nodes in the graph represented by the given edges
    
    :raises TErrror: if the specified number of nodes is less than or equal to the number of distinct nodes present in the edges
    """
    max_nid = int(edges.max())
    num_nodes = max_nid + 1 \
        if num_nodes is None else num_nodes
    if num_nodes <= max_nid:
        raise TError("number of nodes must be greater than max node id")
    return num_nodes


def create_tcsr(edges: np.ndarray, times: np.ndarray, num_nodes: int = None):
    check_edges_times(edges, times)
    num_nodes = check_num_nodes(edges, num_nodes)
    return _c.create_tcsr(edges, times, num_nodes)

import torch
import numpy as np
from torch import Tensor
from pathlib import Path
from typing import Optional, Union

from ._core import TError
from ._frame import TFrame
from ._memory import Memory
from ._mailbox import Mailbox
from ._utils import create_tcsr, check_edges_times, check_num_nodes


class TGraph(object):
    """A container for temporal graph and related tensor data. It initially stores temporal edges in COO format,
    sorted based on timestamp. While performing neighborhood sampling, it uses CSR format for faster lookups. TGLite
    automatically handles the construction and management of these graph formats without intervention from the user."""

    def __init__(self, edges: np.ndarray, times: np.ndarray, num_nodes: int = None):
        """
        Internal constructor for creating a TGraph

        :param np.ndarray edges:
        :param np.ndarray times:
        :param int num_nodes:
        """
        check_edges_times(edges, times)
        self._num_nodes = check_num_nodes(edges, num_nodes)
        self._efeat_frame = TFrame(dim=edges.shape[0])
        self._nfeat_frame = TFrame(dim=self._num_nodes)
        self._edata = TFrame(dim=edges.shape[0])
        self._ndata = TFrame(dim=self._num_nodes)
        self._edges = edges
        self._times = times
        self._tcsr = None
        self._mem = None
        self._mailbox = None
        self._storage_dev = torch.device('cpu')
        self._compute_dev = torch.device('cpu')

    @property
    def efeat(self) -> Optional[Tensor]:
        """Returns edge feature"""
        return self._efeat_frame.get('f')

    @efeat.setter
    def efeat(self, value):
        """
        Sets edge feature

        :param value: edge feature
        """
        if value is None:
            self._efeat_frame.clear()
        else:
            self._efeat_frame['f'] = value

    @property
    def nfeat(self) -> Optional[Tensor]:
        """Returns node feature"""
        return self._nfeat_frame.get('f')

    @nfeat.setter
    def nfeat(self, value):
        """
        Sets node feature

        :param value: edge feature
        """
        if value is None:
            self._nfeat_frame.clear()
        else:
            self._nfeat_frame['f'] = value

    @property
    def edata(self) -> TFrame:
        """Returns edge data"""
        return self._edata

    @property
    def ndata(self) -> TFrame:
        """Returns node data"""
        return self._ndata

    @property
    def mem(self) -> Optional[Memory]:
        """Returns node memory"""
        return self._mem

    @mem.setter
    def mem(self, value: Memory):
        """
        Sets node memory

        :param Memory value: node memory to set
        :raises TError: if value is not a Memory instance or its length doesn't equal to number of nodes,
        or value is not on this TGraph's storage device.
        """
        if not isinstance(value, Memory):
            raise TError('invalid memory object')
        if len(value) != self._num_nodes:
            raise TError('memory number of nodes mismatch')
        if value.device != self._storage_dev:
            raise TError('memory storage device mismatch')
        self._mem = value

    @property
    def mailbox(self) -> Optional[Mailbox]:
        """Returns node mailbox"""
        return self._mailbox

    @mailbox.setter
    def mailbox(self, value: Mailbox):
        """
        Sets mailbox

        :param Mailbox value: mailbox to set
        :raises TError: if value is not a Mailbox instance or its length doesn't equal to number of nodes,
        or value is not on this TGraph's storage device.
        """
        if not isinstance(value, Mailbox):
            raise TError('invalid mailbox object')
        if value.device != self._storage_dev:
            raise TError('mailbox storage device mismatch')
        # ... more checks here ...
        self._mailbox = value

    def storage_device(self) -> torch.device:
        """Returns TGraph's storage device"""
        return self._storage_dev

    def compute_device(self) -> torch.device:
        """Returns TGraph's computing device"""
        return self._compute_dev

    def num_nodes(self) -> int:
        """
        Total number of nodes

        :rtype: int
        """
        return self._num_nodes

    def num_edges(self) -> int:
        """
        Total number of edges

        :rtype: int
        """
        return self._edges.shape[0]

    def set_compute(self, device):
        """Sets computing device"""
        self._compute_dev = torch.device(device)

    def move_data(self, device, **kwargs):
        """Moves tensor data to device while keeping graph on CPU"""
        if self._storage_dev == device:
            return
        self._efeat_frame = self._efeat_frame.to(device, **kwargs)
        self._nfeat_frame = self._nfeat_frame.to(device, **kwargs)
        self._edata = self._edata.to(device, **kwargs),
        self._ndata = self._ndata.to(device, **kwargs),
        if self._mem is not None:
            self._mem.move_to(device, **kwargs)
        if self._mailbox is not None:
            self._mailbox.move_to(device, **kwargs)
        self._storage_dev = device

    def _init_tcsr(self):
        """Creates tcsr of the graph if it doesn't exist"""
        if self._tcsr is None:
            self._tcsr = create_tcsr(self._edges, self._times, num_nodes=self._num_nodes)

    def _get_tcsr(self):
        """Returns the tcsr of the graph"""
        self._init_tcsr()
        return self._tcsr


def from_csv(path: Union[str, Path], skip_first=True) -> TGraph:
    """
    Creates a TGraph from a csv file

    :param path: csv file path
    :type path: str or Path
    :param bool skip_first: whether to skip the first line
    :rtype: TGraph
    :raises TError: if path doesn't exist
    """
    src, dst, ts = [], [], []

    path = Path(path)
    if not path.exists():
        raise TError(f'file does not exist: {path}')

    with path.open() as file:
        if skip_first:
            next(file)
        for line in file:
            line = line.strip().split(',')
            src.append(int(line[0]))
            dst.append(int(line[1]))
            ts.append(float(line[2]))

    src = np.array(src, dtype=np.int32).reshape(-1, 1)
    dst = np.array(dst, dtype=np.int32).reshape(-1, 1)
    edges = np.concatenate([src, dst], axis=1)
    del src
    del dst

    etime = np.array(ts, dtype=np.float32)
    del ts

    return TGraph(edges, etime)

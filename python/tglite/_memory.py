import torch
import numpy as np
from torch import Tensor
from typing import Tuple, Union

from ._core import TError


class Memory(object):
    """A container for node memory."""

    def __init__(self, num_nodes: int, dim: int, device=None):
        """
        Internal constructor for creating a Memory. Initialize node memory and timestamps to zero.

        :param int num_nodes: Number of nodes
        :param int dim: Length of memory vector for a single node
        :param device: Which device to put node memory
        :type device: None or str or torch.device
        """
        self._device = torch.device('cpu' if device is None else device)

        self._data = torch.zeros((num_nodes, dim), device=self._device)
        self._time = torch.zeros(num_nodes, device=self._device)

        if list(self._data.shape) != [num_nodes, dim]:
            raise TError('memory data dimension mismatch')
        if self._time.shape[0] != num_nodes:
            raise TError('memory timestamp dimension mismatch')

    def __len__(self) -> int:
        """
        Return number of nodes in Memory

        :rtype: int
        """
        return self._data.shape[0]

    @property
    def data(self) -> Tensor:
        """Return node memory

        :rtype: Tensor
        """
        return self._data

    @property
    def time(self) -> Tensor:
        """Return timestamps of current node memory

        :rtype: Tensor
        """
        return self._time

    @property
    def device(self) -> torch.device:
        """Return the device where Memory is located

        :rtype: torch.device
        """
        return self._device

    def dim(self) -> int:
        """Return length of memory vector for a single node"""
        return self._data.shape[1]

    def reset(self):
        """Reset node memory and timestamps to zero"""
        self._data.zero_()
        self._time.zero_()

    def update(self, nids: Union[np.ndarray, Tensor], newdata: Tensor, newtime: Tensor):
        if not isinstance(nids, Tensor):
            nids = torch.from_numpy(nids).long()
        nids = nids.to(self._device)
        self._data[nids] = newdata.detach().to(self._device)
        self._time[nids] = newtime.detach().to(self._device)

    def move_to(self, device, **kwargs):
        if device is None or self._device == device:
            return
        self._data = self._data.to(device, **kwargs)
        self._time = self._time.to(device, **kwargs)
        self._device = device

    def backup(self) -> Tuple[Tensor, Tensor]:
        return (self._data.cpu().clone(), self._time.cpu().clone())

    def restore(self, state: Tuple[Tensor, Tensor]):
        data, time = state
        if self._data.shape != data.shape:
            raise TError('memory data dimension mismatch')
        if self._time.shape != time.shape:
            raise TError('memory timestamp dimension mismatch')
        self._data = data.clone().to(self._device)
        self._time = time.clone().to(self._device)

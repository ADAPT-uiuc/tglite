from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional, Tuple
if TYPE_CHECKING:
    from ._graph import TGraph
    from ._context import TContext

import torch
import numpy as np
from torch import Tensor

from ._core import TError
from ._frame import TFrame
from ._stats import tt


class TBlock(object):
    """Captures 1-hop relations between node/time pairs and their neighbors for doing computations, such as segmented
     softmax and message-passing aggregation."""

    def __init__(self, ctx: 'TContext', layer: int, dstnodes: np.ndarray, dsttimes: np.ndarray,
                 dstindex: np.ndarray = None, srcnodes: np.ndarray = None,
                 eid: np.ndarray = None, ets: np.ndarray = None):
        """
        Internal constructor for creating a TBlock.

        :param TContext ctx: The TContext.
        :param int layer: The layer in the GNN.
        :param np.ndarray dstnodes: The destination nodes.
        :param np.ndarray dstindex: The indices of destination nodes in dstnodes.
        :param np.ndarray srcnodes: The source nodes of the edges.
        :param np.ndarray eid: The edge indices.
        :param np.ndarray ets: The edge timestamps.
        """

        self._ctx = ctx
        self._g = ctx.graph()
        self._layer = layer
        self._dstnodes = dstnodes
        self._dsttimes = dsttimes

        # metadata
        self._prev = None
        self._next = None
        self._hooks = []
        self._include_prev_dst = False

        # neighbor attributes
        self._dstindex = dstindex
        self._srcnodes = srcnodes
        self._eid = eid
        self._ets = ets
        self._has_nbrs = (
            self._dstindex is not None and
            self._srcnodes is not None and
            self._eid is not None and
            self._ets is not None)

        # data attributes
        self._dstdata = TFrame(len(dstnodes))
        self._srcdata = TFrame(0 if srcnodes is None else len(srcnodes))
        self._edata = TFrame(0 if eid is None else len(eid))

        # cached attributes
        self._c_efeat = None
        self._c_nfeat = None
        self._c_allnodes = None
        self._c_uniq_src = None
        self._c_mem_data = None
        self._c_mail = None

    @property
    def g(self) -> 'TGraph':
        """Returns the TGraph associated with this TBlock."""
        return self._g

    @property
    def layer(self) -> int:
        """Returns the GNN layer to which this TBlock belongs."""
        return self._layer
    
    @property
    def dstnodes(self) -> np.ndarray:
        """Returns the full destination nodes."""
        return self._dstnodes

    @property
    def dsttimes(self) -> np.ndarray:
        """Returns the destination node timestamps."""
        return self._dsttimes

    @property
    def dstindex(self) -> Optional[np.ndarray]:
        """Returns the indices of destination nodes in self.dstnodes() that are used for sampling."""
        return self._dstindex

    @property
    def srcnodes(self) -> Optional[np.ndarray]:
        """Return the source nodes (temporally sampled neighbors)."""
        return self._srcnodes

    @property
    def eid(self) -> Optional[np.ndarray]:
        """Returns the edge indices, which are connections towards sampled neighbors."""
        return self._eid

    @property
    def ets(self) -> Optional[np.ndarray]:
        """Returns the edge timestamps."""
        return self._ets
    
    @property
    def dstdata(self) -> TFrame:
        """Returns the destination node data."""
        return self._dstdata

    @property
    def srcdata(self) -> TFrame:
        """Returns the source node data."""
        return self._srcdata
    
    @property
    def edata(self) -> TFrame:
        """Returns the edge data."""
        return self._edata

    @property
    def prev(self) -> Optional[TBlock]:
        """Returns the previous TBlock."""
        return self._prev

    @prev.setter
    def prev(self, value: 'TBlock'):
        """
        Sets the previous TBlock.

        :param TBlock value: The previous TBlock.
        :raises TError: If value is not an instance of TBlock or its layer is not the current TBlock's layer - 1.
        """
        if not isinstance(value, TBlock) or value.layer != self.layer - 1:
            raise TError('invalid previous block')
        self._prev = value

    @property
    def next(self) -> Optional[TBlock]:
        """Returns the next TBlock."""
        return self._next

    @next.setter
    def next(self, value: 'TBlock'):
        """
        Sets the next TBlock.

        :param TBlock value: The next TBlock.
        :raises TError: If value is not an instance of TBlock or its layer is not current TBlock's layer + 1.
        """
        if not isinstance(value, TBlock) or value.layer != self.layer + 1:
            raise TError('invalid next block')
        self._next = value

    def num_edges(self) -> int:
        """Returns the number of edges in the block when self.has_nbrs() is True."""
        return len(self._eid) if self._has_nbrs else 0

    def num_src(self) -> int:
        """Returns the length of source node array when self.has_nbrs() is True."""
        return len(self._srcnodes) if self._has_nbrs else 0

    def num_dst(self) -> int:
        """Returns the length of destination node array."""
        return len(self._dstnodes)

    def has_nbrs(self) -> bool:
        """Returns True when the block has the neighbor attributes, including dstindex, srcnodes, eid and ets."""
        return self._has_nbrs

    def set_nbrs(self, dstindex: np.ndarray, srcnodes: np.ndarray,
                      eid: np.ndarray, ets: np.ndarray):
        """Sets the neighbor attributes for the block."""
        self.clear_nbrs()
        self._has_nbrs = True
        self._dstindex = dstindex
        self._srcnodes = srcnodes
        self._eid = eid
        self._ets = ets
        self._edata = TFrame(len(eid))
        self._srcdata = TFrame(len(srcnodes))

    def clear_nbrs(self):
        """Clears the neighbor attributes and related cache."""
        self._has_nbrs = False
        self._dstindex = None
        self._srcnodes = None
        self._eid = None
        self._ets = None
        self._edata = TFrame(0)
        self._srcdata = TFrame(0)
        self._c_efeat = None
        self._c_nfeat = None
        self._c_allnodes = None
        self._c_uniq_src = None

    def clear_data(self):
        """Clears the data and all cache."""
        self._edata.clear()
        self._srcdata.clear()
        self._dstdata.clear()
        self._c_efeat = None
        self._c_nfeat = None
        self._c_allnodes = None
        self._c_uniq_src = None
        self._c_mem_data = None
        self._c_mail = None

    def next_block(self, include_dst=False, use_dst_times=False) -> TBlock:
        """
        Generates the next TBlock with neighbor information.

        :param bool include_dst: Whether to include current TBlocks's destination nodes in the next TBlock.
        :param bool use_dst_times: True when using timestamps of current TBlock's destination nodes as timestamps of
         next TBlock's destination nodes, False when using edges' timestamps.
        """
        if self._next is not None:
            return self._next
        t_start = tt.start()
        if self.num_dst() == 0:
            blk = TBlock(self._ctx, self._layer + 1, np.empty(0), np.empty(0))
        else:
            self._check_has_nbrs()
            next_dstnodes = self._srcnodes
            next_dsttimes = self._ets if not use_dst_times else self._dsttimes[self._dstindex]
            if include_dst:
                next_dstnodes = np.concatenate([self._dstnodes, next_dstnodes])
                next_dsttimes = np.concatenate([self._dsttimes, next_dsttimes])
            blk = TBlock(self._ctx, self._layer + 1, next_dstnodes, next_dsttimes)
        blk._include_prev_dst = True
        blk._prev = self
        self._next = blk
        tt.t_prep_input += tt.elapsed(t_start)
        return blk

    def allnodes(self) -> Tensor:
        """Returns a tensor containing the destination nodes concatenated with the source nodes (if available) in pre-defined TGraph's storage device."""
        if self._c_allnodes is None:
            sdev = self._g.storage_device()
            nodes = np.concatenate([self._dstnodes, self._srcnodes]) \
                if self._has_nbrs else self._dstnodes
            nodes = torch.from_numpy(nodes).long().to(sdev)
            self._c_allnodes = nodes
        return self._c_allnodes

    def uniq_src(self) -> Optional[Tuple[Tensor, Tensor]]:
        """Returns a Tensor tuple: (the unique source node indices, the indices of the unique array that reconstructs the source
         node array) in pre-defined TGraph's storage device."""
        if self._c_uniq_src is None and self._srcnodes is not None:
            sdev = self._g.storage_device()
            uniq_nids, idx = np.unique(self._srcnodes, return_inverse=True)
            uniq_nids = torch.from_numpy(uniq_nids).long().to(sdev)
            idx = torch.from_numpy(idx).long().to(sdev)
            self._c_uniq_src = (uniq_nids, idx)
        return self._c_uniq_src

    def efeat(self) -> Optional[Tensor]:
        """Returns the edge features in TGraph's computation device, always use pinned memory if possible."""
        self._load_efeat(use_pin=True)
        return self._c_efeat

    def nfeat(self) -> Optional[Tensor]:
        """Returns the node features in TGraph's computation device, always use pinned memory if possible."""
        self._load_nfeat(use_pin=True)
        return self._c_nfeat

    def srcfeat(self) -> Optional[Tensor]:
        """Returns the source node features in TGraph's computation device, always use pinned memory if possible."""
        self._load_nfeat(use_pin=True)
        return self._c_nfeat[self.num_dst():]

    def dstfeat(self) -> Optional[Tensor]:
        """Returns the destination node features in TGraph's computation device, always use pinned memory if possible."""
        self._load_nfeat(use_pin=True)
        return self._c_nfeat[:self.num_dst()]

    def mem_data(self) -> Optional[Tensor]:
        """Returns node memory in TGraph's computation device, always use pinned memory if possible."""
        self._load_mem_data(use_pin=True)
        return self._c_mem_data

    def mail(self) -> Optional[Tensor]:
        """Returns the node mails in TGraph's computation device, always use pinned memory if possible."""
        self._load_mail(use_pin=True)
        return self._c_mail

    def time_deltas(self) -> Tensor:
        """Computes the timestamp differences between destination nodes (used for sampling) and edges and returns
        them as a tensor in pre-defined TGraph's computation device.
        
        :return: A tensor containing the time deltas.
        """
        self._check_has_nbrs()
        dts = self._dsttimes[self._dstindex]
        dts = dts - self._ets
        dev = self._g.compute_device()
        return torch.from_numpy(dts).to(device=dev, dtype=torch.float)

    def apply(self, fn: Callable, need_nbrs=True, run_hooks=True):
        """
        Applies an operator to the block itself and returns the output.

        :param Callable fn: The function to apply to the block itself.
        :param bool need_nbrs: Whether sampled neighbors are needed.
        :param run_hooks: Whether to run registered hooks.
        """
        if need_nbrs:
            self._check_has_nbrs()
        output = fn(self)
        if run_hooks:
            output = self.run_hooks(output)
        return output

    def run_hooks(self, input: Optional[Tensor]) -> Optional[Tensor]:
        """Runs registered hooks on the input tensor in reversed order."""
        for hook in reversed(self._hooks):
            input = hook(self, input)
        return input

    def clear_hooks(self):
        """Clears all the hooks."""
        self._hooks.clear()

    def register_hook(self, hook: Callable):
        """Registers a hook for running post-processing."""
        self._hooks.append(hook)

    def _check_has_nbrs(self):
        """
        Makes sure the block has all the neighbor attributes, including dstindex, srcnodes, eid and ets.

        :raises TError: If self.has_neighbor is False.
        """
        if not self._has_nbrs:
            raise TError('block neighbors has not been sampled yet')

    def _replace_dst_empty(self):
        """Replaces the destination nodes and timestamps with empty arrays."""
        self._replace_dst(np.empty(0), np.empty(0))

    def _replace_dst(self, dstnodes: np.ndarray, dsttimes: np.ndarray):
        """Replaces destination nodes and timestamps with given arrays."""
        self.clear_nbrs()
        self.clear_data()
        self._dstnodes = dstnodes
        self._dsttimes = dsttimes
        self._dstdata = TFrame(len(dstnodes))

    def _load_efeat(self, use_pin=False):
        """Loads the edge features to the TGraph's computation device."""
        if self._c_efeat is None and self._g.efeat is not None:
            self._c_efeat = self._load_feat(
                self._g.efeat, self._eid, use_pin,
                self._ctx._get_efeat_pin)

    def _load_nfeat(self, use_pin=False):
        """Loads the node features to the TGraph's computation device."""
        if self._c_nfeat is None and self._g.nfeat is not None:
            self._c_nfeat = self._load_feat(
                self._g.nfeat, self.allnodes().cpu().numpy(), use_pin,
                self._ctx._get_nfeat_pin)

    def _load_feat(self, feat: Tensor, idx: np.ndarray, use_pin: bool, pin_getter: Callable) -> Tensor:
        """Loads selected feature data from the TGraph's storage device to computation device.

        :param Tensor feat: Feature tensor.
        :param np.ndarray idx: The indices of selected features.
        :param bool use_pin: Whether to use pinned memory, only applicable when the storage device is cpu and the computation device.
        is cuda.
        :param Callable pin_getter: Function to get the pinned buffer.
        :return: A tensor containing the loaded feature data.
        """
        t_start = tt.start()
        sdev = self._g.storage_device()
        cdev = self._g.compute_device()
        idx = torch.from_numpy(idx).long()
        if sdev.type == 'cpu' and cdev.type == 'cuda' and use_pin:
            pin = pin_getter(self.layer, len(idx), feat.shape[1])
            torch.index_select(feat, 0, idx, out=pin)
            data = pin.to(cdev, non_blocking=True)
        else:
            data = feat[idx.to(sdev)].to(cdev)
        tt.t_prep_input += tt.elapsed(t_start)
        return data

    def _load_mem_data(self, use_pin=False):
        """Loads the node memory to the TGraph's computation device."""
        if self._c_mem_data is None and self._g.mem is not None:
            t_start = tt.start()
            sdev = self._g.storage_device()
            cdev = self._g.compute_device()
            nodes = self.allnodes()
            if sdev.type == 'cpu' and cdev.type == 'cuda' and use_pin:
                pin = self._ctx._get_mem_data_pin(self.layer, len(nodes))
                torch.index_select(self._g.mem.data, 0, nodes, out=pin)
                data = pin.to(cdev, non_blocking=True)
            else:
                data = self._g.mem.data[nodes].to(cdev)
            tt.t_prep_input += tt.elapsed(t_start)
            self._c_mem_data = data

    def _load_mail(self, use_pin=False):
        """Loads the mail to the TGraph's computation device"""
        if self._c_mail is None and self._g.mailbox is not None:
            t_start = tt.start()
            sdev = self._g.storage_device()
            cdev = self._g.compute_device()
            nodes = self.allnodes()
            if sdev.type == 'cpu' and cdev.type == 'cuda' and use_pin:
                pin = self._ctx._get_mail_pin(self.layer, len(nodes))
                torch.index_select(self._g.mailbox.mail, 0, nodes, out=pin)
                data = pin.to(cdev, non_blocking=True)
            else:
                data = self._g.mailbox.mail[nodes].to(cdev)
            tt.t_prep_input += tt.elapsed(t_start)
            self._c_mail = data

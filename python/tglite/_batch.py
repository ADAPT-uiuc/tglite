from typing import TYPE_CHECKING, Optional, Tuple
if TYPE_CHECKING:
    from ._graph import TGraph
    from ._context import TContext

import numpy as np
from torch import Tensor

from ._core import TError
from ._block import TBlock
from ._stats import tt


class TBatch(object):
    """
    Represents a batch of temporal edges to process. A thin wrapper with a TGraph reference and without actually
    materializing any arrays until they are needed.
    """

    def __init__(self, g: 'TGraph', range: Tuple[int, int]):
        """
        Internal constructor for creating a TBatch.

        :param TGraph g: The TGraph.
        :param Tuple[int, int] range: The range of edge indices: beginning and ending edge index.
        """
        self._g = g
        self._beg_idx = range[0]
        self._end_idx = range[1]
        self._neg_nodes = None

    def __len__(self) -> int:
        """Returns the total number of edges in the batch."""
        return self._end_idx - self._beg_idx

    @property
    def g(self) -> 'TGraph':
        """Returns the TGraph."""
        return self._g

    @property
    def neg_nodes(self) -> Optional[np.ndarray]:
        """Get the negative nodes."""
        return self._neg_nodes

    @neg_nodes.setter
    def neg_nodes(self, value: np.ndarray):
        """
        Set the negative nodes.

        :param np.ndarray value: An array of negative node samples.
        :raises TError: if value is not a 1-dimensional ndarray.
        """
        if not isinstance(value, np.ndarray):
            raise TError('negative samples must be an ndarray')
        if len(value.shape) != 1:
            raise TError('negative samples must be 1-dimensional')
        self._neg_nodes = value

    def block(self, ctx: 'TContext') -> TBlock:
        """Creates the head TBlock of the batch, including negative nodes if set."""
        t_start = tt.start()
        blk = TBlock(ctx, 0, self.nodes(), self.times())
        tt.t_prep_input += tt.elapsed(t_start)
        return blk

    def block_adj(self, ctx: 'TContext') -> TBlock:
        """Creates the head TBlock with batch edges as neighbors (excluding negative nodes)."""
        dstnodes = self.nodes(include_negs=False)
        srcnodes = self.nodes(include_negs=False, reverse=True)
        dstindex = np.arange(len(dstnodes))
        eids = np.tile(self.eids(), 2)
        ets = self.times(include_negs=False)
        return TBlock(ctx, 0, dstnodes, ets, dstindex, srcnodes, eids, ets)

    def eids(self) -> np.ndarray:
        """
        Returns edge ids of the batch.

        rtype: np.ndarray
        """
        return np.arange(self._beg_idx, self._end_idx, dtype=np.int32)

    def edges(self) -> np.ndarray:
        """
        Returns the edges in the batch as a two-column ndarray, where the first column represents the source
          node index and the second column represents the destination node index.

        rtype: np.ndarray
        """
        return self.g._edges[self._beg_idx:self._end_idx]

    def nodes(self, include_negs=True, reverse=False) -> np.ndarray:
        """
        Returns a node index array: [src, des(, neg)] if reverse is False or [des, src(, src)] if reverse is True.

        :param bool include_negs: Whether to include negative nodes.
        :param bool reverse: Whether to reverse the edges.
        :rtype: np.ndarray
        """
        nids = self.g._edges[self._beg_idx:self._end_idx]
        nids = np.flip(nids, axis=1) if reverse else nids
        nids = nids.T.reshape(-1)
        if self._neg_nodes is not None and include_negs:
            negs = nids[len(self):] if reverse else self._neg_nodes
            nids = np.concatenate([nids, negs])
        return nids.astype(np.int32)

    def times(self, include_dsts=True, include_negs=True) -> np.ndarray:
        """
        Returns timestamps corresponding to the nodes. It retrieves timestamps of the batch edges (as the timestamps for source nodes),
          repeating the timestamps to include destination nodes and negative nodes.

        :param bool include_dst: Whether to include destination nodes of the edges as positive nodes.
        :param bool include_negs: Whether to include negative nodes.
        """
        n_repeats = 2 if include_dsts else 1
        if include_negs and self._neg_nodes is not None:
            n_repeats += 1
        times = self.g._times[self._beg_idx:self._end_idx]
        return np.tile(times, n_repeats).astype(np.float32)

    def split_data(self, data: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Splits the data into multiple arrays, with each array containing a number of rows equal to the batch size.

        :param Tensor data: The source data to be split.
        :raises TError: If the length of data is not three times the batch size when negative nodes are included or two times otherwise.
        :return: A tuple (src, dst, neg), where neg is None if no negative nodes are specified.
        """
        size = len(self)
        if self._neg_nodes is not None:
            if data.shape[0] != 3 * size:
                raise TError('expected data to have 3 times batch size')
            dst = data[size:2 * size]
            neg = data[2 * size:]
        else:
            if data.shape[0] != 2 * size:
                raise TError('expected data to have 2 times batch size')
            dst = data[size:]
            neg = None
        src = data[:size]
        return (src, dst, neg)
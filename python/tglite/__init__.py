"""
tglite: Temporal GNN Lightweight Framework
"""

from __future__ import annotations

__version__ = "0.1.0"

### Re-exports

from ._graph import TGraph, from_csv
from ._batch import TBatch
from ._block import TBlock
from ._frame import TFrame
from ._memory import Memory
from ._mailbox import Mailbox
from ._sampler import TSampler
from ._context import TContext
from ._core import TError
from . import _utils as utils
from . import nn
from . import op


def iter_edges(g: TGraph, size=1, start=None, end=None) -> EdgesIter:
    """
    Create and return an iterator to generate TBatch

    :param TGraph g: The graph to iterate on.
    :param int size: Number of edges in each mini-batch.
    :rtype: EdgesIter
    :param start: The starting edge index.
    :type start: int or None
    :param end: The ending edge index.
    :type end: int or None
    """
    return EdgesIter(g, size=size, start=start, end=end)


class EdgesIter(object):
    """ An edge iterator of a TGraph."""
    def __init__(self, g: TGraph, size=1, start=None, end=None):
        """
        Create an edge iterator.

        :param TGraph g: The graph it iterates on.
        :param int size: Number of edges in each mini-batch.
        :param start: The starting edge index.
        :type start: int or None
        :param end: The ending edge index.
        :type end: int or None
        """
        self._g = g
        self._size = size
        self._curr = 0 if start is None else start
        self._last = g.num_edges() if end is None else end

    def __iter__(self) -> EdgesIter:
        return self

    def __next__(self) -> TBatch:
        if self._curr < self._last:
            idx = self._curr
            self._curr += self._size
            end = min(self._curr, self._last)
            return TBatch(self._g, range=(idx, end))
        raise StopIteration

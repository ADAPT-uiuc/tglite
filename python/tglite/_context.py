from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from ._graph import TGraph

import torch
from torch import Tensor

from ._core import TError


class TContext(object):
    """Graph-level context and scratch space used by the tglite runtime."""

    def __init__(self, g: 'TGraph'):
        """
        Internal constructor for creating a TContext.

        :param TGraph g: The TGraph to operate on.
        """
        self._g = g
        self._training = True

        # pin buffers
        self._efeat_pins = {}
        self._nfeat_pins = {}
        self._mem_data_pins = {}
        self._mail_pins = {}

        # embed caching
        self._cache_enabled = False
        self._cache_dim_emb = None
        self._cache_limit = int(2e6)
        self._cache_tables = {}

        # time precomputation
        self._time_enabled = False
        self._time_window = int(1e4)
        self._time_tables = {}

    @property
    def graph(self) -> 'TGraph':
        """Returns the TGraph it associated with."""
        return self._g

    def train(self):
        """Enables training mode and clear time tables and embedding cache tables."""
        self._training = True
        self._time_tables.clear()
        self._cache_tables.clear()

    def eval(self):
        """Disables training mode."""
        self._training = False

    def need_sampling(self, need: bool):
        """
        Creates tcsr within the TGraph if sampling is needed.

        :param bool need: Whether sampling is required.
        """
        if need:
            self._g._init_tcsr()
        else:
            self._g._tcsr = None

    def enable_embed_caching(self, enabled: bool, dim_embed: int = None):
        """
        Performs embedding cache settings and clear cache tables.

        :param bool enabled: Whether to enable embedding caching.
        :param dim_embed: Dimension of node embeddings.
        :raises TError: If enable is True and dim_embed is None.
        """
        self._cache_enabled = enabled
        if enabled and dim_embed is None:
            raise TError('need dimension of embeddings')
        elif enabled:
            self._cache_dim_emb = dim_embed
        self._cache_tables.clear()

    def set_cache_limit(self, limit: int):
        """
        Sets embedding cache limit and clear cache tables.

        :param int limit: Number of embeddings to cache.
        """
        self._cache_limit = limit
        self._cache_tables.clear()

    def enable_time_precompute(self, enabled: bool):
        """
        Performs time precomputation settings and clear time tables.

        :param bool enabled: Whether to enable embedding caching.
        """
        self._time_enabled = enabled
        self._time_tables.clear()

    def set_time_window(self, window: int):
        """
        Sets length of time window and clear time tables.

        :param int window: Length of time window.
        :raises TError: If int is negative.
        """

        if window < 0:
            raise TError('time window must be non-negative')
        self._time_window = window
        self._time_tables.clear()

    def _get_efeat_pin(self, layer: int, rows: int, dim: int) -> Tensor:
        return self._get_pin(self._efeat_pins, layer, rows, [dim])

    def _get_nfeat_pin(self, layer: int, rows: int, dim: int) -> Tensor:
        return self._get_pin(self._nfeat_pins, layer, rows, [dim])

    def _get_mem_data_pin(self, layer: int, rows: int) -> Tensor:
        return self._get_pin(self._mem_data_pins, layer, rows, [self._g.mem.dim()])

    def _get_mail_pin(self, layer: int, rows: int) -> Tensor:
        return self._get_pin(self._mail_pins, layer, rows, self._g.mailbox.dims())

    def _get_pin(self, cache: dict, layer: int, rows: int, dims: List[int]) -> Tensor:
        """
        Creates/reshapes pinned buffer and returns it.

        :param dict cache: The dictionary of the buffer with layer numbers as keys.
        :param int layer: Which layer pinned data belongs to.
        :param int rows: Number of rows of pinned data.
        :param List[int] dims: Size of each pinned data.
        :return: Pinned buffer.
        :rtype: Tensor
        """
        if layer not in cache:
            shape = tuple([rows] + dims)
            pin = torch.zeros(shape, pin_memory=True)
            cache[layer] = pin
            return pin
        pin = cache[layer]
        if pin.shape[0] < rows or list(pin.shape[1:]) != dims:
            shape = tuple([rows] + dims)
            pin.resize_(shape)
        return pin[:rows]

from . import _c
from ._core import TError
from ._block import TBlock
from ._utils import get_num_cpus
from ._stats import tt


class TSampler(object):

    def __init__(self, num_nbrs: int, strategy='recent', num_threads: int = None):
        """
        Internal constructor for creating a TSampler

        :param int num_nbrs: number of neighbors
        :param str strategy: sampling strategy, 'recent' or 'uniform'
        :param int num_threads: number of threads for parallel sampling, set to number of cpus if not provided
        :raises TError: if strategy is not in ['recent', 'uniform']
        """

        if strategy not in ['recent', 'uniform']:
            raise TError(f'sampling strategy not supported: {strategy}')

        self._n_nbrs = num_nbrs
        self._strategy = strategy
        self._n_threads = get_num_cpus() \
            if num_threads is None else num_threads

        self._sampler = _c.TemporalSampler(
            self._n_threads,
            self._n_nbrs,
            self._strategy == 'recent')

    def sample(self, blk: TBlock) -> TBlock:
        """Updates block with sampled 1-hop source neighbors
        
        :returns: updated block
        """
        t_start = tt.start()
        if blk.num_dst() > 0:
            block = self._sampler.sample(blk._g._get_tcsr(), blk._dstnodes, blk._dsttimes)
            blk.set_nbrs(
                block.copy_dstindex(),
                block.copy_srcnodes(),
                block.copy_eid(),
                block.copy_ets())
        tt.t_sample += tt.elapsed(t_start)
        return blk

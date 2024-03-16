import pytest
import torch
from pathlib import Path

import tglite as tg


def test_tblock():
    g = tg.from_csv(Path(__file__).parent / 'data/edges.csv')
    ctx = tg.TContext(g)
    batch = next(tg.iter_edges(g, size=10))
    assert len(batch) == 10

    blk = batch.block(ctx)
    assert blk.layer == 0
    assert blk.num_dst() == len(batch) * 2  # src, dst nodes of batch edges
    assert blk.num_dst() == len(blk.dsttimes)
    assert blk.has_nbrs() == False  # no neighbor info before sampling

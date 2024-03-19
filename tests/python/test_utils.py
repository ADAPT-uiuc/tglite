import pytest
import numpy as np

import tglite as tg


def test_tcsr():
    edges = np.array([[0,1], [0,2], [1,2]], dtype=np.int32)
    etime = np.array([10, 11, 12], dtype=np.float32)
    tcsr = tg.utils.create_tcsr(edges, etime)

    assert len(tcsr.ind) == len(edges) + 1
    assert list(tcsr.ind) == [0, 2, 4, 6]
    assert list(tcsr.nbr) == [1, 2, 0, 2, 0, 1]
    assert list(tcsr.eid) == [0, 1, 0, 2, 1, 2]
    assert list(tcsr.ets) == [10, 11, 10, 12, 11, 12]

import pytest
import torch
import numpy as np
from pathlib import Path

import tglite as tg


def test_from_csv_nofile():
    with pytest.raises(tg.TError) as exinfo:
        tg.from_csv("foobar")
    assert "file does not exist" in str(exinfo.value)


def test_from_csv():
    path = Path(__file__).parent / 'data/edges.csv'
    g = tg.from_csv(path)
    assert g.num_edges() == 100
    assert g.num_nodes() == 8263


def test_tgraph():
    edges = np.array([[0,1], [0,2], [1,2]], dtype=np.int32)
    etime = np.array([10, 11, 12], dtype=np.float32)

    g = tg.TGraph(edges, etime)
    g.edata['f'] = torch.randn((3, 2))
    g.ndata['f'] = torch.randn((3, 2))

    assert g.num_edges() == 3
    assert g.num_nodes() == 3
    assert g.edata['f'].shape[1] == 2
    assert g.ndata['f'].shape[1] == 2
    assert str(g.storage_device()) == 'cpu'
    assert str(g.compute_device()) == 'cpu'

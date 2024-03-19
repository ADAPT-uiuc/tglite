import pytest
import torch

import tglite as tg


def test_tframe():
    frame = tg.TFrame(dim=3)
    frame['f'] = torch.ones((3, 2))
    assert frame['f'].shape == (3, 2)
    assert frame['f'].sum() == 6


def test_tframe_dim():
    frame = tg.TFrame()
    assert frame.dim() == 0
    frame = tg.TFrame(dim=16)
    assert frame.dim() == 16


def test_tframe_checks_dim():
    with pytest.raises(tg.TError) as exinfo:
        frame = tg.TFrame(dim=3)
        frame['f'] = torch.ones((1, 2))
    assert "dimension of 3, got 1" in str(exinfo.value)


def test_tframe_only_tensors():
    frame = tg.TFrame(dim=3)
    frame['tensor'] = torch.randn(3, 2)
    with pytest.raises(tg.TError) as exinfo:
        frame['list'] = [1, 2, 3]
    assert "expected value to be a tensor" in str(exinfo.value)


def test_tframe_dict_behavior():
    frame = tg.TFrame(dim=3)
    frame['a'] = torch.randn((3, 1))
    frame['b'] = torch.randn((3, 2))
    frame['c'] = torch.randn((3, 3))
    frame['d'] = torch.randn((3, 4))

    assert 'a' in frame
    assert len(frame) == 4
    for key, val in frame.items():
        if key == 'a': assert val.shape[1] == 1
        if key == 'b': assert val.shape[1] == 2
        if key == 'c': assert val.shape[1] == 3
        if key == 'd': assert val.shape[1] == 4

    frame.clear()
    assert frame.dim() == 3
    assert len(frame) == 0

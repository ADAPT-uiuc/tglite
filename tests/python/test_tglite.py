import tglite as tg


def test_terror():
    err = tg.TError("message")
    assert str(err) == "message"

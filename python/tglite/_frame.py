from __future__ import annotations

from torch import Tensor

from ._core import TError


class TFrame(dict):
    """A container for storing tensor features."""

    def __init__(self, dim=None):
        """Initialize a TFrame object"""
        super().__init__()
        self._dim = 0 if dim is None else dim

    def dim(self) -> int:
        """Get the leading dimension of stored tensors"""
        return self._dim

    def to(self, device, **kwargs) -> TFrame:
        """Move tensor data in this frame to the specified device.

        :param device: the device to which the tensor data should be moved
        :param **kwargs: additional keyword arguments for the pytorch Tensor.to() method
        :returns: a new TFrame object with the tensor data copied to the specified device
        """
        copy = TFrame(dim=self._dim)
        for key, val in self.items():
            copy[key] = val.to(device, **kwargs)
        return copy

    def __setitem__(self, key, value):
        """
        Set an item in the TFrame

        :raises TError: if the value is not a tensor or if it does not have the expected leading dimension
        """
        if not isinstance(value, Tensor):
            raise TError("expected value to be a tensor")
        if len(value) != self._dim:
            raise TError(f"expected value to have leading dimension of {self._dim}, got {len(value)}")
        super().__setitem__(key, value)

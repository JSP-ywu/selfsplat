from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from torch import nn, Tensor
from jaxtyping import Float

from ...dataset.types import BatchedViews, DataShim
from ..types import Gaussians

T = TypeVar("T")


class Encoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        context: BatchedViews,
        target: BatchedViews,
        val_or_test: bool,
        supervised: bool,
    ) -> tuple[
        Gaussians,
        Float[Tensor, "b p 4 4"] | None,
        Float[Tensor, "b p 4 4"] | None, 
        Float[Tensor, "b v 1 h w"],
        Float[Tensor, "b v+1 3 h w"]]:
        pass

    def get_data_shim(self) -> DataShim:
        """The default shim doesn't modify the batch."""
        return lambda x: x

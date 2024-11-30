from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, Literal, get_args, TypeVar
from typing_extensions import Self


@dataclass
class Dims:
    L: int


@dataclass
class Qty:
    val: float

    def __class_getitem__(cls, params):
        if not isinstance(params, tuple):
            params = (params,)

        return Dims(params[0])

    def __add__(self, qty: Self) -> Self:
        return Qty(self.val + qty.val)

    # def __mul__(self, qty: Qty[_L2]) -> Qty[_L]:


LengthT = Qty[Literal[1], Literal[0], Literal[0]]
TimeT = Qty[Literal[0], Literal[1], Literal[0]]


def func(dx: LengthT):
    pass


func(1, 2)
func("1")


def func2(x: int) -> str:
    return x


func2("s")

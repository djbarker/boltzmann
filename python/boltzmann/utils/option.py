from dataclasses import dataclass
from typing import TypeVar, Generic

T = TypeVar("T")


@dataclass
class Some(Generic[T]):
    val: T


def to_opt(val: T | Some[T] | None) -> Some[T] | None:
    """
    Maybe wrap a value in `Some` so we can use it pattern matches nicely.
    Idempotent, so calling on an instance of Some[T] will not nest.
    """
    match val:
        case Some():
            return val
        case None:
            return None
        case _:
            return Some(val)

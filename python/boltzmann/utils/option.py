from dataclasses import dataclass
from typing import Callable, TypeVar, Generic

T = TypeVar("T")
S = TypeVar("S")


@dataclass
class Some(Generic[T]):
    val: T


Option = T | Some[T] | None


def to_opt(val: Option[T]) -> Some[T] | None:
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


def map_opt(val: Option[T], func: Callable[[T], S]) -> Some[S] | None:
    """
    Maybe map the contents of a `Some[T]`.
    If a raw `T` is passed will wrap it first.
    """
    match to_opt(val):
        case Some(t):
            return Some(func(t))
        case None:
            return None
        case _:
            assert False


def unwrap(some: T | Some[T]) -> T:
    """
    Get the value from inside a `Some[T]`, or just the value if it's already `T`.
    Throws a TypeError if None is passed.
    """
    match some:
        case Some(t):
            return t
        case None:
            raise TypeError()
        case _:
            return some

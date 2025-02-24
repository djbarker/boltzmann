from contextlib import contextmanager
import datetime
import logging
import os

from dataclasses import dataclass, field
from typing import Any, Generator

__all__ = [
    "basic_config",
    "PerfInfo",
    "PerfTimer",
    "time",
    "dotted",
]


def basic_config(logger: logging.Logger | None = None, level: int | str | None = None):
    logger = logger or logging.getLogger()

    match level:
        case int():
            pass
        case str():
            level = logging.getLevelName(level)
        case None:
            level = logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO"))
        case _:
            raise TypeError("Expected int, str or None")

    for h in logger.handlers:
        logger.removeHandler(h)

    fmt = "{asctime:s} - {levelname:5.5s} - [{module}] {message}"
    formatter = logging.Formatter(fmt=fmt, style=r"{")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(handler)


@dataclass
class PerfInfo:
    events: int = field(default=0)
    micros: int = field(default=0)

    @property
    def seconds(self) -> float:
        return self.micros * 1e-6

    def fmt(self, events_name: str) -> str:
        # TODO: time delta formatting
        return f"{self.events} {events_name} in {self.seconds:.1f} seconds => {self.events / self.seconds:.1f} {events_name}/sec"

    def __str__(self) -> str:
        return self.fmt("events")

    def __add__(self, rhs: "PerfInfo") -> "PerfInfo":
        if not isinstance(rhs, PerfInfo):
            raise TypeError(f"rhs must be PerfInfo. [{rhs=!r}]")

        return PerfInfo(
            self.events + rhs.events,
            self.micros + rhs.micros,
        )

    def __sub__(self, rhs: "PerfInfo") -> "PerfInfo":
        if not isinstance(rhs, PerfInfo):
            raise TypeError(f"rhs must be PerfInfo. [{rhs=!r}]")

        return PerfInfo(
            self.events - rhs.events,
            self.micros - rhs.micros,
        )

    def __iadd__(self, rhs: "PerfInfo") -> "PerfInfo":
        if not isinstance(rhs, PerfInfo):
            raise TypeError(f"rhs must be PerfInfo. [{rhs=!r}]")

        self.events += rhs.events
        self.micros += rhs.micros

        return self


@dataclass
class PerfTimer:
    start: datetime.datetime
    events: int

    @property
    def elapsed(self) -> datetime.timedelta:
        return datetime.datetime.now() - self.start

    def tock(self, *, events: int = 0) -> PerfInfo:
        self.add_events(events)
        delta = self.elapsed
        return PerfInfo(self.events, delta.seconds * 1e6 + delta.microseconds)

    def add_events(self, events: int) -> "PerfTimer":
        self.events += events
        return self

    def add_event(self) -> "PerfTimer":
        return self.add_events(1)


def tick() -> PerfTimer:
    return PerfTimer(datetime.datetime.now(), 0)


@contextmanager
def time(
    logger: logging.Logger, message: str = "", events: int = 1, *, silent: bool = False
) -> Generator[PerfTimer, Any, Any]:
    """
    Time the contents of a with-statement, and log the result.
    """
    timer = tick()
    try:
        logger.debug(f"Starting {message}")
        yield timer
    finally:
        if not silent:
            if message != "":
                message = f"Done {message}: "
            logger.debug(f"{message}{timer.tock(events=events)}")


# TODO: this seems to print "logger" as the name of the logger, why?
def dotted(
    logger: logging.Logger, label: str, value: Any, width: int = 40, level: int = logging.INFO
):
    value = str(value)
    dots = width - len(label) - len(value) - 2
    dots = max(dots, 3)
    dots = "." * dots
    logger.log(level, f"{label} {dots} {value}")

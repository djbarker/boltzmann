"""
Utils for logging & timing.
"""

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
    "timed",
    "dotted",
]


def basic_config(logger: logging.Logger | None = None, level: int | str | None = None):
    """
    Set up the logger configuration.

    :param logger: The logger to configure. If :py:const:`None`, the root logger is used.
    :param level: The log level. If :py:const:`None`, look at the :bash:`$LOG_LEVEL` environment variable.

    .. role:: bash(code)
        :language: bash
    """
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
    """
    Stores information about performance; namely a number of events and the time taken to run them.

    :py:class:`PerfInfo` objects can be added together (& subtracted) which is useful for measuring long-running performance.
    """

    events: int = field(default=0)  #: The number of events that processed.
    micros: int = field(default=0)  #: The number of microseconds taken to process the events.

    @property
    def seconds(self) -> float:
        """
        The number of seconds taken to process the events.
        """
        return self.micros * 1e-6

    @property
    def rate(self) -> float:
        """
        The rate, measured in events/second.
        """
        return self.events / self.seconds

    def fmt(self, events_name: str) -> str:
        """
        A human-readable string formatted with the performance information.

        .. code-block:: python

            perf_info = PerfInfo(2500, 1_250_000)
            perf_info.fmt("frobinations") # == "2500 frobinations in 1.3 seconds => 2000.0 frobinations/sec"

        :param events_name: The plural 'name' of the events being measured, e.g. "updates" or "frames".
        """
        # TODO: time delta formatting
        return f"{self.events} {events_name} in {self.seconds:.1f} seconds => {self.rate:.1f} {events_name}/sec"

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
    """
    Measure time and return a :py:class:`PerfInfo` object when done.

    .. code-block:: python

        timer = tick()
        do_something_expensive()
        perf = timer.tock()
    """

    start: datetime.datetime
    events: int

    @property
    def elapsed(self) -> datetime.timedelta:
        return datetime.datetime.now() - self.start

    def tock(self, *, events: int = 0) -> PerfInfo:
        """
        Return a :py:class:`PerfInfo` with the given numer of processed events & elapsed time.

        .. code-block:: python

            timer = tick()
            for _ in range(10):
                do_something()
            perf = timer.tock(events=10)

        :param events: Any additional events to add which have not already been added by calls to :py:meth:`add_events`.
        """
        self.add_events(events)
        delta = self.elapsed
        return PerfInfo(self.events, delta.seconds * 1e6 + delta.microseconds)

    def add_events(self, events: int) -> "PerfTimer":
        """
        Add the given number of events to the timer.

        .. code-block:: python

            timer = tick()
            for _ in range(10):
                do_something()
            timer.add_events(10)
            perf = timer.tock()

        :param events: The number of events to add.
        """
        self.events += events
        return self

    def add_event(self) -> "PerfTimer":
        """
        Add the a single event to the timer.

        .. code-block:: python

            timer = tick()
            for _ in range(10):
                do_something()
                timer.add_event()
            perf = timer.tock()
        """
        return self.add_events(1)


def tick() -> PerfTimer:
    """
    Get a new :py:class:`PerfTimer` object.
    """
    return PerfTimer(datetime.datetime.now(), 0)


@contextmanager
def timed(
    logger: logging.Logger, message: str, events: int = 1
) -> Generator[PerfTimer, None, None]:
    """
    Time the contents of a with-statement, and log the result.

    .. code-block:: python

        with timed(logger, "the thing"):
            do_the_thing()

    :param logger: The logger to write the timing information to.
    :param message: An identifier for the block.
    :param events: How many events are being processed in the block.
    """
    timer = tick()
    try:
        logger.debug(f"Starting {message}")
        yield timer
    finally:
        message = f"Finished {message}: "
        perf = timer.tock(events=events)
        logger.debug(f"{message} in {perf.seconds:.1f} seconds ({perf.rate} / sec)")


# TODO: this seems to print "logger" as the name of the logger, why?
def dotted(
    logger: logging.Logger,
    label: str,
    value: Any,
    units: str = "",
    width: int = 40,
    level: int = logging.INFO,
):
    """
    Log a value with some formatting to align consecuative calls.

    .. code-block:: python

        dotted(logger, "x", 42)
        dotted(logger, "pi", 3.14159)
        dotted(logger, "g", 9.81, "m/s")

        # output:
        # x ................................... 42
        # pi ............................. 3.14159
        # g ................................. 9.81 m/s

    :param logger: The logger to write the information to.
    :param label: The label for the value.
    :param value: The value to log.
    :param units: The (optional) units for the value.
    :param width: The width of the label and value (but not the units).
    :param level: The log level to use (deault is :py:const:`logging.INFO`).
    """
    value = str(value)
    dots = width - len(label) - len(value) - 2
    dots = max(dots, 3)
    dots = "." * dots
    logger.log(level, f"{label} {dots} {value} {units}")

import datetime
import logging
import os

from dataclasses import dataclass

LOG_FMT_STRING = "{asctime} - {levelname:5.5s} - [{module}] {message}"


class UtilHandler(logging.Handler):
    pass


def basic_config(logger: logging.Logger | None = None, level: int | str | None = None):

    logger = logger or logging.getLogger()

    match level:
        case int():
            pass
        case str():
            level = logging.getLevelName(level)  # confusing function name :shrug:
        case None:
            level = logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO"))
        case _:
            raise TypeError(f"Expected int, str or None")

    logger.setLevel(level)


@dataclass
class PerfInfo:
    start: datetime.datetime
    end: datetime.datetime
    events: int

    @property
    def seconds(self) -> float:
        delta = self.end - self.start
        return delta.seconds + delta.microseconds * 1e-6

    def fmt(self, events_name: str) -> str:
        # TODO: time delta formatting
        return f"{self.events} {events_name} in {self.seconds:.1f} seconds => {self.events / self.seconds:.1f} {events_name}/sec"

    def __str__(self) -> str:
        return self.fmt("events")


@dataclass
class PerfInfoProg:
    start: datetime.datetime
    events: int

    def tock(self) -> "PerfInfo":
        return PerfInfo(
            self.start,
            datetime.datetime.now(),
            self.events,
        )

    def add_events(self, events: int):
        self.events += events

    def add_event(self):
        self.add_events(1)


def tick() -> PerfInfoProg:
    return PerfInfoProg(datetime.datetime.now(), 0)

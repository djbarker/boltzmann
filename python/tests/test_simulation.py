import datetime
import time

from boltzmann.simulation import EveryN, EveryT, parse_checkpoints


def test_every_n():
    gater = EveryN(10)

    for i in range(1, 31):
        assert gater.allow() == ((i + 1) % 10 == 0)


def test_every_t():
    interval_ms = 100
    gater = EveryT(datetime.timedelta(milliseconds=interval_ms))

    # starts with current time so should be false initially
    print(
        datetime.datetime.now(),
        gater._last,
    )
    assert not gater.allow()

    for _ in range(3):
        # wait >100ms to reset
        time.sleep(2 * interval_ms / 1000)

        # time has passed => should be true
        assert gater.allow()

        # fired recently => should be false
        for _ in range(5):
            time.sleep(0.1 * interval_ms / 1000)
            assert not gater.allow()


def test_parse_checkpoints():
    assert parse_checkpoints("5") == EveryN(5)
    assert parse_checkpoints("10") == EveryN(10)
    everyt = parse_checkpoints("5m")
    assert isinstance(everyt, EveryT) and (everyt.interval == datetime.timedelta(minutes=5))
    everyt = parse_checkpoints("10m")
    assert isinstance(everyt, EveryT) and (everyt.interval == datetime.timedelta(minutes=10))

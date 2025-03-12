import datetime
import time

from boltzmann.simulation import EveryN, EveryT, parse_checkpoints


def test_every_n():
    gater = EveryN(10)
    for i in range(30):
        assert gater.allow() == (i % 10 == 0)


def test_every_t():
    gater = EveryT(datetime.timedelta(milliseconds=100))
    for i in range(3):
        # haven't fired in a while => should be true
        assert gater.allow()

        # fired recently => should be false
        for i in range(5):
            time.sleep(10 / 1000)
            assert not gater.allow()

        # wait 100ms to reset
        time.sleep(100 / 1000)


def test_parse_checkpoints():
    assert parse_checkpoints("5") == EveryN(5)
    assert parse_checkpoints("10") == EveryN(10)
    assert parse_checkpoints("5m") == EveryT(datetime.timedelta(minutes=5))
    assert parse_checkpoints("10m") == EveryT(datetime.timedelta(minutes=10))

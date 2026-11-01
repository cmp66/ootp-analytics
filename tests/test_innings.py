import numpy as np
import pandas as pd
import pytest

from util.innings import dump_ip_to_outs, ip_to_innings, outs_to_ip


@pytest.mark.parametrize(
    "outs, expected",
    [(0, 0.0), (1, 0.1), (2, 0.2), (3, 1.0), (301, 100.1), (302, 100.2), (300, 100.0)],
)
def test_outs_to_ip(outs, expected):
    assert outs_to_ip(outs) == pytest.approx(expected)


@pytest.mark.parametrize(
    "ip, expected",
    [(100.0, 100.0), (100.1, 100 + 1 / 3), (59.2, 59 + 2 / 3), (0.1, 1 / 3)],
)
def test_ip_to_innings(ip, expected):
    assert ip_to_innings(ip) == pytest.approx(expected)


def test_round_trip_for_every_out_count():
    outs = np.arange(0, 1000)
    innings = ip_to_innings(outs_to_ip(outs))
    assert innings * 3 == pytest.approx(outs)


def test_dump_innings_include_extra_outs():
    df = pd.DataFrame({"IP": [100, 5], "IPF": [2, 0]})
    assert dump_ip_to_outs(df).tolist() == [302, 15]

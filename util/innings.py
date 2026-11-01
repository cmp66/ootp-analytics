"""Conversions between outs, baseball innings notation (100.1 = 100 and 1/3) and true decimal innings."""

import numpy as np
import pandas as pd


def outs_to_ip(outs):
    """Outs recorded -> baseball notation, e.g. 301 outs -> 100.1"""
    outs = np.asarray(outs)
    return outs // 3 + (outs % 3) / 10


def ip_to_innings(ip):
    """Baseball notation -> decimal innings, e.g. 100.1 -> 100.333..., 59.2 -> 59.666..."""
    ip = np.asarray(ip, dtype=float)
    whole = np.floor(ip + 1e-9)
    thirds = np.rint((ip - whole) * 10)
    return whole + thirds / 3


def dump_ip_to_outs(df: pd.DataFrame) -> pd.Series:
    """The dumps store whole innings in IP and the extra outs (0-2) in IPF."""
    return df["IP"] * 3 + df["IPF"]

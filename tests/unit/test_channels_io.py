from datetime import datetime, timedelta

import pandas as pd
import pytest

from esa_adb.io.channels import _read_channel_pickle


def test_reads_value_prefixed_categorical_strings(tmp_path):
    # Real ESA-AD categorical channels store "value_N" strings rather than raw floats
    # (channels.csv "Categorical" = YES) -- confirmed against a real Mission2 download.
    times = [datetime(2020, 1, 1) + timedelta(seconds=i) for i in range(4)]
    series = pd.Series(["value_0", "value_0", "value_2", "value_10"], index=pd.DatetimeIndex(times))
    path = tmp_path / "channel_4.pkl"
    series.to_pickle(path)

    df = _read_channel_pickle(path, "channel_4")

    assert df["value"].to_list() == [0.0, 0.0, 2.0, 10.0]


def test_reads_plain_numeric_floats(tmp_path):
    times = [datetime(2020, 1, 1) + timedelta(seconds=i) for i in range(3)]
    series = pd.Series([0.1, 0.2, 0.3], index=pd.DatetimeIndex(times))
    path = tmp_path / "channel_1.pkl"
    series.to_pickle(path)

    df = _read_channel_pickle(path, "channel_1")

    assert df["value"].to_list() == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)

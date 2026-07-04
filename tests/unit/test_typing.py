import polars as pl

from esa_adb.config import MissionConfig
from esa_adb.preprocessing.typing_ import ChannelType, infer_channel_types


def _config(monotonic_groups=None):
    return MissionConfig(
        mission_number=1,
        raw_data_dir="x",
        interim_dir="x",
        processed_dir="x",
        target_frequency_seconds=30,
        channel_count=6,
        target_channel_count=6,
        monotonic_groups=monotonic_groups or [],
        categorical_max_unique=4,
    )


def test_infers_expected_type_per_channel():
    n = 100
    df = pl.DataFrame(
        {
            "constant_ch": [5.0] * n,
            "binary_ch": [0.0, 1.0] * (n // 2),
            "categorical_ch": [float(i % 3) for i in range(n)],
            "monotonic_data_driven_ch": [float(i) for i in range(n)],
            "monotonic_group_ch": [float(i % 7) for i in range(n)],  # not monotonic by data, only by group
            "continuous_ch": [float((i * 37) % 101) / 3.7 for i in range(n)],
            "any_target_anomaly": [False] * n,
        }
    )
    channel_meta = pl.DataFrame(
        {
            "channel": ["monotonic_group_ch"],
            "group": [2],
            "is_target": [True],
        }
    )
    config = _config(monotonic_groups=[2])
    channels = [
        "constant_ch",
        "binary_ch",
        "categorical_ch",
        "monotonic_data_driven_ch",
        "monotonic_group_ch",
        "continuous_ch",
    ]

    result = infer_channel_types(df.lazy(), channel_meta, config, channels)
    types = dict(zip(result["channel"], result["type"]))
    typed_by = dict(zip(result["channel"], result["typed_by"]))

    assert types["constant_ch"] == ChannelType.CONSTANT.value
    assert types["binary_ch"] == ChannelType.BINARY.value
    assert types["categorical_ch"] == ChannelType.CATEGORICAL.value
    assert types["monotonic_data_driven_ch"] == ChannelType.MONOTONIC.value
    assert typed_by["monotonic_data_driven_ch"] == "data-driven"
    assert types["monotonic_group_ch"] == ChannelType.MONOTONIC.value
    assert typed_by["monotonic_group_ch"] == "group-metadata"
    assert types["continuous_ch"] == ChannelType.CONTINUOUS.value

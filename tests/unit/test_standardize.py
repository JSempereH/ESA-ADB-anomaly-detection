from datetime import datetime, timedelta

import polars as pl

from esa_adb.preprocessing.standardize import apply_standardization, compute_standardization_stats


def _times(n):
    return [datetime(2020, 1, 1) + timedelta(seconds=10 * i) for i in range(n)]


def test_standardization_per_channel_type():
    n = 20
    times = _times(n)
    df = pl.DataFrame(
        {
            "time": times,
            "continuous_ch": [float(i) for i in range(n)],
            "binary_ch": [0.0, 1.0] * (n // 2),
            "constant_ch": [7.0] * n,
            "monotonic_ch": [float(i) for i in range(n)],  # cumulative counter
            "categorical_ch": [float(i % 3) for i in range(n)],
        }
    )
    channel_types = pl.DataFrame(
        {
            "channel": ["continuous_ch", "binary_ch", "constant_ch", "monotonic_ch", "categorical_ch"],
            "type": ["continuous", "binary", "constant", "monotonic", "categorical"],
        }
    )
    labels_long = pl.DataFrame({"time": [], "channel": [], "category_code": []})
    channels = channel_types["channel"].to_list()

    stats, categorical_mappings = compute_standardization_stats(df.lazy(), labels_long, channel_types, channels)
    transform_by_channel = dict(zip(stats["channel"], stats["transform"]))

    assert transform_by_channel["continuous_ch"] == "zscore"
    assert transform_by_channel["binary_ch"] == "minmax"
    assert transform_by_channel["constant_ch"] == "mean_subtract"
    assert transform_by_channel["monotonic_ch"] == "diff_zscore"
    assert transform_by_channel["categorical_ch"] == "categorical_zscore"

    standardized = apply_standardization(df, stats, categorical_mappings)

    # Binary channel maps to [0, 1] exactly.
    assert sorted(standardized["binary_ch"].unique().to_list()) == [0.0, 1.0]
    # Constant channel: mean subtracted, no division -> all zeros.
    assert standardized["constant_ch"].to_list() == [0.0] * n
    # Continuous channel: z-scored -> mean ~0.
    assert abs(standardized["continuous_ch"].mean()) < 1e-9


def test_standardization_excludes_non_nominal_train_points_per_channel():
    n = 10
    times = _times(n)
    # Without exclusion, mean would be pulled toward the huge outlier at index 5.
    values = [1.0] * n
    values[5] = 1000.0
    df = pl.DataFrame({"time": times, "ch": values})
    channel_types = pl.DataFrame({"channel": ["ch"], "type": ["continuous"]})
    labels_long = pl.DataFrame({"time": [times[5]], "channel": ["ch"], "category_code": [1]})

    stats, _ = compute_standardization_stats(df.lazy(), labels_long, channel_types, ["ch"])
    row = stats.row(0, named=True)

    assert row["mean"] == 1.0  # outlier excluded from nominal-only stats

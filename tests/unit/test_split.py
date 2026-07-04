from datetime import datetime

import polars as pl
import pytest
from dateutil.relativedelta import relativedelta

from esa_adb.preprocessing.split import compute_split_boundaries, label_split


def test_split_boundaries_partition_range_with_no_gaps():
    start = datetime(2019, 1, 1)
    end = datetime(2021, 1, 1)
    b = compute_split_boundaries(start, end, val_months=3)

    midpoint = start + (end - start) / 2
    assert b.train_start == start
    assert b.test_end == end
    assert b.train_end == b.test_start == midpoint
    assert b.val_start == midpoint - relativedelta(months=3)  # last 3 calendar months before the midpoint
    assert b.train_start <= b.val_start <= b.train_end <= b.test_start <= b.test_end


def test_invalid_boundaries_raise():
    with pytest.raises(ValueError):
        compute_split_boundaries(datetime(2020, 1, 1), datetime(2020, 1, 2), val_months=36)


def test_label_split_assigns_expected_bucket():
    b = compute_split_boundaries(datetime(2020, 1, 1), datetime(2022, 1, 1), val_months=3)
    df = pl.DataFrame(
        {
            "time": [
                datetime(2020, 6, 1),  # train
                datetime(2020, 11, 1),  # val
                datetime(2021, 6, 1),  # test
            ]
        }
    )
    result = df.with_columns(label_split("time", b))
    assert result["split"].to_list() == ["train", "val", "test"]

from datetime import datetime

from esa_adb.models.walk_forward import aggregate_fold_results, generate_walk_forward_folds


def test_expanding_window_train_start_stays_fixed():
    folds = list(generate_walk_forward_folds(datetime(2020, 1, 1), datetime(2020, 5, 1), step_months=1, window="expanding"))

    assert len(folds) == 3  # eval periods: Feb, Mar, Apr (Jan is consumed as the first training period)
    assert all(f.train_start == datetime(2020, 1, 1) for f in folds)
    assert [f.train_end for f in folds] == [datetime(2020, 2, 1), datetime(2020, 3, 1), datetime(2020, 4, 1)]


def test_folds_tile_without_gaps_or_overlap():
    folds = list(generate_walk_forward_folds(datetime(2020, 1, 1), datetime(2020, 6, 1), step_months=1))
    for f in folds:
        assert f.eval_start == f.train_end  # eval period starts exactly where training ends
    for a, b in zip(folds, folds[1:]):
        assert a.eval_end == b.train_end  # consecutive folds tile with no gap


def test_sliding_window_train_start_trails_train_end():
    folds = list(
        generate_walk_forward_folds(
            datetime(2020, 1, 1), datetime(2020, 6, 1), step_months=1, window="sliding", sliding_window_months=1
        )
    )
    for f in folds:
        assert f.train_start >= datetime(2020, 1, 1)
        # A 1-month sliding window trains on exactly [train_end - 1mo, train_end).
        assert f.train_end.month - f.train_start.month in (1, -11)  # handles year wraparound


def test_sliding_window_clamped_at_the_start_of_available_data():
    # A 3-month sliding window can't reach before `start` for the very first fold.
    folds = list(
        generate_walk_forward_folds(
            datetime(2020, 1, 1), datetime(2020, 4, 1), step_months=1, window="sliding", sliding_window_months=3
        )
    )
    assert folds[0].train_start == datetime(2020, 1, 1)


def test_no_folds_when_range_shorter_than_two_steps():
    folds = list(generate_walk_forward_folds(datetime(2020, 1, 1), datetime(2020, 1, 20), step_months=1))
    assert folds == []


def test_aggregate_fold_results_micro_averages_confusion_counts():
    fold_results = [
        {"tp": 10, "fp": 5, "fn": 2, "f_beta": 0.7, "beta": 0.5},
        {"tp": 4, "fp": 20, "fn": 1, "f_beta": 0.3, "beta": 0.5},
    ]
    agg = aggregate_fold_results(fold_results)

    total_tp, total_fp, total_fn = 14, 25, 3
    expected_precision = total_tp / (total_tp + total_fp)
    expected_recall = total_tp / (total_tp + total_fn)
    assert agg["micro"]["precision"] == expected_precision
    assert agg["micro"]["recall"] == expected_recall
    assert agg["macro"]["f_beta"] == (0.7 + 0.3) / 2
    assert agg["n_folds"] == 2


def test_aggregate_fold_results_empty():
    agg = aggregate_fold_results([])
    assert agg["n_folds"] == 0

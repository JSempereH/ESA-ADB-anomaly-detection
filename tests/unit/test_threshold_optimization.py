import numpy as np
import pytest

from esa_adb.evaluation import find_best_threshold


def test_finds_a_threshold_that_perfectly_separates_when_possible():
    y_true = np.array([0, 0, 0, 1, 1, 0, 0])
    scores = np.array([0.1, 0.2, 0.15, 0.9, 0.8, 0.05, 0.3])

    result = find_best_threshold(scores, y_true, beta=0.5)

    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f_beta"] == 1.0
    assert 0.3 < result["threshold"] <= 0.8


def test_picks_the_threshold_with_the_actual_highest_f_beta():
    # y=1 scores: 0.9, 0.8, 0.3. y=0 scores: 0.85 (sits between the two highest positives,
    # so no threshold can separate {0.9, 0.8} from it without also excluding 0.8), plus
    # several low negatives. Hand-verified via precision_recall_curve: the true F0.5
    # maximum is at threshold=0.3 (precision=0.75, recall=1.0, F0.5=0.789) -- better than
    # threshold=0.9 (precision=1.0, recall=1/3, F0.5=0.714) or threshold=0.8
    # (precision=2/3, recall=2/3, F0.5=0.667, since 0.85 gets caught alongside 0.8).
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.3, 0.85, 0.2, 0.1, 0.15, 0.05, 0.12, 0.08])

    result = find_best_threshold(scores, y_true, beta=0.5)
    assert result["precision"] == 0.75
    assert result["recall"] == 1.0
    assert result["f_beta"] == pytest.approx(0.7895, abs=1e-3)


def test_empty_or_all_one_class_does_not_crash():
    result = find_best_threshold(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 0]))
    assert result["f_beta"] == 0.0

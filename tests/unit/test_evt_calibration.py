import numpy as np

from esa_adb.models.evt_calibration import EVTCalibrator, _gpd_quantile


def test_gpd_quantile_above_peak_level_on_heavy_tailed_data():
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0, 1, 5000), rng.exponential(3, 200) + 5])
    q = _gpd_quantile(scores, q=0.98, alpha=0.01, min_excesses=30)
    assert q is not None
    assert q > np.percentile(scores, 98)


def test_gpd_quantile_falls_back_when_too_few_excesses():
    rng = np.random.default_rng(1)
    scores = rng.normal(0, 1, 50)
    assert _gpd_quantile(scores, q=0.98, alpha=0.01, min_excesses=30) is None


def test_calibrator_first_fold_has_no_smoothing():
    rng = np.random.default_rng(2)
    scores = rng.normal(0, 1, 2000)
    cal = EVTCalibrator()
    t1 = cal.calibrate(scores)
    assert cal.prev_threshold == t1


def test_calibrator_clips_large_jumps_between_folds():
    rng = np.random.default_rng(3)
    cal = EVTCalibrator(delta=0.05)
    cal.calibrate(rng.normal(0, 1, 2000))
    prev = cal.prev_threshold
    huge_jump_scores = rng.normal(0, 1, 2000) * 100  # a wildly different-scale fold
    new_threshold = cal.calibrate(huge_jump_scores)
    assert abs(new_threshold - prev) <= 1.0001 * 0.05 * abs(prev)


def test_calibrator_is_deterministic_given_same_inputs():
    rng = np.random.default_rng(4)
    scores = rng.normal(0, 1, 2000)
    a, b = EVTCalibrator(), EVTCalibrator()
    assert a.calibrate(scores) == b.calibrate(scores)

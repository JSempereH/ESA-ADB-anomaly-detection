import numpy as np

from esa_adb.models.ensemble import PairedLearnerEnsemble


class _FixedPredictor:
    """A stub detector that always predicts a given fixed bool array, regardless of X."""

    def __init__(self, predictions: np.ndarray):
        self._predictions = predictions

    def predict(self, X, threshold=None):
        return self._predictions


def test_and_combination_flags_only_where_both_agree():
    stable = _FixedPredictor(np.array([True, True, False, False]))
    reactive = _FixedPredictor(np.array([True, False, True, False]))
    ensemble = PairedLearnerEnsemble(stable, reactive)

    result = ensemble.predict(np.zeros((4, 1)))

    assert result.tolist() == [True, False, False, False]


def test_settled_drift_scenario_only_stable_flags_not_ensemble():
    # The stable (long-memory) detector still expects the old normal and flags a
    # genuinely-drifted-but-now-normal region; the reactive (short-memory) detector has
    # already adapted and doesn't. The ensemble should NOT alarm here.
    stable_flags_drifted_region = np.array([False, False, True, True, True])
    reactive_already_adapted = np.array([False, False, False, False, False])
    ensemble = PairedLearnerEnsemble(_FixedPredictor(stable_flags_drifted_region), _FixedPredictor(reactive_already_adapted))

    result = ensemble.predict(np.zeros((5, 1)))

    assert not result.any()


def test_agreement_stats_breakdown():
    stable = _FixedPredictor(np.array([True, True, False, False]))
    reactive = _FixedPredictor(np.array([True, False, True, False]))
    ensemble = PairedLearnerEnsemble(stable, reactive)

    stats = ensemble.agreement_stats(np.zeros((4, 1)))

    assert stats == {"both_flag": 1, "only_stable": 1, "only_reactive": 1, "neither": 1}


def test_predict_passes_through_explicit_thresholds():
    calls = {}

    class _RecordingPredictor:
        def predict(self, X, threshold=None):
            calls["threshold"] = threshold
            return np.array([True])

    ensemble = PairedLearnerEnsemble(_RecordingPredictor(), _FixedPredictor(np.array([True])))
    ensemble.predict(np.zeros((1, 1)), stable_threshold=0.42)
    assert calls["threshold"] == 0.42

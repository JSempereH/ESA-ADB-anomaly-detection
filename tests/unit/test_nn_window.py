import numpy as np

from esa_adb.models.lightweight import NearestNominalWindow
from esa_adb.preprocessing.windowing import WindowSpec


def _periodic(n_rows: int = 1000, n_channels: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n_rows)
    base = np.stack([np.sin(2 * np.pi * t / 50 + phase) for phase in np.linspace(0, 1, n_channels)], axis=1)
    return base + 0.01 * rng.normal(size=(n_rows, n_channels))


class TestNearestNominalWindow:
    def test_novel_pattern_scores_higher(self):
        spec = WindowSpec(length=10, stride=10)
        model = NearestNominalWindow(spec, max_reference_windows=500).fit(_periodic())
        nominal_scores = model.score(_periodic(seed=1))
        anomalous = _periodic(seed=1)
        anomalous[500:520] = 5.0  # flat segment at a level never seen
        anomalous_scores = model.score(anomalous)
        assert anomalous_scores[50:52].max() > 10 * np.percentile(nominal_scores, 99)

    def test_rare_but_seen_pattern_scores_low(self):
        # The matrix-profile selling point: a rare nominal pattern present in training
        # stays quiet at test time because its exact window is memorized.
        spec = WindowSpec(length=10, stride=10)
        train = _periodic()
        train[300:320] = 3.0  # rare maneuver, but nominal
        model = NearestNominalWindow(spec, max_reference_windows=500).fit(train)
        test = _periodic(seed=1)
        test[600:620] = 3.0  # same maneuver again
        scores = model.score(test)
        assert scores[60:62].max() < 5 * np.percentile(model.score(_periodic(seed=2)), 99)

    def test_reservoir_bound_respected(self):
        spec = WindowSpec(length=10, stride=10)
        model = NearestNominalWindow(spec, max_reference_windows=20)
        model.fit(_periodic(n_rows=5000))
        assert model._reference.shape[0] == 20
        model.partial_fit(_periodic(n_rows=5000, seed=3))
        assert model._reference.shape[0] == 20
        assert model._windows_seen == 1000

    def test_partial_fit_absorbs_new_regime(self):
        spec = WindowSpec(length=10, stride=10)
        model = NearestNominalWindow(spec, max_reference_windows=1000).fit(_periodic())
        shifted = _periodic(seed=4) + 5.0
        before = model.score(shifted).mean()
        model.partial_fit(shifted)
        after = model.score(shifted).mean()
        assert after < 0.5 * before

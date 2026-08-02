import numpy as np
import pytest
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest

from esa_adb.models.lightweight import PyODDetector

FACTORIES = {
    "ECOD": lambda: ECOD(),
    "COPOD": lambda: COPOD(),
    "IForest": lambda: IForest(random_state=42),
}


def _nominal(n_rows: int = 1000, n_channels: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n_rows)
    base = np.stack([np.sin(2 * np.pi * t / 50 + phase) for phase in np.linspace(0, 1, n_channels)], axis=1)
    return base + 0.05 * rng.normal(size=(n_rows, n_channels))


@pytest.mark.parametrize("name", list(FACTORIES))
class TestPyODDetector:
    def test_anomaly_scores_above_nominal(self, name):
        # Confirms both the wrapper's fit/score plumbing AND PyOD's own "higher = more
        # abnormal" score convention (evaluation.py/AnomalyDetector assume this).
        X = _nominal(seed=0)
        model = PyODDetector(FACTORIES[name]).fit(X)
        X_test = _nominal(seed=1)
        nominal_level = np.percentile(model.score(X_test), 99)
        X_anom = X_test.copy()
        X_anom[500] += 15.0
        scores = model.score(X_anom)
        assert scores[500] > nominal_level

    def test_partial_fit_runs_and_still_detects(self, name):
        X = _nominal(seed=0)
        model = PyODDetector(FACTORIES[name], buffer_rows=2000)
        model.partial_fit(X[:500])
        model.partial_fit(X[500:])
        X_anom = _nominal(seed=1)
        nominal_level = np.percentile(model.score(X_anom), 99)
        X_anom[500] += 15.0
        assert model.score(X_anom)[500] > nominal_level

    def test_reservoir_bound_respected(self, name):
        model = PyODDetector(FACTORIES[name], buffer_rows=50)
        model.partial_fit(_nominal(n_rows=1000))
        assert model._buffer.shape[0] == 50
        model.partial_fit(_nominal(n_rows=1000, seed=3))
        assert model._buffer.shape[0] == 50
        assert model._buffer_seen == 2000

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from esa_adb.models.runners import build_runners  # noqa: E402
from esa_adb.preprocessing.windowing import WindowSpec  # noqa: E402

SPEC = WindowSpec(length=10, stride=10)


def _data(seed=0, n_channels=5):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(600, n_channels)),
        rng.normal(size=(300, n_channels)),
        rng.normal(size=(300, n_channels)),
    )


def test_multiseed_runners_average_three_members():
    fit, calib, ev = _data()
    runners = build_runners(["CRTFAutoencoder-multiseed", "RoCALite-multiseed"], 5, {}, SPEC)
    for name, r in runners.items():
        assert len(r.members) == 3
        c, e = r.online_scores(fit, calib, ev)
        assert c.shape == (300,) and e.shape == (300,)
        assert np.isfinite(c).all() and np.isfinite(e).all()


def test_multiseed_matches_manual_average():
    fit, calib, ev = _data(seed=1)
    runners = build_runners(["CRTFAutoencoder-multiseed"], 5, {}, SPEC)
    r = runners["CRTFAutoencoder-multiseed"]
    c, e = r.online_scores(fit, calib, ev)
    manual_c = np.mean([m.score_rows(calib) for m in r.members], axis=0)
    np.testing.assert_allclose(c, manual_c, atol=1e-6)


def test_gdn_runner_online_scores_and_run_config():
    fit, calib, ev = _data(seed=2)
    runner = build_runners(["GDN"], 5, {}, SPEC)["GDN"]
    c, e = runner.online_scores(fit, calib, ev)
    assert c.shape == (300,) and e.shape == (300,)
    assert np.isfinite(c).all() and np.isfinite(e).all()
    assert runner.run_config()["seed"] == 42


def test_gdn_runner_calibrates_each_call():
    fit, calib, ev = _data(seed=3)
    runner = build_runners(["GDN"], 5, {}, SPEC)["GDN"]
    runner.online_scores(fit, calib, ev)
    median_after_first = runner.online_model.median_.copy()
    fit2, calib2, _ = _data(seed=4)
    runner.online_scores(fit2, calib2, ev)
    # calibrate_threshold is called again each fold with the NEW calib slice
    assert not np.array_equal(runner.online_model.median_, median_after_first)


def test_fusion_runner_combines_five_members_via_rank_average():
    fit, calib, ev = _data(seed=5, n_channels=6)
    runner = build_runners(["HeterogeneousFusion"], 6, {}, SPEC)["HeterogeneousFusion"]
    assert set(runner.members) == {"PyOD-ECOD", "LSTMForecaster", "WindowedIsolationForest", "PyOD-COPOD", "PyOD-IForest"}
    c, e = runner.online_scores(fit, calib, ev)
    assert c.shape == (300,) and e.shape == (300,)
    assert (c >= 0).all() and (c <= 1).all()  # rank-averaged, so bounded in [0, 1]


def test_fusion_rank_helper_is_a_valid_permutation_rank():
    from esa_adb.models.runners import FusionRunner

    x = np.array([5.0, 1.0, 3.0, 1.0, 9.0])
    ranks = FusionRunner._rank(x)
    assert ranks.min() == 0.0 and ranks.max() == 1.0
    assert ranks[4] == 1.0  # the largest value gets the top rank

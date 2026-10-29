import numpy as np

from esa_adb.models.lightweight.matrix_profile import MatrixProfileDetector


def test_fit_and_score_finite():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2))
    model = MatrixProfileDetector(m=10, ref_rows=200).fit(X)
    scores = model.score(rng.normal(size=(100, 2)))
    assert scores.shape == (100,)
    assert np.isfinite(scores).all()


def test_scores_a_novel_pattern_higher_than_reference_pattern():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 40 * np.pi, 600)
    ref_pattern = np.sin(t)[:, None] + rng.normal(scale=0.01, size=(600, 1))
    model = MatrixProfileDetector(m=10, ref_rows=500).fit(ref_pattern)

    same_pattern = np.sin(np.linspace(0, 4 * np.pi, 60))[:, None] + rng.normal(scale=0.01, size=(60, 1))
    novel_pattern = np.full((60, 1), 50.0) + rng.normal(scale=0.01, size=(60, 1))

    assert model.score(novel_pattern).mean() > model.score(same_pattern).mean()


def test_ref_rows_caps_reference_window():
    rng = np.random.default_rng(2)
    model = MatrixProfileDetector(m=5, ref_rows=100).fit(rng.normal(size=(1000, 3)))
    assert model.reference_.shape[0] == 100


def test_score_per_channel_shape():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 4))
    model = MatrixProfileDetector(m=10, ref_rows=200).fit(X)
    z = model.score_per_channel(rng.normal(size=(50, 4)))
    assert z.shape == (50, 4)
    assert np.isfinite(z).all()


def test_fusion_sum_vs_max():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 3))
    eval_values = rng.normal(size=(50, 3))
    sum_model = MatrixProfileDetector(m=10, ref_rows=200, fusion="sum").fit(X)
    max_model = MatrixProfileDetector(m=10, ref_rows=200, fusion="max").fit(X)
    sum_scores = sum_model.score(eval_values)
    max_scores = max_model.score(eval_values)
    assert (sum_scores >= max_scores - 1e-9).all()

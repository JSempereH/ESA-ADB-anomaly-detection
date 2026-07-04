import numpy as np

from esa_adb.models.window_features import FEATURE_NAMES, extract_window_features
from esa_adb.preprocessing.windowing import WindowSpec


def test_output_shape():
    n_channels = 3
    values = np.random.default_rng(0).normal(size=(100, n_channels))
    spec = WindowSpec(length=10, stride=10)

    X = extract_window_features(values, spec)

    assert X.shape == (10, len(FEATURE_NAMES) * n_channels)


def test_mean_and_std_features_match_numpy_directly():
    values = np.arange(40, dtype=np.float64).reshape(20, 2)
    spec = WindowSpec(length=20, stride=20)

    X = extract_window_features(values, spec)

    # Feature layout is [mean(n_channels), std(n_channels), skew(...), ...] concatenated.
    n_channels = 2
    np.testing.assert_allclose(X[0, 0:n_channels], values.mean(axis=0))
    np.testing.assert_allclose(X[0, n_channels : 2 * n_channels], values.std(axis=0))


def test_distinguishes_flat_from_oscillating_window_with_same_mean_and_variance():
    length = 40
    t = np.arange(length)
    flat_with_one_spike = np.zeros(length)
    flat_with_one_spike[0] = np.sqrt(length)
    flat_with_one_spike -= flat_with_one_spike.mean()  # exactly zero-mean, concentrates variance in one point
    oscillating = np.sin(t * 2 * np.pi * 5 / length) * np.sqrt(2)  # zero-mean by construction, spread across frequency

    # Scale oscillating to match variance exactly for a fair comparison.
    oscillating = oscillating / oscillating.std() * flat_with_one_spike.std()

    window_a = flat_with_one_spike.reshape(-1, 1)
    window_b = oscillating.reshape(-1, 1)
    spec = WindowSpec(length=length, stride=length)

    fa = extract_window_features(window_a, spec)[0]
    fb = extract_window_features(window_b, spec)[0]

    # Same mean and std (by construction), but spectral/kurtosis features must differ --
    # this is exactly the point of richer features over raw-value flattening.
    n = 1
    np.testing.assert_allclose(fa[0:n], fb[0:n], atol=1e-6)  # mean
    np.testing.assert_allclose(fa[n : 2 * n], fb[n : 2 * n], atol=1e-6)  # std
    assert not np.allclose(fa[3 * n : 4 * n], fb[3 * n : 4 * n])  # kurtosis differs
    assert not np.allclose(fa[5 * n : 6 * n], fb[5 * n : 6 * n])  # spectral_energy differs


def test_slope_captures_linear_trend():
    length = 30
    trending = np.arange(length, dtype=np.float64).reshape(-1, 1)  # slope should be ~1
    flat = np.zeros((length, 1))
    spec = WindowSpec(length=length, stride=length)

    f_trend = extract_window_features(trending, spec)[0]
    f_flat = extract_window_features(flat, spec)[0]

    slope_index = 4  # [mean, std, skew, kurtosis, slope, ...] -> index 4 for n_channels=1
    assert f_trend[slope_index] > 0.9
    assert abs(f_flat[slope_index]) < 1e-6


def test_empty_input_returns_correctly_shaped_empty_array():
    values = np.zeros((3, 2))
    spec = WindowSpec(length=10, stride=10)
    X = extract_window_features(values, spec)
    assert X.shape == (0, len(FEATURE_NAMES) * 2)

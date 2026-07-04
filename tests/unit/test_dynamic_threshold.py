import time

import numpy as np

from esa_adb.models.dynamic_threshold import POTThreshold, RollingPercentileThreshold


def test_threshold_reflects_percentile_of_recent_window():
    scores = list(range(1, 101))  # 1..100
    thr = RollingPercentileThreshold(window_size=1000, percentile=99.0, initial_scores=np.array(scores))
    assert thr.threshold == np.percentile(scores, 99.0)


def test_window_is_bounded_and_forgets_old_scores():
    thr = RollingPercentileThreshold(window_size=10, percentile=100.0)
    for s in range(20):  # only the last 10 (10..19) should remain
        thr.update(s)
    assert thr.threshold == 19.0


def test_protect_from_contamination_keeps_flagged_scores_out_of_the_window():
    thr = RollingPercentileThreshold(
        window_size=1000, percentile=99.0, initial_scores=np.zeros(100), protect_from_contamination=True
    )
    baseline_threshold = thr.threshold
    thr.predict_and_update(1000.0)  # a huge outlier -- must not drag the window up
    assert thr.threshold == baseline_threshold


def test_default_folds_in_every_score_including_flagged_ones():
    # Default (protect_from_contamination=False) prioritizes drift-tracking: even a
    # flagged score updates the window, unlike the protected mode above. A single
    # outlier among many zeros won't move a 99th-percentile threshold (correctly), so
    # check the underlying buffer directly rather than the threshold value.
    thr = RollingPercentileThreshold(window_size=1000, percentile=99.0, initial_scores=np.zeros(100))
    thr.predict_and_update(1000.0)
    assert 1000.0 in thr._buffer


def test_predict_and_update_flags_scores_above_threshold():
    thr = RollingPercentileThreshold(window_size=100, percentile=99.0, initial_scores=np.zeros(100))
    assert thr.predict_and_update(0.0) is False
    assert thr.predict_and_update(100.0) is True


def test_predict_stream_adapts_after_a_regime_shift():
    # A nominal baseline that jumps to a new, stable level (a regime shift -- the kind of
    # concept drift a bounded rolling window handles well) rather than a continuous
    # drift (for which a plain rolling window necessarily lags behind by an amount
    # proportional to window_size x drift_rate, a real and separate limitation worth
    # knowing about but not what this test targets).
    rng = np.random.default_rng(0)
    before = rng.normal(0, 0.1, size=1000)
    after = rng.normal(3, 0.1, size=2000)
    stream = np.concatenate([before, after])

    thr = RollingPercentileThreshold(window_size=300, percentile=99.0, initial_scores=before[:300])
    preds = thr.predict_stream(stream[300:])

    # Once the window has fully turned over into the new regime (a bit more than
    # window_size samples past the shift), it should have adapted and mostly stop
    # flagging the new, now-stable nominal level.
    tail = preds[-500:]
    assert tail.mean() < 0.05


def test_empty_threshold_is_infinite_so_nothing_is_flagged_before_any_data():
    thr = RollingPercentileThreshold(window_size=10)
    assert thr.threshold == float("inf")
    assert thr.predict_and_update(1e9) is False


def test_recompute_every_gives_large_streams_tractable_runtime():
    # `recompute_every` amortizes the window sort over many updates instead of doing it
    # on every single sample, which is what makes large streams (millions of rows)
    # tractable.
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 1, size=200_000)

    thr = RollingPercentileThreshold(window_size=5000, percentile=99.0, recompute_every=500)
    t0 = time.time()
    thr.predict_stream(scores)
    elapsed = time.time() - t0

    assert elapsed < 5.0  # unbounded without recompute_every amortizing the sort cost


def test_predict_batch_matches_predict_stream_for_rolling_threshold():
    rng = np.random.default_rng(10)
    nominal = rng.normal(0, 1, size=1000)
    stream = rng.normal(0, 1, size=3000)

    a = RollingPercentileThreshold(window_size=300, percentile=99.0, initial_scores=nominal, recompute_every=50)
    stream_preds = a.predict_stream(stream)

    b = RollingPercentileThreshold(window_size=300, percentile=99.0, initial_scores=nominal, recompute_every=50)
    batch_preds = b.predict_batch(stream)

    np.testing.assert_array_equal(stream_preds, batch_preds)
    assert a.threshold == b.threshold


def test_predict_batch_handles_large_streams_quickly():
    # RollingPercentileThreshold's own per-sample loop overhead is modest (its
    # recompute_every already does the heavy lifting -- see
    # test_recompute_every_gives_large_streams_tractable_runtime); predict_batch's real
    # payoff shows up combined with POT's much costlier genpareto.fit recomputation (see
    # test_pot_predict_batch_is_faster_than_predict_stream_at_scale below). This just
    # confirms predict_batch itself stays fast in isolation, at a larger scale than the
    # correctness test above.
    rng = np.random.default_rng(11)
    scores = rng.normal(0, 1, size=200_000)
    thr = RollingPercentileThreshold(window_size=5000, percentile=99.0, recompute_every=200)
    t0 = time.time()
    thr.predict_batch(scores)
    assert time.time() - t0 < 5.0


def test_pot_recompute_every_is_the_real_lever_on_runtime():
    # predict_batch barely beats predict_stream for POT -- genpareto.fit's own MLE cost,
    # identical between the two, dominates either way -- so recompute_every (how often
    # that MLE fit runs at all) is the actual lever on total runtime, not which
    # predict_* method is used.
    rng = np.random.default_rng(13)
    nominal = np.abs(rng.normal(0, 1, size=2000))
    stream = np.abs(rng.normal(0, 1, size=5000))

    frequent = POTThreshold(risk=1e-3, initial_scores=nominal, recompute_every=50)
    t0 = time.time()
    frequent.predict_batch(stream)
    frequent_elapsed = time.time() - t0

    sparse = POTThreshold(risk=1e-3, initial_scores=nominal, recompute_every=1000)
    t0 = time.time()
    sparse.predict_batch(stream)
    sparse_elapsed = time.time() - t0

    assert sparse_elapsed < frequent_elapsed * 0.5


def test_pot_predict_batch_matches_predict_stream():
    rng = np.random.default_rng(12)
    nominal = np.abs(rng.normal(0, 1, size=2000))
    stream = np.abs(rng.normal(0, 1, size=2000))

    a = POTThreshold(risk=1e-3, initial_scores=nominal, recompute_every=100)
    stream_preds = a.predict_stream(stream)

    b = POTThreshold(risk=1e-3, initial_scores=nominal, recompute_every=100)
    batch_preds = b.predict_batch(stream)

    np.testing.assert_array_equal(stream_preds, batch_preds)
    assert a.threshold == b.threshold


def test_pot_falls_back_to_percentile_with_too_few_excesses():
    # A tiny initial batch can't support a GPD fit (min_excesses=20 by default) -- must
    # gracefully fall back to the plain rolling-percentile threshold rather than error
    # out or produce a degenerate result.
    thr = POTThreshold(initial_scores=np.random.default_rng(0).normal(size=30), min_excesses=20)
    fallback_only = RollingPercentileThreshold(percentile=99.0, initial_scores=np.random.default_rng(0).normal(size=30))
    assert thr.threshold == fallback_only.threshold


def test_pot_flags_extreme_outlier_after_enough_tail_data():
    rng = np.random.default_rng(1)
    # Heavy-tailed-ish nominal data (folded normal) with a genuine, well-populated tail.
    nominal = np.abs(rng.normal(0, 1, size=5000))

    thr = POTThreshold(risk=1e-3, initial_scores=nominal)
    assert thr.predict_and_update(0.1) is False  # deep in the bulk of the distribution
    assert thr.predict_and_update(50.0) is True  # far beyond anything in the fitted tail


def test_pot_threshold_is_at_or_above_the_peak_level():
    rng = np.random.default_rng(2)
    nominal = rng.normal(0, 1, size=5000)
    thr = POTThreshold(risk=1e-3, initial_scores=nominal)
    assert thr.threshold >= thr._t


def test_pot_same_input_gives_same_threshold():
    rng = np.random.default_rng(3)
    nominal = rng.normal(0, 1, size=3000)
    a = POTThreshold(risk=1e-3, initial_scores=nominal)
    b = POTThreshold(risk=1e-3, initial_scores=nominal)
    assert a.threshold == b.threshold


def test_pot_predict_stream_matches_iterative_predict_and_update():
    rng = np.random.default_rng(4)
    nominal = rng.normal(0, 1, size=2000)
    stream = rng.normal(0, 1, size=500)

    a = POTThreshold(risk=1e-3, initial_scores=nominal, recompute_every=50)
    stream_preds = a.predict_stream(stream)

    b = POTThreshold(risk=1e-3, initial_scores=nominal, recompute_every=50)
    manual_preds = [b.predict_and_update(s) for s in stream]

    assert stream_preds.tolist() == manual_preds


def test_pot_excess_deque_is_memory_bounded():
    rng = np.random.default_rng(5)
    thr = POTThreshold(window_size=50, min_excesses=5, init_threshold_percentile=50.0, initial_scores=rng.normal(size=200))
    for s in rng.normal(10, 1, size=5000):  # a long run of values, most above the peak level
        thr.update(s)
    assert len(thr._excesses) <= 50


def test_pot_empty_threshold_is_infinite_before_any_data():
    thr = POTThreshold()
    assert thr.threshold == float("inf")


def test_recompute_every_one_matches_exact_per_sample_behaviour():
    # recompute_every=1 (the default) must still behave exactly as before the
    # caching change -- no regression in correctness, only in how often it re-sorts.
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 1, size=2000)

    exact = RollingPercentileThreshold(window_size=300, percentile=95.0, recompute_every=1)
    exact_preds = exact.predict_stream(scores)

    reference_flags = []
    buffer: list[float] = []
    for s in scores:
        thr = np.percentile(buffer, 95.0) if buffer else float("inf")
        flagged = s > thr
        reference_flags.append(flagged)
        buffer.append(float(s))
        if len(buffer) > 300:
            buffer.pop(0)

    assert exact_preds.tolist() == reference_flags

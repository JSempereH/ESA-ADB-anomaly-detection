import numpy as np
import pytest

from esa_adb.models.postprocess import PostprocessSpec, enforce_min_duration, ewma, merge_short_gaps


class TestEwma:
    def test_none_alpha_is_passthrough(self):
        x = np.array([1.0, 5.0, 2.0])
        assert ewma(x, None) is x

    def test_alpha_one_is_passthrough(self):
        x = np.array([1.0, 5.0, 2.0])
        assert ewma(x, 1.0) is x

    def test_matches_recursive_definition(self):
        x = np.array([1.0, 5.0, 2.0, 0.0])
        alpha = 0.3
        expected = [x[0]]
        for v in x[1:]:
            expected.append(alpha * v + (1 - alpha) * expected[-1])
        np.testing.assert_allclose(ewma(x, alpha), expected, rtol=1e-12)

    def test_suppresses_single_spike_below_plateau(self):
        # A 1-sample spike to 10 must smooth to well under a genuine plateau at 10.
        spike = np.zeros(50)
        spike[25] = 10.0
        plateau = np.zeros(50)
        plateau[20:31] = 10.0
        assert ewma(spike, 0.3).max() < 0.5 * ewma(plateau, 0.3).max()

    def test_empty_input(self):
        assert ewma(np.array([]), 0.3).shape == (0,)


class TestMergeShortGaps:
    def test_fills_small_gap(self):
        pred = np.array([True, True, False, False, True, True])
        merged = merge_short_gaps(pred, max_gap_rows=2)
        assert merged.all()

    def test_keeps_large_gap(self):
        pred = np.array([True, False, False, False, True])
        np.testing.assert_array_equal(merge_short_gaps(pred, max_gap_rows=2), pred)

    def test_no_segments(self):
        pred = np.zeros(5, dtype=bool)
        np.testing.assert_array_equal(merge_short_gaps(pred, 3), pred)

    def test_does_not_extend_edges(self):
        pred = np.array([False, True, False])
        np.testing.assert_array_equal(merge_short_gaps(pred, 5), pred)


class TestEnforceMinDuration:
    def test_drops_short_segment(self):
        pred = np.array([False, True, False, True, True, True])
        out = enforce_min_duration(pred, min_rows=2)
        np.testing.assert_array_equal(out, [False, False, False, True, True, True])

    def test_keeps_exact_length(self):
        pred = np.array([True, True, False])
        np.testing.assert_array_equal(enforce_min_duration(pred, min_rows=2), pred)

    def test_all_false_unchanged(self):
        pred = np.zeros(4, dtype=bool)
        np.testing.assert_array_equal(enforce_min_duration(pred, 3), pred)


class TestPostprocessSpec:
    def test_merge_runs_before_min_duration(self):
        # Two 3-row fragments separated by a 1-row gap: individually shorter than
        # min_duration=5, but fused they span 7 rows and must survive.
        pred = np.zeros(20, dtype=bool)
        pred[5:8] = True
        pred[9:12] = True
        out = PostprocessSpec(merge_gap_rows=2, min_duration_rows=5).apply(pred)
        assert out[5:12].all() and out.sum() == 7

    def test_isolated_spike_removed(self):
        pred = np.zeros(20, dtype=bool)
        pred[10] = True
        assert not PostprocessSpec(merge_gap_rows=2, min_duration_rows=5).apply(pred).any()


class TestFindBestThresholdEventWise:
    @pytest.fixture()
    def events(self):
        import polars as pl

        return pl.DataFrame(
            {
                "ID": ["ev_1"],
                "Channel": ["channel_1"],
                "Category": ["anomaly"],
                "StartTime": [np.datetime64("2001-01-01T01:00:00").astype("datetime64[us]").astype(object)],
                "EndTime": [np.datetime64("2001-01-01T02:00:00").astype("datetime64[us]").astype(object)],
            }
        )

    def test_prefers_quiet_threshold_that_still_catches_event(self, events):
        from esa_adb.evaluation import find_best_threshold_event_wise

        n = 1000
        times = np.datetime64("2001-01-01T00:00:00") + np.arange(n) * np.timedelta64(18, "s")
        rng = np.random.default_rng(0)
        scores = rng.normal(0.0, 1.0, n)
        in_event = (times >= np.datetime64("2001-01-01T01:00:00")) & (times <= np.datetime64("2001-01-01T02:00:00"))
        scores[in_event] += 10.0

        best = find_best_threshold_event_wise(scores, times, events)
        assert not best["fallback"]
        assert best["f_beta"] > 0.9
        # The chosen threshold must sit above the nominal noise.
        assert best["threshold"] > scores[~in_event].max() * 0.5

    def test_no_events_falls_back_to_quietest(self, events):
        from esa_adb.evaluation import find_best_threshold_event_wise

        n = 500
        times = np.datetime64("2005-06-01T00:00:00") + np.arange(n) * np.timedelta64(18, "s")  # far from any event
        scores = np.random.default_rng(1).normal(size=n)
        best = find_best_threshold_event_wise(scores, times, events)
        assert best["fallback"]
        assert best["threshold"] >= np.quantile(scores, 0.9999) - 1e-12

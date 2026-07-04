#!/usr/bin/env python3
"""Four follow-up explainability analyses on real Mission2 data, building on
scripts/demo_explain_detections.py's finding that `subsystem_1` (channels 17-20)
explains most of Autoencoder's detections. Trains PCAReconstructor, Autoencoder, and GDN
ONCE and reuses them across all four analyses (each retrains ~4-5 min on its own; no
reason to pay that four times over).

1. **GDN's learned graph vs. known subsystem structure**: does `graph_edges()` connect
   more channels WITHIN the same `channels.csv` subsystem than a random top-k graph
   would, without ever being told about subsystems?
2. **Cross-model agreement**: for events both `Autoencoder` and `PCAReconstructor`
   detect, do they blame the same channel? Agreement suggests real, model-independent
   signal; disagreement is informative about how differently linear vs. nonlinear
   reconstruction "see" the same anomaly.
3. **True positives vs. false positives**: does `Autoencoder`'s explainability point to
   different channels for detections that DID overlap a real event vs. ones that
   didn't (false alarms)? A channel that dominates false positives but not true
   positives is a candidate "noisy" channel worth investigating.
4. **Temporal drift of the "usual suspect"**: grouping detected events by calendar
   month across Mission2's ~21-month test period, does `channel_18` stay dominant
   throughout, or does the top explaining channel shift over the mission's lifetime?

Every analysis is logged to data/experiments/results.jsonl via log_experiment (same as
every other script in this project), model_name prefixed "Explainability-".

Usage: python scripts/demo_explainability_deep_dive.py --mission 2
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import polars as pl

from esa_adb.config import load_mission_config
from esa_adb.evaluation import _contiguous_segments
from esa_adb.experiments import log_experiment
from esa_adb.io.channels import load_channel_metadata
from esa_adb.models.base import channel_matrix
from esa_adb.models.heavy import GDN, Autoencoder
from esa_adb.models.lightweight import PCAReconstructor
from esa_adb.preprocessing.labeling import filter_nominal_rows


def _load_split(lf, config, target_channels, labels_long):
    train_raw = lf.filter(pl.col("split") == "train").select(["time", *target_channels]).collect()
    train_nominal = filter_nominal_rows(train_raw, labels_long, target_channels)
    val_df = (
        lf.filter((pl.col("time") >= config.split.val_start) & (pl.col("time") < config.split.train_end))
        .select(["time", *target_channels])
        .sort("time")
        .collect()
    )
    val_nominal = filter_nominal_rows(val_df, labels_long, target_channels)
    test_df = (
        lf.filter((pl.col("time") >= config.split.test_start) & (pl.col("time") < config.split.test_end))
        .select(["time", *target_channels])
        .sort("time")
        .collect()
    )
    return train_nominal, val_df, val_nominal, test_df


def _peak_top_channels(model, values, times, start, end, target_channels, top_n=3):
    mask = (times >= start) & (times <= end)
    if not mask.any():
        return None
    idx = np.where(mask)[0]
    local_scores = model.score(values[idx])
    peak_row = values[idx[np.argmax(local_scores)]][None, :]
    return model.most_anomalous_channels(peak_row, channel_names=target_channels, top_n=top_n)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=int, required=True, choices=[2])
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--ae-epochs", type=int, default=30)
    parser.add_argument("--gdn-epochs", type=int, default=15)
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config_dir) / f"mission{args.mission}.yaml"
    config = load_mission_config(config_path)
    channel_meta = load_channel_metadata(Path(config.raw_data_dir) / "channels.csv")
    target_channels = channel_meta.filter(pl.col("is_target"))["channel"].to_list()
    channel_to_subsystem = dict(zip(channel_meta["channel"], channel_meta["subsystem"]))
    n_channels = len(target_channels)
    print(f"Mission{args.mission}: {n_channels} target channels")

    chunks_glob = str(Path(config.processed_dir) / "chunks" / "*.parquet")
    lf = pl.scan_parquet(chunks_glob)
    labels_long_files = sorted((Path(config.processed_dir) / "labels_long").glob("*.parquet"))
    labels_long = pl.concat([pl.read_parquet(f) for f in labels_long_files], how="vertical")
    events = pl.read_csv(Path(config.interim_dir) / f"anomalies_mission{args.mission}.csv", try_parse_dates=True).filter(
        pl.col("Channel").is_in(target_channels)
    )

    train_nominal, val_df, val_nominal, test_df = _load_split(lf, config, target_channels, labels_long)
    train_values = channel_matrix(train_nominal, target_channels)
    val_nominal_values = channel_matrix(val_nominal, target_channels)
    test_values = channel_matrix(test_df, target_channels)
    test_times = test_df["time"].to_numpy()

    print("\n=== Training PCAReconstructor, Autoencoder, GDN (once, reused below) ===")
    pca = PCAReconstructor(n_components=0.95).fit(train_values)
    pca.calibrate_threshold(val_nominal_values, percentile=99.0)
    print("PCAReconstructor done.")

    ae = Autoencoder(n_channels=n_channels, hidden_sizes=(32, 16), latent_size=8)
    ae.fit(train_values, epochs=args.ae_epochs, batch_size=2048, verbose=False)
    ae.calibrate_threshold(val_nominal_values, percentile=99.0)
    print("Autoencoder done.")

    gdn = GDN(n_channels=n_channels, window_length=10, embed_dim=32, top_k=15, hidden_size=64)
    gdn.fit(train_values, epochs=args.gdn_epochs, batch_size=2048, stride=10, verbose=False)
    gdn.calibrate_threshold(val_nominal_values, percentile=99.0)
    print("GDN done.")

    eval_start, eval_end = test_times.min(), test_times.max()
    in_range_events = events.filter((pl.col("EndTime") >= eval_start) & (pl.col("StartTime") <= eval_end))

    pca_pred = pca.predict(test_values)
    ae_pred = ae.predict(test_values)

    # === 1. GDN's learned graph vs. known subsystem structure ===
    print("\n=== 1. GDN graph vs. subsystem structure ===")
    edges = gdn.graph_edges(channel_names=target_channels)
    same_subsystem = sum(1 for src, dst in edges if channel_to_subsystem.get(src) == channel_to_subsystem.get(dst))
    observed_fraction = same_subsystem / len(edges)

    subsystem_sizes = Counter(channel_to_subsystem.get(c) for c in target_channels)
    # expected fraction under a random top-k graph: average, over channels, of
    # (that channel's subsystem size - 1) / (n_channels - 1) -- the chance any single
    # uniformly-random OTHER channel shares its subsystem.
    expected_fraction = np.mean([(subsystem_sizes[channel_to_subsystem[c]] - 1) / (n_channels - 1) for c in target_channels])

    print(f"Observed same-subsystem edge fraction: {observed_fraction:.3f}")
    print(f"Expected under a random graph: {expected_fraction:.3f}")
    print(f"Ratio: {observed_fraction / expected_fraction:.2f}x")
    log_experiment(
        model_name="Explainability-GDN-graph-vs-subsystems",
        mission=args.mission,
        config={"top_k": gdn.top_k, "embed_dim": gdn.embed_dim},
        metrics={
            "observed_same_subsystem_fraction": observed_fraction,
            "expected_same_subsystem_fraction": float(expected_fraction),
            "ratio": observed_fraction / expected_fraction,
            "n_edges": len(edges),
        },
        notes="Does GDN's learned top-k graph recover known subsystem structure it was never told about?",
        mirror_to_mlflow=args.mlflow,
    )

    # === 2. Cross-model agreement: Autoencoder vs. PCAReconstructor ===
    print("\n=== 2. Cross-model explainability agreement (Autoencoder vs. PCAReconstructor) ===")
    both_detected, agree_top1, disagreements = 0, 0, []
    for row in in_range_events.iter_rows(named=True):
        start, end = np.datetime64(row["StartTime"]), np.datetime64(row["EndTime"])
        mask = (test_times >= start) & (test_times <= end)
        if not mask.any() or not (ae_pred[mask].any() and pca_pred[mask].any()):
            continue
        both_detected += 1
        ae_top = _peak_top_channels(ae, test_values, test_times, start, end, target_channels, top_n=1)
        pca_top = _peak_top_channels(pca, test_values, test_times, start, end, target_channels, top_n=1)
        if ae_top and pca_top and ae_top[0] == pca_top[0]:
            agree_top1 += 1
        elif ae_top and pca_top and len(disagreements) < 10:
            disagreements.append((row["ID"], ae_top[0], pca_top[0]))

    agreement_rate = agree_top1 / both_detected if both_detected else 0.0
    print(f"Events detected by both: {both_detected}, agree on #1 channel: {agree_top1} ({agreement_rate:.1%})")
    print(f"Example disagreements (event_id, autoencoder_channel, pca_channel): {disagreements[:5]}")
    log_experiment(
        model_name="Explainability-CrossModel-agreement",
        mission=args.mission,
        config={},
        metrics={"n_events_both_detected": both_detected, "n_agree_top1": agree_top1, "agreement_rate": agreement_rate},
        notes="Do Autoencoder and PCAReconstructor blame the same #1 channel for events they both detect?",
        mirror_to_mlflow=args.mlflow,
    )

    # === 3. True positives vs. false positives (Autoencoder) ===
    print("\n=== 3. True-positive vs. false-positive explainability (Autoencoder) ===")
    segments = _contiguous_segments(ae_pred, test_times)
    event_starts = in_range_events["StartTime"].to_numpy()
    event_ends = in_range_events["EndTime"].to_numpy()
    tp_channels: Counter = Counter()
    fp_channels: Counter = Counter()
    for seg_start, seg_end in segments:
        overlap = ((event_starts <= seg_end) & (event_ends >= seg_start)).any() if len(event_starts) else False
        top = _peak_top_channels(ae, test_values, test_times, seg_start, seg_end, target_channels, top_n=1)
        if not top:
            continue
        (tp_channels if overlap else fp_channels).update(top)

    print(f"True-positive detections: {sum(tp_channels.values())}, top channels: {tp_channels.most_common(5)}")
    print(f"False-positive detections: {sum(fp_channels.values())}, top channels: {fp_channels.most_common(5)}")
    log_experiment(
        model_name="Explainability-Autoencoder-TP-vs-FP",
        mission=args.mission,
        config={},
        metrics={
            "tp_top_channels": dict(tp_channels.most_common(10)),
            "fp_top_channels": dict(fp_channels.most_common(10)),
            "n_tp_detections": sum(tp_channels.values()),
            "n_fp_detections": sum(fp_channels.values()),
        },
        notes="Do false-positive detections blame different channels than true-positive ones?",
        mirror_to_mlflow=args.mlflow,
    )

    # === 4. Temporal drift of the "usual suspect" channel (Autoencoder) ===
    print("\n=== 4. Temporal drift of the top explaining channel, by calendar month ===")
    monthly_channels: dict[str, Counter] = defaultdict(Counter)
    for row in in_range_events.iter_rows(named=True):
        start, end = np.datetime64(row["StartTime"]), np.datetime64(row["EndTime"])
        mask = (test_times >= start) & (test_times <= end)
        if not mask.any() or not ae_pred[mask].any():
            continue
        top = _peak_top_channels(ae, test_values, test_times, start, end, target_channels, top_n=1)
        if top:
            month_key = str(row["StartTime"])[:7]  # "YYYY-MM"
            monthly_channels[month_key].update(top)

    monthly_top = {month: counter.most_common(1)[0] for month, counter in sorted(monthly_channels.items())}
    for month, (channel, count) in monthly_top.items():
        print(f"  {month}: {channel} ({count} events)")
    unique_top_channels = {channel for channel, _ in monthly_top.values()}
    print(f"Distinct #1 channels across all months: {unique_top_channels}")
    log_experiment(
        model_name="Explainability-Autoencoder-temporal-drift",
        mission=args.mission,
        config={},
        metrics={"monthly_top_channel": {m: c for m, (c, _) in monthly_top.items()}, "n_distinct_top_channels": len(unique_top_channels)},
        notes="Does the dominant explaining channel change across the mission's ~21-month test period, or stay fixed?",
        mirror_to_mlflow=args.mlflow,
    )

    print(f"\nAll 4 analyses logged to {Path('data/experiments/results.jsonl').resolve()}")


if __name__ == "__main__":
    main()

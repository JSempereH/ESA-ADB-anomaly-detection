from __future__ import annotations

import resource
import sys
from contextlib import contextmanager


def _peak_rss_mb() -> float:
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KB on Linux, bytes on macOS.
    divisor = 1024 if sys.platform != "darwin" else 1024 * 1024
    return peak_kb / divisor


@contextmanager
def log_peak_rss(label: str):
    before = _peak_rss_mb()
    yield
    after = _peak_rss_mb()
    print(f"[memlog] {label}: peak RSS {before:.0f} MB -> {after:.0f} MB")

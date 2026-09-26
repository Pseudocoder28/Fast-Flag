"""Per-tick latency of the replay pipeline, per stage.

Stages (seconds, measured with time.perf_counter):
- tick:    building the tick from the race data (before it is emitted)
- frame:   slicing the tick's rows for the processors
- one per processor, named after its package (src.detect.* -> "detect",
  src.predict.* -> "predict"), so new processors are timed automatically
- send:    sending the tick's envelopes (broadcast in the server, JSON encoding in the benchmark)
- total:   from the moment the tick is emitted until its detection and risk envelopes are sent

Nothing here touches the envelopes, so the contract is unchanged.
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

BUDGET_MS = 250.0          # one tick period at 4 Hz
REPORT_EVERY_S = 30.0
MAX_SAMPLES = 500_000      # per stage, for the cumulative summary


def print_now(msg: str) -> None:
    print(msg, flush=True)      # flush, so reports show up at once in logs and piped output


def stage_name(processor) -> str:
    """'detect' for src.detect.detectors.DetectorSuite, 'predict' for src.predict.*, and so on.
    A processor can override this with a `stage` attribute."""
    explicit = getattr(processor, "stage", None)
    if explicit:
        return str(explicit)
    parts = type(processor).__module__.split(".")
    return parts[1] if len(parts) > 2 and parts[0] == "src" else type(processor).__name__


def stats(samples_s: list[float]) -> dict:
    a = np.asarray(samples_s) * 1000.0
    return {"n": int(len(a)), "mean_ms": round(float(a.mean()), 3), "p95_ms": round(float(np.percentile(a, 95)), 3),
            "max_ms": round(float(a.max()), 3)}


class LatencyTracker:
    def __init__(self, report_every_s: float = REPORT_EVERY_S, budget_ms: float = BUDGET_MS,
                 printer=print_now, label: str = "latency") -> None:
        self.report_every_s, self.budget_ms, self.printer, self.label = report_every_s, budget_ms, printer, label
        self.window: dict[str, list[float]] = defaultdict(list)
        self.all: dict[str, list[float]] = defaultdict(list)
        self.window_over = self.total_over = self.window_ticks = self.total_ticks = 0
        self.last_report = time.perf_counter()

    def record(self, stage_s: dict[str, float], total_s: float) -> None:
        for name, s in {**stage_s, "total": total_s}.items():
            self.window[name].append(s)
            if len(self.all[name]) < MAX_SAMPLES:
                self.all[name].append(s)
        over = total_s * 1000.0 > self.budget_ms
        self.window_over += over
        self.total_over += over
        self.window_ticks += 1
        self.total_ticks += 1

    def maybe_report(self, now: float | None = None) -> None:
        """Print the last window every report_every_s of wall time (only if ticks were processed)."""
        now = time.perf_counter() if now is None else now
        if now - self.last_report < self.report_every_s:
            return
        if self.window_ticks:
            self.printer(self.format(self.window, self.window_ticks, self.window_over, now - self.last_report))
        self.window = defaultdict(list)
        self.window_over = self.window_ticks = 0
        self.last_report = now

    def format(self, samples: dict[str, list[float]], ticks: int, over: int, span_s: float) -> str:
        lines = [f"[{self.label}] last {span_s:.0f} s: {ticks} ticks, over {self.budget_ms:.0f} ms: {over}",
                 f"  {'stage':<10}{'mean ms':>10}{'p95 ms':>10}{'worst ms':>10}"]
        for name in self.order(samples):
            s = stats(samples[name])
            lines.append(f"  {name:<10}{s['mean_ms']:>10.3f}{s['p95_ms']:>10.3f}{s['max_ms']:>10.3f}")
        return "\n".join(lines)

    @staticmethod
    def order(samples: dict[str, list[float]]) -> list[str]:
        first, last = ["tick", "frame"], ["send", "total"]
        middle = sorted(k for k in samples if k not in first + last)
        return [k for k in first if k in samples] + middle + [k for k in last if k in samples]

    def summary(self) -> dict:
        return {"ticks": self.total_ticks, "budget_ms": self.budget_ms, "over_budget": self.total_over,
                "stages": {name: stats(self.all[name]) for name in self.order(self.all)}}

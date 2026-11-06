"""In-memory operational metrics collector for ReversalLab service."""

from __future__ import annotations

import os
import threading
import time
from typing import Any

try:
    import psutil  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    psutil = None


class ServiceMetrics:
    """Thread-safe in-memory operational metrics collector."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "_lock", threading.Lock()):
            self._requests_total: dict[str, int] = {}
            self._request_duration_ms: dict[str, list[float]] = {}
            self._evaluations_total: dict[str, int] = {}
            self._diagnostic_gate_failures: dict[str, int] = {}
            self._active_requests = 0

    def record_request_start(self) -> None:
        with self._lock:
            self._active_requests += 1

    def record_request_end(self, endpoint: str, status_code: int, duration_ms: float) -> None:
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            key = f"{endpoint}:{status_code}"
            self._requests_total[key] = self._requests_total.get(key, 0) + 1
            if endpoint not in self._request_duration_ms:
                self._request_duration_ms[endpoint] = []
            # Keep rolling window of last 1000 measurements per endpoint
            times = self._request_duration_ms[endpoint]
            times.append(duration_ms)
            if len(times) > 1000:
                times.pop(0)

    def record_evaluation(self, estimator: str, outcome: str) -> None:
        with self._lock:
            key = f"{estimator}:{outcome}"
            self._evaluations_total[key] = self._evaluations_total.get(key, 0) + 1

    def record_gate_failure(self, reason: str) -> None:
        with self._lock:
            self._diagnostic_gate_failures[reason] = (
                self._diagnostic_gate_failures.get(reason, 0) + 1
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            # Memory measurement
            rss_mb: float | None = None
            if psutil is not None:
                try:
                    process = psutil.Process(os.getpid())
                    rss_mb = round(process.memory_info().rss / (1024 * 1024), 2)
                except Exception:  # pragma: no cover
                    rss_mb = None

            latencies: dict[str, dict[str, float]] = {}
            for ep, times in self._request_duration_ms.items():
                if times:
                    latencies[ep] = {
                        "count": len(times),
                        "mean_ms": round(sum(times) / len(times), 2),
                        "min_ms": round(min(times), 2),
                        "max_ms": round(max(times), 2),
                    }

            return {
                "active_requests": self._active_requests,
                "requests_total": dict(self._requests_total),
                "latencies_ms": latencies,
                "evaluations_total": dict(self._evaluations_total),
                "diagnostic_gate_failures": dict(self._diagnostic_gate_failures),
                "memory_rss_mb": rss_mb,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }


# Global singleton instance
metrics = ServiceMetrics()

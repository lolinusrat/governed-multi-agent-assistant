"""Tests for the development performance baseline.

The script is a measuring instrument, so the parts worth testing are the ones that
could report a wrong number: the percentile calculation and the success/failure
split. No server is started and no request is made.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from perf_baseline import Result, as_dict, percentile, summarise  # noqa: E402


class TestPercentile:
    def test_nearest_rank_without_interpolation(self):
        values = [10.0, 20.0, 30.0, 40.0]
        assert percentile(values, 50) == 20.0
        assert percentile(values, 100) == 40.0

    def test_p95_of_ten_samples_is_the_top_sample(self):
        # Stated plainly because the report leans on it: at n=10 the p95 is the
        # maximum, and must not be presented as a stable tail measurement.
        values = [float(i) for i in range(1, 11)]
        assert percentile(values, 95) == 10.0
        assert percentile(values, 95) == max(values)

    def test_order_of_input_does_not_matter(self):
        assert percentile([30.0, 10.0, 20.0], 50) == percentile([10.0, 20.0, 30.0], 50)

    def test_empty_input_is_zero_not_an_error(self):
        assert percentile([], 95) == 0.0

    def test_single_sample(self):
        assert percentile([7.5], 50) == percentile([7.5], 95) == 7.5


class TestSummary:
    RESULTS = [
        Result(ok=True, latency_ms=100.0, http_status=200, outcome="ANSWERED"),
        Result(ok=True, latency_ms=20.0, http_status=200, outcome="ABSTAINED"),
        Result(ok=True, latency_ms=40.0, http_status=200, outcome="ABSTAINED"),
        Result(ok=False, latency_ms=500.0, http_status=503, outcome="UNAVAILABLE", error="HTTP 503"),
        Result(ok=False, latency_ms=5.0, outcome="TRANSPORT_ERROR", error="ConnectError: refused"),
    ]

    def test_counts_successes_and_failures(self):
        summary = summarise(self.RESULTS)
        assert summary.total == 5
        assert summary.successes == 3
        assert summary.failures == 2
        assert summary.successes + summary.failures == summary.total

    def test_failures_are_grouped_by_cause(self):
        errors = summarise(self.RESULTS).errors
        assert errors["HTTP 503"] == 1
        assert errors["ConnectError: refused"] == 1

    def test_latency_is_broken_down_by_outcome(self):
        # A blended average is misleading here: an abstention makes no model calls
        # and an answered question makes three.
        report = as_dict(summarise(self.RESULTS), concurrency=5, wall_ms=600.0)
        assert report["by_outcome"]["ABSTAINED"]["count"] == 2
        assert report["by_outcome"]["ABSTAINED"]["average_ms"] == 30.0
        assert report["by_outcome"]["ANSWERED"]["average_ms"] == 100.0

    def test_report_reports_every_required_figure(self):
        report = as_dict(summarise(self.RESULTS), concurrency=5, wall_ms=600.0)
        assert report["successful_requests"] == 3
        assert report["failed_requests"] == 2
        for key in ("average", "p50", "p95"):
            assert key in report["latency_ms"]

    def test_the_report_marks_itself_as_a_development_baseline(self):
        report = as_dict(summarise(self.RESULTS), concurrency=5, wall_ms=600.0)
        assert report["development_baseline"] is True
        assert report["not_a_production_capacity_test"] is True

    def test_timed_out_requests_still_contribute_a_latency(self):
        summary = summarise([Result(ok=False, latency_ms=120_000.0, outcome="TRANSPORT_ERROR")])
        assert summary.latencies_ms == [120_000.0]
        assert summary.failures == 1

"""Development performance baseline for POST /ask.

**This is a development baseline, not proof of production capacity.** It measures a
handful of requests against a local, single-worker process, over a network path and a
model provider that neither the sample size nor the environment controls for. It sets
no SLA and implies none. Treat it as a smoke-level sanity check on latency and
concurrency behaviour, and nothing more.

Deliberately no load-testing framework: a dependency, a config format and a report
renderer would all be heavier than the question being answered here, which is only
"does the service stay upright under a few simultaneous callers, and roughly how long
does a request take".

Usage:

    uv run python scripts/perf_baseline.py
    uv run python scripts/perf_baseline.py --requests 10 --concurrency 10
    uv run python scripts/perf_baseline.py --url http://127.0.0.1:8000 --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import httpx

# A mix on purpose. An abstention makes no model calls and an answered question makes
# three, so a single blended average would hide the only thing that drives latency here.
QUESTIONS = [
    "A customer does not recognise a card transaction. What should I do?",
    "Can I immediately block the customer's account?",
    "Can I guarantee the customer will get their money back?",
    "How quickly must I acknowledge a customer complaint?",
    "What is the weather forecast in Sydney tomorrow?",
]

TRANSPORT_ERROR = "TRANSPORT_ERROR"


@dataclass
class Result:
    """One request's outcome."""

    ok: bool
    latency_ms: float
    http_status: int | None = None
    outcome: str = ""
    error: str = ""


@dataclass
class Summary:
    total: int
    successes: int
    failures: int
    latencies_ms: list[float] = field(default_factory=list)
    by_outcome: dict[str, list[float]] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=dict)


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile.

    No interpolation: with a sample this small, interpolating between two points
    would dress up a number that does not deserve the precision.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


async def ask_once(client: httpx.AsyncClient, url: str, question: str, timeout: float) -> Result:
    started = perf_counter()
    try:
        response = await client.post(
            f"{url}/ask", json={"question": question}, timeout=timeout
        )
    except Exception as exc:  # noqa: BLE001 - any transport failure is a failed request
        return Result(
            ok=False,
            latency_ms=(perf_counter() - started) * 1000,
            outcome=TRANSPORT_ERROR,
            error=f"{type(exc).__name__}: {exc}",
        )

    latency_ms = (perf_counter() - started) * 1000
    try:
        outcome = response.json().get("status", "")
    except ValueError:
        outcome = ""

    return Result(
        ok=response.status_code == 200,
        latency_ms=latency_ms,
        http_status=response.status_code,
        outcome=outcome or f"HTTP_{response.status_code}",
        error="" if response.status_code == 200 else f"HTTP {response.status_code}",
    )


async def run_baseline(url: str, requests: int, concurrency: int, timeout: float) -> list[Result]:
    """Fire `requests` requests, at most `concurrency` of them in flight at once."""
    gate = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:

        async def guarded(index: int) -> Result:
            async with gate:
                return await ask_once(client, url, QUESTIONS[index % len(QUESTIONS)], timeout)

        return await asyncio.gather(*(guarded(i) for i in range(requests)))


def summarise(results: list[Result]) -> Summary:
    summary = Summary(
        total=len(results),
        successes=sum(1 for r in results if r.ok),
        failures=sum(1 for r in results if not r.ok),
        latencies_ms=[r.latency_ms for r in results],
    )
    for result in results:
        summary.by_outcome.setdefault(result.outcome or "UNKNOWN", []).append(result.latency_ms)
        if not result.ok:
            summary.errors[result.error or "unknown"] = (
                summary.errors.get(result.error or "unknown", 0) + 1
            )
    return summary


def as_dict(summary: Summary, concurrency: int, wall_ms: float) -> dict[str, Any]:
    latencies = summary.latencies_ms
    return {
        "development_baseline": True,
        "not_a_production_capacity_test": True,
        "concurrency": concurrency,
        "requests": summary.total,
        "successful_requests": summary.successes,
        "failed_requests": summary.failures,
        "wall_clock_ms": round(wall_ms, 1),
        "latency_ms": {
            "average": round(statistics.fmean(latencies), 1) if latencies else 0.0,
            "p50": round(percentile(latencies, 50), 1),
            "p95": round(percentile(latencies, 95), 1),
            "min": round(min(latencies), 1) if latencies else 0.0,
            "max": round(max(latencies), 1) if latencies else 0.0,
        },
        "by_outcome": {
            outcome: {
                "count": len(values),
                "average_ms": round(statistics.fmean(values), 1),
            }
            for outcome, values in sorted(summary.by_outcome.items())
        },
        "errors": summary.errors,
    }


def render(report: dict[str, Any]) -> str:
    lat = report["latency_ms"]
    lines = [
        "Development performance baseline — NOT proof of production capacity.",
        "",
        f"  concurrency          {report['concurrency']}",
        f"  requests             {report['requests']}",
        f"  successful           {report['successful_requests']}",
        f"  failed               {report['failed_requests']}",
        f"  wall clock           {report['wall_clock_ms']:.0f} ms",
        "",
        f"  average latency      {lat['average']:.0f} ms",
        f"  p50 latency          {lat['p50']:.0f} ms",
        f"  p95 latency          {lat['p95']:.0f} ms",
        f"  min / max            {lat['min']:.0f} / {lat['max']:.0f} ms",
        "",
        "  by outcome:",
    ]
    for outcome, stats in report["by_outcome"].items():
        lines.append(f"    {outcome:22} n={stats['count']:<3} avg {stats['average_ms']:.0f} ms")
    if report["errors"]:
        lines.append("")
        lines.append("  errors:")
        for error, count in report["errors"].items():
            lines.append(f"    {error} ×{count}")
    lines += [
        "",
        "Read with care:",
        f"  - {report['requests']} samples is too few for p95 to mean what it usually means;",
        "    at this sample size it is close to the maximum observed. It is reported",
        "    because it was asked for, not because it is statistically meaningful.",
        "  - Latency here is dominated by the model provider, not by this application.",
        "    Abstained requests make no model calls at all; answered ones make three,",
        "    sequentially. Compare like with like using the by-outcome breakdown.",
        "  - Single local process, no warm-up, shared machine, uncontrolled network.",
        "  - Failures at concurrency are usually the provider rate limiting (HTTP 429",
        "    from Groq surfacing as a 503 here), not the application falling over. A",
        "    503 means the workflow stopped and fenced the request: no answer, human",
        "    review required. Check the `errors` and by-outcome lines before reading",
        "    a failure count as an application defect.",
        "  - This sets no service level objective and none should be inferred from it.",
    ]
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--requests", type=int, default=10, help="total requests to send")
    parser.add_argument("--concurrency", type=int, default=10, help="requests in flight at once")
    parser.add_argument("--timeout", type=float, default=120.0, help="per-request timeout, seconds")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args()

    started = perf_counter()
    results = await run_baseline(args.url, args.requests, args.concurrency, args.timeout)
    wall_ms = (perf_counter() - started) * 1000

    report = as_dict(summarise(results), args.concurrency, wall_ms)
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 0 if report["failed_requests"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

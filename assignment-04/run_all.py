"""
Run all 12 golden-dataset cases through the pipeline and print a summary.

For each case the full pipeline console output is shown as it runs, followed
by a one-line verdict. At the end a table lists every case with its result
and latency, plus totals.

Usage:
    python run_all.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from langfuse_client import langfuse
from test_agent import _ALL_CASES, _SkipCase, _run_case

BAR  = "=" * 62
THIN = "-" * 62


def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    print(BAR)
    print("  MEETING NOTES PIPELINE — FULL RUN")
    print(f"  Cases : {len(_ALL_CASES)}")
    print("  NOTE  : Every case makes real Anthropic API calls.")
    print(BAR)

    records: list[dict] = []
    total_start = time.monotonic()

    for case in _ALL_CASES:
        tid      = case["test_id"]
        category = case["category"]

        print(f"\n{THIN}")
        print(f"  {tid}  [{category.upper()}]")
        print(f"  {case['description'][:70]}")
        print(THIN)

        t0 = time.monotonic()
        try:
            checks = _run_case(case)
        except _SkipCase as exc:
            elapsed = time.monotonic() - t0
            print(f"\n  SKIP ({elapsed:.1f}s): {exc}")
            records.append({"tid": tid, "category": category,
                            "status": "SKIP", "elapsed": elapsed, "failures": []})
            continue
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - t0
            print(f"\n  ERROR ({elapsed:.1f}s): {exc}")
            records.append({"tid": tid, "category": category,
                            "status": "FAIL", "elapsed": elapsed, "failures": [str(exc)]})
            continue

        elapsed  = time.monotonic() - t0
        failures = [r for r in checks if not r["passed"]]
        status   = "PASS" if not failures else "FAIL"

        print(f"\n  RESULT: {status}  ({elapsed:.1f}s)")
        for f in failures:
            label = f"{f['check']}({f['field']})" if f["field"] else f["check"]
            print(f"  FAIL  {label}: {f['reason']}")

        records.append({"tid": tid, "category": category,
                        "status": status, "elapsed": elapsed,
                        "failures": [f["reason"] for f in failures]})

    # ----------------------------------------------------------------
    # Final summary table
    # ----------------------------------------------------------------
    total_elapsed = time.monotonic() - total_start
    n_pass  = sum(1 for r in records if r["status"] == "PASS")
    n_fail  = sum(1 for r in records if r["status"] == "FAIL")
    n_skip  = sum(1 for r in records if r["status"] == "SKIP")

    print(f"\n{BAR}")
    print("  FINAL SUMMARY")
    print(THIN)
    print(f"  {'TEST ID':<10} {'CATEGORY':<16} {'RESULT':<6}  TIME")
    print(f"  {'-'*8:<10} {'-'*14:<16} {'-'*6:<6}  {'-'*6}")
    for r in records:
        mark = "✓" if r["status"] == "PASS" else ("~" if r["status"] == "SKIP" else "✗")
        print(f"  {r['tid']:<10} {r['category']:<16} {r['status']:<6}  {r['elapsed']:>5.1f}s  {mark}")
    print(THIN)
    print(f"  Passed  : {n_pass}")
    print(f"  Failed  : {n_fail}")
    print(f"  Skipped : {n_skip}")
    print(f"  Total   : {len(records)}")
    print(f"  Time    : {total_elapsed:.0f}s")
    print(BAR)

    if langfuse:
        langfuse.flush()

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()

"""
Automated test runner for the meeting-notes pipeline.

Loads tests/golden_dataset.json and runs each test case against two suites:

  tool_call_validation   — happy_path, edge_case, adversarial cases
    Checks that each expected agent was called and returned valid JSON
    whose fields satisfy the evaluation constraints in the dataset.

  failure_mode_testing   — failure_mode cases
    Checks that invalid or out-of-scope inputs are rejected by the Vision
    Agent before reaching the Note Parser.

Usage:
    pytest tests/test_agent.py -v --tb=short      # recommended
    python tests/test_agent.py                     # standalone summary
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Path setup — allow imports from the project root regardless of cwd
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import interaction_log, run_action_planner, run_critic, run_note_parser
from langfuse_client import langfuse
from monitoring import PipelineLogger
from schema import STATUS_APPROVED, STATUS_ERROR
from vision_agent import run_vision_agent
from main import run_pipeline

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
MAX_REVISIONS = 3


def _load_dataset() -> dict:
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


_DATASET = _load_dataset()
_ALL_CASES: list[dict] = _DATASET["test_cases"]

TOOL_VALIDATION_CASES = [
    c for c in _ALL_CASES if c["category"] in ("happy_path", "edge_case", "adversarial")
]
FAILURE_MODE_CASES = [
    c for c in _ALL_CASES if c["category"] == "failure_mode"
]


# ---------------------------------------------------------------------------
# Skip helper — works in both pytest and standalone contexts
# ---------------------------------------------------------------------------
class _SkipCase(Exception):
    pass


def _skip(reason: str) -> None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        pytest.skip(reason)
    raise _SkipCase(reason)


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------

def _find_new_log(existing: set[Path]) -> dict | None:
    """Return parsed content of the newest log file not in `existing`."""
    log_dir = ROOT / "logs"
    if not log_dir.exists():
        return None
    current = set(log_dir.glob("run_*.json"))
    new = current - existing
    if not new:
        return None
    newest = max(new, key=lambda p: p.stat().st_mtime)
    try:
        with open(newest, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _existing_logs() -> set[Path]:
    log_dir = ROOT / "logs"
    return set(log_dir.glob("run_*.json")) if log_dir.exists() else set()


def _run_full_pipeline(case: dict) -> dict:
    """
    Runner for full_pipeline entry point (image-based cases).

    Calls run_vision_agent() once to capture VisionOutput for assertions,
    then calls run_pipeline() to exercise the full orchestration path and
    populate the interaction_log and structured log file.  Pipeline entries
    in interaction_log are isolated by snapshotting the log length after
    the pre-run vision call.
    """
    image_path = ROOT / case["input"]["path"]
    if not image_path.exists():
        _skip(f"Fixture not found: {image_path}. See tests/fixtures/README.md")

    # Pre-run: call vision agent separately to obtain VisionOutput for assertions.
    # This avoids parsing internal pipeline state from the log file.
    vision_output = run_vision_agent(str(image_path))

    # Snapshot log position and existing files so pipeline outputs are isolated
    pipeline_log_start = len(interaction_log)
    before_logs = _existing_logs()
    before_registers = set(ROOT.glob("action_register_*.json"))

    exception_raised = None
    try:
        run_pipeline(str(image_path))
    except Exception as exc:  # noqa: BLE001
        exception_raised = exc

    pipeline_entries = interaction_log[pipeline_log_start:]
    pipeline_log = _find_new_log(before_logs)

    # Read the action register saved by run_pipeline (absent on early halt)
    action_register = None
    new_registers = set(ROOT.glob("action_register_*.json")) - before_registers
    if new_registers:
        newest_reg = max(new_registers, key=lambda p: p.stat().st_mtime)
        try:
            with open(newest_reg, encoding="utf-8") as f:
                action_register = json.load(f)
        except Exception:
            pass

    return {
        "vision_output": {
            "rejected": vision_output.rejected,
            "document_type": vision_output.document_type,
            "confidence": vision_output.confidence,
            "extracted_text": vision_output.extracted_text,
            "rejection_reason": vision_output.rejection_reason,
        },
        "parser_message": None,
        "action_register": action_register,
        "critic_message": None,
        "exception": exception_raised,
        "interaction_log_entries": pipeline_entries,
        "pipeline_log": pipeline_log,
    }


def _run_note_parser_direct(case: dict) -> dict:
    """
    Runner for note_parser_direct entry point (text-based cases).

    Mirrors main.py's orchestration loop and Langfuse span structure:
    a root span wraps child spans for each agent call so text-based cases
    appear in Langfuse alongside image-based ones.
    """
    content = case["input"]["content"]

    # Snapshot BEFORE creating the logger — PipelineLogger.__init__ writes
    # the file immediately, so capturing before_logs after construction would
    # include the new file in the "existing" set and _find_new_log would miss it.
    log_start = len(interaction_log)
    before_logs = _existing_logs()

    logger = PipelineLogger()
    logger.pipeline_start("text_input")

    exception_raised = None
    parser_message = None
    planner_message = None
    critic_message = None

    ctx = (
        langfuse.start_as_current_observation(
            name="meeting-notes-pipeline",
            as_type="span",
            input={"text_preview": content[:200]},
            end_on_exit=True,
        )
        if langfuse
        else contextlib.nullcontext(None)
    )

    with ctx as trace:
        try:
            if trace:
                parser_span = trace.start_observation(
                    name="note_parser", as_type="span", input={"content": content}
                )
            parser_message = run_note_parser(content, logger=logger)
            if trace:
                parser_span.update(output=parser_message["payload"])
                parser_span.end()

            if parser_message["status"] != STATUS_ERROR:
                revision_round = 0
                revision_instructions = ""

                while True:
                    if trace:
                        planner_span = trace.start_observation(
                            name="action_planner",
                            as_type="span",
                            input=parser_message["payload"],
                            metadata={"round": revision_round},
                        )
                    planner_message = run_action_planner(
                        parser_message,
                        revision_instructions=revision_instructions,
                        revision_round=revision_round,
                        logger=logger,
                    )
                    if trace:
                        planner_span.update(output=planner_message["payload"])
                        planner_span.end()

                    if planner_message["status"] == STATUS_ERROR:
                        break

                    if trace:
                        critic_span = trace.start_observation(
                            name="critic",
                            as_type="span",
                            input=planner_message["payload"],
                            metadata={"round": revision_round},
                        )
                    critic_message = run_critic(planner_message, parser_message, logger=logger)
                    if trace:
                        critic_approved = critic_message["status"] == STATUS_APPROVED
                        critic_span.update(
                            output={
                                "score": critic_message["payload"].get("overall_score"),
                                "issues": critic_message["payload"].get("issues", []),
                                "approved": critic_message["payload"].get("approved", False),
                            },
                            metadata={"round": revision_round, "approved": critic_approved},
                        )
                        critic_span.end()

                    if critic_message["status"] in (STATUS_ERROR, STATUS_APPROVED):
                        break
                    if revision_round >= MAX_REVISIONS - 1:
                        break

                    logger.critic_revision_required(
                        revision_round=revision_round,
                        score=critic_message["payload"].get("overall_score", 0),
                        issues=critic_message["payload"].get("issues", []),
                    )
                    revision_round += 1
                    revision_instructions = critic_message["payload"].get("revision_instructions", "")

            # Close the structured log with an appropriate outcome event
            planner_errored = planner_message and planner_message["status"] == STATUS_ERROR
            critic_errored = critic_message and critic_message["status"] == STATUS_ERROR
            if planner_errored or critic_errored:
                logger.pipeline_error(error="agent_error", stage="pipeline")
                if trace:
                    trace.update(output={"error": "agent_error"})
            else:
                approved = critic_message and critic_message["status"] == STATUS_APPROVED
                outcome = "approved" if approved else "max_revisions_reached"
                logger.pipeline_complete(outcome=outcome)
                if trace:
                    trace.update(
                        output=planner_message["payload"] if planner_message else {},
                        metadata={"outcome": outcome},
                    )

        except Exception as exc:  # noqa: BLE001
            exception_raised = exc
            logger.pipeline_error(error=str(exc), stage="unknown")
            if trace:
                trace.update(output={"error": str(exc)})

    pipeline_entries = interaction_log[log_start:]
    pipeline_log = _find_new_log(before_logs)

    return {
        "vision_output": None,
        "parser_message": parser_message,
        "action_register": planner_message["payload"] if planner_message else None,
        "critic_message": critic_message,
        "exception": exception_raised,
        "interaction_log_entries": pipeline_entries,
        "pipeline_log": pipeline_log,
    }


# ---------------------------------------------------------------------------
# Field resolution
# ---------------------------------------------------------------------------

def _get_field(context: dict, field_path: str) -> Any:
    """Resolve a dotted field path through the context dict."""
    if not field_path:
        return context
    value: Any = context
    for part in field_path.split("."):
        if value is None:
            return None
        value = value.get(part) if isinstance(value, dict) else getattr(value, part, None)
    return value


def _get_item_field(item: Any, field_name: str) -> Any:
    """Extract a named field from a list item (dict or object)."""
    if item is None:
        return None
    return item.get(field_name) if isinstance(item, dict) else getattr(item, field_name, None)


# ---------------------------------------------------------------------------
# Check engine
# ---------------------------------------------------------------------------

def _apply_sub_check(sub: dict, item_val: Any) -> tuple[bool, str]:
    """Evaluate a sub-check against a single field value from a list item."""
    sc = sub["check"]
    field = sub.get("field", "field")
    expected = sub.get("value", "")

    if sc == "field_non_empty":
        ok = bool(item_val and str(item_val).strip())
        return ok, f"'{field}' is empty or None" if not ok else ""

    if sc == "field_equals":
        ok = item_val == expected
        return ok, f"'{field}' = {item_val!r}, expected {expected!r}" if not ok else ""

    if sc == "string_contains":
        ok = str(expected).lower() in str(item_val or "").lower()
        return ok, f"'{field}' does not contain {expected!r}" if not ok else ""

    if sc == "string_not_contains":
        ok = str(expected).lower() not in str(item_val or "").lower()
        return ok, f"'{field}' contains forbidden substring {expected!r}" if not ok else ""

    return False, f"Unknown sub-check type: {sc!r}"


def _run_check(check: dict, context: dict) -> tuple[bool, str]:
    """Execute one evaluation check. Returns (passed, failure_reason)."""
    ct = check["check"]
    field = check.get("field", "")
    field_val = _get_field(context, field) if field else None
    expected = check.get("value")

    # ---- simple field checks -----------------------------------------------

    if ct == "no_exception":
        exc = context.get("exception")
        return exc is None, f"Exception raised: {exc}" if exc else ""

    if ct == "field_equals":
        ok = field_val == expected
        return ok, f"'{field}' = {field_val!r}, expected {expected!r}" if not ok else ""

    if ct == "field_in_set":
        ok = field_val in expected
        return ok, f"'{field}' = {field_val!r}, expected one of {expected}" if not ok else ""

    if ct == "field_non_empty":
        ok = bool(field_val and str(field_val).strip())
        return ok, f"'{field}' is empty or None" if not ok else ""

    if ct == "field_gte":
        if field_val is None:
            return False, f"'{field}' is None (expected >= {expected})"
        ok = field_val >= expected
        return ok, f"'{field}' = {field_val}, expected >= {expected}" if not ok else ""

    if ct == "field_lte":
        if field_val is None:
            return False, f"'{field}' is None (expected <= {expected})"
        ok = field_val <= expected
        return ok, f"'{field}' = {field_val}, expected <= {expected}" if not ok else ""

    # ---- list checks --------------------------------------------------------

    if ct == "list_non_empty":
        ok = bool(field_val)
        return ok, f"'{field}' is empty or None" if not ok else ""

    if ct == "list_min_length":
        if field_val is None:
            return False, f"'{field}' is None"
        ok = len(field_val) >= expected
        return ok, f"'{field}' has {len(field_val)} item(s), expected >= {expected}" if not ok else ""

    if ct == "list_all_match":
        if not field_val:
            return False, f"'{field}' is empty or None"
        sub = check["sub_check"]
        failures = []
        for i, item in enumerate(field_val):
            item_val = _get_item_field(item, sub.get("field", ""))
            ok, reason = _apply_sub_check(sub, item_val)
            if not ok:
                failures.append(f"item[{i}]: {reason}")
        ok = not failures
        return ok, "; ".join(failures[:3]) if not ok else ""

    if ct == "list_no_item_matches":
        # Passes when NO item in the list satisfies the sub_check.
        # Use sub_check: string_contains to assert no item contains a string.
        if field_val is None:
            return True, ""
        sub = check["sub_check"]
        violations = []
        for i, item in enumerate(field_val):
            item_val = _get_item_field(item, sub.get("field", ""))
            ok, _ = _apply_sub_check(sub, item_val)
            if ok:
                violations.append(f"item[{i}].{sub.get('field', '')!r} = {item_val!r}")
        ok = not violations
        return ok, f"Unexpected match(es): {'; '.join(violations[:3])}" if not ok else ""

    if ct == "set_min_distinct":
        if not field_val:
            return False, f"'{field}' is empty or None"
        sub_field = check["sub_field"]
        distinct = {
            _get_item_field(item, sub_field)
            for item in field_val
            if _get_item_field(item, sub_field) is not None
        }
        ok = len(distinct) >= expected
        return (
            ok,
            f"'{sub_field}' has only {len(distinct)} distinct value(s): {distinct}; expected >= {expected}"
            if not ok
            else "",
        )

    # ---- agent / log checks ------------------------------------------------

    if ct == "agent_ran":
        entries = context.get("interaction_log_entries", [])
        ran = any(e.get("agent") == expected for e in entries)
        return ran, f"Agent '{expected}' not found in interaction_log" if not ran else ""

    if ct == "agent_not_ran":
        entries = context.get("interaction_log_entries", [])
        ran = any(e.get("agent") == expected for e in entries)
        return not ran, f"Agent '{expected}' unexpectedly ran" if ran else ""

    if ct == "pipeline_log_event":
        log = context.get("pipeline_log")
        if log is None:
            return False, "No pipeline log file found"
        found = any(e.get("event_type") == expected for e in log.get("events", []))
        return found, f"Event '{expected}' not found in pipeline log" if not found else ""

    if ct == "string_not_contains":
        if field_val is None:
            return True, ""
        ok = str(expected).lower() not in str(field_val).lower()
        return ok, f"'{field}' contains forbidden substring {expected!r}" if not ok else ""

    # ---- compound checks ---------------------------------------------------

    if ct == "any_true":
        results = [_run_check(c, context) for c in check["conditions"]]
        ok = any(r[0] for r in results)
        reasons = [r[1] for r in results if not r[0]]
        return ok, f"All conditions failed: {'; '.join(reasons)}" if not ok else ""

    return False, f"Unknown check type: {ct!r}"


# ---------------------------------------------------------------------------
# Case runner
# ---------------------------------------------------------------------------

def _run_case(case: dict) -> list[dict]:
    """
    Execute a test case end-to-end and return a list of check results.

    Each result is a dict: {check, field, passed, reason}.
    Raises _SkipCase when a prerequisite (e.g. fixture image) is missing.
    """
    entry = case["entry_point"]

    if entry == "full_pipeline":
        context = _run_full_pipeline(case)
    elif entry == "note_parser_direct":
        context = _run_note_parser_direct(case)
    else:
        raise ValueError(f"Unknown entry_point: {entry!r}")

    results = []
    for chk in case.get("evaluation_method", []):
        passed, reason = _run_check(chk, context)
        results.append(
            {
                "check": chk["check"],
                "field": chk.get("field", ""),
                "passed": passed,
                "reason": reason,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Pytest test functions
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def flush_langfuse():
    """Flush Langfuse at session end so all spans are sent before process exits."""
    yield
    if langfuse:
        langfuse.flush()


@pytest.fixture(autouse=True)
def require_api_key():
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")


@pytest.mark.parametrize("case", TOOL_VALIDATION_CASES, ids=[c["test_id"] for c in TOOL_VALIDATION_CASES])
def test_tool_call_validation(case):
    """
    Tool call validation — happy_path, edge_case, adversarial.

    Verifies that each expected agent was called and produced output
    conforming to its schema and the evaluation constraints in the dataset.
    """
    results = _run_case(case)
    failures = [r for r in results if not r["passed"]]
    if failures:
        lines = [f"[{case['test_id']}] {case['description'][:80]}"]
        for f in failures:
            label = f"{f['check']}({f['field']})" if f["field"] else f["check"]
            lines.append(f"  FAIL  {label}: {f['reason']}")
        pytest.fail("\n" + "\n".join(lines))


@pytest.mark.parametrize("case", FAILURE_MODE_CASES, ids=[c["test_id"] for c in FAILURE_MODE_CASES])
def test_failure_mode(case):
    """
    Failure mode testing — failure_mode category.

    Verifies that invalid or out-of-scope inputs are rejected by the Vision
    Agent before reaching the Note Parser.
    """
    results = _run_case(case)
    failures = [r for r in results if not r["passed"]]
    if failures:
        lines = [f"[{case['test_id']}] {case['description'][:80]}"]
        for f in failures:
            label = f"{f['check']}({f['field']})" if f["field"] else f["check"]
            lines.append(f"  FAIL  {label}: {f['reason']}")
        pytest.fail("\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# Standalone runner (python tests/test_agent.py)
# ---------------------------------------------------------------------------

def _standalone_run() -> None:
    bar = "-" * 62

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Exiting.")
        sys.exit(1)

    print("=" * 62)
    print("  PIPELINE TEST RUNNER")
    print(f"  Dataset : {DATASET_PATH.relative_to(ROOT)}")
    print(f"  Cases   : {len(_ALL_CASES)}")
    print("  NOTE    : Tests make real Anthropic API calls.")
    print("=" * 62)

    total = len(_ALL_CASES)
    n_passed = 0
    n_failed = 0
    n_skipped = 0
    failure_log: list[tuple[str, list[str]]] = []

    for case in _ALL_CASES:
        tid = case["test_id"]
        category = case["category"]

        print(f"\n{bar}")
        print(f"  {tid}  [{category}]")
        print(f"  {case['description'][:70]}")
        print(bar)

        t0 = time.monotonic()
        try:
            results = _run_case(case)
        except _SkipCase as exc:
            elapsed = round(time.monotonic() - t0, 1)
            print(f"  SKIP  ({elapsed}s): {exc}")
            n_skipped += 1
            continue
        except Exception as exc:  # noqa: BLE001
            elapsed = round(time.monotonic() - t0, 1)
            print(f"  ERROR  ({elapsed}s): {exc}")
            traceback.print_exc()
            n_failed += 1
            failure_log.append((tid, [f"Unexpected exception: {exc}"]))
            continue

        elapsed = round(time.monotonic() - t0, 1)
        case_failures = [r for r in results if not r["passed"]]

        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            label = f"{r['check']}({r['field']})" if r["field"] else r["check"]
            detail = f"  — {r['reason']}" if not r["passed"] else ""
            print(f"  [{status}]  {label}{detail}")

        if case_failures:
            n_failed += 1
            failure_log.append((tid, [r["reason"] for r in case_failures]))
            print(f"\n  VERDICT: FAIL  ({elapsed}s)")
        else:
            n_passed += 1
            print(f"\n  VERDICT: PASS  ({elapsed}s)")

    # Summary
    print(f"\n{'=' * 62}")
    print("  SUMMARY")
    print("=" * 62)
    print(f"  Total   : {total}")
    print(f"  Passed  : {n_passed}")
    print(f"  Failed  : {n_failed}")
    print(f"  Skipped : {n_skipped}")

    if failure_log:
        print(f"\n  FAILURES")
        print(f"  {'-' * 58}")
        for tid, reasons in failure_log:
            print(f"  {tid}:")
            for reason in reasons:
                print(f"    • {reason}")

    print("=" * 62)

    if langfuse:
        langfuse.flush()

    sys.exit(0 if n_failed == 0 else 1)


if __name__ == "__main__":
    _standalone_run()

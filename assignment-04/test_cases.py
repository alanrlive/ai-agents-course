"""
Integration tests for the Vision Agent pipeline.

Each test makes real API calls. Place test images in test_images/ before running.
Run: python test_cases.py
"""

import sys

from agents import interaction_log
from main import run_pipeline
from vision_agent import run_vision_agent

BAR = "-" * 62
VALID_CONFIDENCES = {"high", "medium", "low"}


def _check(label: str, expected: str, actual: str, passed: bool) -> None:
    verdict = "PASS" if passed else "FAIL"
    print(f"  [{verdict}] {label}")
    print(f"         Expected : {expected}")
    print(f"         Actual   : {actual}")


def test_happy_path() -> bool:
    """
    Input   : test_images/whiteboard_sample.jpg
    Expected: document_type=whiteboard, rejected=False,
              extracted_text non-empty, full pipeline completes
    """
    image_path = "test_images/whiteboard_sample.jpg"

    print(f"\n{BAR}")
    print("TEST: test_happy_path")
    print(BAR)
    print(f"  Input    : {image_path}")
    print(
        "  Expected : document_type in [whiteboard, digital], rejected=False, "
        "extracted_text non-empty, pipeline completes"
    )
    print()

    vision_output = run_vision_agent(image_path)
    results = []

    passed = vision_output.document_type in ("whiteboard", "digital")
    _check("document_type", "whiteboard or digital", vision_output.document_type, passed)
    results.append(passed)

    passed = vision_output.rejected is False
    _check("rejected", "False", str(vision_output.rejected), passed)
    results.append(passed)

    has_text = bool(vision_output.extracted_text.strip())
    passed = has_text
    _check(
        "extracted_text",
        "non-empty string",
        f"{'non-empty' if has_text else 'empty'} ({len(vision_output.extracted_text)} chars)",
        passed,
    )
    results.append(passed)

    print()
    pipeline_passed = False
    if not vision_output.rejected:
        print("  Running full pipeline...")
        try:
            run_pipeline(image_path)
            pipeline_passed = True
        except Exception as exc:
            print(f"  Pipeline raised exception: {exc}")
    else:
        print("  Skipping full pipeline — image was rejected by Vision Agent.")

    _check(
        "full pipeline completes",
        "no exception raised",
        "completed" if pipeline_passed else "failed or skipped",
        pipeline_passed,
    )
    results.append(pipeline_passed)

    overall = all(results)
    print(f"\n  VERDICT: {'PASS' if overall else 'FAIL'}")
    return overall


def test_alternate_path() -> bool:
    """
    Input   : test_images/handwritten_sample.jpg
    Expected: document_type=handwritten, confidence is a valid value,
              pipeline completes with action register output
    """
    image_path = "test_images/handwritten_sample.jpg"

    print(f"\n{BAR}")
    print("TEST: test_alternate_path")
    print(BAR)
    print(f"  Input    : {image_path}")
    print(
        "  Expected : document_type in [handwritten, digital], confidence logged (high/medium/low), "
        "pipeline completes"
    )
    print()

    vision_output = run_vision_agent(image_path)
    results = []

    passed = vision_output.document_type in ("handwritten", "digital")
    _check("document_type", "handwritten or digital", vision_output.document_type, passed)
    results.append(passed)

    passed = vision_output.confidence in VALID_CONFIDENCES
    _check(
        "confidence logged",
        f"one of {sorted(VALID_CONFIDENCES)}",
        vision_output.confidence,
        passed,
    )
    results.append(passed)

    print()
    pipeline_passed = False
    if not vision_output.rejected:
        print("  Running full pipeline...")
        try:
            run_pipeline(image_path)
            pipeline_passed = True
        except Exception as exc:
            print(f"  Pipeline raised exception: {exc}")
    else:
        print("  Skipping full pipeline — image was rejected by Vision Agent.")

    _check(
        "pipeline completes with action register",
        "no exception raised",
        "completed" if pipeline_passed else "failed or skipped",
        pipeline_passed,
    )
    results.append(pipeline_passed)

    overall = all(results)
    print(f"\n  VERDICT: {'PASS' if overall else 'FAIL'}")
    return overall


def test_edge_case() -> bool:
    """
    Input   : test_images/blurry_sample.jpg
    Expected: rejected=True, rejection_reason non-empty,
              pipeline stops before Note Parser runs
    """
    image_path = "test_images/blurry_sample.jpg"

    print(f"\n{BAR}")
    print("TEST: test_edge_case")
    print(BAR)
    print(f"  Input    : {image_path}")
    print(
        "  Expected : rejected=True, rejection_reason non-empty, "
        "pipeline stops before Note Parser"
    )
    print()

    vision_output = run_vision_agent(image_path)
    results = []

    passed = vision_output.rejected is True
    _check("rejected", "True", str(vision_output.rejected), passed)
    results.append(passed)

    has_reason = bool(vision_output.rejection_reason.strip())
    passed = has_reason
    _check(
        "rejection_reason",
        "non-empty string",
        (
            f"non-empty ({len(vision_output.rejection_reason)} chars)"
            if has_reason
            else "empty"
        ),
        passed,
    )
    results.append(passed)

    # Verify the pipeline halts before Note Parser by checking no new
    # note_parser entry appears in the interaction log after run_pipeline().
    print()
    log_len_before = len(interaction_log)
    try:
        run_pipeline(image_path)
    except Exception as exc:
        print(f"  Pipeline raised unexpected exception: {exc}")

    new_entries = interaction_log[log_len_before:]
    note_parser_called = any(e["agent"] == "note_parser" for e in new_entries)
    stopped_early = not note_parser_called

    _check(
        "pipeline stops before Note Parser",
        "note_parser not in interaction log",
        "stopped early — note_parser not called" if stopped_early else "note_parser was called",
        stopped_early,
    )
    results.append(stopped_early)

    overall = all(results)
    print(f"\n  VERDICT: {'PASS' if overall else 'FAIL'}")
    return overall


if __name__ == "__main__":
    print("=" * 62)
    print("  VISION PIPELINE — INTEGRATION TESTS")
    print("  Assignment 03B")
    print("=" * 62)

    results = {
        "test_happy_path": test_happy_path(),
        "test_alternate_path": test_alternate_path(),
        "test_edge_case": test_edge_case(),
    }

    passed_count = sum(1 for v in results.values() if v)

    print(f"\n{'=' * 62}")
    print("  TEST SUMMARY")
    print("=" * 62)
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"\n  {passed_count}/{len(results)} tests passed")
    print("=" * 62)

    sys.exit(0 if all(results.values()) else 1)

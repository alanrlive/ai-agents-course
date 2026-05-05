import argparse
import json
import os
import sys

from agents import interaction_log, run_action_planner, run_critic, run_note_parser, save_log
from schema import STATUS_APPROVED, STATUS_ERROR
from vision_agent import run_vision_agent

MAX_REVISIONS = 3

BAR = "-" * 62


def print_handoff(message: dict) -> None:
    """Print the inter-agent envelope so the handoff is visible in the console."""
    print(
        f"  >> {message['sender']} -> {message['recipient']} "
        f"| status: {message['status']} "
        f"| round: {message['revision_round']}"
    )


def print_action_register(payload: dict) -> None:
    actions = payload.get("actions", [])
    open_questions = payload.get("open_questions", [])

    print("\nACTION REGISTER")
    print("=" * 62)

    if not actions:
        print("  (no actions produced)")
    else:
        for action in actions:
            action_id = action.get("action_id", "?")
            priority = action.get("priority", "?").upper()
            description = action.get("description", "?")
            owner = action.get("owner", "?")
            deadline = action.get("deadline", "?")
            basis_owner = action.get("basis_for_owner", "")
            basis_deadline = action.get("basis_for_deadline", "")
            depends = action.get("depends_on", [])

            print(f"\n  {action_id}  [{priority}]  {description}")
            print(f"    Owner    : {owner}")
            print(f"    Deadline : {deadline}")
            if basis_owner:
                print(f"    Owner basis    : {basis_owner}")
            if basis_deadline:
                print(f"    Deadline basis : {basis_deadline}")
            if depends:
                print(f"    Depends on : {', '.join(depends)}")

    if open_questions:
        print("\nOPEN QUESTIONS")
        print("=" * 62)
        for q in open_questions:
            print(f"  • {q}")


def run_pipeline(image_path: str) -> None:
    # ----------------------------------------------------------------
    # Step 0 — Vision Agent
    # ----------------------------------------------------------------
    print(f"\n{BAR}")
    print("STEP 0  |  Vision Agent")
    print(BAR)

    vision_output = run_vision_agent(image_path)

    if vision_output.rejected:
        print(f"\n[Orchestrator] Pipeline halted — Vision Agent rejected the image.")
        print(f"  Reason: {vision_output.rejection_reason}")
        return

    print(
        f"  >> vision_agent -> note_parser "
        f"| document_type: {vision_output.document_type} "
        f"| confidence: {vision_output.confidence}"
    )
    if vision_output.processing_notes:
        print(f"  Notes: {vision_output.processing_notes}")

    # ----------------------------------------------------------------
    # Step 1 — Note Parser
    # ----------------------------------------------------------------
    print(f"\n{BAR}")
    print("STEP 1  |  Note Parser")
    print(BAR)

    parser_message = run_note_parser(vision_output.extracted_text)
    print_handoff(parser_message)

    if parser_message["status"] == STATUS_ERROR:
        print("\n[Orchestrator] Pipeline halted — Note Parser returned an error.")
        print(f"  Detail: {parser_message['payload'].get('parse_error', 'unknown')}")
        return

    # ----------------------------------------------------------------
    # Step 2 — Action Planner + Critic revision loop
    # ----------------------------------------------------------------
    revision_round = 0
    revision_instructions = ""
    planner_message = None
    critic_message = None

    while True:
        print(f"\n{BAR}")
        print(f"STEP 2  |  Action Planner  (round {revision_round})")
        print(BAR)

        planner_message = run_action_planner(
            parser_message,
            revision_instructions=revision_instructions,
            revision_round=revision_round,
        )
        print_handoff(planner_message)

        if planner_message["status"] == STATUS_ERROR:
            print("\n[Orchestrator] Pipeline halted — Action Planner returned an error.")
            print(f"  Detail: {planner_message['payload'].get('parse_error', 'unknown')}")
            break

        print(f"\n{BAR}")
        print(f"STEP 2  |  Critic Review  (round {revision_round})")
        print(BAR)

        critic_message = run_critic(planner_message, parser_message)
        print_handoff(critic_message)

        if critic_message["status"] == STATUS_ERROR:
            print("\n[Orchestrator] Pipeline halted — Critic returned an error.")
            print(f"  Detail: {critic_message['payload'].get('parse_error', 'unknown')}")
            break

        if critic_message["status"] == STATUS_APPROVED:
            print(
                f"\n[Orchestrator] Register approved by Critic "
                f"after {revision_round + 1} round(s)."
            )
            break

        if revision_round >= MAX_REVISIONS - 1:
            print(
                f"\n[Orchestrator] Max revisions ({MAX_REVISIONS}) reached. "
                "Proceeding with best available register."
            )
            break

        revision_round += 1
        revision_instructions = critic_message["payload"].get("revision_instructions", "")
        print(
            f"\n[Orchestrator] Revision requested — "
            f"passing instructions to Action Planner for round {revision_round}."
        )

    if planner_message is None:
        return

    # ----------------------------------------------------------------
    # Step 3 — Final output
    # ----------------------------------------------------------------
    print(f"\n{BAR}")
    print("STEP 3  |  Final Output")
    print(BAR)

    print_action_register(planner_message["payload"])

    total_tokens = sum(entry["tokens_used"] for entry in interaction_log)
    total_calls = len(interaction_log)
    final_score = (
        critic_message["payload"].get("overall_score", "?") if critic_message else "?"
    )
    approved = critic_message["status"] == STATUS_APPROVED if critic_message else False

    print(f"\n{BAR}")
    print("EXECUTION SUMMARY")
    print(BAR)
    print(f"  LLM calls      : {total_calls}")
    print(f"  Total tokens   : {total_tokens}")
    print(f"  Revision rounds: {revision_round}")
    print(f"  Final score    : {final_score}/10")
    print(f"  Approved       : {'Yes' if approved else 'No — max revisions reached'}")

    # ----------------------------------------------------------------
    # Step 4 — Save outputs
    # ----------------------------------------------------------------
    print(f"\n{BAR}")
    print("STEP 4  |  Saving Outputs")
    print(BAR)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    register_path = f"action_register_{timestamp}.json"
    with open(register_path, "w", encoding="utf-8") as f:
        json.dump(planner_message["payload"], f, indent=2)
    print(f"  Saved: {register_path}")

    log_path = save_log()
    print(f"  Saved: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meeting Notes Processor — Vision Pipeline")
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the image file containing meeting notes (jpg, jpeg, png)",
    )
    args = parser.parse_args()
    image_path = args.image

    print("=" * 62)
    print("  MEETING NOTES PROCESSOR")
    print("  Multi-Agent Pipeline  |  Assignment 03B")
    print(f"  Input : {image_path}")
    print("=" * 62)

    if not os.path.exists(image_path):
        print(f"\nERROR: {image_path} not found.")
        sys.exit(1)

    run_pipeline(image_path)

    print(f"\n{'=' * 62}")
    print("  Pipeline complete.")
    print("  Output files saved with datetime stamp.")
    print("=" * 62)

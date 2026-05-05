import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import json_repair

# Ensure stdout can handle any Unicode character on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv

from prompts import ACTION_PLANNER_PROMPT, CRITIC_PROMPT, NOTE_PARSER_PROMPT
from schema import (
    ACTION_PLANNER,
    CRITIC,
    NOTE_PARSER,
    ORCHESTRATOR,
    STATUS_APPROVED,
    STATUS_ERROR,
    STATUS_REVISION_REQUIRED,
    STATUS_SUCCESS,
    create_message,
    validate_message,
)

load_dotenv()

MODEL = "claude-haiku-4-5"
MAX_RETRIES = 3
RETRY_DELAY = 2

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

interaction_log: list[dict] = []


def log_interaction(
    agent_name: str,
    input_summary: str,
    output_preview: str,
    tokens_used: int,
    revision_round: int,
) -> None:
    interaction_log.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent_name,
        "input_summary": input_summary[:200],
        "output_preview": output_preview[:300],
        "tokens_used": tokens_used,
        "revision_round": revision_round,
    })


def _clean_json(text: str) -> str:
    text = text.strip()
    # Full fence: ```json ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    else:
        # Opening fence only (response truncated before closing fence)
        text = re.sub(r"^```(?:json)?\s*", "", text)

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    try:
        repaired = json_repair.repair_json(text)
        json.loads(repaired)
        return repaired
    except (json.JSONDecodeError, Exception):
        pass

    return text


def save_log() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"agent_interaction_log_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(interaction_log, f, indent=2)
    return path


def call_llm(
    system_prompt: str,
    user_message: str,
    agent_name: str,
    revision_round: int = 0,
) -> tuple[str, int]:
    last_error = None
    response = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            break
        except anthropic.RateLimitError as exc:
            last_error = exc
            print(
                f"[{agent_name}] Rate limit hit (attempt {attempt}/{MAX_RETRIES}), "
                f"retrying in {RETRY_DELAY}s..."
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
        except anthropic.APIError as exc:
            last_error = exc
            print(f"[{agent_name}] API error (attempt {attempt}/{MAX_RETRIES}): {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    if response is None:
        raise RuntimeError(
            f"[{agent_name}] All {MAX_RETRIES} API attempts failed: {last_error}"
        ) from last_error

    if response.stop_reason == "tool_use":
        raise ValueError(
            f"[{agent_name}] Unexpected tool_use stop_reason — no tools declared for this agent"
        )

    text = response.content[0].text
    tokens = response.usage.input_tokens + response.usage.output_tokens
    log_interaction(
        agent_name=agent_name,
        input_summary=user_message[:200],
        output_preview=text[:300],
        tokens_used=tokens,
        revision_round=revision_round,
    )
    return text, tokens


def run_note_parser(raw_notes: str) -> dict:
    user_message = (
        "Here are the raw meeting notes to parse:\n\n"
        f"<notes>\n{raw_notes}\n</notes>\n\n"
        "Return the extracted data as JSON matching the schema in your instructions."
    )

    text, _ = call_llm(NOTE_PARSER_PROMPT, user_message, NOTE_PARSER)

    try:
        payload = json.loads(_clean_json(text))
        status = STATUS_SUCCESS
    except json.JSONDecodeError as exc:
        payload = {"raw_response": text, "parse_error": str(exc)}
        status = STATUS_ERROR

    if status == STATUS_SUCCESS:
        decisions = len(payload.get("decisions_made", []))
        blockers = len(payload.get("blockers", []))
        unresolved = len(payload.get("unresolved_items", []))
        print(
            f"[Note Parser] Extraction complete — "
            f"{decisions} decision(s), {blockers} blocker(s), {unresolved} unresolved item(s)"
        )
    else:
        print("[Note Parser] ERROR — failed to parse JSON response")

    message = create_message(
        sender=NOTE_PARSER,
        recipient=ACTION_PLANNER,
        task="parsed_meeting_notes",
        payload=payload,
        status=status,
    )
    validate_message(message)
    return message


def run_action_planner(
    parser_message: dict,
    revision_instructions: str = "",
    revision_round: int = 0,
) -> dict:
    extracted = json.dumps(parser_message["payload"], indent=2)

    if revision_instructions:
        user_message = (
            "Here is the structured data extracted from the meeting notes:\n\n"
            f"<extracted_data>\n{extracted}\n</extracted_data>\n\n"
            "Your previous action register was reviewed and requires revision. "
            "Here are the specific instructions:\n\n"
            f"<revision_instructions>\n{revision_instructions}\n</revision_instructions>\n\n"
            "Produce a revised action register as JSON matching the schema in your instructions."
        )
    else:
        user_message = (
            "Here is the structured data extracted from the meeting notes:\n\n"
            f"<extracted_data>\n{extracted}\n</extracted_data>\n\n"
            "Produce the action register as JSON matching the schema in your instructions."
        )

    text, _ = call_llm(ACTION_PLANNER_PROMPT, user_message, ACTION_PLANNER, revision_round)

    try:
        payload = json.loads(_clean_json(text))
        status = STATUS_SUCCESS
    except json.JSONDecodeError as exc:
        payload = {"raw_response": text, "parse_error": str(exc)}
        status = STATUS_ERROR

    if status == STATUS_SUCCESS:
        actions = len(payload.get("actions", []))
        open_q = len(payload.get("open_questions", []))
        print(
            f"[Action Planner] Register produced — "
            f"{actions} action(s), {open_q} open question(s) (round {revision_round})"
        )
    else:
        print(f"[Action Planner] ERROR — failed to parse JSON response (round {revision_round})")

    message = create_message(
        sender=ACTION_PLANNER,
        recipient=CRITIC,
        task="action_register",
        payload=payload,
        status=status,
        revision_round=revision_round,
    )
    validate_message(message)
    return message


def run_critic(planner_message: dict, parser_message: dict) -> dict:
    extracted = json.dumps(parser_message["payload"], indent=2)
    register = json.dumps(planner_message["payload"], indent=2)
    revision_round = planner_message["revision_round"]

    user_message = (
        "Here is the original extracted meeting data:\n\n"
        f"<extracted_data>\n{extracted}\n</extracted_data>\n\n"
        "Here is the action register to review:\n\n"
        f"<action_register>\n{register}\n</action_register>\n\n"
        "Review the action register against the original extracted data and return "
        "your assessment as JSON matching the schema in your instructions."
    )

    text, _ = call_llm(CRITIC_PROMPT, user_message, CRITIC, revision_round)

    try:
        payload = json.loads(_clean_json(text))
        status = STATUS_APPROVED if payload.get("approved") else STATUS_REVISION_REQUIRED
    except json.JSONDecodeError as exc:
        debug_path = f"critic_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[Critic] Raw LLM response written to {debug_path}")
        payload = {"raw_response": text, "parse_error": str(exc)}
        status = STATUS_ERROR

    if status in (STATUS_APPROVED, STATUS_REVISION_REQUIRED):
        score = payload.get("overall_score", "?")
        approved = payload.get("approved", False)
        issues = len(payload.get("issues", []))
        print(
            f"[Critic] Score: {score}/10 — "
            f"{'APPROVED' if approved else 'REVISION REQUIRED'} — "
            f"{issues} issue(s) found (round {revision_round})"
        )
    else:
        print(f"[Critic] ERROR — failed to parse JSON response (round {revision_round})")

    message = create_message(
        sender=CRITIC,
        recipient=ORCHESTRATOR,
        task="critic_review",
        payload=payload,
        status=status,
        revision_round=revision_round,
    )
    validate_message(message)
    return message

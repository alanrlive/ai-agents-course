from datetime import datetime, timezone

# Agent name constants — used as sender/recipient values in every message
NOTE_PARSER = "note_parser"
ACTION_PLANNER = "action_planner"
CRITIC = "critic"
ORCHESTRATOR = "orchestrator"

# Valid status values for a message envelope
STATUS_SUCCESS = "success"
STATUS_ERROR = "error"
STATUS_REVISION_REQUIRED = "revision_required"
STATUS_APPROVED = "approved"

VALID_STATUSES = {STATUS_SUCCESS, STATUS_ERROR, STATUS_REVISION_REQUIRED, STATUS_APPROVED}

REQUIRED_FIELDS = {"sender", "recipient", "task", "payload", "status", "timestamp", "revision_round"}


def create_message(
    sender: str,
    recipient: str,
    task: str,
    payload: str,
    status: str,
    revision_round: int = 0,
) -> dict:
    """Build a message envelope with a current UTC ISO timestamp."""
    return {
        "sender": sender,           # agent that produced this message
        "recipient": recipient,     # agent that will consume this message
        "task": task,               # short label describing what the payload contains
        "payload": payload,         # the actual content being handed off
        "status": status,           # outcome of the producing agent's work
        "timestamp": datetime.now(timezone.utc).isoformat(),  # UTC time of creation
        "revision_round": revision_round,  # how many critic revision cycles have occurred
    }


def validate_message(message: dict) -> None:
    """Raise ValueError if any required field is missing or status is not a known value."""
    missing = REQUIRED_FIELDS - message.keys()
    if missing:
        raise ValueError(f"Message is missing required fields: {sorted(missing)}")

    if message["status"] not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status '{message['status']}'. Must be one of: {sorted(VALID_STATUSES)}"
        )

# memory_manager.py

import json
import os
from datetime import datetime

MEMORY_FILE = "memory/conversation_history.json"
MAX_HISTORY_ENTRIES = 20


def load_history() -> list[dict]:
    """Load conversation history from JSON file. Returns empty list if file does not exist."""
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_history(history: list[dict]) -> None:
    """Save conversation history to JSON file. Creates the memory/ folder if needed."""
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def add_interaction(history: list[dict], user_input: str, agent_response: str) -> list[dict]:
    """
    Append a new interaction to history and trim to MAX_HISTORY_ENTRIES.
    Each entry has: timestamp, user_input, agent_response.
    """
    history = list(history)
    history.append({
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "agent_response": agent_response,
    })
    return history[-MAX_HISTORY_ENTRIES:]


def format_history_for_prompt(history: list[dict]) -> str:
    """
    Format the last 5 interactions as a readable string for injection into the system prompt.
    Returns empty string if history is empty.
    """
    if not history:
        return ""
    lines = []
    for entry in history[-5:]:
        ts = entry.get("timestamp", "")
        lines.append(f"[{ts}] User: {entry.get('user_input', '')}")
        lines.append(f"[{ts}] Agent: {entry.get('agent_response', '')}")
    return "\n".join(lines)

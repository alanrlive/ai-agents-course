# tools.py
TOOL_DEFINITIONS = [
    {
        "name": "get_current_date",
        "description": "Returns today's date as a string in YYYY-MM-DD format.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_public_holidays",
        "description": (
            "Fetches the list of public holidays for a given country and year "
            "from the Nager.Date API. Returns a list of objects each containing "
            "a 'date' (YYYY-MM-DD) and a 'name' for the holiday."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "country_code": {
                    "type": "string",
                    "description": "ISO 3166-1 alpha-2 country code (e.g. 'US', 'GB', 'DE').",
                },
                "year": {
                    "type": "integer",
                    "description": "The calendar year to fetch holidays for (e.g. 2025).",
                },
            },
            "required": ["country_code", "year"],
        },
    },
    {
        "name": "calculate_working_days",
        "description": (
            "Counts the number of working days between two dates, excluding weekends "
            "and any provided public holidays. The start date is exclusive and the "
            "end date is inclusive."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "The start date in YYYY-MM-DD format (not counted).",
                },
                "end_date": {
                    "type": "string",
                    "description": "The end date in YYYY-MM-DD format (counted if a working day).",
                },
                "holidays": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of holiday dates in YYYY-MM-DD format to exclude.",
                },
            },
            "required": ["start_date", "end_date", "holidays"],
        },
    },
    {
        "name": "search_scheduling_knowledge",
        "description": (
            "Search the Elevator Financial Services scheduling knowledge base for office-specific "
            "working hours, cultural norms, meeting preferences, time zone guidance, and global "
            "scheduling policies. Use this tool when the meeting notes include attendees from "
            "specific offices, when you need to check office-specific constraints, or when you "
            "need to apply the firm's global scheduling policy. Always search before proposing "
            "a meeting date when attendee office locations are known."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural language query describing what scheduling information you need. For example: 'London office meeting preferences' or 'Dubai working hours and cultural considerations' or 'global notice period for steering committee'.",
                }
            },
            "required": ["query"],
        },
    },
]

import urllib.request
import urllib.error
import json
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from openai import OpenAI

load_dotenv()

_openai_client = OpenAI()
_chroma_client = chromadb.PersistentClient(
    path=str(Path(__file__).parent / "chroma_db")
)
_kb_collection = _chroma_client.get_or_create_collection(
    name="elevator_fs_kb",
    metadata={"hnsw:space": "cosine"},
)


def get_current_date() -> str:
    """Returns today's date as a YYYY-MM-DD string."""
    try:
        return date.today().isoformat()
    except Exception as e:
        raise RuntimeError(f"Failed to get current date: {e}")


def get_public_holidays(country_code: str, year: int) -> list[dict]:
    """
    Fetches public holidays from the Nager.Date API.
    Returns a list of dicts with 'date' (YYYY-MM-DD) and 'name' keys.
    """
    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"API returned status {response.status} for {country_code}/{year}"
                )
            data = json.loads(response.read().decode("utf-8"))
            return [{"date": h["date"], "name": h["localName"]} for h in data]
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"HTTP error fetching holidays for {country_code}/{year}: {e.code} {e.reason}"
        )
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Network error fetching holidays for {country_code}/{year}: {e.reason}"
        )
    except (KeyError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"Unexpected response format for {country_code}/{year}: {e}"
        )


def calculate_working_days(
    start_date: str, end_date: str, holidays: list[str]
) -> int:
    """
    Counts working days between start_date and end_date (exclusive of start, inclusive of end).
    Excludes weekends and any dates in the holidays list (YYYY-MM-DD strings).
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError(f"Invalid date format (expected YYYY-MM-DD): {e}")

    if end < start:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}")

    holiday_set = set()
    for h in holidays:
        try:
            holiday_set.add(datetime.strptime(h, "%Y-%m-%d").date())
        except ValueError:
            raise ValueError(f"Invalid holiday date format (expected YYYY-MM-DD): {h}")

    count = 0
    current = start + timedelta(days=1)
    while current <= end:
        if current.weekday() < 5 and current not in holiday_set:
            count += 1
        current += timedelta(days=1)

    return count


def search_scheduling_knowledge(query: str, n_results: int = 3) -> str:
    print(f"[RAG] Searching knowledge base for: {query}")
    embedding_response = _openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    )
    query_embedding = embedding_response.data[0].embedding

    results = _kb_collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return "No relevant information found in the scheduling knowledge base."

    parts = []
    for doc, meta in zip(documents, metadatas):
        source = meta.get("source", "unknown")
        parts.append(f"[Source: {source}] {doc}")

    return "\n\n".join(parts)

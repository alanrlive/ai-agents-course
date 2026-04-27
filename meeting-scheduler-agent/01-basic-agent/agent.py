# agent.py
import os
import json
from dotenv import load_dotenv
import anthropic
import tools as tool_module
from tools import TOOL_DEFINITIONS

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"
MAX_ITERATIONS = 10

SYSTEM_PROMPT = """You are a Meeting Scheduler Agent helping delivery managers coordinate meetings across distributed teams.

When given meeting notes containing attendee names, their country locations, and the date of the last meeting, you must:
1. Identify all countries represented by the attendees.
2. Use get_current_date to confirm today's date.
3. Use get_public_holidays to fetch holidays for each country for the relevant year(s).
4. Use calculate_working_days to count working days between dates.
5. Propose a next meeting date that falls between 10 and 15 working days after the last meeting.
6. Ensure the proposed date is not a weekend or a public holiday in any attendee country.

Always explain your reasoning and clearly state the proposed meeting date."""

TOOL_DISPATCH = {
    "get_current_date": tool_module.get_current_date,
    "get_public_holidays": tool_module.get_public_holidays,
    "calculate_working_days": tool_module.calculate_working_days,
}


def call_tool(name: str, inputs: dict):
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        raise ValueError(f"Unknown tool: {name}")
    return fn(**inputs)


def run_agent(user_input: str):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    messages = [{"role": "user", "content": user_input}]

    print(f"\n{'='*60}")
    print("MEETING SCHEDULER AGENT")
    print(f"{'='*60}")
    print(f"Input: {user_input}\n")

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"--- Iteration {iteration} ---")

        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        print(f"Stop reason: {response.stop_reason}")

        # Collect any text the model produced this turn
        for block in response.content:
            if block.type == "text":
                print(f"Model: {block.text}")

        # No tool calls — agent is done
        if response.stop_reason == "end_turn":
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                print(f"\n{'='*60}")
                print("FINAL ANSWER")
                print(f"{'='*60}")
                for block in response.content:
                    if block.type == "text":
                        print(block.text)
                return response

        # Append the assistant turn (full content list)
        messages.append({"role": "assistant", "content": response.content})

        # Process every tool_use block and build tool_result list
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            print(f"  Tool call: {block.name}({json.dumps(block.input)})")

            try:
                result = call_tool(block.name, block.input)
                result_content = json.dumps(result)
                print(f"  Tool result: {result_content}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_content,
                })
            except Exception as e:
                error_msg = f"Error: {e}"
                print(f"  Tool error: {error_msg}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": error_msg,
                    "is_error": True,
                })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    print(f"\nReached max iterations ({MAX_ITERATIONS}) without a final answer.")
    return None


if __name__ == "__main__":
    sample_notes = """
    Meeting notes - Last meeting: 2025-01-15
    Attendees:
    - Alice (United States)
    - Bob (Germany)
    - Carol (United Kingdom)
    Please schedule our next meeting.
    """
    run_agent(sample_notes)

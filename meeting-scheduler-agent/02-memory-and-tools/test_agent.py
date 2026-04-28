# test_agent.py
# Runs 4 test scenarios to demonstrate multi-step agent behaviour
# Each scenario exercises different combinations of offices and constraints

import sys
sys.stdout.reconfigure(encoding="utf-8")

from agent import run_agent

TEST_SCENARIOS = [
    {
        "name": "Scenario 1: London and Warsaw (EMEA pair)",
        "notes": """
        Meeting notes - Last meeting: 2025-04-10
        Attendees:
        - Sarah (London office)
        - Marek (Warsaw office)
        - Tom (London office)
        Please schedule our next delivery review meeting.
        """
    },
    {
        "name": "Scenario 2: London, New York and Dubai (cross-regional)",
        "notes": """
        Meeting notes - Last meeting: 2025-04-15
        Attendees:
        - James (London office)
        - Maria (New York office)
        - Khalid (Dubai office)
        This is a steering committee meeting. Please schedule accordingly.
        """
    },
    {
        "name": "Scenario 3: Singapore and Sydney (APAC pair)",
        "notes": """
        Meeting notes - Last meeting: 2025-04-20
        Attendees:
        - Priya (Singapore office)
        - Liam (Sydney office)
        Please schedule our next working session.
        """
    },
    {
        "name": "Scenario 4: All offices (global call)",
        "notes": """
        Meeting notes - Last meeting: 2025-04-05
        Attendees:
        - Alice (London office)
        - Bob (New York office)
        - Chen (Singapore office)
        - Fatima (Dubai office)
        - Anna (Warsaw office)
        - Jack (Sydney office)
        Please schedule our next all-hands coordination call.
        """
    },
]


if __name__ == "__main__":
    print("=" * 70)
    print("ELEVATOR FINANCIAL SERVICES — MEETING SCHEDULER AGENT TEST RUN")
    print("=" * 70)

    for i, scenario in enumerate(TEST_SCENARIOS, 1):
        print(f"\n{'#' * 70}")
        print(f"# {scenario['name']}")
        print(f"{'#' * 70}")
        run_agent(scenario["notes"])
        print("\n")

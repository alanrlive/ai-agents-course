# Assignment 03A — Meeting Notes Processor

A multi-agent pipeline that transforms raw, messy meeting notes into a clean, structured action register.

## How it works

Three agents collaborate in sequence:

1. **Note Parser** — extracts structured data from raw notes without interpretation
2. **Action Planner** — reasons over the extracted data to assign owners, deadlines, and priorities
3. **Action Register Critic** — quality-checks the register and either approves it or requests specific revisions (up to 3 cycles)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your Anthropic API key to .env
```

## Run

```bash
# Default input (sample_notes.txt)
python main.py

# Custom input file
python main.py --notes path/to/your_notes.txt
```

The demo input is `sample_notes.txt` — a Sprint 4 Review for a CRM-to-payments integration project.

## Output

A structured action register printed to stdout, approved by the Critic agent.

Output files are written to the working directory with datetime stamps:

- `action_register_YYYYMMDD_HHMMSS.json` — the final approved action register
- `agent_interaction_log_YYYYMMDD_HHMMSS.json` — full interaction log with per-agent token usage

If the Critic agent fails to parse a response, a `critic_debug_YYYYMMDD_HHMMSS.txt` file is written containing the raw LLM output for inspection.

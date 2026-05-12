# Assignment 03B — Meeting Notes Processor with Vision Agent

Extends the Assignment 03A multi-agent pipeline with a Vision Agent at the front. The pipeline accepts an image file (whiteboard photo, handwritten notes, or digital screenshot) and processes it through to a structured, critic-approved action register. All pipeline stages are traced in Langfuse.

## How it works

Four agents collaborate in sequence:

1. **Vision Agent** — classifies the image type, extracts all visible text, and passes structured output downstream. Rejects unreadable or unrecognised images and halts the pipeline immediately.
2. **Note Parser** — extracts structured data from the Vision Agent's text output without interpretation
3. **Action Planner** — reasons over the extracted data to assign owners, deadlines, and priorities
4. **Action Register Critic** — quality-checks the register and either approves it or requests specific revisions (up to 3 cycles)

### Document types recognised by the Vision Agent

| Type | Description |
|---|---|
| `whiteboard` | Physical whiteboard or flip chart; spatial groupings and low-contrast areas noted |
| `handwritten` | Handwritten notes on paper; per-word confidence flags added for difficult words |
| `digital` | Screenshot, slide, or digital document; highest confidence extraction path |
| `unrecognised` | Blank, blurry, or non-meeting content — pipeline stops with a structured error |

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...

# Optional — omit to run without tracing
LANGFUSE_PUBLIC_KEY=your_public_key_here
LANGFUSE_SECRET_KEY=your_secret_key_here
LANGFUSE_HOST=https://cloud.langfuse.com
```

Langfuse tracing is optional. If the keys are absent the pipeline runs normally and prints `Langfuse tracing : SKIPPED` at the end.

## Run

```bash
python main.py --image path/to/your_image.jpg
```

Supported image formats: `jpg`, `jpeg`, `png`.

## Observability

Each pipeline run creates a single Langfuse trace (`meeting-notes-pipeline`) with a child span per agent stage:

| Span | Input | Output |
|---|---|---|
| `vision_agent` | image path | document_type, confidence, extracted_text_length, rejected |
| `note_parser` | extracted text | parsed notes payload |
| `action_planner` | parsed notes | action register payload (one span per revision round) |
| `critic` | action register | score, issues, approved (one span per revision round) |

The root span records the final action register as output and total token count as metadata.

## Output

A structured action register printed to stdout. Output files are written to the working directory with datetime stamps:

- `action_register_YYYYMMDD_HHMMSS.json` — the final action register
- `agent_interaction_log_YYYYMMDD_HHMMSS.json` — per-agent token counts and response previews

If the Critic fails to parse a response, a `critic_debug_YYYYMMDD_HHMMSS.txt` file is written with the raw LLM output.

## Testing

Generate sample images and run the integration tests:

```bash
python test_images/create_samples.py   # one-time setup
python test_cases.py                   # makes real API calls
```

| Test | Input | Pass criterion |
|---|---|---|
| `test_happy_path` | `whiteboard_sample.jpg` | Not rejected, text extracted, full pipeline completes |
| `test_alternate_path` | `handwritten_sample.jpg` | Not rejected, confidence logged, pipeline completes |
| `test_edge_case` | `blurry_sample.jpg` | Rejected with reason, Note Parser never called |

## Project structure

```
assignment-03b/
├── main.py                 # Orchestrator — runs the four-agent pipeline
├── vision_agent.py         # Vision Agent — image classification and text extraction
├── agents.py               # Note Parser, Action Planner, Critic agents
├── prompts.py              # System prompts for all four agents
├── schema.py               # Message envelope schema and VisionOutput dataclass
├── langfuse_client.py      # Langfuse client initialisation (graceful no-op if keys absent)
├── test_cases.py           # Integration tests
├── test_images/
│   ├── create_samples.py   # Generates the three sample test images
│   ├── whiteboard_sample.jpg
│   ├── handwritten_sample.jpg
│   └── blurry_sample.jpg
├── requirements.txt        # anthropic, python-dotenv, json-repair, Pillow, langfuse
└── .env                    # API keys (not committed)
```

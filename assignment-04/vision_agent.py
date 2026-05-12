from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import anthropic
from dotenv import load_dotenv

if TYPE_CHECKING:
    from monitoring.logger import PipelineLogger

load_dotenv()

from agents import MODEL, client, _clean_json, log_interaction
from prompts import VISION_AGENT_PROMPT
from schema import VisionOutput

SUPPORTED_MEDIA_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
}

MAX_RETRIES = 3
RETRY_DELAY = 2


def run_vision_agent(image_path: str, logger: PipelineLogger | None = None) -> VisionOutput:
    if logger:
        logger.agent_start("vision_agent", image_path=str(image_path))

    path = Path(image_path)

    if not path.exists():
        print(f"[Vision Agent] ERROR — file not found: {image_path}")
        reason = f"File not found: {image_path}"
        if logger:
            logger.agent_error("vision_agent", error=reason)
        return VisionOutput(
            document_type="unrecognised",
            extracted_text="",
            confidence="low",
            processing_notes="",
            rejected=True,
            rejection_reason=reason,
        )

    ext = path.suffix.lstrip(".").lower()
    media_type = SUPPORTED_MEDIA_TYPES.get(ext)
    if media_type is None:
        print(f"[Vision Agent] ERROR — unsupported file type: .{ext}")
        reason = f"Unsupported file type: .{ext}. Supported: jpg, jpeg, png"
        if logger:
            logger.agent_error("vision_agent", error=reason)
        return VisionOutput(
            document_type="unrecognised",
            extracted_text="",
            confidence="low",
            processing_notes="",
            rejected=True,
            rejection_reason=reason,
        )

    with open(path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Analyse this image and return the structured JSON "
                        "as specified in your instructions."
                    ),
                },
            ],
        }
    ]

    last_error = None
    response = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                temperature=0,
                system=VISION_AGENT_PROMPT,
                messages=messages,
            )
            break
        except anthropic.RateLimitError as exc:
            last_error = exc
            print(
                f"[Vision Agent] Rate limit hit (attempt {attempt}/{MAX_RETRIES}), "
                f"retrying in {RETRY_DELAY}s..."
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
        except anthropic.APIError as exc:
            last_error = exc
            print(f"[Vision Agent] API error (attempt {attempt}/{MAX_RETRIES}): {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    if response is None:
        print(f"[Vision Agent] ERROR — all {MAX_RETRIES} attempts failed: {last_error}")
        reason = f"API call failed after {MAX_RETRIES} attempts: {last_error}"
        if logger:
            logger.agent_error("vision_agent", error=reason)
        return VisionOutput(
            document_type="unrecognised",
            extracted_text="",
            confidence="low",
            processing_notes="",
            rejected=True,
            rejection_reason=reason,
        )

    if response.stop_reason == "tool_use":
        print("[Vision Agent] ERROR — unexpected tool_use stop_reason; no tools declared")
        reason = "Unexpected tool_use stop_reason from API"
        if logger:
            logger.agent_error("vision_agent", error=reason)
        return VisionOutput(
            document_type="unrecognised",
            extracted_text="",
            confidence="low",
            processing_notes="",
            rejected=True,
            rejection_reason=reason,
        )

    raw_text = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    log_interaction(
        agent_name="vision_agent",
        input_summary=f"image:{path.name} media_type:{media_type}",
        output_preview=raw_text[:300],
        tokens_used=input_tokens + output_tokens,
        revision_round=0,
    )

    try:
        data = json.loads(_clean_json(raw_text))
        output = VisionOutput(
            document_type=data.get("document_type", "unrecognised"),
            extracted_text=data.get("extracted_text", ""),
            confidence=data.get("confidence", "low"),
            processing_notes=data.get("processing_notes", ""),
            rejected=bool(data.get("rejected", False)),
            rejection_reason=data.get("rejection_reason", ""),
        )
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"[Vision Agent] ERROR — failed to parse JSON response: {exc}")
        reason = f"JSON parse error: {exc}"
        if logger:
            logger.agent_error("vision_agent", error=reason)
        return VisionOutput(
            document_type="unrecognised",
            extracted_text="",
            confidence="low",
            processing_notes="",
            rejected=True,
            rejection_reason=reason,
        )

    if logger:
        logger.agent_complete(
            "vision_agent",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            document_type=output.document_type,
            confidence=output.confidence,
            rejected=output.rejected,
        )

    print(
        f"[Vision Agent] document_type={output.document_type}, "
        f"confidence={output.confidence}"
    )
    if output.rejected:
        print(f"[Vision Agent] Rejected — {output.rejection_reason}")

    return output

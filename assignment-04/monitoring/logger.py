"""Structured JSON logger and operational metrics for the meeting-notes pipeline.

Log file format (logs/run_{timestamp}_{trace_id}.json):
  {
    "events": [ ... ],   # appended on every event; written incrementally
    "metrics": { ... }   # written once on pipeline_complete / pipeline_error
  }

Usage:
    logger = PipelineLogger()
    logger.pipeline_start(image_path)

    logger.agent_start("vision_agent", image_path=image_path)
    ...
    logger.agent_complete("vision_agent", input_tokens=400, output_tokens=120)

    logger.pipeline_complete(outcome="approved")
"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


class PipelineLogger:
    def __init__(self) -> None:
        self.trace_id: str = uuid.uuid4().hex[:12]
        self._start_wall: datetime = datetime.now(timezone.utc)

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        ts = self._start_wall.strftime("%Y%m%d_%H%M%S")
        self._log_path: Path = log_dir / f"run_{ts}_{self.trace_id}.json"

        self._events: list[dict] = []
        self._agent_t0: dict[str, float] = {}
        self._pipeline_t0: float | None = None

        # Accumulator state for the metrics summary block
        self._agent_metrics: dict[str, dict] = {}
        self._total_tokens: int = 0
        self._critic_revisions: int = 0
        self._success: bool | None = None
        self._total_latency_s: float | None = None

        # Write an empty skeleton so the file exists immediately
        self._write(metrics=None)

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _record(self, event_type: str, data: dict) -> None:
        self._events.append(
            {"timestamp": self._now(), "trace_id": self.trace_id, "event_type": event_type, **data}
        )
        self._write(metrics=None)

    def _write(self, metrics: dict | None) -> None:
        with open(self._log_path, "w", encoding="utf-8") as fh:
            json.dump({"events": self._events, "metrics": metrics}, fh, indent=2)

    def _agent_stats(self, agent: str) -> dict:
        return self._agent_metrics.setdefault(
            agent,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "total_latency_s": 0.0},
        )

    def _build_metrics(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "success": self._success,
            "total_latency_s": self._total_latency_s,
            "total_tokens": self._total_tokens,
            "critic_revision_count": self._critic_revisions,
            "agents": {name: dict(stats) for name, stats in self._agent_metrics.items()},
        }

    def _close(self) -> None:
        self._write(metrics=self._build_metrics())

    # --------------------------------------------------------------- pipeline

    def pipeline_start(self, image_path: str) -> None:
        self._pipeline_t0 = time.monotonic()
        self._record("pipeline_start", {"image_path": str(image_path)})

    def pipeline_complete(self, outcome: str) -> None:
        self._total_latency_s = (
            round(time.monotonic() - self._pipeline_t0, 3) if self._pipeline_t0 is not None else None
        )
        self._success = True
        self._record("pipeline_complete", {"outcome": outcome, "total_latency_s": self._total_latency_s})
        self._close()

    def pipeline_error(self, error: str, stage: str) -> None:
        self._total_latency_s = (
            round(time.monotonic() - self._pipeline_t0, 3) if self._pipeline_t0 is not None else None
        )
        self._success = False
        self._record(
            "pipeline_error",
            {"error": error, "stage": stage, "total_latency_s": self._total_latency_s},
        )
        self._close()

    # ----------------------------------------------------------------- agents

    def agent_start(self, agent: str, **kwargs) -> None:
        self._agent_t0[agent] = time.monotonic()
        self._record(f"{agent}_start", kwargs)

    def agent_complete(
        self,
        agent: str,
        input_tokens: int,
        output_tokens: int,
        revision_round: int = 0,
        **kwargs,
    ) -> None:
        t0 = self._agent_t0.get(agent)
        latency = round(time.monotonic() - t0, 3) if t0 is not None else 0.0
        total = input_tokens + output_tokens

        stats = self._agent_stats(agent)
        stats["calls"] += 1
        stats["input_tokens"] += input_tokens
        stats["output_tokens"] += output_tokens
        stats["total_tokens"] += total
        stats["total_latency_s"] = round(stats["total_latency_s"] + latency, 3)

        self._total_tokens += total

        self._record(
            f"{agent}_complete",
            {
                "latency_s": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total,
                "revision_round": revision_round,
                **kwargs,
            },
        )

    def agent_error(self, agent: str, error: str, revision_round: int = 0, **kwargs) -> None:
        t0 = self._agent_t0.get(agent)
        latency = round(time.monotonic() - t0, 3) if t0 is not None else 0.0
        self._record(
            f"{agent}_error",
            {"error": error, "latency_s": latency, "revision_round": revision_round, **kwargs},
        )

    def critic_revision_required(self, revision_round: int, score: int, issues: list) -> None:
        self._critic_revisions += 1
        self._record(
            "critic_revision_required",
            {"revision_round": revision_round, "score": score, "issues": issues},
        )

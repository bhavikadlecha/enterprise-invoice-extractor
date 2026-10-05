"""
app/observability/tracker.py
-----------------------------
Lightweight SQLite-backed LLM cost and token tracker.

Tracks per-extraction:
  - Input / output tokens
  - Estimated cost in USD (using published OpenAI / Anthropic pricing)
  - Latency in seconds
  - Number of self-healing retries needed
  - Whether a PO was matched

Exposes:
  - `record()` — async write a new extraction record
  - `get_summary()` — aggregate stats for the /metrics endpoint
  - `get_recent()` — last N extraction records

Pricing table (update when model pricing changes):
  https://openai.com/api/pricing / https://www.anthropic.com/pricing
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import aiosqlite

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-million-token pricing (USD) — update these as pricing changes
# ---------------------------------------------------------------------------
_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    # Gemini models (litellm prefixes model names with "gemini/", stripped below)
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-flash-latest": {"input": 0.15, "output": 0.60},
    "gemini-pro-latest": {"input": 1.25, "output": 10.00},
}
_DEFAULT_PRICING = {"input": 1.25, "output": 5.00}

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS extraction_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    model           TEXT    NOT NULL,
    provider        TEXT    NOT NULL,
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    cost_usd        REAL    NOT NULL DEFAULT 0.0,
    latency_seconds REAL    NOT NULL DEFAULT 0.0,
    retries         INTEGER NOT NULL DEFAULT 0,
    success         INTEGER NOT NULL DEFAULT 1,
    po_matched      INTEGER NOT NULL DEFAULT 0,
    vendor_name     TEXT
)
"""


# ---------------------------------------------------------------------------
# Data class passed by the extractor into record()
# ---------------------------------------------------------------------------
@dataclass
class ExtractionMetrics:
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    retries: int = 0
    success: bool = True
    po_matched: bool = False
    vendor_name: Optional[str] = None
    # Computed automatically
    cost_usd: float = field(init=False)

    def __post_init__(self) -> None:
        # Strip the "gemini/" prefix that litellm adds to model names
        model_key = self.model.replace("gemini/", "")
        pricing = _PRICING.get(model_key, _DEFAULT_PRICING)
        self.cost_usd = round(
            (self.input_tokens / 1_000_000) * pricing["input"]
            + (self.output_tokens / 1_000_000) * pricing["output"],
            6,
        )


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
async def init_db() -> None:
    """Create the metrics table if it does not exist."""
    async with aiosqlite.connect(settings.metrics_db_path) as db:
        await db.execute(_CREATE_TABLE_SQL)
        await db.commit()
    logger.info("Metrics database initialised at: %s", settings.metrics_db_path)


async def record(metrics: ExtractionMetrics) -> None:
    """Persist one extraction record to SQLite."""
    ts = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(settings.metrics_db_path) as db:
        await db.execute(
            """
            INSERT INTO extraction_metrics
              (timestamp, model, provider, input_tokens, output_tokens,
               cost_usd, latency_seconds, retries, success, po_matched, vendor_name)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                ts,
                metrics.model,
                metrics.provider,
                metrics.input_tokens,
                metrics.output_tokens,
                metrics.cost_usd,
                metrics.latency_seconds,
                metrics.retries,
                int(metrics.success),
                int(metrics.po_matched),
                metrics.vendor_name,
            ),
        )
        await db.commit()


async def get_summary() -> dict[str, Any]:
    """Return aggregate statistics across all recorded extractions."""
    async with aiosqlite.connect(settings.metrics_db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT
                COUNT(*)                                   AS total_extractions,
                SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) AS successful,
                SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) AS failed,
                ROUND(SUM(cost_usd), 6)                    AS total_cost_usd,
                ROUND(AVG(cost_usd), 6)                    AS avg_cost_per_extraction_usd,
                SUM(input_tokens)                          AS total_input_tokens,
                SUM(output_tokens)                         AS total_output_tokens,
                ROUND(AVG(latency_seconds), 3)             AS avg_latency_seconds,
                ROUND(AVG(retries), 2)                     AS avg_retries,
                SUM(CASE WHEN po_matched=1 THEN 1 ELSE 0 END) AS po_matches,
                ROUND(100.0 * SUM(CASE WHEN retries>0 THEN 1 ELSE 0 END)
                      / MAX(COUNT(*), 1), 1)               AS self_healing_trigger_pct
            FROM extraction_metrics
            """
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else {}


async def get_recent(limit: int = 20) -> list[dict[str, Any]]:
    """Return the last `limit` extraction records."""
    async with aiosqlite.connect(settings.metrics_db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM extraction_metrics ORDER BY id DESC LIMIT ?", (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

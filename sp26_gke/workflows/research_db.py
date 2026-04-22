"""CloudSQL persistence layer for research runs, checkpoints, and evidence."""

from __future__ import annotations

import hashlib
import json
import os
from typing import TYPE_CHECKING, Any

import asyncpg  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from sp26_gke.workflows.research_agent import Evidence

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS research_runs (
  run_id TEXT PRIMARY KEY,
  thread_id TEXT UNIQUE,
  topic TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'running',
  current_node TEXT,
  last_checkpoint_id TEXT,
  paused_reason TEXT,
  last_error TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS thread_id TEXT;
ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS current_node TEXT;
ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS last_checkpoint_id TEXT;
ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS paused_reason TEXT;
ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS last_error TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_research_runs_thread_id
  ON research_runs(thread_id);

CREATE TABLE IF NOT EXISTS evidence (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id TEXT NOT NULL,
  section_id TEXT NOT NULL,
  claim TEXT NOT NULL,
  source_url TEXT NOT NULL,
  retrieval_query TEXT,
  claim_hash TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_evidence_run_section
  ON evidence(run_id, section_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_dedup
  ON evidence(run_id, section_id, claim_hash);

CREATE TABLE IF NOT EXISTS workflow_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  node_name TEXT,
  payload JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_workflow_events_run_created
  ON workflow_events(run_id, created_at);

CREATE TABLE IF NOT EXISTS workflow_checkpoints (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id TEXT NOT NULL,
  checkpoint_id TEXT NOT NULL,
  node_name TEXT,
  state_json JSONB,
  attempt INT NOT NULL DEFAULT 1,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_workflow_checkpoints_unique
  ON workflow_checkpoints(run_id, checkpoint_id);

CREATE TABLE IF NOT EXISTS section_reports (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id TEXT NOT NULL,
  section_id TEXT NOT NULL,
  content TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'completed',
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_section_reports_unique
  ON section_reports(run_id, section_id);

CREATE TABLE IF NOT EXISTS final_reports (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id TEXT UNIQUE NOT NULL,
  content TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'completed',
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS paragraph_evidence_map (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id TEXT NOT NULL,
  report_level TEXT NOT NULL,
  section_id TEXT,
  paragraph_index INT NOT NULL,
  evidence_id UUID NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);
"""


def _get_dsn() -> str:
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError(
            "Missing DATABASE_URL. Set it in your environment (or .env) to persist "
            "research evidence."
        )
    return dsn


def _claim_hash(claim: str, source_url: str) -> str:
    return hashlib.sha256((claim + source_url).encode()).hexdigest()


class ResearchDB:
    """Async persistence for research runs and evidence objects."""

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn or _get_dsn()

    async def _connect(self) -> asyncpg.Connection:
        return await asyncpg.connect(self._dsn)

    async def ensure_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        conn = await self._connect()
        try:
            await conn.execute(SCHEMA_SQL)
        finally:
            await conn.close()

    async def create_run(
        self, run_id: str, topic: str, thread_id: str | None = None
    ) -> None:
        """Insert or reactivate a research run record."""
        conn = await self._connect()
        try:
            await conn.execute(SCHEMA_SQL)
            await conn.execute(
                "INSERT INTO research_runs (run_id, thread_id, topic, status) "
                "VALUES ($1, $2, $3, 'running') "
                "ON CONFLICT (run_id) DO UPDATE "
                "SET topic = EXCLUDED.topic, "
                "thread_id = COALESCE(research_runs.thread_id, EXCLUDED.thread_id), "
                "status = 'running', last_error = NULL, paused_reason = NULL, "
                "updated_at = now()",
                run_id,
                thread_id,
                topic,
            )
        finally:
            await conn.close()

    async def mark_run_status(
        self,
        run_id: str,
        *,
        status: str,
        current_node: str | None = None,
        last_checkpoint_id: str | None = None,
        paused_reason: str | None = None,
        last_error: str | None = None,
    ) -> None:
        """Update lifecycle fields for a run."""
        conn = await self._connect()
        try:
            await conn.execute(
                "UPDATE research_runs SET status = $2, "
                "current_node = COALESCE($3, current_node), "
                "last_checkpoint_id = COALESCE($4, last_checkpoint_id), "
                "paused_reason = COALESCE($5, paused_reason), "
                "last_error = COALESCE($6, last_error), "
                "updated_at = now() WHERE run_id = $1",
                run_id,
                status,
                current_node,
                last_checkpoint_id,
                paused_reason,
                last_error,
            )
        finally:
            await conn.close()

    async def log_event(
        self,
        run_id: str,
        *,
        event_type: str,
        node_name: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Append an event row for run observability."""
        conn = await self._connect()
        try:
            payload_json = json.dumps(payload) if payload is not None else None
            await conn.execute(
                "INSERT INTO workflow_events (run_id, event_type, node_name, payload) "
                "VALUES ($1, $2, $3, $4::jsonb)",
                run_id,
                event_type,
                node_name,
                payload_json,
            )
        finally:
            await conn.close()

    async def upsert_checkpoint_meta(
        self,
        run_id: str,
        *,
        checkpoint_id: str,
        node_name: str | None,
        attempt: int,
        state_json: dict[str, Any] | None = None,
    ) -> None:
        """Store lightweight checkpoint metadata for DB inspection."""
        conn = await self._connect()
        try:
            state_payload = json.dumps(state_json) if state_json is not None else None
            await conn.execute(
                "INSERT INTO workflow_checkpoints "
                "(run_id, checkpoint_id, node_name, state_json, attempt) "
                "VALUES ($1, $2, $3, $4::jsonb, $5) "
                "ON CONFLICT (run_id, checkpoint_id) DO UPDATE "
                "SET node_name = EXCLUDED.node_name, "
                "state_json = COALESCE(EXCLUDED.state_json, workflow_checkpoints.state_json), "
                "attempt = EXCLUDED.attempt",
                run_id,
                checkpoint_id,
                node_name,
                state_payload,
                attempt,
            )
        finally:
            await conn.close()

    async def insert_evidence_batch(self, run_id: str, items: list[Evidence]) -> int:
        """Insert evidence, deduplicating by (run_id, section_id, claim_hash)."""
        if not items:
            return 0

        conn = await self._connect()
        try:
            await conn.execute(SCHEMA_SQL)
            rows = [
                (
                    run_id,
                    item.query_plan_id,
                    item.claim,
                    item.source_url,
                    item.retrieval_query,
                    _claim_hash(item.claim, item.source_url),
                )
                for item in items
            ]
            await conn.executemany(
                "INSERT INTO evidence "
                "(run_id, section_id, claim, source_url, retrieval_query, claim_hash) "
                "VALUES ($1, $2, $3, $4, $5, $6) "
                "ON CONFLICT (run_id, section_id, claim_hash) DO NOTHING",
                rows,
            )
            return len(rows)
        finally:
            await conn.close()

    async def complete_run(self, run_id: str) -> None:
        """Mark a research run as completed."""
        conn = await self._connect()
        try:
            await conn.execute(
                "UPDATE research_runs SET status = 'completed', "
                "updated_at = now() WHERE run_id = $1",
                run_id,
            )
        finally:
            await conn.close()

    async def get_run_persistence_summary(self, run_id: str) -> dict[str, Any]:
        """Return compact per-run DB summary for demo verification."""
        conn = await self._connect()
        try:
            run_row = await conn.fetchrow(
                "SELECT run_id, thread_id, status, current_node, last_checkpoint_id "
                "FROM research_runs WHERE run_id = $1",
                run_id,
            )
            evidence_count = await conn.fetchval(
                "SELECT COUNT(*) FROM evidence WHERE run_id = $1",
                run_id,
            )
            event_count = await conn.fetchval(
                "SELECT COUNT(*) FROM workflow_events WHERE run_id = $1",
                run_id,
            )
            return {
                "run": dict(run_row) if run_row else None,
                "evidence_count": int(evidence_count or 0),
                "event_count": int(event_count or 0),
            }
        finally:
            await conn.close()

    async def get_run_checkpoint_timeline(
        self, run_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Return latest checkpoint metadata rows for a run."""
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                "SELECT checkpoint_id, node_name, attempt, created_at "
                "FROM workflow_checkpoints WHERE run_id = $1 "
                "ORDER BY created_at DESC LIMIT $2",
                run_id,
                limit,
            )
            return [dict(row) for row in rows]
        finally:
            await conn.close()

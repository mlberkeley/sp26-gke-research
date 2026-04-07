"""
CloudSQL persistence layer for research evidence.

Requires DATABASE_URL env var (asyncpg-compatible DSN) and asyncpg installed.
"""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING

import asyncpg  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from sp26_gke.workflows.research_agent import Evidence

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS research_runs (
  run_id      TEXT PRIMARY KEY,
  topic       TEXT NOT NULL,
  status      TEXT NOT NULL DEFAULT 'running',
  created_at  TIMESTAMPTZ DEFAULT now(),
  updated_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evidence (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id          TEXT NOT NULL,
  section_id      TEXT NOT NULL,
  claim           TEXT NOT NULL,
  source_url      TEXT NOT NULL,
  retrieval_query TEXT,
  claim_hash      TEXT NOT NULL,
  created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_evidence_run_section
  ON evidence(run_id, section_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_dedup
  ON evidence(run_id, section_id, claim_hash);
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

    async def create_run(self, run_id: str, topic: str) -> None:
        """Insert a new research run record."""
        conn = await self._connect()
        try:
            await conn.execute(SCHEMA_SQL)
            await conn.execute(
                "INSERT INTO research_runs (run_id, topic) VALUES ($1, $2) "
                "ON CONFLICT (run_id) DO NOTHING",
                run_id,
                topic,
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

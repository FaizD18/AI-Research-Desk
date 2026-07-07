"""SQLite persistence layer.

One database file holds all pipeline state. Each stage writes its own table
and reads its inputs from the previous stage's table, so every stage can be
re-run independently and the whole pipeline is inspectable with plain SQL.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from research_desk import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS filings (
    accession        TEXT PRIMARY KEY,
    ticker           TEXT NOT NULL,
    cik              INTEGER NOT NULL,
    form             TEXT NOT NULL,
    filing_date      TEXT NOT NULL,   -- date the filing became public (signal date)
    report_date      TEXT NOT NULL,   -- fiscal period end
    primary_document TEXT NOT NULL,
    doc_url          TEXT NOT NULL,
    local_path       TEXT NOT NULL,
    sha256           TEXT NOT NULL,
    size_bytes       INTEGER NOT NULL,
    fetched_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS extractions (
    accession         TEXT PRIMARY KEY REFERENCES filings(accession),
    method            TEXT NOT NULL,  -- toc-anchor | heading-scan | llm-boundary
    word_count        INTEGER NOT NULL,
    n_paragraphs      INTEGER NOT NULL,
    text_path         TEXT NOT NULL,  -- extracted plain text on disk
    extractor_version TEXT NOT NULL,
    validation_json   TEXT NOT NULL,  -- metrics from the validation gate
    extracted_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS risk_changes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    prior_accession TEXT NOT NULL REFERENCES filings(accession),
    curr_accession  TEXT NOT NULL REFERENCES filings(accession),
    change_type     TEXT NOT NULL CHECK (change_type IN
                        ('NEW', 'REMOVED', 'ESCALATED', 'UNCHANGED')),
    similarity      REAL,            -- best-match cosine; NULL for NEW/REMOVED
    paragraph_text  TEXT NOT NULL,   -- current-year text (prior-year for REMOVED)
    matched_text    TEXT,            -- best-matching other-year paragraph, if any
    paragraph_index INTEGER NOT NULL,
    UNIQUE (curr_accession, change_type, paragraph_index)
);

CREATE TABLE IF NOT EXISTS risk_scores (
    change_id INTEGER PRIMARY KEY REFERENCES risk_changes(id),
    category  TEXT NOT NULL,
    severity  INTEGER NOT NULL CHECK (severity BETWEEN 1 AND 5),
    rationale TEXT NOT NULL,
    model     TEXT NOT NULL,
    scored_at TEXT NOT NULL
);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open (and initialize if needed) the project database."""
    path = db_path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn

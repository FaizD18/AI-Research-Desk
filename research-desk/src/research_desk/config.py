"""Single source of truth for all project configuration.

Every threshold, path, ticker list, and model name lives here — nothing is
hardcoded inline in modules. Secrets (ANTHROPIC_API_KEY) come from the
environment only and are never stored in this file.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("RESEARCH_DESK_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"  # data/raw/<ticker>/<accession>.html
EXTRACTED_DIR = DATA_DIR / "extracted"  # plain-text Item 1A sections
LLM_CACHE_DIR = DATA_DIR / "cache" / "llm"  # disk cache for every LLM call
EMBED_CACHE_DIR = DATA_DIR / "cache" / "embeddings"  # cached paragraph vectors
DEBATES_DIR = DATA_DIR / "debates"  # full debate transcripts (Phase 3)
DB_PATH = DATA_DIR / "research_desk.db"

# --- Universe ----------------------------------------------------------

TICKERS: list[str] = ["AAPL", "NVDA", "JPM"]
N_YEARS = 5  # how many fiscal years of 10-Ks to ingest per ticker

# --- SEC EDGAR fair access (sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)

# SEC requires a declared User-Agent identifying the app and a contact email.
EDGAR_USER_AGENT = "AIResearchDesk faizpro2018@gmail.com"
# SEC's published cap is 10 req/s; we throttle below it to stay clearly inside.
EDGAR_REQUESTS_PER_SEC = 8.0
EDGAR_MAX_RETRIES = 3
EDGAR_BACKOFF_SECONDS = 2.0  # doubled on each retry after a 403/429/5xx

EDGAR_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/{filename}"
EDGAR_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"

# --- Item 1A extraction ------------------------------------------------

EXTRACTOR_VERSION = "1.0"
# Validation gate (observed real range across AAPL/NVDA/JPM: ~9.7k-17.2k words).
EXTRACT_MIN_WORDS = 2_000
EXTRACT_MAX_WORDS = 40_000
EXTRACT_MIN_PARAGRAPHS = 10
# A heading block must be short; every observed cross-reference is a full sentence.
HEADING_MAX_CHARS = 60

# --- Year-over-year diffing --------------------------------------------

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # symmetric similarity: no query prefix
# Cosine-similarity bands for matching risk paragraphs across years.
MATCH_THRESHOLD = 0.70  # below this: no match -> NEW (current yr) / REMOVED (prior yr)
UNCHANGED_THRESHOLD = 0.95  # at/above this: same risk, same language -> UNCHANGED
MIN_PARAGRAPH_WORDS = 25  # ignore headings/fragments when splitting paragraphs

# --- Earnings-call transcripts (Phase 2) --------------------------------

TRANSCRIPTS_DIR = DATA_DIR / "transcripts"  # data/transcripts/<ticker>/fy<year>q<q>.json
TRANSCRIPT_SOURCE = "defeatbeta"  # open HuggingFace dataset via defeatbeta-api; no key
TRANSCRIPT_MIN_WORDS = 500  # below this a "transcript" is a stub; skip it

# Deterministic tone word-lists, inspired by the Loughran-McDonald financial
# sentiment dictionaries (weak-modal / uncertainty categories). Counted as
# whole words or phrases, case-insensitive, per 1,000 words of call text.
HEDGING_TERMS: list[str] = [
    "may", "might", "could", "possibly", "perhaps", "appears", "appeared",
    "seems", "somewhat", "roughly", "approximately", "we believe", "we think",
    "sort of", "kind of",
]
UNCERTAINTY_TERMS: list[str] = [
    "uncertain", "uncertainty", "uncertainties", "risk", "risks", "risky",
    "volatile", "volatility", "unpredictable", "unclear", "unknown",
    "fluctuate", "fluctuation", "fluctuations", "headwind", "headwinds",
]
GUIDANCE_TERMS: list[str] = [
    "guidance", "outlook", "forecast", "expect", "expects", "expected",
    "expectations", "projection", "projections", "target", "targets",
]

# --- LLM scoring -------------------------------------------------------

LLM_MODEL = "claude-sonnet-5"
LLM_MAX_TOKENS = 1_000
# Use the Message Batches API (50% discount) when this many calls miss cache.
LLM_BATCH_THRESHOLD = 8
LLM_BATCH_POLL_SECONDS = 15.0

RISK_CATEGORIES: list[str] = [
    "regulatory",
    "competitive",
    "macro",
    "operational",
    "cyber",
    "litigation",
    "supply-chain",
]

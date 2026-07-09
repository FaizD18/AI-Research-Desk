"""Year-over-year risk-language diffing.

For each ticker, consecutive 10-K Item 1A sections are compared paragraph by
paragraph. Each paragraph is embedded with a sentence-transformer; a current
paragraph is matched to its most similar prior-year paragraph by cosine
similarity and classified:

* ``UNCHANGED``  — cosine ≥ ``UNCHANGED_THRESHOLD``: essentially the same risk,
  same language.
* ``ESCALATED``  — ``MATCH_THRESHOLD`` ≤ cosine < ``UNCHANGED_THRESHOLD``: the
  risk is retained but its language was materially revised (usually expanded).
  Whether a revision is a genuine escalation is a judgment call left to the LLM
  scoring stage; this stage only flags that the wording changed.
* ``NEW``        — cosine < ``MATCH_THRESHOLD``: no prior-year counterpart.

A prior-year paragraph whose best match to *any* current paragraph falls below
``MATCH_THRESHOLD`` is ``REMOVED``.

The embedding model is symmetric (BAAI/bge-small-en-v1.5), so no retrieval
query prefix is prepended. Classification is pure and deterministic given the
embeddings; only the embedding step needs the model, and its output is cached
to disk so full-pipeline reruns are fast.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from research_desk import config, db

log = logging.getLogger(__name__)

_UNIT = "␟"  # record separator used to hash a paragraph list


@dataclass(frozen=True)
class RiskChange:
    """One classified paragraph in a year-over-year comparison."""

    change_type: str  # NEW | REMOVED | ESCALATED | UNCHANGED
    paragraph_index: int
    paragraph_text: str
    matched_text: str | None
    similarity: float | None


def load_paragraphs(text: str) -> list[str]:
    """Split an extracted Item 1A into substantive risk paragraphs.

    Blocks are separated by blank lines (as written by ``extract.py``). Short
    blocks — the ``Item 1A`` heading, category subheadings, stray fragments —
    are dropped so the diff compares comparable units of risk prose.
    """
    paras = [p.strip() for p in text.split("\n\n")]
    return [p for p in paras if len(p.split()) >= config.MIN_PARAGRAPH_WORDS]


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row so a dot product equals cosine similarity."""
    if matrix.size == 0:
        return matrix.reshape(0, 0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def classify_changes(
    prior_paras: list[str],
    curr_paras: list[str],
    prior_emb: np.ndarray,
    curr_emb: np.ndarray,
    *,
    match_threshold: float = config.MATCH_THRESHOLD,
    unchanged_threshold: float = config.UNCHANGED_THRESHOLD,
) -> list[RiskChange]:
    """Classify every current and removed prior paragraph. Pure function."""
    changes: list[RiskChange] = []
    have_prior = len(prior_paras) > 0
    have_curr = len(curr_paras) > 0

    # Similarity matrix: rows = current paragraphs, cols = prior paragraphs.
    if have_prior and have_curr:
        sim = _unit_rows(curr_emb) @ _unit_rows(prior_emb).T
    else:
        sim = None

    for i, para in enumerate(curr_paras):
        if sim is None:  # no prior year to compare against
            changes.append(RiskChange("NEW", i, para, None, None))
            continue
        j = int(sim[i].argmax())
        score = float(sim[i, j])
        if score >= unchanged_threshold:
            changes.append(RiskChange("UNCHANGED", i, para, prior_paras[j], score))
        elif score >= match_threshold:
            changes.append(RiskChange("ESCALATED", i, para, prior_paras[j], score))
        else:
            changes.append(RiskChange("NEW", i, para, None, None))

    for j, para in enumerate(prior_paras):
        if sim is None:  # nothing in the current year — everything removed
            changes.append(RiskChange("REMOVED", j, para, None, None))
            continue
        best = float(sim[:, j].max())
        if best < match_threshold:
            changes.append(RiskChange("REMOVED", j, para, None, None))

    return changes


@lru_cache(maxsize=1)
def _model():
    """Load the sentence-transformer once (downloads on first use)."""
    from sentence_transformers import SentenceTransformer

    log.info("loading embedding model %s", config.EMBEDDING_MODEL)
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed(accession: str, paragraphs: list[str]) -> np.ndarray:
    """Embed paragraphs (unit-normalized), caching the result to disk.

    The cache key includes a hash of the paragraph list, so a re-extraction
    that changes the text invalidates the stale vectors automatically.
    """
    if not paragraphs:
        return np.zeros((0, 0), dtype=np.float32)
    digest = hashlib.sha256(_UNIT.join(paragraphs).encode()).hexdigest()[:16]
    path = config.EMBED_CACHE_DIR / f"{accession}.{digest}.npy"
    if path.exists():
        return np.load(path)
    vectors = _model().encode(
        paragraphs, normalize_embeddings=True, show_progress_bar=False
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors


def run_diff(tickers: list[str]) -> dict[str, dict[str, int]]:
    """Diff consecutive filings for each ticker into the ``risk_changes`` table.

    Idempotent: a filing pair's changes are replaced on re-run. One bad pair
    logs and is skipped. Returns {ticker: {change_type: count}}.
    """
    conn = db.connect()
    summary: dict[str, dict[str, int]] = {}
    for ticker in tickers:
        filings = conn.execute(
            """SELECT f.accession, f.report_date, e.text_path
               FROM extractions e JOIN filings f ON f.accession = e.accession
               WHERE f.ticker = ? ORDER BY f.report_date""",
            (ticker.upper(),),
        ).fetchall()
        counts: dict[str, int] = {k: 0 for k in ("NEW", "REMOVED", "ESCALATED", "UNCHANGED")}
        for prior, curr in zip(filings, filings[1:], strict=False):
            try:
                prior_paras = load_paragraphs(Path(prior["text_path"]).read_text())
                curr_paras = load_paragraphs(Path(curr["text_path"]).read_text())
                changes = classify_changes(
                    prior_paras,
                    curr_paras,
                    embed(prior["accession"], prior_paras),
                    embed(curr["accession"], curr_paras),
                )
            except Exception:
                log.exception(
                    "skipping %s pair %s→%s", ticker, prior["accession"], curr["accession"]
                )
                continue
            conn.execute(
                "DELETE FROM risk_changes WHERE curr_accession = ?", (curr["accession"],)
            )
            pair_counts: dict[str, int] = {
                k: 0 for k in ("NEW", "REMOVED", "ESCALATED", "UNCHANGED")
            }
            for ch in changes:
                conn.execute(
                    """INSERT INTO risk_changes
                       (ticker, prior_accession, curr_accession, change_type,
                        similarity, paragraph_text, matched_text, paragraph_index)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        ticker.upper(),
                        prior["accession"],
                        curr["accession"],
                        ch.change_type,
                        ch.similarity,
                        ch.paragraph_text,
                        ch.matched_text,
                        ch.paragraph_index,
                    ),
                )
                pair_counts[ch.change_type] += 1
                counts[ch.change_type] += 1
            conn.commit()
            log.info(
                "%s %s→%s: %s",
                ticker, prior["report_date"], curr["report_date"], pair_counts,
            )
        summary[ticker] = counts
    conn.close()
    return summary

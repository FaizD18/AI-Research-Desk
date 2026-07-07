"""AI Research Desk — SEC 10-K risk intelligence pipeline.

Phase 1 (RiskDelta): ingest 10-Ks from EDGAR, extract Item 1A risk factors,
diff risk language year-over-year, and score new/escalated risks with an LLM.
"""

__version__ = "0.1.0"

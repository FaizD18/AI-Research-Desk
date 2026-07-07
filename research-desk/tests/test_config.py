"""Sanity checks on configuration invariants other modules rely on."""

from research_desk import config


def test_edgar_user_agent_declares_contact() -> None:
    """SEC fair-access policy requires an identifying UA with a contact email."""
    assert "@" in config.EDGAR_USER_AGENT
    assert config.EDGAR_USER_AGENT.split()[0]  # app/company name present


def test_edgar_rate_below_sec_cap() -> None:
    assert 0 < config.EDGAR_REQUESTS_PER_SEC <= 10


def test_similarity_thresholds_ordered() -> None:
    assert 0 < config.MATCH_THRESHOLD < config.UNCHANGED_THRESHOLD <= 1


def test_risk_categories_nonempty_and_unique() -> None:
    assert config.RISK_CATEGORIES
    assert len(set(config.RISK_CATEGORIES)) == len(config.RISK_CATEGORIES)

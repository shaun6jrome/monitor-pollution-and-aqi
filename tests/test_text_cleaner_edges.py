"""Tests for edge-case text normalization and chemical formula preservation."""

from src.preprocessing.text_cleaner import clean_text, normalize_unicode


def test_unicode_subscript_conversion():
    # Subscripts PM₂.₅ and SO₂ normalized to ASCII via NFKC
    raw = "The concentration of PM₂.₅ reached 350 ug/m³ while SO₂ was 80 ug/m³."
    cleaned = clean_text(raw)
    assert "PM2.5" in cleaned
    assert "SO2" in cleaned
    assert "ug/m3" in cleaned or "ug/m³" in cleaned


def test_chemical_formulas_are_preserved():
    raw = "Emissions of NO2, CO, C6H6, and NMHC exceeded prescribed CPCB standards."
    cleaned = clean_text(raw)
    assert "NO2" in cleaned
    assert "CO" in cleaned
    assert "C6H6" in cleaned
    assert "NMHC" in cleaned


def test_curly_quotes_and_em_dashes_normalized():
    raw = '“Severe smog”—reported local authorities—‘delayed twelve flights’.'
    cleaned = clean_text(raw)
    assert '"' in cleaned
    assert "'" in cleaned
    assert "-" in cleaned
    assert "“" not in cleaned
    assert "”" not in cleaned
    assert "—" not in cleaned


def test_email_and_tracking_url_stripping():
    raw = "Report filed by env.reporter@newsagency.org. Read details at https://news.example.com/delhi-smog?utm_source=feed"
    cleaned = clean_text(raw)
    assert "env.reporter@newsagency.org" not in cleaned
    assert "https://news.example.com" not in cleaned
    assert "Report filed by" in cleaned


def test_extreme_whitespace_and_newlines():
    from src.preprocessing.text_cleaner import normalize_text
    raw = "Smog     alert\n\n\n\tin    \t   Delhi   NCR.\r\n"
    cleaned = clean_text(raw)
    assert cleaned == "Smog alert\nin Delhi NCR."
    single_line = normalize_text(raw)
    assert single_line == "smog alert in delhi ncr."

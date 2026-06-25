"""Fetch the latest 10-K for a ticker from SEC EDGAR and extract key sections.

We pull the "Risk Factors" (Item 1A) and "Management's Discussion & Analysis"
(Item 7) sections because they are the narrative, forward-looking parts of a
10-K — exactly the qualitative context a price/technicals model cannot see and
that the judge can use to corroborate or challenge a forecast.

SEC etiquette is mandatory, not optional:
  * Every request sends a descriptive ``User-Agent`` with a contact address;
    SEC blocks anonymous/default user agents.
  * Requests are throttled below SEC's 10 req/s ceiling via a small delay.

Discovery uses EDGAR full-text search (efts.sec.gov). Because full-text search
is keyed on phrases rather than tickers, we resolve the ticker to a CIK first
(via EDGAR's official ticker map) and keep only hits for that CIK. If full-text
search returns nothing usable, we fall back to the EDGAR submissions API so the
pipeline still finds the filing.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

# SEC requires a User-Agent identifying the requester with a contact address.
SEC_USER_AGENT = "FinSentinel/1.0 (research contact: amanshekhar000@gmail.com)"
_HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}

# Stay well under SEC's 10 requests/second limit.
_MIN_REQUEST_INTERVAL = 0.2
_last_request_ts = 0.0

EFTS_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"


@dataclass
class TenKSections:
    """Extracted 10-K narrative for one ticker."""

    ticker: str
    cik: str
    accession: str
    filing_url: str
    sections: dict[str, str]  # {"Risk Factors": "...", "MD&A": "..."}


def _throttled_get(url: str, **kwargs) -> requests.Response:
    """GET with SEC headers and a global minimum inter-request delay."""
    global _last_request_ts
    elapsed = time.time() - _last_request_ts
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    resp = requests.get(url, headers=_HEADERS, timeout=20, **kwargs)
    _last_request_ts = time.time()
    resp.raise_for_status()
    return resp


def resolve_cik(ticker: str) -> str | None:
    """Resolve a ticker to its zero-padded 10-digit CIK via EDGAR's ticker map."""
    data = _throttled_get(TICKER_MAP_URL).json()
    target = ticker.upper()
    for row in data.values():
        if row.get("ticker", "").upper() == target:
            return str(row["cik_str"]).zfill(10)
    return None


def _find_10k_via_efts(cik: str, ticker: str) -> tuple[str, str] | None:
    """Use full-text search to find the latest 10-K; return (accession, primary_doc).

    ``q`` is a phrase every 10-K contains; results are filtered to our CIK and
    sorted by filing date so we take the most recent annual report.
    """
    params = {"q": "\"risk factors\"", "forms": "10-K", "ciks": cik}
    try:
        hits = _throttled_get(EFTS_SEARCH_URL, params=params).json().get("hits", {}).get("hits", [])
    except Exception:  # noqa: BLE001 - efts hiccup -> caller falls back
        return None

    relevant = [h for h in hits if cik in (h.get("_source", {}).get("ciks") or [])]
    if not relevant:
        return None
    relevant.sort(key=lambda h: h["_source"].get("file_date", ""), reverse=True)
    # _id format: "<accession-with-dashes>:<primary-document-filename>"
    doc_id = relevant[0]["_id"]
    accession, _, primary_doc = doc_id.partition(":")
    return accession, primary_doc


def _find_10k_via_submissions(cik: str) -> tuple[str, str] | None:
    """Fallback: read the submissions API and pick the latest 10-K."""
    data = _throttled_get(SUBMISSIONS_URL.format(cik=cik)).json()
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    for i, form in enumerate(forms):
        if form == "10-K":
            return recent["accessionNumber"][i], recent["primaryDocument"][i]
    return None


def _build_filing_url(cik: str, accession: str, primary_doc: str) -> str:
    """Construct the Archives URL for the filing's primary HTML document."""
    accession_nodashes = accession.replace("-", "")
    cik_int = str(int(cik))  # Archives path uses the un-padded CIK
    return f"{ARCHIVES_BASE}/{cik_int}/{accession_nodashes}/{primary_doc}"


def _html_to_text(html: str) -> str:
    """Strip a 10-K HTML document to normalised plain text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    # Collapse the heavy whitespace EDGAR HTML produces.
    return re.sub(r"\s+", " ", text).strip()


def _extract_between(text: str, start_pattern: str, end_patterns: list[str]) -> str:
    """Slice the text from the LAST start-header match to the next end header.

    Using the *last* occurrence of the item header skips the table-of-contents
    entry (which mentions the same "Item 1A. Risk Factors" string) and lands on
    the real section body.
    """
    starts = list(re.finditer(start_pattern, text, flags=re.IGNORECASE))
    if not starts:
        return ""
    begin = starts[-1].start()
    end = len(text)
    for pat in end_patterns:
        m = re.search(pat, text[begin + 50:], flags=re.IGNORECASE)
        if m:
            end = min(end, begin + 50 + m.start())
    return text[begin:end].strip()


def _extract_sections(html: str) -> dict[str, str]:
    """Pull Risk Factors (Item 1A) and MD&A (Item 7) from filing HTML."""
    text = _html_to_text(html)
    risk = _extract_between(
        text,
        r"item\s*1a\.?\s*[-–—:]?\s*risk\s+factors",
        [r"item\s*1b\.", r"item\s*2\.\s*properties"],
    )
    mdna = _extract_between(
        text,
        r"item\s*7\.?\s*[-–—:]?\s*management.{0,3}s\s+discussion",
        [r"item\s*7a\.", r"item\s*8\.\s*financial"],
    )
    return {"Risk Factors": risk, "MD&A": mdna}


def load_latest_10k(ticker: str) -> TenKSections:
    """Fetch and parse the latest 10-K for ``ticker``.

    Returns a :class:`TenKSections`. Missing sections come back as empty strings
    (the retriever then reports weak/absent evidence) rather than raising.

    Raises:
        ValueError: If the ticker cannot be resolved or no 10-K can be located.
    """
    cik = resolve_cik(ticker)
    if cik is None:
        raise ValueError(f"Could not resolve CIK for ticker '{ticker}'.")

    located = _find_10k_via_efts(cik, ticker) or _find_10k_via_submissions(cik)
    if located is None:
        raise ValueError(f"No 10-K filing found for '{ticker}' (CIK {cik}).")

    accession, primary_doc = located
    filing_url = _build_filing_url(cik, accession, primary_doc)
    resp = _throttled_get(filing_url)
    resp.encoding = "utf-8"  # EDGAR HTML is UTF-8; avoid requests' latin-1 guess.
    sections = _extract_sections(resp.text)

    return TenKSections(
        ticker=ticker.upper(),
        cik=cik,
        accession=accession,
        filing_url=filing_url,
        sections=sections,
    )

"""Separate deployment gate for numbers that are not in the allowed evidence.

Source-copy overlap stays in ``rag_source_copying``. This module only checks
quantities.
"""
from __future__ import annotations

import re

_MONEY = re.compile(
    r"\$\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*"
    r"(billion|million|thousand|k|m|b)?\+?",
    re.IGNORECASE,
)
_MAGNITUDE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(thousand|million|billion)\b",
    re.IGNORECASE,
)
_PERCENT = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.IGNORECASE)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_DATE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\.?\s+\d{1,2}(?:\s*[-–]\s*\d{1,2})?(?:,\s*\d{4})?",
    re.IGNORECASE,
)
_INTEGER = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{3,}\b")
_SCALE = {
    None: 1,
    "": 1,
    "k": 1_000,
    "thousand": 1_000,
    "m": 1_000_000,
    "million": 1_000_000,
    "b": 1_000_000_000,
    "billion": 1_000_000_000,
}
_MONTHS = {
    "jan": "jan",
    "feb": "feb",
    "mar": "mar",
    "apr": "apr",
    "may": "may",
    "jun": "jun",
    "jul": "jul",
    "aug": "aug",
    "sep": "sep",
    "sept": "sep",
    "oct": "oct",
    "nov": "nov",
    "dec": "dec",
}


def _covered(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < stop and end > begin for begin, stop in spans)


def _money_canonical(raw_number: str, suffix: str | None) -> str:
    amount = float(raw_number.replace(",", ""))
    scale = _SCALE[(suffix or "").casefold() or None]
    return f"usd:{int(round(amount * scale))}"


def _percent_canonical(raw_number: str) -> str:
    amount = float(raw_number)
    if amount == int(amount):
        return f"pct:{int(amount)}"
    return f"pct:{amount}"


def _date_canonical(surface: str) -> str:
    match = re.match(
        r"([A-Za-z]+)\.?\s+(\d{1,2})(?:\s*[-–]\s*(\d{1,2}))?(?:,\s*(\d{4}))?",
        surface.strip(),
    )
    month = _MONTHS[match.group(1)[:4].casefold() if match.group(1).casefold().startswith("sept") else match.group(1)[:3].casefold()]
    days = match.group(2)
    if match.group(3):
        days = f"{days}-{match.group(3)}"
    year = f":{match.group(4)}" if match.group(4) else ""
    return f"date:{month}:{days}{year}"


def _add(found: list[dict], spans: list[tuple[int, int]], text: str, match: re.Match, canonical: str) -> None:
    if _covered(match.start(), match.end(), spans):
        return
    spans.append((match.start(), match.end()))
    found.append(
        {
            "value": match.group(0),
            "canonical": canonical,
            "sentence": _sentence(text, match.start(), match.end()),
        }
    )


def _sentence(text: str, start: int, end: int) -> str:
    left_breaks = [text.rfind(mark, 0, start) for mark in ("\n", ". ", "! ", "? ")]
    left = max(left_breaks)
    right_candidates = [text.find(mark, end) for mark in ("\n", ". ", "! ", "? ")]
    right_candidates = [index for index in right_candidates if index >= 0]
    right = min(right_candidates) if right_candidates else len(text)
    if left < 0:
        left = 0
    elif text[left:left + 2] in {". ", "! ", "? "}:
        left += 2
    elif text[left] == "\n":
        left += 1
    return text[left:right].strip()


def extract_quantities(text: str) -> list[dict]:
    """Percentages, money, dates, years, and other multi-digit counts."""
    found: list[dict] = []
    spans: list[tuple[int, int]] = []
    source = text or ""
    for match in _MONEY.finditer(source):
        _add(found, spans, source, match, _money_canonical(match.group(1), match.group(2)))
    for match in _MAGNITUDE.finditer(source):
        _add(found, spans, source, match, _money_canonical(match.group(1), match.group(2)))
    for match in _PERCENT.finditer(source):
        _add(found, spans, source, match, _percent_canonical(match.group(1)))
    for match in _DATE.finditer(source):
        _add(found, spans, source, match, _date_canonical(match.group(0)))
    for match in _YEAR.finditer(source):
        _add(found, spans, source, match, f"year:{match.group(0)}")
    for match in _INTEGER.finditer(source):
        digits = match.group(0).replace(",", "")
        if len(digits) < 3:
            continue
        _add(found, spans, source, match, f"n:{int(digits)}")
    return found


def numbers_in(text: str) -> set[str]:
    return {item["canonical"] for item in extract_quantities(text)}


def allowed_numbers(topic: str, claims: list[dict]) -> set[str]:
    """Topic text plus accepted claim text. Support spans are not included."""
    allowed = numbers_in(topic)
    for claim in claims:
        allowed |= numbers_in(claim.get("claim") or "")
    return allowed


def evaluate_numeric_grounding(
    answer: str,
    topic: str,
    claims: list[dict] | None,
    *,
    enabled: bool,
) -> dict:
    """Fail when the answer introduces a quantity the evidence does not allow."""
    if not enabled:
        return {
            "applicable": False,
            "pass": None,
            "failure": None,
            "allowed": [],
            "values": [],
            "unsupported": [],
        }
    allowed = allowed_numbers(topic, claims or [])
    values = extract_quantities(answer)
    unsupported = [item for item in values if item["canonical"] not in allowed]
    return {
        "applicable": True,
        "pass": not unsupported,
        "failure": None if not unsupported else "unsupported_numeric_grounding",
        "allowed": sorted(allowed),
        "values": values,
        "unsupported": unsupported,
    }


def gates_failed(copy_report: dict, numeric_report: dict) -> bool:
    """Either deployment guard can request the single retry."""
    if copy_report.get("applicable") and not copy_report.get("pass"):
        return True
    if numeric_report.get("applicable") and not numeric_report.get("pass"):
        return True
    return False

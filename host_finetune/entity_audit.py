"""Flags for low-information graph entity names.

These flags are an audit. They do not delete nodes from a built graph.
"""

from __future__ import annotations

import re

_WEEKDAYS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}
_MONTHS = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}
_GENERIC = {
    "basic", "company", "partner", "investment", "investor", "round", "rounds",
    "startup", "startups", "update", "correct",
}


def entity_name_flags(name: str) -> list[str]:
    """Return the reasons a name is a weak entity. Empty means no rule fired."""
    text = (name or "").strip()
    lowered = text.casefold()
    flags = []
    if lowered in _WEEKDAYS:
        flags.append("weekday")
    if lowered in _MONTHS:
        flags.append("month")
    if re.search(r"\d|%", text):
        flags.append("numeric_name")
    token = lowered.split()[0] if lowered else ""
    if lowered in _GENERIC or token in _GENERIC:
        flags.append("generic_noun")
    return flags


def false_employer_descriptions(descriptions: list[str], speaker: str, organization: str) -> list[str]:
    """Descriptions that make a quoted speaker the leader of a later-mentioned organization."""
    speaker_re = re.compile(re.escape(speaker), re.I)
    org_re = re.compile(re.escape(organization), re.I)
    role_re = re.compile(r"chief product|cpo", re.I)
    hits = []
    for description in descriptions:
        if speaker_re.search(description) and org_re.search(description) and role_re.search(description):
            hits.append(description)
    return hits

"""Restore punctuation, casing, and strip verbal fillers from transcript text.

Used by ``clean_dataset_v2`` to repair auto-captioned YouTube transcripts before
they enter SFT. Drops filler words ("um", "uh", "you know" etc.), restores
sentence-ending punctuation with ``oliverguhr/fullstop-punctuation-multilang-large``,
then applies a heuristic casing pass (start-of-sentence caps + a curated proper-noun
list).

Usage::

    from host_finetune.repunctuate import Repunctuator
    rp = Repunctuator(device="cuda")
    fixed = rp.repunctuate("a six bit of the sixth one is just a bit of hubris")
    # -> "A six bit of the sixth one is just a bit of hubris."

Public surface kept small so the host venv doesn't need extra packages beyond
``deepmultilingualpunctuation`` (which itself depends only on transformers).
"""
from __future__ import annotations

import logging
import os
import re
from typing import Iterable

logger = logging.getLogger(__name__)

# ── Filler-word stripping ───────────────────────────────────────────────────
# These are clear non-word verbal fillers from auto-captioned speech. We
# deliberately leave "you know", "i mean", "like" alone because Lemkin uses them
# in writing too; removing them would distort his actual voice.
_FILLER_INTERJECTIONS = (
    "um", "umm", "ummm",
    "uh", "uhh", "uhm", "uhmm",
    "ah", "ahh", "ahem",
    "er", "erm", "errr",
    "hmm", "hmmm", "mhm", "mmhm", "mm-hmm", "uh-huh", "huh",
)
_FILLER_RE = re.compile(
    r"(?<![\w-])(?:" + "|".join(re.escape(f) for f in _FILLER_INTERJECTIONS) + r")(?![\w-])",
    re.IGNORECASE,
)
# "you know," / "I mean," / "kind of" / "sort of" used as discourse markers
# (only when surrounded by commas or at sentence transitions). Conservative —
# preserves these phrases when they carry meaning.
_DISCOURSE_FILLERS = (
    r",\s*you know\s*,",
    r",\s*i mean\s*,",
    r",\s*kind of\s*,",
    r",\s*sort of\s*,",
)
_DISCOURSE_RE = re.compile("|".join(_DISCOURSE_FILLERS), re.IGNORECASE)
# Repeated tokens like "the the the" or "i i i" (caption stutter).
_REPEAT_RE = re.compile(r"\b(\w{1,4})(?:\s+\1\b){1,}", re.IGNORECASE)
# Multiple spaces collapse.
_WS_RE = re.compile(r"[ \t]{2,}")


def strip_fillers(text: str) -> str:
    """Remove verbal fillers + repeated-token stutters from transcript text."""
    if not text:
        return text
    t = _FILLER_RE.sub(" ", text)
    t = _DISCOURSE_RE.sub(", ", t)
    t = _REPEAT_RE.sub(r"\1", t)
    t = _WS_RE.sub(" ", t)
    return t.strip()


# ── Casing heuristics ───────────────────────────────────────────────────────
# Acronyms / proper nouns that appear regularly in the Lemkin corpus. Order
# matters only for the longest-match pass; the regex below uses word-boundary
# replacement so order-within-list is irrelevant once compiled.
_PROPER_NOUNS = {
    # People / brands Lemkin mentions often.
    "Jason", "Lemkin", "Aaron", "Levie", "Henry", "Schuck", "Eric", "Keith",
    "Rabois", "Sequoia", "Andreessen", "Horowitz", "OpenAI", "Anthropic",
    "Microsoft", "Google", "Amazon", "Apple", "Meta", "IBM", "Salesforce",
    "HubSpot", "Datadog", "ZoomInfo", "DiscoverOrg", "Snowflake", "Shopify",
    "Replit", "Elevenlabs", "Qualified", "Artisan", "Delphi", "Slack",
    "Stripe", "Twilio", "Box", "Dropbox", "Atlassian", "Zendesk", "Intercom",
    "ServiceNow", "Adobe", "Workday", "Oracle", "Klarna", "Calendly",
    # Companies via SaaStr ecosystem.
    "SaaStr", "EchoSign", "DocuSign",
    # Cities / regions.
    "London", "Austin", "New", "York", "Tokyo", "Bay", "Area", "Silicon",
    "Valley", "California", "America", "European",
    # Models / products.
    "Claude", "ChatGPT", "GPT", "LLaMA", "Llama",
    # Months + days.
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    # Generic capitalised words used by Lemkin.
    "American", "British", "Indian", "Chinese", "Japanese",
}
# All-caps acronyms — must stay uppercase regardless of position. We
# deliberately exclude ``US`` / ``USA`` / ``UK`` because the pronoun ``us`` is
# vastly more common than the country in this corpus and was being wrongly
# uppercased everywhere ("Thanks for joining US"). Add a context-aware rule if
# we ever need to recover country casing.
_ACRONYMS_UPPER = {
    "SaaS", "B2B", "B2C", "AI", "ML", "API", "APIs", "SDK", "SDKs",
    "CEO", "CEOs", "CFO", "CFOs", "COO", "COOs", "CTO", "CTOs", "CRO",
    "CROs", "CMO", "CMOs", "CPO", "CPOs", "EVP", "EVPs", "SVP", "SVPs",
    "VP", "VPs", "SDR", "SDRs", "BDR", "BDRs", "AE", "AEs", "CSM", "CSMs",
    "GTM", "ICP", "ACV", "ARR", "NRR", "MRR", "LTV", "CAC", "IPO", "IPOs",
    "VC", "VCs", "PE", "TAM", "SAM", "SOM", "PLG", "RTO", "HR", "FDE",
    "SaaStr", "AMA", "FAQ", "ROI", "KPI", "KPIs", "OKR", "OKRs",
    "Q1", "Q2", "Q3", "Q4", "EU", "EMEA", "APAC", "YC",
}
# Build a single regex for proper-noun + acronym casing in one pass.
_VOCAB_CANONICAL: dict[str, str] = {}
for word in _PROPER_NOUNS:
    _VOCAB_CANONICAL[word.lower()] = word
for word in _ACRONYMS_UPPER:
    _VOCAB_CANONICAL[word.lower()] = word
_VOCAB_RE = re.compile(
    r"(?<![\w'])(" + "|".join(re.escape(w) for w in _VOCAB_CANONICAL) + r")(?![\w'])",
    re.IGNORECASE,
)


def _apply_vocab_casing(text: str) -> str:
    """Rewrite known proper-noun / acronym tokens to their canonical casing."""

    def repl(m: re.Match) -> str:
        return _VOCAB_CANONICAL[m.group(1).lower()]

    return _VOCAB_RE.sub(repl, text)


_SENT_START_RE = re.compile(r"(^|[.!?]\s+|\n\s*)([a-z])")
_I_STANDALONE_RE = re.compile(r"(?<![\w'])i(?![\w'])")
_I_CONTRACTION_RE = re.compile(r"(?<![\w])i(?='[a-z])", re.IGNORECASE)


def _capitalize_sentence_starts(text: str) -> str:
    """Capitalize the first letter of every sentence."""
    return _SENT_START_RE.sub(lambda m: m.group(1) + m.group(2).upper(), text)


def _fix_i_pronoun(text: str) -> str:
    """Capitalize standalone 'i' and 'i'm / i'll / i've' contractions."""
    text = _I_STANDALONE_RE.sub("I", text)
    text = _I_CONTRACTION_RE.sub("I", text)
    return text


def restore_casing(text: str) -> str:
    """Best-effort recase: sentence starts + I + proper-noun vocab.

    Idempotent — running it twice produces the same output.
    """
    if not text:
        return text
    out = _apply_vocab_casing(text)
    out = _capitalize_sentence_starts(out)
    out = _fix_i_pronoun(out)
    return out


# ── Punctuation restoration ─────────────────────────────────────────────────
# ``deepmultilingualpunctuation`` ships a thin wrapper around the model below
# but pins a deprecated ``grouped_entities`` argument that transformers 5.x has
# removed. So we drive the model directly with ``aggregation_strategy="first"``.
class Repunctuator:
    """Restore punctuation with ``oliverguhr/fullstop-punctuation-multilang-large``.

    Loads the model lazily on first call so import-time stays cheap. The model
    is ~440 MB at fp16 (or ~880 MB at fp32) — keep it on GPU when possible.
    """

    # Label vocabulary for the multilang model: see
    # https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large
    _LABEL_TO_PUNCT = {
        "0": "",
        ".": ".",
        ",": ",",
        "?": "?",
        "-": "-",
        ":": ":",
    }
    # Window size — model context limit is 512 tokens. Process long texts in
    # word-aligned windows with a small overlap to avoid losing token labels
    # near the seam.
    _WINDOW_WORDS = 200
    _OVERLAP_WORDS = 5

    def __init__(
        self,
        *,
        device: str | None = None,
        model_name: str = "oliverguhr/fullstop-punctuation-multilang-large",
    ) -> None:
        self.model_name = model_name
        self._device_pref = device or os.environ.get(
            "REPUNCTUATE_DEVICE", "cuda"
        )
        self._pipe = None

    def _ensure_pipe(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        device = self._device_pref
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        device_idx = 0 if device == "cuda" else -1

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self._pipe = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="first",
            device=device_idx,
        )
        return self._pipe

    def _restore_window(self, words: list[str]) -> str:
        """Run the model on a single ≤512-token window and stitch punctuation."""
        if not words:
            return ""
        pipe = self._ensure_pipe()
        joined = " ".join(words)
        try:
            preds = pipe(joined)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "punct pipe failed for %d words: %r — leaving unpunctuated",
                len(words),
                e,
            )
            return joined
        # Map each predicted span (start, end) back to whole-word indices so we
        # can append the appropriate punctuation after that word.
        word_starts: list[int] = []
        cur = 0
        for w in words:
            word_starts.append(cur)
            cur += len(w) + 1  # +1 for the space we added in ``joined``
        out_tokens: list[str] = list(words)
        for p in preds:
            label = p.get("entity_group") or p.get("entity") or "0"
            punct = self._LABEL_TO_PUNCT.get(label, "")
            if not punct:
                continue
            end = p.get("end")
            if end is None:
                continue
            # Find the last word whose start < end.
            idx = -1
            for i, s in enumerate(word_starts):
                if s < end:
                    idx = i
                else:
                    break
            if idx < 0:
                continue
            # Don't double-stack punctuation.
            if not out_tokens[idx].endswith(punct):
                out_tokens[idx] = out_tokens[idx] + punct
        return " ".join(out_tokens)

    def restore_punctuation(self, text: str) -> str:
        """Add sentence-ending punctuation. Casing is unchanged by this step.

        Chunks long inputs into ≤``_WINDOW_WORDS``-sized windows with a small
        overlap so we stay within the model's 512-token context.
        """
        if not text or not text.strip():
            return text
        words = text.split()
        if not words:
            return text
        if len(words) <= self._WINDOW_WORDS:
            return self._restore_window(words)
        out_parts: list[str] = []
        i = 0
        step = self._WINDOW_WORDS - self._OVERLAP_WORDS
        while i < len(words):
            window = words[i : i + self._WINDOW_WORDS]
            restored = self._restore_window(window)
            # Drop the leading overlap from windows after the first so we don't
            # double-count words.
            if i > 0:
                rw = restored.split(" ")
                rw = rw[self._OVERLAP_WORDS :]
                restored = " ".join(rw)
            out_parts.append(restored)
            i += step
        return " ".join(p for p in out_parts if p)

    def repunctuate(self, text: str) -> str:
        """End-to-end: strip fillers → restore punctuation → restore casing."""
        if not text or not text.strip():
            return text
        t = strip_fillers(text)
        t = self.restore_punctuation(t)
        t = restore_casing(t)
        return t

    def repunctuate_many(self, texts: Iterable[str]) -> list[str]:
        """Convenience: apply ``repunctuate`` over an iterable."""
        return [self.repunctuate(t) for t in texts]


def repunctuate(text: str, *, device: str | None = None) -> str:
    """Module-level helper. Builds a fresh model per call — not for bulk use."""
    return Repunctuator(device=device).repunctuate(text)


if __name__ == "__main__":
    sample = (
        "a six bit of the sixth one is just a bit of hubris that i hear it's "
        "a small flag but it can scale over time which is oh they're not a "
        "significant competitor i mean shopify started with snowboard sales "
        "i guess but um after that you start to compete with zendesk or or "
        "hubspot or datadogger or whoever it is you actually should start "
        "losing deals because of course you're feature poor"
    )
    rp = Repunctuator()
    print("INPUT:\n", sample)
    print()
    print("OUTPUT:\n", rp.repunctuate(sample))

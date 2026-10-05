"""The RAG source-copying gate is separate from training-memorization."""
import pytest

from host_finetune.rag_source_copying import (
    RAG_SOURCE_COPY_THRESHOLD,
    RETRY_SEED_OFFSET,
    assemble_copy_guard,
    copy_guard_seeds,
    evaluate_rag_source_copying,
    retry_seed,
)


def _words(n: int) -> str:
    names = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
        "india", "juliet", "kilo", "lima", "mike", "november", "oscar",
    ]
    return " ".join(names[:n])


def _report(answer: str, hits: list[dict], enabled: bool = True):
    return evaluate_rag_source_copying(answer, hits, enabled=enabled)


def test_more_than_twelve_copied_words_fail():
    text = _words(13)
    report = _report(text, [{"id": "train_a", "text": text}])
    assert report["pass"] is False
    assert report["max_contiguous_words"] == 13
    assert report["document_id"] == "train_a"
    assert report["matched_generated_span"] == text
    assert report["matched_retrieved_span"] == text


def test_exactly_threshold_length_fails():
    text = _words(RAG_SOURCE_COPY_THRESHOLD)
    report = _report(text, [{"id": "train_a", "text": f"intro {text} outro"}])
    assert RAG_SOURCE_COPY_THRESHOLD == 12
    assert report["pass"] is False
    assert report["max_contiguous_words"] == 12


def test_eleven_copied_words_pass():
    text = _words(11)
    report = _report(text, [{"id": "train_a", "text": text}])
    assert report["pass"] is True
    assert report["max_contiguous_words"] == 11


def test_case_and_punctuation_differences_still_match():
    source = "Monday.com grew 91 percent at five hundred fifty million in annual recurring revenue last year."
    answer = "MONDAY.COM GREW 91 PERCENT, AT FIVE HUNDRED FIFTY MILLION, IN ANNUAL RECURRING REVENUE LAST YEAR."
    report = _report(answer, [{"id": "train_b", "text": source}])
    assert report["pass"] is False
    assert report["max_contiguous_words"] >= 12
    assert report["document_id"] == "train_b"


def test_overlap_against_a_later_excerpt_is_detected():
    copied = _words(12)
    report = _report(
        f"Here is the post. {copied}",
        [
            {"id": "train_near", "text": "A short unrelated note about hiring."},
            {"id": "train_far", "text": copied},
        ],
    )
    assert report["pass"] is False
    assert report["document_id"] == "train_far"


def test_ordinary_short_shared_phrases_pass():
    report = _report(
        "The company should hire a leader and keep the customer.",
        [{"id": "train_c", "text": "The company reported earnings and the customer stayed."}],
    )
    assert report["pass"] is True
    assert report["max_contiguous_words"] < 12


def test_retrieval_off_does_not_apply_the_gate():
    copied = _words(14)
    report = _report(copied, [{"id": "train_a", "text": copied}], enabled=False)
    assert report["applicable"] is False
    assert report["pass"] is None
    assert report["max_contiguous_words"] is None
    assert report["document_id"] is None
    cell = assemble_copy_guard(
        {"seed": 42, "answer": copied, "report": report},
        None,
    )
    assert cell["deployment_accepted"] is True
    assert cell["retry_answer"] is None


def test_failed_attempt_is_preserved_when_retry_passes():
    copied = _words(12)
    fresh = "A different draft that does not repeat the source."
    cell = assemble_copy_guard(
        {"seed": 42, "answer": copied, "report": _report(copied, [{"id": "train_a", "text": copied}])},
        {"seed": retry_seed(42), "answer": fresh, "report": _report(fresh, [{"id": "train_a", "text": copied}])},
    )
    assert cell["first_attempt_answer"] == copied
    assert cell["retry_answer"] == fresh
    assert cell["answer"] == fresh
    assert cell["deployment_accepted"] is True
    assert cell["rag_source_copying"]["accepted_attempt"] == 2
    assert cell["rag_source_copying"]["first_attempt"]["pass"] is False


def test_retry_seed_is_the_fixed_offset():
    assert retry_seed(42) == 42 + RETRY_SEED_OFFSET
    assert RETRY_SEED_OFFSET == 10000
    assert copy_guard_seeds(49, first_passed=False) == [49, 49 + RETRY_SEED_OFFSET]


def test_at_most_one_retry_is_planned():
    assert copy_guard_seeds(42, first_passed=True) == [42]
    assert len(copy_guard_seeds(42, first_passed=False)) == 2
    with pytest.raises(ValueError):
        assemble_copy_guard(
            {"seed": 42, "answer": "ok", "report": _report("ok", [{"id": "t", "text": "other words"}])},
            {"seed": retry_seed(42), "answer": "again", "report": _report("again", [])},
        )

"""Unified evaluation gates, calibration, and selection."""
from host_finetune.eval_gold_rag import grounding_scores
from host_finetune.unified_eval import (
    REFUSED_DIAGNOSTICS,
    UTILITY_WEIGHTS,
    CopyIndex,
    completion_gate,
    copying_gate,
    evaluate_row,
    format_gate,
    persona_utility_score,
    rank_models,
    summarize_model,
)
from host_finetune.voice_distance import (
    GENERIC_PROSE,
    held_out_reference_texts,
    metric_calibrated,
    perspective_features,
    style_features,
)
from host_finetune.writing_quality import (
    WRITING_DIMENSIONS,
    calibration_gate,
    judge_prompt,
    parse_judge_scores,
    writing_quality_unavailable,
)


def test_held_out_excludes_train_gold_and_matches_medium():
    dataset = [
        {"output": "Train blog post about quotas."},
        {"output": "Valid blog post. I've seen this."},
        {"output": "Gold family blog."},
        {"output": "Valid linkedin post. We're hiring AEs."},
        {"output": "Continuation blog should not count."},
        {"output": "Test blog must stay out."},
    ]
    assignments = [
        {"split": "train", "group_id": "t", "source_platform": "blog", "sft_role": "unchanged"},
        {"split": "validation", "group_id": "v", "source_platform": "blog", "sft_role": "unchanged"},
        {"split": "validation", "group_id": "g", "source_platform": "blog", "sft_role": "unchanged"},
        {"split": "validation", "group_id": "l", "source_platform": "linkedin", "sft_role": "unchanged"},
        {"split": "validation", "group_id": "c", "source_platform": "blog", "sft_role": "continuation"},
        {"split": "test", "group_id": "h", "source_platform": "blog", "sft_role": "unchanged"},
    ]
    refs = held_out_reference_texts(dataset, assignments, {"g"})
    assert refs["blog"] == ["Valid blog post. I've seen this."]
    assert refs["linkedin"] == ["Valid linkedin post. We're hiring AEs."]
    assert refs["talk"] == []
    blob = " ".join(refs["blog"] + refs["linkedin"])
    assert "Train blog" not in blob
    assert "Gold family" not in blob
    assert "Test blog" not in blob


def test_calibration_separates_real_from_generic_and_fails_closed():
    assert metric_calibrated([0.2, 0.3, 0.25, 0.4], [1.4, 1.5, 1.6])["calibrated"] is True
    failed = metric_calibrated([1.2, 1.3, 1.1, 1.4], [0.2, 0.3, 0.4])
    assert failed["calibrated"] is False
    assert failed["reason"] == "real_not_closer_than_generic"
    assert metric_calibrated([0.2], [1.0, 1.1])["calibrated"] is False
    base_fail = metric_calibrated(
        [0.2, 0.3, 0.2, 0.4],
        [1.5, 1.6],
        base_distances=[0.1, 0.1],
    )
    assert base_fail["calibrated"] is False
    assert base_fail["reason"] == "real_not_closer_than_base"


def test_style_features_cover_the_requested_surface_counts():
    features = style_features("So I've seen 120% NRR. Really? We're not done.\n\nNext paragraph.")
    for name in (
        "words_per_sentence",
        "questions_per_100w",
        "contractions_per_100w",
        "first_person_per_100w",
        "numbers_per_100w",
        "punctuation_per_100w",
        "type_token_ratio",
        "paragraphs_per_100w",
        "fw_the_per_100w",
    ):
        assert name in features
    perspective = perspective_features("You should watch churn, but the tradeoff is pipeline.")
    assert perspective["advice_cues_per_100w"] > 0
    assert perspective["saas_terms_per_100w"] > 0
    assert "words_per_sentence" not in perspective


def test_writing_quality_schema_and_calibration_gate():
    prompt = judge_prompt("Draft text.", "pricing", "blog")
    for name in WRITING_DIMENSIONS:
        assert name in prompt
    assert parse_judge_scores({name: 4 for name in WRITING_DIMENSIONS}) is None
    full = {name: 4 for name in (*WRITING_DIMENSIONS, "voice")}
    assert parse_judge_scores(full)["coherence"] == 4.0
    good = calibration_gate(
        [
            {"bucket": "real", "voice": 5, "coherence": 5},
            {"bucket": "good_persona", "voice": 4, "coherence": 5},
            {"bucket": "base", "voice": 2, "coherence": 4},
            {"bucket": "degraded", "voice": 1, "coherence": 1},
        ]
    )
    assert good["calibrated"] is True
    assert good["role"] == "selector"
    bad = calibration_gate(
        [
            {"bucket": "real", "voice": 2, "coherence": 5},
            {"bucket": "base", "voice": 5, "coherence": 4},
            {"bucket": "degraded", "voice": 1, "coherence": 1},
        ]
    )
    assert bad["calibrated"] is False
    assert bad["role"] == "diagnostic_only"
    unavailable = writing_quality_unavailable()
    assert unavailable["role"] == "diagnostic_only"
    assert unavailable["mean"] is None


def test_completion_copying_and_format_gates():
    finished = completion_gate({"answer": "Ships.", "early_eos": True, "stop_reason": "eos"})
    assert finished["pass"] is True
    assert finished["early_eot"] is True
    cutoff = completion_gate({"answer": "cut off mid", "early_eos": False, "stop_reason": "length"})
    assert cutoff["pass"] is False
    assert cutoff["mid_sentence_stop"] is True
    missing = completion_gate({"answer": "Ships."})
    assert missing["early_eot_available"] is False

    copied = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
    index = CopyIndex([copied])
    fail = copying_gate({"answer": copied, "prompt": "Write about pricing"}, index)
    assert fail["pass"] is False
    assert fail["longest_shared_word_span"] >= 12
    allowed = copying_gate({"answer": copied, "prompt": copied}, index)
    assert allowed["pass"] is True
    missing_index = copying_gate({"answer": "Ships today.", "prompt": "Write"}, None)
    assert missing_index["pass"] is False
    assert missing_index["available"] is False

    short_x = format_gate({"medium": "x", "answer": "Hire the AE.", "prompt": "Write an X post"})
    assert short_x["pass"] is True
    assert "Post 1" not in short_x["rule"]


def test_writing_quality_gate_requires_degraded_separation():
    from host_finetune.writing_quality import writing_quality_gate

    passed = writing_quality_gate([
        {"bucket": "real", "coherence": 5},
        {"bucket": "normal", "coherence": 4},
        {"bucket": "degraded", "coherence": 1},
    ])
    assert passed["role"] == "selector"
    failed = writing_quality_gate([
        {"bucket": "real", "coherence": 2},
        {"bucket": "normal", "coherence": 2},
        {"bucket": "degraded", "coherence": 4},
    ])
    assert failed["role"] == "diagnostic_only"


def test_retrieval_off_has_no_rag_grounding_and_stance_can_score_utility():
    from host_finetune.claim_judge import (
        evidence_report,
        parse_claims,
        rag_grounding_report,
        stance_calibration_gate,
        stance_report,
    )
    from host_finetune.judge_client import judge_family
    from host_finetune.unified_eval import apply_judge_overlay

    try:
        judge_family("llama3.1")
    except ValueError:
        refused = True
    else:
        refused = False
    assert refused
    assert judge_family("hf.co/QuantFactory/Qwen2.5-7B-GGUF:latest") == "qwen"
    claims = parse_claims(
        {"claims": [{"text": "Charge for the product.", "kind": "advice", "rag": "supported", "evidence": "supported", "stance": "consistent", "confidence": 0.8}]},
        retrieval_on=False,
    )
    assert claims[0]["rag"] == "not_applicable"
    assert rag_grounding_report(claims, False)["rag_grounding"] is None
    assert stance_calibration_gate(0.8, 0.1)["role"] == "selector"
    assert stance_calibration_gate(0.2, 0.8)["role"] == "diagnostic_only"
    words = " ".join(["word"] * 40) + "."
    pack = {
        "media": {"blog": {
            "style_profile": {"words_per_sentence": {"mean": 8.0, "stdev": 2.0}},
            "perspective_profile": {"advice_cues_per_100w": {"mean": 1.0, "stdev": 1.0}},
            "style_role": "selector",
            "perspective_role": "diagnostic_only",
        }},
        "writing_quality_calibration": {"calibrated": True, "role": "selector"},
    }
    base = evaluate_row(
        {
            "id": "gold_001", "medium": "blog", "topic": "pricing", "goal": "teach",
            "audience": "founders", "answer": words,
            "prompt": "Write a blog post in the style of Jason Lemkin about: pricing",
            "gold_reference": "A strong answer argues about pricing.",
            "expected_facts": ["pricing"], "retrieval": False,
        },
        pack,
        CopyIndex(["unrelated training sentence about something else entirely here now."]),
        {},
    )
    writing = {
        "available": True, "mean": 4.0, "dimensions": {"coherence": 4},
        "judge_model": "qwen", "judge_family_differs_from_candidate": True,
    }
    overlaid = apply_judge_overlay(
        base, writing,
        rag_grounding_report(claims, False),
        evidence_report(claims, 2),
        stance_report(claims),
        writing_role="selector", writing_calibrated=True, stance_role="selector",
    )
    assert overlaid["grounding"]["rag_grounding"] is None
    assert overlaid["diagnostics"]["retrieved_lexical_overlap"] is None or isinstance(
        overlaid["diagnostics"]["retrieved_lexical_overlap"], float
    )
    assert overlaid["lemkin_evidence_support"]["not_rag_grounding"] is True
    assert overlaid["perspective_fidelity"]["framing_role"] == "diagnostic_only"
    assert overlaid["perspective_fidelity"]["stance_consistency_rate"] == 1.0
    assert overlaid["persona_utility"] is not None


def test_retrieval_hits_fallback_is_not_called_faithfulness():
    row = {
        "answer": "The NRR was 120 percent and founders should hire.",
        "query": "NRR",
        "retrieval": True,
        "contexts": [],
        "retrieval_hits": [{
            "text": "The NRR was 120 percent and founders should hire more AEs this year.",
        }],
    }
    scored = grounding_scores(row)
    assert scored["passage_source"] == "retrieval_hits"
    assert scored["lexical_overlap"] > 0
    assert "faithfulness" not in scored
    assert scored["not_claim_entailment"] is True
    absent = grounding_scores({"answer": "Hello there.", "query": "Hello", "retrieval": False})
    assert absent["applicable"] is False
    assert absent["lexical_overlap"] is None


def test_composite_requires_gates_and_calibrated_components():
    assert persona_utility_score(
        0.2, 0.3, 4.0, 0.8,
        eligible=False, style_role="selector", perspective_role="selector", writing_role="selector",
    ) is None
    assert persona_utility_score(
        0.2, 0.3, None, 0.8,
        eligible=True, style_role="selector", perspective_role="selector", writing_role="diagnostic_only",
    ) is None
    value = persona_utility_score(
        0.0, 0.0, 5.0, 1.0,
        eligible=True, style_role="selector", perspective_role="selector", writing_role="selector",
    )
    assert value == round(sum(UTILITY_WEIGHTS.values()), 4)


def test_bleu_is_reference_overlap_without_nltk():
    from host_finetune.unified_eval import bleu, rouge_l

    text = "Founders should hire the account executive after the motion works."
    assert bleu(text, text) > 0.9
    assert rouge_l(text, text) == 1
    assert bleu(text, "The committee convened regarding procedural compliance today.") < 0.2


def test_selection_refuses_diagnostic_overlap():
    diagnostic = {
        "model_id": "high_bleu",
        "n_eligible": 10,
        "persona_style": {"role": "diagnostic_only", "mean_distance_eligible": 0.1},
        "perspective_fidelity": {"role": "diagnostic_only", "mean_distance_eligible": 0.1},
        "writing_quality": {"role": "diagnostic_only", "mean_eligible": 5, "available": False},
        "task_adherence": {"mean_eligible": 0.9},
        "persona_utility": None,
        "diagnostics": {"reference_similarity": {"bleu": 0.8}},
    }
    closer = {
        "model_id": "closer_voice",
        "n_eligible": 10,
        "persona_style": {"role": "selector", "mean_distance_eligible": 0.4},
        "perspective_fidelity": {"role": "selector", "mean_distance_eligible": 0.5},
        "writing_quality": {"role": "diagnostic_only", "mean_eligible": None, "available": False},
        "task_adherence": {"mean_eligible": 0.4},
        "persona_utility": None,
        "diagnostics": {"reference_similarity": {"bleu": 0.01}},
    }
    farther = dict(closer)
    farther["model_id"] = "farther_voice"
    farther["persona_style"] = {"role": "selector", "mean_distance_eligible": 1.2}
    refused = rank_models([diagnostic])
    assert refused["selected"] is None
    assert refused["reason"] == "persona_style_not_calibrated"
    assert "bleu" in refused["refused_diagnostics"]
    chosen = rank_models([diagnostic, farther, closer])
    assert chosen["selected"] == "closer_voice"
    assert chosen["used_for_selection"][0] == "persona_style"
    assert not set(chosen["used_for_selection"]) & set(REFUSED_DIAGNOSTICS)


def test_evaluate_row_keeps_families_separate():
    words = " ".join(["word"] * 30) + "."
    profile = {
        "words_per_sentence": {"mean": 8.0, "stdev": 2.0},
        "questions_per_100w": {"mean": 1.0, "stdev": 1.0},
    }
    pack = {
        "media": {
            "blog": {
                "style_profile": profile,
                "perspective_profile": {"advice_cues_per_100w": {"mean": 2.0, "stdev": 1.0}},
                "style_role": "diagnostic_only",
                "perspective_role": "diagnostic_only",
            }
        },
        "writing_quality_calibration": {"calibrated": False, "role": "diagnostic_only"},
    }
    row = evaluate_row(
        {
            "id": "gold_001",
            "medium": "blog",
            "topic": "pricing",
            "goal": "teach",
            "audience": "founders",
            "answer": words,
            "prompt": "Write a blog post in the style of Jason Lemkin about: pricing",
            "gold_reference": "A strong answer argues about pricing.",
            "expected_facts": ["pricing"],
            "retrieval": False,
        },
        pack,
        CopyIndex(["unrelated training sentence about something else entirely."]),
        {},
    )
    assert set(row) >= {
        "gates", "persona_style", "perspective_fidelity", "writing_quality",
        "task_adherence", "grounding", "diagnostics", "persona_utility",
    }
    assert row["diagnostics"]["brief_overlap"]["role"] == "brief_task_agreement_not_persona_quality"
    assert row["persona_utility"] is None
    summary = summarize_model([row], "toy", "EXP-test")
    assert summary["perspective_fidelity"]["method"] == "framing_rate_distance_vs_held_out"
    assert summary["writing_quality"]["role"] == "diagnostic_only"
    assert "gold_rubric_overall" in str(summary["diagnostics"])

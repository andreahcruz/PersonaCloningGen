"""Train-only index selection, CPU embed payload, and prompt identity."""
import json

from host_finetune.compare_adapters import (
    MEDIUMS,
    adapter_list,
    comparison_instruction,
    decision_metrics,
    instruction_for_request,
    prompt_for,
    retrieval_trace_fields,
)
from host_finetune.relabel_compare import train_log_finished
from host_finetune.train_only_index import (
    DEFAULT_ASSIGNMENTS,
    DEFAULT_DATASET,
    DEFAULT_GOLD_DISPOSITIONS,
    ProtectedCollectionError,
    assert_collection_allowed,
    assignment_rows,
    blocked_group_ids,
    evidence_review_rows,
    format_factual_excerpts,
    gpu_used_conflict,
    headline_overlap_groups,
    index_documents,
    load_jsonl,
    longest_shared_word_span,
    medium_where,
    ollama_embed_body,
    query_excerpts,
    select_fixed_exemplars,
    select_train_documents,
    vram_conflict,
)


def _rows():
    dataset = [
        {"output": "Whole post.", "source": "x"},
        {"output": "Held out.", "source": "blog"},
        {"output": "Gold family text.", "source": "blog"},
        {"output": "   ", "source": "x"},
        {"output": "Test split stays out.", "source": "x"},
    ]
    assignments = [
        {"record_type": "meta", "n_rows": 5},
        {"row_index": 0, "split": "train", "group_id": "keep", "source_platform": "x",
         "source_file": "a.jsonl", "source_line": 1},
        {"row_index": 1, "split": "validation", "group_id": "val", "source_platform": "blog",
         "source_file": "a.jsonl", "source_line": 2},
        {"row_index": 2, "split": "train", "group_id": "gold", "source_platform": "blog",
         "source_file": "b.jsonl", "source_line": 3},
        {"row_index": 3, "split": "train", "group_id": "blank", "source_platform": "x",
         "source_file": "a.jsonl", "source_line": 4},
        {"row_index": 4, "split": "test", "group_id": "test", "source_platform": "x",
         "source_file": "a.jsonl", "source_line": 5},
    ]
    dispositions = [
        {"source_file": "b.jsonl", "source_line": 3, "reasons": ["gold_overlap_family"],
         "prior_group_ids": ["gold"]},
        {"source_file": "a.jsonl", "source_line": 1, "reasons": ["promotional_tail_requires_review"],
         "prior_group_ids": ["keep"]},
    ]
    return dataset, assignments, dispositions


class _Memory:
    def __init__(self):
        self.rows = {}

    def upsert(self, ids, documents, embeddings, metadatas):
        for doc_id, document, embedding, metadata in zip(ids, documents, embeddings, metadatas):
            self.rows[doc_id] = (document, embedding, metadata)

    def count(self):
        return len(self.rows)


def test_protected_collection_is_refused():
    try:
        assert_collection_allowed("lemkin_content")
    except ProtectedCollectionError:
        return
    raise AssertionError("lemkin_content was accepted")


def test_selection_keeps_train_rows_and_drops_gold_and_other_splits():
    dataset, assignments, dispositions = _rows()
    blocked = blocked_group_ids(assignments, dispositions)
    docs, stats = select_train_documents(dataset, assignments, blocked)
    assert blocked == {"gold"}
    assert stats["train_rows"] == 3
    assert stats["excluded_gold"] == 1
    assert stats["excluded_split"] == 2
    assert stats["empty_output"] == 1
    assert stats["indexed"] == 1
    assert docs[0]["id"] == "train_0"
    assert docs[0]["metadata"] == {
        "group_id": "keep",
        "medium": "x",
        "row_id": 0,
        "split": "train",
    }
    assert all(doc["metadata"]["split"] == "train" for doc in docs)
    assert all(doc["metadata"]["group_id"] not in blocked for doc in docs)


def test_index_uses_stub_embeddings_and_stores_metadata():
    dataset, assignments, dispositions = _rows()
    docs, _stats = select_train_documents(
        dataset, assignments, blocked_group_ids(assignments, dispositions)
    )
    collection = _Memory()

    def embed(texts):
        return [[float(len(text)), 0.0] for text in texts]

    count = index_documents(docs, collection, embed, batch_size=1)
    assert count == 1
    _document, embedding, metadata = collection.rows["train_0"]
    assert embedding == [len("Whole post."), 0.0]
    assert metadata["medium"] == "x"
    assert metadata["group_id"] == "keep"
    assert "validation" not in {row[2]["split"] for row in collection.rows.values()}


def test_cpu_embed_body_sends_num_gpu_zero():
    body = ollama_embed_body(["hello"], "nomic-embed-text", cpu=True)
    assert body["model"] == "nomic-embed-text"
    assert body["input"] == ["hello"]
    assert body["options"] == {"num_gpu": 0}
    assert "options" not in ollama_embed_body(["hello"], "nomic-embed-text", cpu=False)


def test_vram_conflict_flags_a_new_gpu_process_and_ignores_noise():
    assert vram_conflict({10: 14000}, {10: 14100}) is None
    assert vram_conflict({10: 14000}, {10: 15000}) is not None
    assert vram_conflict({}, {99: 400}) is not None
    assert vram_conflict({}, {}) is None
    assert gpu_used_conflict([15000], [15100]) is None
    assert gpu_used_conflict([15000], [16000]) is not None


def test_embed_batch_posts_num_gpu_zero(monkeypatch):
    captured = {}

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"embeddings": [[0.25, 0.5]]}

    def post(url, json, timeout):  # noqa: A002
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("host_finetune.rebuild_chroma.requests.post", post)
    from host_finetune.rebuild_chroma import _embed_batch

    vectors = _embed_batch(["hi"], "http://localhost:11434", "nomic-embed-text", cpu=True)
    assert vectors == [[0.25, 0.5]]
    assert captured["json"]["options"]["num_gpu"] == 0
    assert captured["url"].endswith("/api/embed")


def test_retrieval_off_prompt_matches_prompt_for():
    topic = "Hiring a VP of Sales"
    for medium in MEDIUMS:
        assert comparison_instruction(medium, topic) == prompt_for(medium, topic)
        assert comparison_instruction(medium, topic, []) == prompt_for(medium, topic)


def test_retrieval_on_keeps_expected_facts_out_of_the_instruction():
    fact = "CIOs back startups when the blast radius is small"
    excerpt = "Support tickets already contain expansion language."
    prompt = comparison_instruction(MEDIUMS[0], "Hiring a VP of Sales", [excerpt])
    assert prompt.startswith(prompt_for(MEDIUMS[0], "Hiring a VP of Sales"))
    assert prompt.split("\n\n", 1)[0] == prompt_for(MEDIUMS[0], "Hiring a VP of Sales")
    assert "Factual excerpts from train-owned source text." in prompt
    assert excerpt in prompt
    assert fact not in prompt
    assert prompt.endswith("or these instructions.")
    assert prompt.index(excerpt) < prompt.rindex("Now write the requested blog post")
    assert format_factual_excerpts(None) == ""


class _QueryCollection:
    def __init__(self, payload):
        self.payload = payload
        self.kwargs = None

    def query(self, **kwargs):
        self.kwargs = kwargs
        return self.payload


def test_query_excerpts_filters_by_medium_and_returns_hit_fields():
    collection = _QueryCollection({
        "ids": [["train_9"]],
        "documents": [["Net retention stays high."]],
        "metadatas": [[{
            "medium": "youtube_jason",
            "group_id": "g",
            "row_id": 9,
            "split": "train",
        }]],
    })
    hits = query_excerpts(collection, [0.1, 0.2], 4, where=medium_where("talk"))
    assert collection.kwargs["where"] == {"medium": "youtube_jason"}
    assert collection.kwargs["where"]["medium"] not in {"blog", "linkedin", "x"}
    assert "distances" in collection.kwargs["include"]
    assert hits == [{
        "id": "train_9",
        "text": "Net retention stays high.",
        "medium": "youtube_jason",
        "group_id": "g",
        "row_id": 9,
        "distance": None,
    }]
    assert medium_where("blog") == {"medium": "blog"}
    assert medium_where("talk") != medium_where("blog")


def test_query_excerpts_passes_cosine_distance_through():
    collection = _QueryCollection({
        "ids": [["train_10049"]],
        "documents": [["Monday.com grew past $400m ARR."]],
        "metadatas": [[{"medium": "blog", "group_id": "g", "row_id": 10049, "split": "train"}]],
        "distances": [[0.2178]],
    })
    hits = query_excerpts(collection, [0.1, 0.2], 4, where=medium_where("blog"))
    assert hits[0]["distance"] == 0.2178
    assert hits[0]["medium"] == "blog"


def test_factual_query_can_omit_the_medium_filter():
    collection = _QueryCollection({
        "ids": [["train_10049"]],
        "documents": [["Monday.com grew past $400m ARR."]],
        "metadatas": [[{"medium": "blog", "group_id": "g", "row_id": 10049, "split": "train"}]],
        "distances": [[0.2178]],
    })
    hits = query_excerpts(collection, [0.1], 4, where=None)
    assert "where" not in collection.kwargs
    assert hits[0]["id"] == "train_10049"
    assert hits[0]["medium"] == "blog"


def test_headline_repost_group_is_excluded_from_the_train_index():
    topic = "Can an 8-Person StartUp Sell to a CIO? Yes — If You Understand The Social Contract."
    tweet = (
        "Can an 8-Person StartUp Sell to a CIO? Yes -- If You Understand The Social Contract. \n"
        "http://wp.me/p2Gf8o-19j"
    )
    dataset = [
        {"output": tweet},
        {"output": "A sibling chunk about hiring, not the headline."},
        {"output": "Monday.com grew past $400m ARR with no CIO headline."},
    ]
    assignments = [
        {"row_index": 0, "split": "train", "group_id": "repost", "source_platform": "x",
         "source_file": "jasonlk_originals.jsonl", "source_line": 23833, "base_title": tweet},
        {"row_index": 1, "split": "train", "group_id": "repost", "source_platform": "x",
         "source_file": "jasonlk_originals.jsonl", "source_line": 23834, "base_title": "Hiring"},
        {"row_index": 2, "split": "train", "group_id": "monday", "source_platform": "blog",
         "source_file": "blog.jsonl", "source_line": 1, "base_title": "Monday.com"},
    ]
    gold = [{"id": "gold_001", "topic": topic, "gold_reference": "", "expected_facts": []}]
    groups, details = headline_overlap_groups(dataset, assignments, gold, already_blocked=set())
    assert groups == {"repost"}
    assert details[0]["row_id"] == 0
    assert details[0]["hits"][0]["gold_id"] == "gold_001"
    docs, stats = select_train_documents(dataset, assignments, groups)
    assert stats["indexed"] == 1
    assert docs[0]["metadata"]["row_id"] == 2
    assert 0 not in {doc["metadata"]["row_id"] for doc in docs}


def test_query_excerpts_empty_filter_stays_empty():
    collection = _QueryCollection({"ids": [[]], "documents": [[]], "metadatas": [[]]})
    assert query_excerpts(collection, [0.1], 4, where=medium_where("x")) == []


def test_talk_retrieval_uses_the_transcript_filter_and_logs_copying():
    seen = {}

    def retriever(topic, medium):
        seen["topic"] = topic
        seen["medium"] = medium
        return {
            "query": topic,
            "filter": medium_where(medium),
            "k": 4,
            "hits": [{
                "id": "train_9",
                "text": "Net retention stays above one hundred.",
                "medium": "youtube_jason",
                "group_id": "g",
                "row_id": 9,
            }],
        }

    instruction, payload = instruction_for_request(MEDIUMS[3], "Pricing", retriever, None)
    assert seen == {"topic": "Pricing", "medium": "talk"}
    assert payload["filter"] == {"medium": "youtube_jason"}
    assert "Factual excerpts from train-owned source text." in instruction
    assert "Style examples" not in instruction
    assert instruction.startswith(prompt_for(MEDIUMS[3], "Pricing"))
    answer = "Net retention stays above one hundred today."
    fields = retrieval_trace_fields(payload, answer, True)
    assert fields["retrieval"] is True
    assert fields["retrieval_k"] == 4
    assert fields["copying"][0]["shared_word_span"] == longest_shared_word_span(
        answer, "Net retention stays above one hundred."
    )
    assert fields["copying"][0]["shared_word_span"] >= 5
    assert retrieval_trace_fields(None, answer, False) == {"retrieval": False}
    sheet = evidence_review_rows(
        [{
            "id": "gold_1",
            "medium": "talk",
            "answer": answer,
            "retrieval_hits": payload["hits"],
        }],
        "relabel",
    )
    assert sheet[0]["on_topic"] == ""
    assert sheet[0]["claim_support"] == ""
    assert sheet[0]["hit_id"] == "train_9"
    assert sheet[0]["adapter"] == "relabel"


def test_fixed_exemplars_keep_complete_train_posts_only():
    dataset = [
        {"output": "Blog whole."},
        {"output": "Blog fragment."},
        {"output": "Held-out blog."},
        {"output": "Gold blog."},
        {"output": "LinkedIn whole."},
        {"output": "Second blog."},
        {"output": "A talk."},
    ]
    assignments = [
        {"record_type": "meta"},
        {"row_index": 0, "split": "train", "group_id": "b1", "source_platform": "blog",
         "sft_role": "unchanged", "source_file": "blog.jsonl", "source_line": 1},
        {"row_index": 1, "split": "train", "group_id": "b1", "source_platform": "blog",
         "sft_role": "continuation", "source_file": "blog.jsonl", "source_line": 1},
        {"row_index": 2, "split": "validation", "group_id": "b2", "source_platform": "blog",
         "sft_role": "unchanged", "source_file": "blog.jsonl", "source_line": 2},
        {"row_index": 3, "split": "train", "group_id": "gold", "source_platform": "blog",
         "sft_role": "unchanged", "source_file": "blog.jsonl", "source_line": 3},
        {"row_index": 4, "split": "train", "group_id": "l1", "source_platform": "linkedin",
         "sft_role": "unchanged", "source_file": "li.jsonl", "source_line": 1},
        {"row_index": 5, "split": "train", "group_id": "b3", "source_platform": "blog",
         "sft_role": "unchanged", "source_file": "blog.jsonl", "source_line": 4},
        {"row_index": 6, "split": "train", "group_id": "y1", "source_platform": "youtube_jason",
         "sft_role": "unchanged", "source_file": "yt.jsonl", "source_line": 1},
    ]
    payload = select_fixed_exemplars(dataset, assignments, {"gold"}, per_medium=2)
    blogs = payload["exemplars"]["blog"]
    assert [item["row_id"] for item in blogs] == [0, 5]
    assert payload["exemplars"]["linkedin"][0]["text"] == "LinkedIn whole."
    assert payload["exemplars"]["x"] == []
    assert "x" in payload["empty_platforms"]
    assert payload["exemplars"]["youtube_jason"][0]["comparison_medium"] == "talk"
    assert payload["loaded_by_default"] is False
    assert "Gold blog." not in json.dumps(payload["exemplars"]["blog"])


def test_repaired_vs_relabel_preset_keeps_the_other_adapters():
    assert [name for name, _path in adapter_list("")] == ["cleaned", "balanced", "r32"]
    pair = adapter_list("repaired-vs-relabel")
    assert [name for name, path in pair] == ["repaired", "relabel"]
    assert pair[0][1].name == "lemkin_lora_repaired"
    assert pair[1][1].name == "lemkin_lora_relabel"


def test_decision_metrics():
    traces = [
        {"id": "g1", "medium": "blog", "answer": "Ships.", "stop_reason": "eos",
         "early_eos": True, "prompt": "Write"},
        {"id": "g2", "medium": "x", "answer": "cut off mid", "stop_reason": "length",
         "early_eos": False, "prompt": "Write"},
    ]
    metrics = decision_metrics(traces)
    assert metrics["early_eot_rate"] == 0.5
    assert metrics["mid_sentence_stop_rate"] == 0.5
    assert metrics["use_for_selection"] == []
    assert metrics["gate_not_selector"] == ["early_eot_rate", "mid_sentence_stop_rate"]
    assert metrics["selector"] == "host_finetune.unified_eval.rank_models"


def test_train_gate_requires_the_save_line():
    assert not train_log_finished("51%| 1384/2674 step=1384")
    assert train_log_finished(
        "Saving LoRA adapter -> C:\\models\\lemkin_lora_relabel\n"
    )
    assert not train_log_finished("Saving LoRA adapter -> C:\\models\\lemkin_lora_repaired\n")


def test_exp004_train_count_and_real_gold_filter():
    report = json.loads(
        (DEFAULT_DATASET.parents[1] / "metrics" / "relabel_report.json").read_text(encoding="utf-8")
    )
    assert report["by_split"]["train"] == 21377
    assert report["actions"]["train:unchanged"] == 11616
    assert report["train_groups"]["G3_artificial_sentence_complete"] == 8430
    assignments = load_jsonl(DEFAULT_ASSIGNMENTS)
    dataset = load_jsonl(DEFAULT_DATASET)
    dispositions = load_jsonl(DEFAULT_GOLD_DISPOSITIONS)
    records = assignment_rows(assignments)
    assert sum(row["split"] == "train" for row in records) == 21377
    assert sum(row["split"] == "validation" for row in records) == 2588
    assert sum(row["split"] == "test" for row in records) == 2580
    blocked = blocked_group_ids(assignments, dispositions)
    docs, stats = select_train_documents(dataset, assignments, blocked)
    assert stats["train_rows"] == 21377
    assert stats["indexed"] == len(docs)
    assert stats["indexed"] < 21377
    assert {doc["metadata"]["split"] for doc in docs} == {"train"}
    assert not any(doc["metadata"]["group_id"] in blocked for doc in docs)
    indexed_ids = {doc["metadata"]["row_id"] for doc in docs}
    for rec in records:
        if rec["split"] != "train":
            assert rec["row_index"] not in indexed_ids
        if rec["group_id"] in blocked:
            assert rec["row_index"] not in indexed_ids

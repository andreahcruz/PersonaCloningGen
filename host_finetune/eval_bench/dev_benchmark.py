"""Development retrieval questions. Grades stay empty.

Candidate ids come from the frozen pilot retrieval files and from the
preselected ids already written in the review files. A preselected id is a
candidate, not a grade. New questions have an empty pool until a person adds
a source. This module does not open a split file or the live index.
"""

from __future__ import annotations

import json
from pathlib import Path

ANNOTATION_SCHEMA = "retrieval-annotation-v2"
GRADE_SCALE = (
    {"grade": 0, "meaning": "irrelevant"},
    {"grade": 1, "meaning": "tangential"},
    {"grade": 2, "meaning": "relevant"},
    {"grade": 3, "meaning": "highly relevant"},
)
PHENOMENA = (
    "factual",
    "entity_identification",
    "direct_relationship",
    "multi_hop",
    "temporal_comparison",
    "conflicting_evidence",
    "attribution",
    "unanswerable",
)

# Question text only. Review notes stay in the withheld file.
QUESTIONS = (
    {"id": "q_brex_price", "phenomenon": "factual", "query": "What price did Capital One announce for acquiring Brex?", "review_file": "host_finetune/graph_relationship_questions.json"},
    {"id": "q_five_percent_share", "phenomenon": "factual", "query": "What does Jason Lemkin say a company should do once it has about 5 percent market share in a niche?", "review_file": "host_finetune/graph_relationship_questions.json"},
    {"id": "q_dev_fact_acquirer", "phenomenon": "factual", "query": "Which company announced an acquisition of Brex?", "review_file": None},
    {"id": "q_dev_fact_acquirer_target", "phenomenon": "factual", "query": "Which company is named as the company Capital One announced it would acquire?", "review_file": None},
    {"id": "q_worst_mishire", "phenomenon": "entity_identification", "query": "Which leadership hire does Jason Lemkin call the worst mis-hire?", "review_file": "host_finetune/pilot_questions.json"},
    {"id": "q_dev_entity_acquisition_parties", "phenomenon": "entity_identification", "query": "Which companies are named as parties to the Brex acquisition?", "review_file": None},
    {"id": "q_dev_entity_speaker", "phenomenon": "entity_identification", "query": "Which person is quoted on personalized experiences in SaaS?", "review_file": None},
    {"id": "q_dev_entity_sales_role", "phenomenon": "entity_identification", "query": "Which leadership role is discussed as a first sales hire?", "review_file": None},
    {"id": "q_cvc_second_check", "phenomenon": "direct_relationship", "query": "What does Jason Lemkin say about whether corporate venture capital firms write a second check?", "review_file": "host_finetune/graph_relationship_questions.json"},
    {"id": "q_exec_craft", "phenomenon": "direct_relationship", "query": "What does the pilot source say happens to executives who stay close to the craft, compared with executives who become pure managers and strategists?", "review_file": "host_finetune/graph_relationship_questions.json"},
    {"id": "q_cap_table", "phenomenon": "direct_relationship", "query": "How does a messy cap table affect future investors and fundraising?", "review_file": "host_finetune/graph_relationship_questions.json"},
    {"id": "q_investor_update", "phenomenon": "direct_relationship", "query": "What does a late investor update indicate about the month it describes?", "review_file": "host_finetune/graph_relationship_questions.json"},
    {"id": "q_dev_rel_director_promotion", "phenomenon": "direct_relationship", "query": "What promotion does the 2021-boom source recommend for a director?", "review_file": None},
    {"id": "q_first_vp_sales", "phenomenon": "multi_hop", "query": "What does Jason Lemkin say about hiring and managing a first VP of Sales across the posts that discuss that role?", "review_file": "host_finetune/graph_relationship_questions.json"},
    {"id": "q_dev_hop_valuation_and_price", "phenomenon": "multi_hop", "query": "How is the earlier Brex valuation connected to the later announced acquisition price?", "review_file": None},
    {"id": "q_dev_hop_share_and_playbook", "phenomenon": "multi_hop", "query": "Does advice about reaching about 5 percent niche share connect to advice about whether the B2C playbook works in B2B?", "review_file": None},
    {"id": "q_dev_hop_update_and_cap_table", "phenomenon": "multi_hop", "query": "Does a late investor update connect to the effect of a messy cap table, or are those separate sources?", "review_file": None},
    {"id": "q_dev_hop_vp_and_craft", "phenomenon": "multi_hop", "query": "Is advice about a first VP of Sales connected by a source to advice about executives who leave the craft?", "review_file": None},
    {"id": "q_boom_promotions", "phenomenon": "temporal_comparison", "query": "During the 2021 boom, what does the pilot source say a company should do about promoting a director and about mediocre customer success people?", "review_file": "host_finetune/graph_relationship_questions.json"},
    {"id": "q_dev_time_valuation_vs_price", "phenomenon": "temporal_comparison", "query": "How does the Brex valuation associated with 2022 compare with the later announced acquisition price?", "review_file": None},
    {"id": "q_dev_time_boom_vs_later", "phenomenon": "temporal_comparison", "query": "Which hiring advice is tied to the 2021 boom, and which hiring advice is not given a 2021 date?", "review_file": None},
    {"id": "q_dev_time_two_updates", "phenomenon": "temporal_comparison", "query": "If one investor update comes later than another, what difference does the source draw between them?", "review_file": None},
    {"id": "q_vp_sales_tension", "phenomenon": "conflicting_evidence", "query": "Does Jason Lemkin say a great VP of Sales should immediately bring in more sales executives, or that criticizing a VP of Sales who cannot do better is counterproductive?", "review_file": "host_finetune/graph_relationship_questions.json"},
    {"id": "q_b2c_playbook", "phenomenon": "conflicting_evidence", "query": "Does the B2C playbook work as well in B2B, and when does a consumer-style approach still matter?", "review_file": "host_finetune/graph_relationship_questions.json"},
    {"id": "q_dev_conflict_second_check", "phenomenon": "conflicting_evidence", "query": "Where the source says corporate venture firms often do not write a second check, what exceptions does it also state?", "review_file": None},
    {"id": "q_dev_conflict_craft", "phenomenon": "conflicting_evidence", "query": "Do the sources agree on whether executives should stay close to the craft or become pure managers?", "review_file": None},
    {"id": "q_personalized_saas_speaker", "phenomenon": "attribution", "query": "Who said that AI means everything in SaaS needs to be a personalized experience?", "review_file": "host_finetune/graph_relationship_questions.json"},
    {"id": "q_dev_attr_speaker_role", "phenomenon": "attribution", "query": "What role or employer, if any, does the source give the person quoted on personalized SaaS experiences?", "review_file": None},
    {"id": "q_dev_attr_lemkin_or_speaker", "phenomenon": "attribution", "query": "Which sentences in the personalized-SaaS discussion are the quoted person's, and which are Jason Lemkin's?", "review_file": None},
    {"id": "q_dev_attr_second_check_speaker", "phenomenon": "attribution", "query": "Who is presented as the author of the claim about corporate venture firms and a second check?", "review_file": None},
    {"id": "q_olympics_ceremony", "phenomenon": "unanswerable", "query": "What did Jason Lemkin say about the choreography of the 2024 Olympic opening ceremony?", "review_file": "host_finetune/graph_relationship_questions.json"},
    {"id": "q_dev_unans_halftime", "phenomenon": "unanswerable", "query": "What did Jason Lemkin write about the choreography of the 2026 Super Bowl halftime show?", "review_file": None},
    {"id": "q_dev_unans_private_board", "phenomenon": "unanswerable", "query": "What was said in a private board meeting that is not described in a published post?", "review_file": None},
    {"id": "q_dev_unans_headcount", "phenomenon": "unanswerable", "query": "What exact employee headcount did the Brex acquisition announcement post state for the day of the announcement?", "review_file": None},
    {"id": "q_dev_unans_dinner_chart", "phenomenon": "unanswerable", "query": "What did Jason Lemkin publish about the seating chart of a private dinner that no post describes?", "review_file": None},
    {"id": "q_dev_fact_niche_boundary", "phenomenon": "factual", "query": "What market-share condition does the source attach to staying present and gathering customers?", "review_file": None},
)


def _load_json(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _review_index(root: Path) -> dict[str, dict]:
    found = {}
    for relative in (
        "host_finetune/graph_relationship_questions.json",
        "host_finetune/pilot_questions.json",
    ):
        for row in _load_json(root / relative):
            found[str(row["id"])] = {
                "review_source_ids": [str(item) for item in (row.get("review_source_ids") or [])],
                "review_note": row.get("review_note"),
                "source_file": relative,
            }
    return found


def _retrieval_pools(root: Path) -> dict[str, list[dict]]:
    """First-seen source ids from the frozen pilot files. Order is not relevance."""
    directory = root / "experiments" / "EXP-20261031-001-graph-relationship-conditioning" / "raw"
    names = (
        "frozen_pilot_dense.jsonl",
        "frozen_naive.jsonl",
        "frozen_hybrid.jsonl",
        "frozen_mix.jsonl",
    )
    by_query: dict[str, list[dict]] = {}
    for name in names:
        path = directory / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                query = str(payload.get("query") or "")
                seen = by_query.setdefault(query, [])
                known = {item["doc_id"] for item in seen}
                for rank, hit in enumerate(payload.get("hits") or [], start=1):
                    doc_id = str(hit.get("id") or "")
                    if not doc_id or doc_id in known:
                        continue
                    known.add(doc_id)
                    seen.append({
                        "doc_id": doc_id,
                        "provenance": [{
                            "origin": "saved_retrieval",
                            "source_file": f"experiments/EXP-20261031-001-graph-relationship-conditioning/raw/{name}",
                            "mode": payload.get("mode"),
                            "backend": payload.get("backend"),
                            "first_seen_rank": rank,
                            "is_grade": False,
                        }],
                    })
    return by_query


def _candidate(doc_id: str, provenance: list[dict]) -> dict:
    return {
        "doc_id": doc_id,
        "provenance": provenance,
        "grade": None,
        "annotator": None,
        "notes": "",
    }


def build_benchmark(root: Path) -> dict:
    reviews = _review_index(root)
    pools = _retrieval_pools(root)
    questions_by_query = {}
    for relative in ("host_finetune/graph_relationship_questions.json",):
        for row in _load_json(root / relative):
            questions_by_query[str(row.get("query") or "")] = str(row["id"])
    retrieval_by_id = {}
    for query, candidates in pools.items():
        question_id = questions_by_query.get(query)
        if question_id:
            retrieval_by_id[question_id] = candidates
    annotator = []
    withheld = []
    for question in QUESTIONS:
        review = reviews.get(question["id"]) or {}
        preselected = list(review.get("review_source_ids") or [])
        merged: dict[str, dict] = {}
        for doc_id in preselected:
            merged[doc_id] = _candidate(doc_id, [{
                "origin": "author_preselection",
                "source_file": review.get("source_file"),
                "is_grade": False,
            }])
        for item in retrieval_by_id.get(question["id"], []):
            current = merged.get(item["doc_id"])
            if current is None:
                merged[item["doc_id"]] = _candidate(item["doc_id"], list(item["provenance"]))
            else:
                current["provenance"].extend(item["provenance"])
        annotator.append({
            "schema": ANNOTATION_SCHEMA,
            "question_id": question["id"],
            "phenomenon": question["phenomenon"],
            "query": question["query"],
            "split_policy": "development_only_do_not_open_the_corpus_test_split",
            "candidates": list(merged.values()),
            "missing_relevant_source_ids": [],
            "annotator_notes": "",
        })
        if review.get("review_note") or preselected:
            withheld.append({
                "question_id": question["id"],
                "do_not_show_to_annotator_until_grades_are_assigned": True,
                "preselected_source_ids": preselected,
                "preselected_ids_are_grades": False,
                "review_note": review.get("review_note"),
                "source_file": review.get("source_file"),
            })
    return {"annotator": annotator, "withheld": withheld}


def write_benchmark(root: Path, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    built = build_benchmark(root)
    annotator_path = directory / "dev_retrieval_benchmark.jsonl"
    withheld_path = directory / "dev_retrieval_benchmark_withheld.jsonl"
    annotator_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in built["annotator"]),
        encoding="utf-8",
    )
    withheld_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in built["withheld"]),
        encoding="utf-8",
    )
    return {
        "questions": len(built["annotator"]),
        "annotator": str(annotator_path),
        "withheld": str(withheld_path),
        "phenomena": sorted({row["phenomenon"] for row in built["annotator"]}),
    }

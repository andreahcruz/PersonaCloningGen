"""Index a graph-ready sample and query LightRAG without writing a final article.

Full-corpus indexing is refused unless ``--full`` and ``--confirm-full-index``
are both set. A smoke test must pass ``--limit``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from host_finetune.graph_ready import load_jsonl, select_index_sample
from host_finetune.lightrag_backend import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_EXTRACTION_MODEL,
    DEFAULT_GRAPH_READY,
    DEFAULT_WORKDIR,
    LIGHTRAG_MODES,
    index_documents_sync,
    query_documents_sync,
)
from host_finetune.retrieval_contract import require_lightrag_mode

ROOT = Path(__file__).resolve().parents[1]


def assert_isolated_workdir(workdir: Path, count: int) -> None:
    """The three-document smoke directory must not receive a larger index."""
    if count > 3 and workdir.resolve() == DEFAULT_WORKDIR.resolve():
        raise SystemExit(
            "refusing to add more than 3 documents to the smoke index at "
            f"{DEFAULT_WORKDIR}. Pass a separate --workdir."
        )


def choose_documents(documents: list[dict], args: argparse.Namespace) -> list[dict]:
    included = [
        row for row in documents if row.get("included", True) and not row.get("exclusion_reasons")
    ]
    included = [row for row in included if row.get("split") == "train"]
    if getattr(args, "id_file", None):
        wanted = set(json.loads(Path(args.id_file).read_text(encoding="utf-8")))
        chosen = [
            row
            for row in included
            if row.get("graph_document_id") in wanted or row.get("parent_graph_document_id") in wanted
        ]
        found = {
            row.get("parent_graph_document_id") or row.get("graph_document_id") for row in chosen
        }
        missing = sorted(wanted - found)
        if missing:
            raise SystemExit(f"id file names {len(missing)} documents that are not in this corpus")
        return chosen
    if args.full:
        if not args.confirm_full_index:
            raise SystemExit("full indexing requires --confirm-full-index")
        chosen = included
    elif args.sample:
        chosen = select_index_sample(
            included,
            args.sample,
            seed=args.seed,
            max_medium_fraction=args.max_medium_fraction,
        )
    else:
        chosen = included
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be at least 1")
        if args.spread:
            buckets: dict[str, list[dict]] = {}
            for document in chosen:
                buckets.setdefault(str(document.get("medium") or "unknown"), []).append(document)
            chosen_spread = []
            while len(chosen_spread) < args.limit:
                progressed = False
                for medium in sorted(buckets):
                    if not buckets[medium]:
                        continue
                    chosen_spread.append(buckets[medium].pop(0))
                    progressed = True
                    if len(chosen_spread) >= args.limit:
                        break
                if not progressed:
                    break
            chosen = chosen_spread
        else:
            chosen = chosen[: args.limit]
    if not chosen:
        raise SystemExit("no graph-ready documents selected")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-ready", type=Path, default=DEFAULT_GRAPH_READY)
    parser.add_argument(
        "--id-file",
        type=Path,
        default=None,
        help="Index these graph ids, or their parent ids, instead of drawing a new sample.",
    )
    parser.add_argument(
        "--entity-guidance",
        choices=("project", "library"),
        default="project",
        help="project is the pilot guidance. library is the LightRAG 1.5.7 default.",
    )
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--index", action="store_true")
    parser.add_argument("--sample", type=int, default=None, help="Representative sample size, normally 100 to 200.")
    parser.add_argument("--limit", type=int, default=None, help="Index only N selected documents.")
    parser.add_argument(
        "--spread",
        action="store_true",
        help="When --limit is set, take documents round-robin by medium instead of id order.",
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--confirm-full-index", action="store_true")
    parser.add_argument("--seed", type=int, default=298)
    parser.add_argument(
        "--max-medium-fraction",
        type=float,
        default=0.4,
        help="Cap one medium's share of a sample so the prototype is not only short posts.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--extraction-model", default=DEFAULT_EXTRACTION_MODEL)
    parser.add_argument("--num-ctx", type=int, default=None, help="Ollama context. The pilot default is 32768.")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--query", default=None)
    parser.add_argument("--modes", default=",".join(LIGHTRAG_MODES))
    parser.add_argument("--evidence-budget", type=int, default=4)
    parser.add_argument("--max-per-source", type=int, default=1)
    args = parser.parse_args()
    modes = [require_lightrag_mode(part.strip()) for part in args.modes.split(",") if part.strip()]
    if args.index:
        if not args.full and not args.sample and not args.id_file:
            raise SystemExit("indexing requires --sample N, --id-file, or --full --confirm-full-index")
        documents = load_jsonl(args.graph_ready / "documents.jsonl")
        chosen = choose_documents(documents, args)
        assert_isolated_workdir(args.workdir, len(chosen))
        print(
            f"indexing {len(chosen)} train-only documents into {args.workdir}",
            flush=True,
        )
        for document in chosen:
            print(
                f"  {document['graph_document_id']} {document.get('medium')} "
                f"words={document.get('word_count')} method={document.get('reconstruction_method')}",
                flush=True,
            )
        extra = {"entity_types_guidance": None} if args.entity_guidance == "library" else {}
        if args.num_ctx is not None:
            extra["num_ctx"] = args.num_ctx
        report = index_documents_sync(
            chosen,
            args.workdir,
            base_url=args.base_url,
            extraction_model=args.extraction_model,
            embed_model=args.embed_model,
            batch_size=args.batch_size,
            **extra,
        )
        (args.workdir / "last_index_report.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({key: report[key] for key in (
            "inserted", "skipped_processed", "extraction_failed", "catalog_documents", "lightrag_version", "entity_guidance"
        )}, indent=2))
    if args.query:
        results = []
        for mode in modes:
            print(f"query mode={mode}", flush=True)
            started = time.perf_counter()
            payload = query_documents_sync(
                args.query,
                args.workdir,
                mode,
                base_url=args.base_url,
                extraction_model=args.extraction_model,
                embed_model=args.embed_model,
                evidence_budget=args.evidence_budget,
                max_per_source=args.max_per_source,
            )
            payload["seconds"] = round(time.perf_counter() - started, 3)
            results.append(
                {
                    "mode": mode,
                    "status": payload.get("status"),
                    "hit_ids": [hit["id"] for hit in payload["hits"]],
                    "hit_sources": [hit.get("graph_document_id") for hit in payload["hits"]],
                    "excluded_reasons": sorted({item.get("reason") for item in payload["excluded"]}),
                    "seconds": payload.get("seconds"),
                    "message": payload.get("message"),
                }
            )
            print(json.dumps(results[-1], indent=2), flush=True)
        out = args.workdir / "last_query_report.json"
        out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

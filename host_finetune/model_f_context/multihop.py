"""Bounded multi-hop paths over a supplied relationship neighborhood.

Each hop must already be a directly source-supported edge. A two-hop path may
name both sources. It does not become a claim that one source stated the
connection between the path's endpoints.
"""

from __future__ import annotations

from host_finetune.model_f_context.schema import EvidencePath, compact_predicate, path_identity

MAX_HOPS = 2


def _direct(relationships: list) -> list:
    direct = []
    for item in relationships:
        if getattr(item, "support_status", None) != "direct":
            continue
        if getattr(item, "validation_decision", None) != "accepted":
            continue
        if not getattr(item, "predicate", None):
            continue
        if not getattr(item, "verbatim_evidence", None):
            continue
        if not (item.supporting_document_ids or item.supporting_chunk_ids or item.graph_source_ids):
            continue
        direct.append(item)
    direct.sort(key=lambda item: item.relationship_id)
    return direct


def _fact(edge) -> str:
    predicate = compact_predicate(edge.keywords, edge.predicate)
    return f"{edge.subject} — {predicate} — {edge.object}"


def _provenance(edge) -> dict:
    return {
        "relationship_id": edge.relationship_id,
        "support_status": edge.support_status,
        "document_ids": list(edge.supporting_document_ids),
        "chunk_ids": list(edge.supporting_chunk_ids or edge.graph_source_ids),
        "validation_decision": edge.validation_decision,
    }


def _make_path(edges: list, direct_pairs: set[tuple[str, str]]) -> EvidencePath:
    nodes = [edges[0].subject]
    for edge in edges:
        nodes.append(edge.object)
    edge_ids = [edge.relationship_id for edge in edges]
    start = nodes[0].casefold()
    end = nodes[-1].casefold()
    explicit = len(edges) == 1 or (start, end) in direct_pairs
    if len(edges) == 1:
        note = "single_supported_edge"
    elif explicit:
        note = "endpoints_have_a_separate_direct_edge"
    else:
        note = "path_is_not_a_single_source_claim"
    display = " || ".join(_fact(edge) for edge in edges)
    return EvidencePath(
        path_id=path_identity(edge_ids),
        hops=len(edges),
        edge_ids=edge_ids,
        nodes=nodes,
        edge_facts=[_fact(edge) for edge in edges],
        edge_provenance=[_provenance(edge) for edge in edges],
        explicit_endpoint_claim=explicit,
        endpoint_claim_note=note,
        display=display,
    )


def select_paths(
    relationships: list,
    *,
    max_hops: int = 2,
    max_paths: int = 24,
    max_edges: int = 64,
) -> dict:
    """Select cycle-free paths. Unsupported edges are not hops."""
    if max_hops not in (1, 2):
        raise ValueError("max_hops must be 1 or 2")
    if max_paths < 1 or max_edges < 1:
        raise ValueError("path limits must be positive")
    direct = _direct(relationships)[:max_edges]
    direct_pairs = {(edge.subject.casefold(), edge.object.casefold()) for edge in direct}
    outgoing: dict[str, list] = {}
    for edge in direct:
        outgoing.setdefault(edge.subject.casefold(), []).append(edge)
    paths: list[EvidencePath] = []
    seen: set[tuple[str, ...]] = set()
    rejected = []
    expansions = 0

    def add(edges: list) -> None:
        key = tuple(edge.relationship_id for edge in edges)
        if key in seen:
            rejected.append({"edge_ids": list(key), "reason": "duplicate_path"})
            return
        nodes = [edges[0].subject.casefold()]
        for edge in edges:
            nodes.append(edge.object.casefold())
        if len(nodes) != len(set(nodes)):
            rejected.append({"edge_ids": list(key), "reason": "cycle"})
            return
        if any(edge.support_status != "direct" for edge in edges):
            rejected.append({"edge_ids": list(key), "reason": "unsupported_edge"})
            return
        seen.add(key)
        paths.append(_make_path(edges, direct_pairs))

    for edge in direct:
        add([edge])
    if max_hops >= 2:
        for first in direct:
            for second in outgoing.get(first.object.casefold(), []):
                expansions += 1
                if expansions > max_edges * max_edges:
                    rejected.append({"edge_ids": [first.relationship_id], "reason": "candidate_limit"})
                    break
                if second.relationship_id == first.relationship_id:
                    continue
                add([first, second])
            else:
                continue
            break
    paths.sort(key=lambda item: (item.hops, item.path_id))
    overflow = []
    if len(paths) > max_paths:
        overflow = [{"path_id": item.path_id, "reason": "candidate_limit"} for item in paths[max_paths:]]
        paths = paths[:max_paths]
    withheld = []
    for item in relationships:
        if getattr(item, "support_status", None) == "direct" and getattr(item, "validation_decision", None) == "accepted":
            continue
        withheld.append({
            "relationship_id": getattr(item, "relationship_id", ""),
            "reason": getattr(item, "support_status", None) or "unsupported",
        })
    return {
        "paths": paths,
        "rejected": rejected + overflow,
        "withheld_edges": withheld,
        "direct_edge_count": len(direct),
        "max_hops": max_hops,
    }

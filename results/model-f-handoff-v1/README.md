# Model F shareable results (v1)

Derived from local artifacts on 2026-11-02. The exporter does not modify `experiments/EXP-20261102-001-model-f-unified-panel`.

| File | Contents |
|---|---|
| `aggregate_summary.json` | Eligible-conditioned `unified_eval.v1` summary for F-REL |
| `architecture_comparison.json` | `medium=all` rows for Base, PersonaRAG, Relabel RAG off, Relabel + RAG, and Model F |
| `per_medium.json` | Model F rows for all, blog, LinkedIn, X, and talk |
| `deterministic_audit_summary.json` | Pre-judge audit counts |
| `provenance.json` | Seeds, hashes, judge command with the local path removed, pilot and stage-2 selection |
| `examples.json` | A pointer to one local draft. The draft text is omitted because the GitHub repo is public |
| `excluded_artifacts.json` | Sizes and hashes of files kept off Git, plus transfer zip hashes recorded in the deployment note |

Rebuild, when the local experiment directory is present:

```powershell
python scripts/export_model_f_shareable.py
```

Narrative: [`docs/model_f_results.md`](../../docs/model_f_results.md).

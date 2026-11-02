import { Card, ScoreBar, SourceTag } from "@/components/ui";
import type { GenerateResult, GenerationMethod } from "@/lib/types";

export type ComparisonState = "queued" | "running" | "complete" | "failed";

export interface ComparisonEntry {
  method: GenerationMethod;
  state: ComparisonState;
  result?: GenerateResult;
  error?: string;
}

const METHOD_LABEL: Record<GenerationMethod, string> = {
  baseline: "Base Llama",
  personarag: "PersonaRAG",
  qlora: "direct QLoRA",
  model_f: "LightRAG + Relabel",
};

function metric(value: string | number, label: string) {
  return (
    <div className="rounded-lg border border-line bg-bg px-2 py-2">
      <p className="text-[11px] uppercase tracking-wide text-mute">{label}</p>
      <p className="mt-1 text-sm font-medium tabular-nums">{value}</p>
    </div>
  );
}

/**
 * Displays operational observations for one shared brief.  It intentionally
 * avoids aggregating them into a quality or voice score: those claims require
 * a separately validated evaluation dataset, not one live draft.
 */
export function LiveComparison({ entries }: { entries: ComparisonEntry[] }) {
  return (
    <div className="space-y-4">
      <div>
        <h2 className="font-semibold">Live architecture comparison</h2>
        <p className="mt-1 text-xs leading-relaxed text-mute">
          One shared content brief, generated sequentially to protect the CPU-only server. Metrics are operational
          observations, not an overall writing-quality or voice score.
        </p>
      </div>
      <div className="grid gap-4 2xl:grid-cols-3">
        {entries.map((entry) => {
          const result = entry.result;
          return (
            <Card key={entry.method} className="min-w-0">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h3 className="font-medium">{METHOD_LABEL[entry.method]}</h3>
                  <p className="text-xs text-mute">
                    {entry.method === "qlora" ? "Relabel GGUF" : "Base Llama 3.1"}
                  </p>
                </div>
                <span className={`rounded-full px-2 py-1 text-xs font-medium ${
                  entry.state === "complete"
                    ? "bg-ok/15 text-ok"
                    : entry.state === "failed"
                      ? "bg-bad/15 text-bad"
                      : "bg-primary/15 text-primary"
                }`}>
                  {entry.state === "queued" ? "Queued" : entry.state === "running" ? "Generating" : entry.state === "complete" ? "Complete" : "Failed"}
                </span>
              </div>

              {entry.state === "queued" || entry.state === "running" ? (
                <p className="mt-5 animate-pulse text-sm text-mute">
                  {entry.state === "running" ? "Generating this method…" : "Waiting for the previous method…"}
                </p>
              ) : null}

              {entry.error ? <p className="mt-4 text-sm text-bad">{entry.error}</p> : null}

              {result ? (
                <>
                  <div className="mt-4 grid grid-cols-2 gap-2">
                    {metric(result.metrics.wordCount, "Words")}
                    {metric(`${((result.latencyMs ?? 0) / 1000).toFixed(1)} s`, "Generation time")}
                    {metric(result.metrics.sentenceCount, "Sentences")}
                    {metric(result.metrics.sourceCount, "Returned sources")}
                  </div>
                  <p className="mt-3 text-[11px] leading-relaxed text-mute">
                    {entry.method === "personarag"
                      ? "Returned sources indicate retrieval was supplied, not that every claim is verified."
                      : "This method does not use retrieved source chunks."}
                  </p>
                  <div className="mt-4 max-h-80 overflow-y-auto whitespace-pre-wrap wrap-break-word rounded-xl border border-line bg-bg p-3 text-sm leading-relaxed">
                    {result.content}
                  </div>
                  {result.sources.length ? (
                    <details className="mt-3 text-xs">
                      <summary className="cursor-pointer text-mute">Show returned PersonaRAG sources</summary>
                      <ul className="mt-2 space-y-2">
                        {result.sources.map((source) => (
                          <li key={source.id} className="rounded-lg border border-line bg-bg p-2">
                            <div className="flex items-center justify-between gap-2">
                              <SourceTag type={source.sourceType} />
                              {source.title ? <span className="truncate text-mute">{source.title}</span> : null}
                            </div>
                            <p className="mt-2 line-clamp-3 leading-relaxed text-mute">{source.text}</p>
                            <div className="mt-2"><ScoreBar score={source.score} /></div>
                          </li>
                        ))}
                      </ul>
                    </details>
                  ) : null}
                </>
              ) : null}
            </Card>
          );
        })}
      </div>
    </div>
  );
}

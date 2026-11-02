"use client";

import { useEffect, useState } from "react";
import { ApiError, getEvaluation } from "@/lib/api";
import type { EvaluationData } from "@/lib/types";
import { Card, Notice, PageHeader } from "@/components/ui";

export default function EvaluationPage() {
  const [data, setData] = useState<EvaluationData | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  useEffect(() => {
    getEvaluation()
      .then(setData)
      .catch((e) => setError(e instanceof ApiError ? e : new ApiError("Could not load evaluation data.")));
  }, []);

  return (
    <>
      <PageHeader
        title="Evaluation evidence"
        subtitle="Versioned research results, displayed without scoring your draft or calling an LLM judge."
      />

      {error ? <Notice tone="error" title={error.message}>{error.hint}</Notice> : null}

      {data ? (
        <div className="space-y-5">
          <Card>
            <h2 className="font-semibold">{data.title}</h2>
            <p className="mt-1 text-sm text-mute">{data.description}</p>
            <p className="mt-3 font-mono text-xs text-mute">Manifest version: {data.version}</p>
          </Card>

          {data.groups.map((group) => (
            <Card key={group.id}>
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <h2 className="font-semibold">{group.title}</h2>
                  <p className="mt-1 text-xs font-medium text-warn">{group.scope}</p>
                </div>
                <a className="text-xs text-primary underline underline-offset-2" href={group.source.report_url} target="_blank" rel="noreferrer">
                  Open source report
                </a>
              </div>
              <p className="mt-3 text-sm leading-relaxed text-mute">{group.summary}</p>
              <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                {group.metrics.map((metric) => (
                  <div key={metric.label} className="rounded-xl border border-line bg-bg p-3">
                    <p className="text-xs text-mute">{metric.label}</p>
                    <p className="mt-1 text-lg font-semibold tabular-nums">{metric.value}</p>
                    <p className="mt-1 text-xs leading-relaxed text-mute">{metric.detail}</p>
                  </div>
                ))}
              </div>

              {group.comparison ? (
                <div className="mt-5 overflow-x-auto">
                  <p className="mb-2 text-sm font-medium">{group.comparison.label}</p>
                  <table className="w-full text-left text-sm">
                    <thead className="text-xs uppercase tracking-wide text-mute">
                      <tr>{group.comparison.columns.map((column) => <th key={column} className="px-2 py-2 font-medium">{column}</th>)}</tr>
                    </thead>
                    <tbody>
                      {group.comparison.rows.map((row) => (
                        <tr key={row[0]} className="border-t border-line">
                          {row.map((value, index) => <td key={`${row[0]}-${index}`} className="px-2 py-2 tabular-nums">{value}</td>)}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : null}

              <div className="mt-5 grid gap-4 border-t border-line pt-4 md:grid-cols-[1fr_2fr]">
                <p className="text-xs leading-relaxed text-mute"><span className="font-medium text-text">Sample:</span> {group.sample_size}</p>
                <div>
                  <p className="text-xs font-medium text-text">Limitations</p>
                  <ul className="mt-1 list-disc space-y-1 pl-4 text-xs leading-relaxed text-mute">
                    {group.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}
                  </ul>
                </div>
              </div>
              <p className="mt-3 text-xs text-mute">Source: {group.source.branch} @ {group.source.commit.slice(0, 12)}</p>
            </Card>
          ))}
        </div>
      ) : !error ? <p className="text-sm text-mute" role="status">Loading evidence…</p> : null}
    </>
  );
}

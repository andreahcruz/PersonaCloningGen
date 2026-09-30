"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
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
        title="Evaluation"
        subtitle="How the generator scores against the gold rubric. Numbers come from the evaluation harness. Systems without results are marked pending."
      />

      {error ? (
        <Notice tone="error" title={error.message}>
          {error.hint}
        </Notice>
      ) : null}

      {data ? (
        <div className="space-y-5">
          <Card>
            <h2 className="mb-3 font-semibold">System comparison</h2>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {data.matrix.map((c) => (
                <div key={c.system} className="rounded-xl border border-line bg-bg p-4">
                  <p className="text-sm font-medium">{c.system}</p>
                  <p className={`mt-1 text-xs ${c.available ? "text-ok" : "text-mute"}`}>
                    {c.available ? "Results available" : "Pending: no results recorded yet"}
                  </p>
                </div>
              ))}
            </div>
            <p className="mt-3 text-xs text-mute">
              The four-way comparison is planned in PUK-13. It fills in here once each system has
              been scored on the same prompts.
            </p>
          </Card>

          {data.runs.map((run) => {
            const rows = run.metrics
              .filter((m) => m.value !== null)
              .map((m) => ({ name: m.label, value: m.value as number }));
            return (
              <Card key={run.id}>
                <h2 className="font-semibold">{run.label}</h2>
                <p className="mt-1 text-xs text-mute">
                  {run.prompts} prompts. {run.note}
                </p>
                <div className="mt-4 h-72" role="img" aria-label={`Bar chart of ${run.label} metrics`}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={rows} margin={{ top: 8, right: 8, left: -12, bottom: 8 }}>
                      <CartesianGrid stroke="#1f2a44" vertical={false} />
                      <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 11 }} interval={0} />
                      <YAxis domain={[0, 1]} stroke="#94a3b8" tick={{ fontSize: 11 }} />
                      <Tooltip
                        contentStyle={{
                          background: "#121a2b",
                          border: "1px solid #1f2a44",
                          borderRadius: 12,
                          color: "#f8fafc",
                        }}
                        formatter={(v) => (typeof v === "number" ? v.toFixed(3) : String(v))}
                      />
                      <Bar
                        dataKey="value"
                        fill="#4f8cff"
                        radius={[6, 6, 0, 0]}
                        isAnimationActive={false}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-4 overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead className="text-xs uppercase tracking-wide text-mute">
                      <tr>
                        <th className="py-2 pr-4 font-medium">Metric</th>
                        <th className="py-2 font-medium">Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {run.metrics.map((m) => (
                        <tr key={m.key} className="border-t border-line">
                          <td className="py-2 pr-4">{m.label}</td>
                          <td className="py-2 tabular-nums">
                            {m.value === null ? "pending" : m.value.toFixed(3)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            );
          })}
        </div>
      ) : !error ? (
        <p className="text-sm text-mute" role="status">
          Loading...
        </p>
      ) : null}
    </>
  );
}

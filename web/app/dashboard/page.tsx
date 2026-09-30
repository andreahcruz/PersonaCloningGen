"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ApiError, getHealth, getStatistics } from "@/lib/api";
import { readHistory } from "@/lib/history";
import { FORMATS, type HistoryItem, type ServiceStatus, type Statistics } from "@/lib/types";
import {
  Card,
  EmptyState,
  Notice,
  PageHeader,
  Stat,
  StatusBadge,
  buttonClass,
} from "@/components/ui";

const fmt = new Intl.NumberFormat("en-US");

const KEY_SERVICES = ["Ollama", "ChromaDB", "Generation model"];

export default function DashboardPage() {
  const [stats, setStats] = useState<Statistics | null>(null);
  const [statsError, setStatsError] = useState<ApiError | null>(null);
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [history, setHistory] = useState<HistoryItem[]>([]);

  useEffect(() => {
    getStatistics()
      .then(setStats)
      .catch((e) => setStatsError(e instanceof ApiError ? e : new ApiError("Could not load statistics.")));
    getHealth()
      .then((h) => setServices(h.services))
      .catch(() => setServices([]));
    // eslint-disable-next-line react-hooks/set-state-in-effect -- browser-only storage read on mount
    setHistory(readHistory());
  }, []);

  const shown = services.filter((s) => KEY_SERVICES.includes(s.name));

  return (
    <>
      <PageHeader
        title="Dashboard"
        subtitle="The corpus behind the generator, current service status and your recent drafts."
        actions={
          <Link href="/generate" className={buttonClass("primary")}>
            New draft
          </Link>
        }
      />

      {statsError ? (
        <div className="mb-5">
          <Notice tone="warn" title={statsError.message}>
            {statsError.hint}
          </Notice>
        </div>
      ) : null}

      <div className="grid gap-4 sm:grid-cols-3">
        <Stat
          label="Total documents"
          value={stats ? fmt.format(stats.totalDocuments) : "-"}
          note="After ingestion filters"
        />
        <Stat
          label="Searchable chunks"
          value={stats ? fmt.format(stats.searchableChunks) : "-"}
          note="Embedded in Chroma"
        />
        <Stat
          label="Training examples"
          value={stats ? fmt.format(stats.trainingExamples) : "-"}
          note="Instruction / response rows"
        />
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        <Card>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="font-semibold">Service status</h2>
            <Link href="/health" className="text-xs text-primary hover:underline">
              All services
            </Link>
          </div>
          {shown.length ? (
            <ul className="space-y-3">
              {shown.map((s) => (
                <li key={s.name} className="flex items-center justify-between gap-3">
                  <span className="text-sm">{s.name}</span>
                  <StatusBadge state={s.status} />
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-mute">Status is not available from the backend yet.</p>
          )}
        </Card>

        <Card>
          <h2 className="mb-3 font-semibold">Recent generations</h2>
          {history.length ? (
            <ul className="space-y-3">
              {history.map((h) => (
                <li key={h.id} className="border-b border-line pb-3 last:border-0 last:pb-0">
                  <p className="truncate text-sm">{h.topic}</p>
                  <p className="text-xs text-mute">
                    {FORMATS.find((f) => f.id === h.format)?.label ?? h.format} - {h.model}
                    {h.latencyMs !== null ? ` - ${(h.latencyMs / 1000).toFixed(1)} s` : ""}
                    {h.mock ? " - demo" : ""}
                  </p>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState title="No drafts yet">Generated drafts from this browser show up here.</EmptyState>
          )}
        </Card>
      </div>
    </>
  );
}

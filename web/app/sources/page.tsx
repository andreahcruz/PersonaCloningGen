"use client";

import { useEffect, useState } from "react";
import { ApiError, NotAvailableError, searchSources } from "@/lib/api";
import type { Source } from "@/lib/types";
import {
  Card,
  EmptyState,
  Notice,
  PageHeader,
  ScoreBar,
  SourceTag,
  inputClass,
} from "@/components/ui";

const TYPES = [
  { id: "all", label: "All" },
  { id: "blog", label: "Blog" },
  { id: "linkedin", label: "LinkedIn" },
  { id: "x", label: "X" },
  { id: "youtube", label: "YouTube" },
];

export default function SourcesPage() {
  const [query, setQuery] = useState("");
  const [type, setType] = useState("all");
  const [items, setItems] = useState<Source[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  useEffect(() => {
    let cancelled = false;
    // Small delay so typing does not fire a request per keystroke.
    const t = setTimeout(() => {
      setLoading(true);
      searchSources(query, type)
        .then((list) => {
          if (cancelled) return;
          setItems(list);
          setError(null);
        })
        .catch((e) => {
          if (cancelled) return;
          setItems([]);
          setError(e instanceof ApiError ? e : new ApiError("Search failed."));
        })
        .finally(() => {
          if (!cancelled) setLoading(false);
        });
    }, 250);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [query, type]);

  return (
    <>
      <PageHeader
        title="Sources and knowledge base"
        subtitle="Browse the passages the generator retrieves from. This is the evidence that the drafts use RAG."
      />

      <Card className="mb-5 space-y-4">
        <input
          className={inputClass}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search the knowledge base, e.g. hiring a VP of Sales"
          aria-label="Search sources"
        />
        <div className="flex flex-wrap gap-2" role="group" aria-label="Filter by source type">
          {TYPES.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setType(t.id)}
              aria-pressed={type === t.id}
              className={`rounded-full border px-3 py-1 text-sm transition ${
                type === t.id
                  ? "border-primary bg-primary/15 text-primary"
                  : "border-line text-mute hover:border-primary/50"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </Card>

      {error instanceof NotAvailableError ? (
        <EmptyState title="Source search is not available yet">
          The backend needs a <code>GET /sources</code> endpoint. It is listed in the API contract.
        </EmptyState>
      ) : error ? (
        <Notice tone="error" title={error.message}>
          {error.hint}
        </Notice>
      ) : loading ? (
        <p className="text-sm text-mute" role="status">
          Searching...
        </p>
      ) : items.length === 0 ? (
        <EmptyState title="No matching sources">Try a different search or filter.</EmptyState>
      ) : (
        <ul className="grid gap-4 lg:grid-cols-2">
          {items.map((s) => (
            <li key={s.id}>
              <Card className="h-full">
                <div className="mb-2 flex items-center justify-between gap-2">
                  <SourceTag type={s.sourceType} />
                  {s.date ? <span className="text-xs text-mute">{s.date}</span> : null}
                </div>
                {s.title ? <p className="mb-1 text-sm font-medium">{s.title}</p> : null}
                <p className="text-sm leading-relaxed text-mute">{s.text}</p>
                <div className="mt-3">
                  <ScoreBar score={s.score} />
                </div>
                {s.url ? (
                  <a
                    href={s.url}
                    target="_blank"
                    rel="noreferrer"
                    className="mt-3 inline-block text-xs text-primary hover:underline"
                  >
                    Open original
                  </a>
                ) : null}
              </Card>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

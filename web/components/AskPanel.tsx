"use client";

import { Fragment, useState } from "react";
import { ApiError, NotAvailableError, askPersona } from "@/lib/api";
import { useElapsed } from "@/lib/motion";
import { PERSONAS } from "@/lib/personas";
import type { AskResult } from "@/lib/types";
import {
  Button,
  Card,
  EmptyState,
  Field,
  Icon,
  Notice,
  PageHeader,
  RangeInput,
  ScoreBar,
  SkeletonLines,
  SourceTag,
  inputClass,
} from "@/components/ui";

const EXAMPLES = [
  "How does he describe the way his companies approach hard engineering problems?",
  "What does he say about the risks and benefits of artificial intelligence?",
  "How has the way he writes on X changed over the years?",
];

/** Turns "[Source 2]" in an answer into a small badge so citations are easy to spot. */
function renderAnswer(text: string) {
  return text.split(/(\[Source \d+\])/g).map((part, i) =>
    /^\[Source \d+\]$/.test(part) ? (
      <span
        key={i}
        className="mx-0.5 inline-block rounded-md bg-primary/12 px-1.5 py-0.5 align-baseline text-xs font-semibold text-primary"
      >
        {part.slice(1, -1)}
      </span>
    ) : (
      <Fragment key={i}>{part}</Fragment>
    ),
  );
}

export default function AskPanel() {
  const persona = PERSONAS.musk;
  const [question, setQuestion] = useState("");
  const [k, setK] = useState(4);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<AskResult | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [copied, setCopied] = useState(false);
  const elapsed = useElapsed(busy);

  async function run() {
    if (!question.trim() || busy) return;
    setBusy(true);
    setError(null);
    setCopied(false);
    try {
      setResult(await askPersona({ persona: "elon_musk", question: question.trim(), k }));
    } catch (e) {
      setResult(null);
      setError(e instanceof ApiError ? e : new ApiError(e instanceof Error ? e.message : "Unknown error"));
    } finally {
      setBusy(false);
    }
  }

  async function copy() {
    if (!result) return;
    try {
      await navigator.clipboard.writeText(result.answer);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setError(new ApiError("Could not copy to the clipboard.", 0, "Select the text and copy it manually."));
    }
  }

  return (
    <>
      <PageHeader
        eyebrow={persona.name}
        title="Ask the sources"
        subtitle="Ask a question and get an answer written from retrieved tweets and interview transcripts, with citations. This mode reports what the sources say. It does not write as Elon Musk."
      />

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_340px]">
        <div className="min-w-0 space-y-5">
          <Card className="space-y-4">
            <Field label="Your question">
              <textarea
                className={`${inputClass} min-h-28 resize-y leading-relaxed`}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run();
                }}
                placeholder="e.g. What does he say about building reusable rockets?"
              />
            </Field>
            <div className="flex flex-wrap gap-2" aria-label="Example questions">
              {EXAMPLES.map((example) => (
                <button
                  key={example}
                  type="button"
                  onClick={() => setQuestion(example)}
                  className="rounded-full border border-line bg-card-2 px-3 py-1 text-left text-xs text-mute transition hover:border-primary/60 hover:text-ink"
                >
                  {example}
                </button>
              ))}
            </div>
            <div>
              <div className="mb-2 flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wider text-mute">Passages to retrieve</span>
                <span key={k} className="pop rounded-md bg-primary/12 px-2 py-0.5 text-xs font-semibold tabular-nums text-primary">
                  {k}
                </span>
              </div>
              <RangeInput value={k} min={1} max={8} onChange={setK} label="Passages to retrieve" />
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <Button onClick={run} disabled={busy || !question.trim()}>
                <Icon name="chat" className="h-4 w-4" />
                {busy ? "Searching and answering..." : result ? "Ask again" : "Ask"}
              </Button>
              <span className="text-xs text-mute">Ctrl or Cmd + Enter also sends.</span>
            </div>
          </Card>

          {error instanceof NotAvailableError ? (
            <EmptyState title="The Elon Musk answer service is not connected yet">
              The retrieval index is built on the <code>feature/elon-musk-etl</code> branch, but the backend has no
              route that serves it. Switch the app to demo mode to preview this screen with sample text.
            </EmptyState>
          ) : error ? (
            <Notice tone="error" title={error.message}>
              {error.hint}
            </Notice>
          ) : null}

          <Card>
            <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
              <h2 className="font-semibold">Answer</h2>
              {result ? (
                <Button variant="ghost" onClick={copy}>
                  <Icon name={copied ? "check" : "copy"} className="h-4 w-4" />
                  {copied ? "Copied" : "Copy"}
                </Button>
              ) : null}
            </div>
            {busy ? (
              <div role="status" aria-label="Writing the answer">
                <div className="mb-4 flex items-center justify-between">
                  <span className="flex items-center gap-2.5 text-sm font-medium">
                    <span className="dot-pulse h-2.5 w-2.5 rounded-full bg-primary" />
                    Searching and answering
                  </span>
                  <span className="rounded-full bg-card-2 px-3 py-1 text-sm font-medium tabular-nums">{elapsed} s</span>
                </div>
                <SkeletonLines lines={6} />
                <p className="mt-4 text-xs text-mute">The local model reports no live progress, so the timer shows real elapsed time. This can take a minute.</p>
              </div>
            ) : result ? (
              <div className="fade-in">
                {result.mock ? (
                  <div className="mb-3">
                    <Notice tone="warn" title="Demo answer">
                      This is sample text, not real model output.
                    </Notice>
                  </div>
                ) : null}
                <div className="whitespace-pre-wrap wrap-break-word rounded-xl border border-line bg-card-2/60 p-6 text-[16px] leading-[1.75]">
                  {renderAnswer(result.answer)}
                </div>
                <p className="mt-3 text-xs text-mute">
                  {result.model}
                  {result.latencyMs !== null ? ` · ${(result.latencyMs / 1000).toFixed(1)} s` : ""} · {result.sources.length} sources
                </p>
              </div>
            ) : error ? null : (
              <EmptyState title="Your answer will appear here">Ask a question and press Ask.</EmptyState>
            )}
          </Card>
        </div>

        <div className="space-y-5">
          <Card className="h-fit">
            <h2 className="mb-3 font-semibold">Active persona</h2>
            <div className="flex items-center gap-3">
              <span className="bg-accent-gradient grid h-11 w-11 place-items-center rounded-xl text-sm font-bold text-primary-ink">
                {persona.initials}
              </span>
              <div>
                <p className="text-sm font-medium">{persona.name}</p>
                <p className="text-xs text-mute">{persona.modeLabel}</p>
              </div>
            </div>
            <p className="mt-3 text-xs leading-relaxed text-mute">{persona.corpus}.</p>
            <p className="mt-2 text-xs leading-relaxed text-mute">{persona.liveNote}</p>
          </Card>

          <Card className="h-fit">
            <h2 className="mb-3 font-semibold">Retrieved sources</h2>
            {result && result.sources.length > 0 ? (
              <ul className="space-y-3">
                {result.sources.map((s, i) => (
                  <li key={s.id} className="lift rounded-xl border border-line bg-card-2/50 p-3.5">
                    <div className="mb-2 flex items-center justify-between gap-2">
                      <span className="flex items-center gap-2">
                        <span className="grid h-5 w-5 place-items-center rounded-md bg-primary/12 text-[11px] font-semibold text-primary">
                          {i + 1}
                        </span>
                        <SourceTag type={s.sourceType} />
                      </span>
                      {s.title ? <span className="truncate text-xs text-mute">{s.title}</span> : null}
                    </div>
                    <p className="line-clamp-4 text-xs leading-relaxed text-mute">{s.text}</p>
                    <div className="mt-2">
                      <ScoreBar score={s.score} />
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-mute">Sources appear after you ask a question.</p>
            )}
          </Card>
        </div>
      </div>
    </>
  );
}

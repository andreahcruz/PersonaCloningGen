"use client";

import { useEffect, useMemo, useState } from "react";
import { ApiError, generate, getModels } from "@/lib/api";
import { addHistory } from "@/lib/history";
import {
  FORMATS,
  type FormatId,
  type GenerateResult,
  type ModelInfo,
} from "@/lib/types";
import {
  Button,
  Card,
  EmptyState,
  Field,
  Notice,
  PageHeader,
  ScoreBar,
  SourceTag,
  inputClass,
} from "@/components/ui";

const DEFAULT_MODEL = "lemkin-clone";

function slug(text: string): string {
  return (
    text
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "")
      .slice(0, 40) || "draft"
  );
}

export default function GeneratorPage() {
  const [format, setFormat] = useState<FormatId>("linkedin_post");
  const [topic, setTopic] = useState("");
  const [audience, setAudience] = useState("");
  const [goal, setGoal] = useState("");
  const [cta, setCta] = useState("none");
  const [k, setK] = useState(8);
  const [temperature, setTemperature] = useState(0.7);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [result, setResult] = useState<GenerateResult | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    getModels()
      .then(({ models: list }) => {
        const usable = list.filter((m) => m.kind !== "embedding" && m.available);
        setModels(usable);
        if (usable.length && !usable.some((m) => m.name === DEFAULT_MODEL)) {
          setModel(usable[0].name);
        }
      })
      .catch(() => setModels([]));
  }, []);

  const missing = useMemo(() => {
    const out: string[] = [];
    if (!topic.trim()) out.push("topic");
    if (!audience.trim()) out.push("audience");
    if (!goal.trim()) out.push("goal");
    return out;
  }, [topic, audience, goal]);

  async function run() {
    if (missing.length || busy) return;
    setBusy(true);
    setError(null);
    setCopied(false);
    try {
      const res = await generate({
        format,
        topic: topic.trim(),
        audience: audience.trim(),
        goal: goal.trim(),
        cta: cta.trim() || "none",
        k,
        model,
        temperature,
      });
      setResult(res);
      addHistory({
        id: `${Date.now()}`,
        at: new Date().toISOString(),
        format,
        topic: topic.trim(),
        model: res.model,
        latencyMs: res.latencyMs,
        mock: res.mock,
      });
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError(e instanceof Error ? e.message : "Unknown error"));
    } finally {
      setBusy(false);
    }
  }

  async function copy() {
    if (!result) return;
    try {
      await navigator.clipboard.writeText(result.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setError(new ApiError("Could not copy to the clipboard.", 0, "Select the text and copy it manually."));
    }
  }

  function download() {
    if (!result) return;
    const blob = new Blob([result.content], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${format}-${slug(topic)}.md`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  const words = result ? result.content.trim().split(/\s+/).filter(Boolean).length : 0;

  return (
    <>
      <PageHeader
        title="Content generator"
        subtitle="Write a brief, retrieve Jason Lemkin sources, and generate a draft in his voice."
      />

      <div className="grid gap-5 xl:grid-cols-[260px_minmax(0,1fr)_320px]">
        {/* Left: settings */}
        <div className="space-y-5">
          <Card className="space-y-4">
            <fieldset>
              <legend className="mb-2 text-xs font-medium uppercase tracking-wide text-mute">
                Format
              </legend>
              <div className="space-y-2">
                {FORMATS.map((f) => (
                  <label
                    key={f.id}
                    className={`block cursor-pointer rounded-xl border p-3 transition ${
                      format === f.id
                        ? "border-primary bg-primary/10"
                        : "border-line hover:border-primary/50"
                    }`}
                  >
                    <input
                      type="radio"
                      name="format"
                      value={f.id}
                      checked={format === f.id}
                      onChange={() => setFormat(f.id)}
                      className="sr-only"
                    />
                    <span className="block text-sm font-medium">{f.label}</span>
                    <span className="block text-xs text-mute">{f.blurb}</span>
                  </label>
                ))}
              </div>
            </fieldset>
          </Card>

          <Card className="space-y-4">
            <Field label="Generation model">
              <select
                className={inputClass}
                value={model}
                onChange={(e) => setModel(e.target.value)}
              >
                {(models.length ? models : [{ name: DEFAULT_MODEL, kind: "fine-tuned", available: true }]).map(
                  (m) => (
                    <option key={m.name} value={m.name}>
                      {m.name}
                      {m.kind === "fine-tuned" ? " (fine-tuned)" : ""}
                    </option>
                  ),
                )}
              </select>
            </Field>
            <Field label={`Retrieved chunks: ${k}`}>
              <input
                type="range"
                min={1}
                max={20}
                value={k}
                onChange={(e) => setK(Number(e.target.value))}
                className="w-full accent-primary"
              />
            </Field>
            <Field
              label={`Temperature: ${temperature.toFixed(1)}`}
              hint="Sent to the API. Applied once the backend supports it."
            >
              <input
                type="range"
                min={0}
                max={1.5}
                step={0.1}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                className="w-full accent-violet"
              />
            </Field>
          </Card>
        </div>

        {/* Center: brief and output */}
        <div className="min-w-0 space-y-5">
          <Card className="space-y-4">
            <Field label="Topic">
              <input
                className={inputClass}
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="e.g. Hiring your first salesperson"
              />
            </Field>
            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Audience">
                <input
                  className={inputClass}
                  value={audience}
                  onChange={(e) => setAudience(e.target.value)}
                  placeholder="e.g. B2B SaaS founders"
                />
              </Field>
              <Field label="Goal">
                <input
                  className={inputClass}
                  value={goal}
                  onChange={(e) => setGoal(e.target.value)}
                  placeholder="e.g. Share one sharp takeaway"
                />
              </Field>
            </div>
            <Field label="Call to action">
              <input className={inputClass} value={cta} onChange={(e) => setCta(e.target.value)} />
            </Field>
            <div className="flex flex-wrap items-center gap-3">
              <Button onClick={run} disabled={busy || missing.length > 0}>
                {busy ? "Generating..." : result ? "Regenerate" : "Generate"}
              </Button>
              {missing.length > 0 ? (
                <span className="text-xs text-mute">Add {missing.join(", ")} to continue.</span>
              ) : null}
            </div>
          </Card>

          {error ? (
            <Notice tone="error" title={error.message}>
              {error.hint}
            </Notice>
          ) : null}

          <Card>
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <h2 className="font-semibold">Draft</h2>
              {result ? (
                <div className="flex gap-2">
                  <Button variant="ghost" onClick={copy}>
                    {copied ? "Copied" : "Copy"}
                  </Button>
                  <Button variant="ghost" onClick={download}>
                    Download .md
                  </Button>
                </div>
              ) : null}
            </div>

            {busy ? (
              <p className="animate-pulse text-sm text-mute" role="status">
                Embedding your brief, retrieving sources and generating. This can take a minute on a
                local model.
              </p>
            ) : result ? (
              <>
                {result.mock ? (
                  <div className="mb-3">
                    <Notice tone="warn" title="Demo draft">
                      This is sample text, not real model output.
                    </Notice>
                  </div>
                ) : null}
                <div className="whitespace-pre-wrap wrap-break-word rounded-xl border border-line bg-bg p-4 text-sm leading-relaxed">
                  {result.content}
                </div>
                <p className="mt-3 text-xs text-mute">
                  {words} words - {result.model}
                  {result.latencyMs !== null ? ` - ${(result.latencyMs / 1000).toFixed(1)} s` : ""}
                </p>
              </>
            ) : (
              <EmptyState title="Your draft will appear here">
                Fill in the brief and press Generate.
              </EmptyState>
            )}
          </Card>
        </div>

        {/* Right: sources */}
        <Card className="h-fit">
          <h2 className="mb-3 font-semibold">Retrieved sources</h2>
          {result && result.sources.length > 0 ? (
            <ul className="space-y-3">
              {result.sources.map((s) => (
                <li key={s.id} className="rounded-xl border border-line bg-bg p-3">
                  <div className="mb-2 flex items-center justify-between gap-2">
                    <SourceTag type={s.sourceType} />
                    {s.title ? <span className="truncate text-xs text-mute">{s.title}</span> : null}
                  </div>
                  <p className="line-clamp-4 text-xs leading-relaxed text-mute">{s.text}</p>
                  <div className="mt-2">
                    <ScoreBar score={s.score} />
                  </div>
                </li>
              ))}
            </ul>
          ) : result ? (
            <p className="text-sm text-mute">
              The backend did not return sources for this draft. The API needs to include them; see
              the API contract.
            </p>
          ) : (
            <p className="text-sm text-mute">Sources appear after you generate a draft.</p>
          )}
        </Card>
      </div>
    </>
  );
}

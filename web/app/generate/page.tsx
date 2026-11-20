"use client";

import { useEffect, useMemo, useState } from "react";
import { ApiError, checkGrammar, generate, getModelFStatus, getPersona } from "@/lib/api";
import DraftPreview from "@/components/DraftPreview";
import Typewriter from "@/components/Typewriter";
import { useElapsed } from "@/lib/motion";
import { type ComparisonEntry, LiveComparison } from "@/components/LiveComparison";
import AskPanel from "@/components/AskPanel";
import { addHistory, readHistory } from "@/lib/history";
import { usePersonaId } from "@/lib/prefs";
import {
  FORMATS,
  type FormatId,
  type GenerationMethod,
  type GenerateResult,
  type DraftCheck,
  type DraftCheckStatus,
  type HistoryItem,
  type GrammarCheckResult,
  type ModelFStatus,
  type PersonaProfile,
} from "@/lib/types";
import {
  Button,
  Card,
  EmptyState,
  Field,
  Icon,
  Notice,
  PageHeader,
  RangeInput,
  Ring,
  ScoreBar,
  SkeletonLines,
  SourceTag,
  Switch,
  Tabs,
  inputClass,
} from "@/components/ui";

const METHODS: Array<{
  id: GenerationMethod;
  label: string;
  model: string;
  detail: string;
}> = [
  {
    id: "baseline",
    label: "Base Llama",
    model: "Base Llama 3.1",
    detail: "Uses your brief directly, without a Jason Lemkin profile or retrieved sources.",
  },
  {
    id: "personarag",
    label: "PersonaRAG",
    model: "Base Llama 3.1",
    detail: "Searches Jason Lemkin source chunks first, then writes a grounded draft.",
  },
  {
    id: "qlora",
    label: "direct QLoRA",
    model: "Relabel GGUF",
    detail: "Writes from the merged Relabel model. It does not retrieve sources.",
  },
  {
    id: "model_f",
    label: "LightRAG + Relabel",
    model: "Relabel GGUF + hybrid retrieval",
    detail: "Retrieves source-grounded passages from the verified LightRAG snapshot, then writes with the same Relabel GGUF.",
  },
];

// Kevin's tested medium grid: a deterministic prompt per format. LinkedIn and
// X constraints deliberately precede the topic so headline-like topics are not
// copied as the entire draft. The backend owns this same mapping.
const QLORA_TEMPLATES: Record<FormatId, string> = {
  linkedin_post:
    "About 80 words. The first line is a claim, not a headline. No hashtags. Write a LinkedIn post in the style of Jason Lemkin about {topic}.",
  blog_draft:
    "Write a blog post in the style of Jason Lemkin about {topic}. Write the post itself in about 250 words, with no title line.",
  x_thread:
    "Write one X post in the style of Jason Lemkin. Exactly 3 sentences, under 60 words, no headline, and no hashtags. Topic: {topic}",
  youtube_script:
    "Write a spoken talk in the style of Jason Lemkin about {topic}. Use short spoken sentences to founders, about 160 words, with no title and no headings.",
};

const LEGACY_QLORA_PREFIXES = [
  "Write a LinkedIn post in the style of Jason Lemkin about:",
  "Write a blog post in the style of Jason Lemkin about:",
  "Write an X post in the style of Jason Lemkin about:",
  "Write a talk in the style of Jason Lemkin about:",
];

const COMPARISON_METHODS: GenerationMethod[] = ["baseline", "personarag", "qlora"];

function qloraTopicOnly(value: string): string {
  const prefixes = LEGACY_QLORA_PREFIXES
    .map((prefix) => prefix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join("|");
  return value.replace(new RegExp(`^\\s*(?:${prefixes})\\s*`, "i"), "");
}

function qloraPrompt(format: FormatId, topic: string): string {
  return QLORA_TEMPLATES[format].replace("{topic}", topic || "[your topic]");
}

const CHECK_STATUS_CLASS: Record<DraftCheckStatus, string> = {
  pass: "border-ok/40 bg-ok/10 text-ok",
  warn: "border-warn/40 bg-warn/10 text-warn",
  fail: "border-bad/40 bg-bad/10 text-bad",
  na: "border-line bg-panel text-mute",
};

function checkLabel(status: DraftCheckStatus): string {
  return status === "na" ? "N/A" : status === "warn" ? "Warning" : status[0].toUpperCase() + status.slice(1);
}

function slug(text: string): string {
  return (
    text
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "")
      .slice(0, 40) || "draft"
  );
}

type StudioTab = "draft" | "preview" | "checks" | "sources";

// Display-only hints per format. The real length rules live in QLORA_TEMPLATES above and on the backend.
const FORMAT_META: Record<FormatId, { icon: string; target: number; length: string; noun: string }> = {
  linkedin_post: { icon: "pen", target: 80, length: "about 80 words", noun: "post" },
  blog_draft: { icon: "doc", target: 250, length: "about 250 words", noun: "article" },
  x_thread: { icon: "chat", target: 60, length: "under 60 words", noun: "thread" },
  youtube_script: { icon: "bolt", target: 160, length: "about 160 words", noun: "script" },
};

const IDEAS = [
  "Hiring your first salesperson",
  "When to hire a VP of Sales",
  "Why founders should still sell",
  "Pricing a new SaaS product",
];
const AUDIENCES = ["B2B SaaS founders", "First-time sales leaders"];
const GOALS = ["Share one sharp takeaway", "Challenge a common belief"];

// What each method does, shown while it runs. These are descriptions, not live progress.
const STAGES: Record<GenerationMethod, (k: number) => string[]> = {
  baseline: () => ["Read your brief", "Write the draft", "Run the draft checks"],
  personarag: (k) => ["Embed your brief", `Retrieve ${k} passages`, "Write a grounded draft", "Run the draft checks"],
  qlora: () => ["Build the training-style request", "Write with the Relabel model", "Run the draft checks"],
  model_f: () => ["Hybrid retrieval from LightRAG", "Write with the Relabel model", "Run the draft checks"],
};

function WriteStudio() {
  const [format, setFormat] = useState<FormatId>("linkedin_post");
  const [topic, setTopic] = useState("");
  const [audience, setAudience] = useState("");
  const [goal, setGoal] = useState("");
  const [cta, setCta] = useState("none");
  const [k, setK] = useState(4);
  const [method, setMethod] = useState<GenerationMethod>("personarag");
  const [compareMode, setCompareMode] = useState(false);
  const [comparison, setComparison] = useState<ComparisonEntry[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [result, setResult] = useState<GenerateResult | null>(null);
  const [copied, setCopied] = useState(false);
  const [persona, setPersona] = useState<PersonaProfile | null>(null);
  const [personaError, setPersonaError] = useState<string | null>(null);
  const [modelFStatus, setModelFStatus] = useState<ModelFStatus | null>(null);
  const [grammarResult, setGrammarResult] = useState<GrammarCheckResult | null>(null);
  const [grammarError, setGrammarError] = useState<string | null>(null);
  const [grammarBusy, setGrammarBusy] = useState(false);
  const [tab, setTab] = useState<StudioTab>("draft");
  const [resultFormat, setResultFormat] = useState<FormatId>("linkedin_post");
  const [recent, setRecent] = useState<HistoryItem[]>([]);
  const elapsed = useElapsed(busy);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- browser-only storage read
    setRecent(readHistory());
  }, [result, comparison]);

  useEffect(() => {
    let active = true;
    getPersona()
      .then((profile) => {
        if (active) setPersona(profile);
      })
      .catch(() => {
        if (active) setPersonaError("Persona profile is unavailable. Generation is unaffected.");
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    getModelFStatus().then((status) => {
      if (active) setModelFStatus(status);
    });
    return () => {
      active = false;
    };
  }, []);

  const missing = useMemo(() => {
    const out: string[] = [];
    if (!topic.trim()) out.push("topic");
    if (compareMode || (method !== "qlora" && method !== "model_f")) {
      if (!audience.trim()) out.push("audience");
      if (!goal.trim()) out.push("goal");
    }
    return out;
  }, [topic, audience, goal, method, compareMode]);

  function requestFor(targetMethod: GenerationMethod) {
    // QLoRA remains training-aligned. The shared brief is the topic; the
    // baseline and RAG paths additionally receive audience/goal/CTA because
    // those are part of their deployed prompt contracts.
    return targetMethod === "qlora" || targetMethod === "model_f"
      ? {
          format,
          topic: qloraTopicOnly(topic).trim(),
          audience: "",
          goal: "",
          cta: "none",
          k: 4,
          method: targetMethod,
        }
      : {
          format,
          topic: topic.trim(),
          audience: audience.trim(),
          goal: goal.trim(),
          cta: cta.trim() || "none",
          k,
          method: targetMethod,
        };
  }

  async function run() {
    if (missing.length || busy) return;
    setBusy(true);
    setError(null);
    setCopied(false);
    setGrammarResult(null);
    setGrammarError(null);
    try {
      if (compareMode) {
        setComparison(COMPARISON_METHODS.map((item, index) => ({
          method: item,
          state: index === 0 ? "running" : "queued",
        })));
        for (let index = 0; index < COMPARISON_METHODS.length; index += 1) {
          const targetMethod = COMPARISON_METHODS[index];
          try {
            const res = await generate(requestFor(targetMethod));
            setComparison((previous) => previous.map((entry) => {
              if (entry.method === targetMethod) return { ...entry, state: "complete", result: res };
              if (entry.method === COMPARISON_METHODS[index + 1] && entry.state === "queued") {
                return { ...entry, state: "running" };
              }
              return entry;
            }));
            addHistory({
              id: `${Date.now()}-${targetMethod}`,
              at: new Date().toISOString(),
              format,
              topic: topic.trim(),
              model: res.model,
              latencyMs: res.latencyMs,
              mock: res.mock,
            });
          } catch (itemError) {
            const message = itemError instanceof ApiError ? itemError.message : "This method could not generate a draft.";
            setComparison((previous) => previous.map((entry) => {
              if (entry.method === targetMethod) return { ...entry, state: "failed", error: message };
              if (entry.method === COMPARISON_METHODS[index + 1] && entry.state === "queued") {
                return { ...entry, state: "running" };
              }
              return entry;
            }));
          }
        }
        return;
      }
      // PersonaRAG uses the full creative brief. Direct QLoRA was not trained
      // on audience/goal/CTA instructions, so do not send stale form values
      // when the user selects that method.
      const res = await generate(requestFor(method));
      setResult(res);
      setResultFormat(format);
      setTab("draft");
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

  async function runGrammarCheck() {
    if (!result || grammarBusy) return;
    setGrammarBusy(true);
    setGrammarError(null);
    try {
      setGrammarResult(await checkGrammar(result.content));
    } catch (e) {
      setGrammarResult(null);
      setGrammarError(e instanceof ApiError ? e.message : "Grammar service unavailable. Try again shortly.");
    } finally {
      setGrammarBusy(false);
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

  const visibleMethods = METHODS.filter((item) => item.id !== "model_f" || modelFStatus?.available);
  const activeMethod = visibleMethods.find((item) => item.id === method) ?? METHODS[0];
  const displayedSourceResult = compareMode ? comparison.find((entry) => entry.method === "personarag")?.result : result;

  const needsBrief = compareMode || (method !== "qlora" && method !== "model_f");
  const requiredCount = needsBrief ? 3 : 1;
  const readiness = (requiredCount - missing.length) / requiredCount;
  const meta = FORMAT_META[resultFormat];
  const checkList = result
    ? ([
        ["Completion", result.checks.completion],
        ["Format compliance", result.checks.formatCompliance],
        ["Prompt echo", result.checks.promptEcho],
        ["Unsupported numbers", result.checks.unsupportedNumbers],
        ["Source copying", result.checks.sourceCopying],
      ] as Array<[string, DraftCheck]>)
    : [];
  const scored = checkList.filter(([, check]) => check.status !== "na");
  const passed = scored.filter(([, check]) => check.status === "pass").length;
  const stages = STAGES[method](k);

  function chip(label: string, onPick: () => void) {
    return (
      <button
        key={label}
        type="button"
        onClick={onPick}
        className="rounded-full border border-line bg-card-2 px-3 py-1 text-xs text-mute transition hover:-translate-y-0.5 hover:border-primary/60 hover:text-ink active:translate-y-0"
      >
        {label}
      </button>
    );
  }

  return (
    <>
      <PageHeader
        eyebrow="Jason Lemkin · Studio"
        title="Write something he would say"
        subtitle="Pick a format, give a brief, choose how it is written. Ctrl or Cmd + Enter generates."
      />

      <div
        className="grid items-start gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]"
        onKeyDown={(e) => {
          if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run();
        }}
      >
        {/* ---------------- Composer ---------------- */}
        <div className="min-w-0 space-y-4">
          <details className="shadow-soft group rounded-2xl border border-line bg-card px-4 py-3">
            <summary className="flex cursor-pointer list-none items-center gap-3">
              <span className="bg-accent-gradient grid h-9 w-9 place-items-center rounded-xl text-xs font-bold text-primary-ink">JL</span>
              <span className="min-w-0 flex-1">
                <span className="block text-sm font-medium">Writing as {persona?.name ?? "Jason Lemkin"}</span>
                <span className="block truncate text-xs text-mute">{persona?.source ?? "Active persona profile"}</span>
              </span>
              <span className="text-xs text-mute transition group-open:rotate-180" aria-hidden>
                ▾
              </span>
            </summary>
            <div className="mt-3 border-t border-line pt-3 text-xs leading-relaxed text-mute">
              {persona ? (
                <>
                  <p className="whitespace-pre-line">{persona.summary}</p>
                  {persona.mock ? <p className="mt-2 text-warn">Sample persona profile (demo mode).</p> : null}
                </>
              ) : personaError ? (
                <p>{personaError}</p>
              ) : (
                <p>Loading active persona profile…</p>
              )}
            </div>
          </details>

          <Card className="space-y-4">
            <fieldset>
              <legend className="mb-3 text-xs font-semibold uppercase tracking-wider text-mute">1 · Format</legend>
              <div className="grid grid-cols-2 gap-2.5">
                {FORMATS.map((f) => {
                  const on = format === f.id;
                  return (
                    <label
                      key={f.id}
                      className={`lift relative block cursor-pointer rounded-xl border p-3.5 ${
                        on ? "border-primary bg-primary/8 ring-1 ring-primary/30" : "border-line hover:bg-card-2"
                      }`}
                    >
                      <input
                        type="radio"
                        name="format"
                        value={f.id}
                        checked={on}
                        onChange={() => setFormat(f.id)}
                        className="sr-only"
                      />
                      <span
                        className={`mb-2 grid h-8 w-8 place-items-center rounded-lg transition ${
                          on ? "bg-accent-gradient text-primary-ink" : "bg-card-2 text-mute"
                        }`}
                      >
                        <Icon name={FORMAT_META[f.id].icon} className="h-4 w-4" />
                      </span>
                      <span className="block text-sm font-medium">{f.label}</span>
                      <span className="block text-xs text-mute">{f.blurb}</span>
                      <span className="mt-1.5 block text-[11px] font-medium text-primary">{FORMAT_META[f.id].length}</span>
                    </label>
                  );
                })}
              </div>
            </fieldset>
          </Card>

          <Card className="space-y-4">
            <p className="text-xs font-semibold uppercase tracking-wider text-mute">2 · Brief</p>
            <Field label="Topic">
              <input
                className={inputClass}
                value={topic}
                onChange={(e) => setTopic(method === "qlora" && !compareMode ? qloraTopicOnly(e.target.value) : e.target.value)}
                placeholder="e.g. Hiring your first salesperson"
              />
            </Field>
            <div className="flex flex-wrap gap-2" aria-label="Topic ideas">
              {IDEAS.map((idea) => chip(idea, () => setTopic(idea)))}
            </div>
            {recent.length ? (
              <div>
                <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-mute">Recent in this browser</p>
                <div className="flex flex-wrap gap-2">
                  {recent
                    .filter((item, i, all) => all.findIndex((other) => other.topic === item.topic && other.format === item.format) === i)
                    .slice(0, 3)
                    .map((item) =>
                    chip(`${item.topic.slice(0, 34)}${item.topic.length > 34 ? "…" : ""}`, () => {
                      setTopic(item.topic);
                      setFormat(item.format);
                    }),
                  )}
                </div>
              </div>
            ) : null}

            {method === "qlora" && !compareMode ? (
              <div className="rounded-xl border border-line bg-card-2/60 p-3 text-xs leading-relaxed text-mute">
                <p className="font-medium text-text">QLoRA receives this tested format-specific request:</p>
                <p className="mt-1 break-words font-mono text-[11px] text-mute">
                  {qloraPrompt(format, qloraTopicOnly(topic).trim())}
                </p>
                <p className="mt-2">
                  Audience, goal, and call to action are for Base Llama and PersonaRAG. Retrieved-chunk settings
                  are PersonaRAG-only and are not sent to QLoRA.
                </p>
              </div>
            ) : null}

            {needsBrief ? (
              <>
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
                <div className="flex flex-wrap gap-2" aria-label="Audience and goal ideas">
                  {AUDIENCES.map((a) => chip(a, () => setAudience(a)))}
                  {GOALS.map((g) => chip(g, () => setGoal(g)))}
                </div>
                <Field label="Call to action">
                  <input className={inputClass} value={cta} onChange={(e) => setCta(e.target.value)} />
                </Field>
              </>
            ) : null}
          </Card>

          <Card className="space-y-4">
            <p className="text-xs font-semibold uppercase tracking-wider text-mute">3 · How it is written</p>
            <fieldset disabled={compareMode} className="grid grid-cols-2 gap-2.5 disabled:opacity-50">
              <legend className="sr-only">Writing method</legend>
              {visibleMethods.map((item) => {
                const on = method === item.id && !compareMode;
                return (
                  <label
                    key={item.id}
                    className={`lift block cursor-pointer rounded-xl border p-3 ${
                      on ? "border-primary bg-primary/8 ring-1 ring-primary/30" : "border-line hover:bg-card-2"
                    }`}
                  >
                    <input
                      type="radio"
                      name="method"
                      value={item.id}
                      checked={method === item.id}
                      onChange={() => setMethod(item.id)}
                      className="sr-only"
                    />
                    <span className="block text-sm font-medium">{item.label}</span>
                    <span className="block text-xs text-mute">{item.model}</span>
                  </label>
                );
              })}
            </fieldset>

            <p key={compareMode ? "compare" : method} className="fade-in text-xs leading-relaxed text-mute">
              {compareMode
                ? "The shared brief is held constant. Each method keeps its own production prompt contract; direct QLoRA receives only its training-aligned topic instruction."
                : activeMethod.detail}
            </p>
            {modelFStatus && !modelFStatus.available ? (
              <p className="text-xs leading-relaxed text-mute">
                LightRAG + Relabel is staged but unavailable: {modelFStatus.detail}
              </p>
            ) : null}

            <div className="flex items-start gap-3 rounded-xl border border-line bg-card-2/50 p-3">
              <Switch
                checked={compareMode}
                label="Compare all deployed methods"
                onChange={(on) => {
                  setCompareMode(on);
                  setComparison([]);
                  setError(null);
                }}
              />
              <div>
                <p className="text-sm font-medium">Compare all deployed methods</p>
                <p className="mt-0.5 text-xs leading-relaxed text-mute">
                  Runs Base Llama, PersonaRAG, and direct QLoRA one at a time. This can take several minutes.
                </p>
              </div>
            </div>

            {method === "personarag" || compareMode ? (
              <div>
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-xs font-semibold uppercase tracking-wider text-mute">Retrieved chunks</span>
                  <span className="pop rounded-md bg-primary/12 px-2 py-0.5 text-xs font-semibold tabular-nums text-primary" key={k}>
                    {k}
                  </span>
                </div>
                <RangeInput value={k} min={1} max={20} onChange={setK} label="Retrieved chunks" />
              </div>
            ) : null}
          </Card>

          <div className="sticky bottom-4 z-10 rounded-2xl border border-line bg-card/95 p-3 shadow-soft backdrop-blur">
            <div className="flex items-center gap-3">
              <div className="min-w-0 flex-1">
                <div className="h-1.5 overflow-hidden rounded-full bg-line" role="progressbar" aria-label="Brief readiness" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(readiness * 100)}>
                  <div className="bg-accent-gradient h-full rounded-full transition-[width] duration-500" style={{ width: `${readiness * 100}%` }} />
                </div>
                <p className="mt-1.5 truncate text-xs text-mute">
                  {missing.length > 0 ? `Add ${missing.join(", ")} to continue.` : "Brief is ready."}
                </p>
              </div>
              <Button onClick={run} disabled={busy || missing.length > 0} className="px-5">
                <Icon name={busy ? "pulse" : "arrow"} className="h-4 w-4" />
                {busy ? (compareMode ? "Comparing..." : "Generating...") : compareMode ? "Compare methods" : result ? "Regenerate" : "Generate"}
              </Button>
            </div>
          </div>
        </div>

        {/* ---------------- Output ---------------- */}
        <section aria-label="Output" className="min-w-0 space-y-4">
          {error ? (
            <Notice tone="error" title={error.message}>
              {error.hint}
            </Notice>
          ) : null}

          {compareMode && comparison.length ? (
            <LiveComparison entries={comparison} />
          ) : busy ? (
            <Card className="fade-in space-y-5" >
              <div role="status" aria-live="polite" className="space-y-5">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <span className="relative grid h-3 w-3 place-items-center">
                      <span className="dot-pulse h-2.5 w-2.5 rounded-full bg-primary" />
                    </span>
                    <p className="font-medium">Writing with {activeMethod.label}</p>
                  </div>
                  <span className="rounded-full bg-card-2 px-3 py-1 text-sm font-medium tabular-nums">{elapsed} s</span>
                </div>
                <SkeletonLines lines={7} />
                <div>
                  <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-mute">What happens</p>
                  <ol className="space-y-1.5 text-sm text-mute">
                    {stages.map((stage, i) => (
                      <li key={stage} className="flex items-center gap-2.5">
                        <span className="grid h-5 w-5 place-items-center rounded-full bg-card-2 text-[11px] font-semibold text-mute">{i + 1}</span>
                        {stage}
                      </li>
                    ))}
                  </ol>
                  <p className="mt-3 text-xs text-mute">
                    {compareMode
                      ? "Each deployed method runs in turn. The next starts after the current one finishes or fails."
                      : "The local model reports no live progress, so the timer shows real elapsed time. This can take a minute."}
                  </p>
                </div>
              </div>
            </Card>
          ) : result ? (
            <Card className="fade-in space-y-5 p-0" >
              <div className="flex flex-wrap items-center justify-between gap-4 p-5 pb-0">
                <div className="flex flex-wrap items-center gap-6">
                  <Ring
                    value={result.metrics.wordCount}
                    max={meta.target}
                    label="Words"
                    sub={`words · aim for ${meta.length}`}
                  />
                  <dl className="grid grid-cols-3 gap-5 text-sm">
                    <div>
                      <dt className="text-xs text-mute">Sentences</dt>
                      <dd className="font-semibold tabular-nums">{result.metrics.sentenceCount}</dd>
                    </div>
                    <div>
                      <dt className="text-xs text-mute">Characters</dt>
                      <dd className="font-semibold tabular-nums">{result.metrics.characterCount}</dd>
                    </div>
                    <div>
                      <dt className="text-xs text-mute">Time</dt>
                      <dd className="font-semibold tabular-nums">
                        {result.latencyMs !== null ? `${(result.latencyMs / 1000).toFixed(1)} s` : "n/a"}
                      </dd>
                    </div>
                  </dl>
                </div>
                <div className="flex gap-2">
                  <Button variant="ghost" onClick={copy}>
                    <Icon name={copied ? "check" : "copy"} className="h-4 w-4" />
                    {copied ? "Copied" : "Copy"}
                  </Button>
                  <Button variant="ghost" onClick={download}>
                    <Icon name="download" className="h-4 w-4" />
                    .md
                  </Button>
                </div>
              </div>

              <div className="px-5">
                <Tabs<StudioTab>
                  label="Draft views"
                  value={tab}
                  onChange={setTab}
                  tabs={[
                    { id: "draft", label: "Draft" },
                    { id: "preview", label: "Preview" },
                    { id: "checks", label: "Checks", badge: scored.length ? `${passed}/${scored.length}` : "n/a" },
                    { id: "sources", label: "Sources", badge: result.sources.length },
                  ]}
                />
              </div>

              <div className="px-5 pb-5" role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`}>
                {tab === "draft" ? (
                  <div className="space-y-3">
                    {result.mock ? (
                      <Notice tone="warn" title="Demo draft">
                        This is sample text, not real model output.
                      </Notice>
                    ) : null}
                    <div className="rounded-xl border border-line bg-card-2/50 p-6">
                      <Typewriter
                        key={result.content}
                        text={result.content}
                        className="whitespace-pre-wrap wrap-break-word text-[16px] leading-[1.75]"
                      />
                    </div>
                    <p className="text-xs text-mute">
                      {result.method === "personarag" ? "PersonaRAG" : result.method === "qlora" ? "direct QLoRA" : result.method === "model_f" ? "LightRAG + Relabel" : "Base Llama"} · {result.model}
                    </p>
                  </div>
                ) : null}

                {tab === "preview" ? (
                  <div className="fade-in space-y-3">
                    <p className="text-xs text-mute">How this {meta.noun} may look when it is read. Layout only.</p>
                    <DraftPreview format={resultFormat} content={result.content} />
                  </div>
                ) : null}

                {tab === "checks" ? (
                  <div className="fade-in space-y-4">
                    <div className="rounded-xl border border-line bg-card-2/50 p-4">
                      <p className="text-sm font-medium">Deterministic draft checks</p>
                      <p className="mt-1 text-xs text-mute">
                        Transparent rule-based diagnostics, not an overall quality or factuality grade.
                      </p>
                      <div className="mt-3 space-y-2">
                        {checkList.map(([label, item]) => (
                          <div key={label} className="flex flex-wrap items-start justify-between gap-2 rounded-lg border border-line bg-card p-2.5 text-xs">
                            <div className="min-w-0">
                              <p className="font-medium text-text">{label}</p>
                              <p className="mt-0.5 text-mute">{item.detail}</p>
                            </div>
                            <span className={`shrink-0 rounded-full border px-2 py-0.5 font-medium ${CHECK_STATUS_CLASS[item.status]}`}>
                              {checkLabel(item.status)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="rounded-xl border border-line bg-card-2/50 p-4">
                      <p className="text-sm font-medium">Live operational measurement</p>
                      <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-mute sm:grid-cols-4">
                        <span>{result.metrics.wordCount} words</span>
                        <span>{result.metrics.sentenceCount} sentences</span>
                        <span>{result.metrics.characterCount} characters</span>
                        <span>{result.metrics.sourceCount} returned sources</span>
                      </div>
                      <p className="mt-2 text-xs text-mute">
                        These are comparable runtime/output measurements, not a score of factuality, writing quality, or persona voice.
                      </p>
                    </div>

                    <div className="rounded-xl border border-line bg-card-2/50 p-4">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div>
                          <p className="text-sm font-medium">Optional grammar screen</p>
                          <p className="mt-1 text-xs text-mute">Screening result, not a grade. Informal Lemkin-style wording can produce false alarms.</p>
                        </div>
                        <Button variant="ghost" onClick={runGrammarCheck} disabled={grammarBusy}>
                          {grammarBusy ? "Checking…" : "Run grammar check"}
                        </Button>
                      </div>
                      {grammarError ? <p className="mt-3 text-xs text-warn">{grammarError}</p> : null}
                      {grammarResult ? (
                        <div className="mt-3 border-t border-line pt-3">
                          <p className="text-sm">
                            {grammarResult.issue_count} screened issue{grammarResult.issue_count === 1 ? "" : "s"} in {grammarResult.word_count} words
                            <span className="text-mute"> ({grammarResult.issues_per_100_words} per 100 words)</span>
                          </p>
                          {grammarResult.issues.length ? (
                            <ul className="mt-2 space-y-2 text-xs text-mute">
                              {grammarResult.issues.slice(0, 8).map((issue, index) => (
                                <li key={`${issue.rule_id}-${issue.offset}-${index}`} className="rounded-lg border border-line bg-card p-2">
                                  <span className="font-medium text-text">“{issue.matched_text}”</span>: {issue.message}
                                  {issue.replacements.length ? <span> Suggested: {issue.replacements.join(", ")}</span> : null}
                                </li>
                              ))}
                            </ul>
                          ) : (
                            <p className="mt-2 text-xs text-mute">No screened issues. This is not a guarantee that the draft is error-free.</p>
                          )}
                        </div>
                      ) : null}
                    </div>
                  </div>
                ) : null}

                {tab === "sources" ? (
                  <div className="fade-in">
                    {displayedSourceResult && displayedSourceResult.sources.length > 0 ? (
                      <ul className="space-y-3">
                        {displayedSourceResult.sources.map((s, i) => (
                          <li key={s.id} className="lift rounded-xl border border-line bg-card-2/50 p-3.5">
                            <div className="mb-2 flex items-center justify-between gap-2">
                              <span className="flex items-center gap-2">
                                <span className="grid h-5 w-5 place-items-center rounded-md bg-primary/12 text-[11px] font-semibold text-primary">{i + 1}</span>
                                <SourceTag type={s.sourceType} />
                              </span>
                              {s.title ? <span className="truncate text-xs text-mute">{s.title}</span> : null}
                            </div>
                            <p className="text-xs leading-relaxed text-mute">{s.text}</p>
                            <div className="mt-2">
                              <ScoreBar score={s.score} />
                            </div>
                          </li>
                        ))}
                      </ul>
                    ) : result.method === "qlora" ? (
                      <p className="text-sm text-mute">direct QLoRA writes from the merged model. It does not retrieve Chroma sources.</p>
                    ) : result.method === "baseline" ? (
                      <p className="text-sm text-mute">Base Llama writes from the user brief only. It does not use a persona profile or retrieve Chroma sources.</p>
                    ) : (
                      <p className="text-sm text-mute">
                        The backend did not return sources for this draft. The API needs to include them; see the API contract.
                      </p>
                    )}
                  </div>
                ) : null}
              </div>
            </Card>
          ) : (
            <EmptyState title={compareMode ? "Your comparison will appear here" : "Your draft will appear here"}>
              Fill in the brief and press {compareMode ? "Compare methods" : "Generate"}. You will get the draft, a
              format preview, rule-based checks and the sources it used.
            </EmptyState>
          )}
        </section>
      </div>
    </>
  );
}

export default function GeneratorPage() {
  const [persona] = usePersonaId();
  return persona === "musk" ? <AskPanel /> : <WriteStudio />;
}

"use client";

import { useEffect, useMemo, useState } from "react";
import { ApiError, checkGrammar, generate, getPersona } from "@/lib/api";
import { type ComparisonEntry, LiveComparison } from "@/components/LiveComparison";
import { addHistory } from "@/lib/history";
import {
  FORMATS,
  type FormatId,
  type GenerationMethod,
  type GenerateResult,
  type GrammarCheckResult,
  type PersonaProfile,
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
  const [grammarResult, setGrammarResult] = useState<GrammarCheckResult | null>(null);
  const [grammarError, setGrammarError] = useState<string | null>(null);
  const [grammarBusy, setGrammarBusy] = useState(false);

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

  const missing = useMemo(() => {
    const out: string[] = [];
    if (!topic.trim()) out.push("topic");
    if (compareMode || method !== "qlora") {
      if (!audience.trim()) out.push("audience");
      if (!goal.trim()) out.push("goal");
    }
    return out;
  }, [topic, audience, goal, method, compareMode]);

  function requestFor(targetMethod: GenerationMethod) {
    // QLoRA remains training-aligned. The shared brief is the topic; the
    // baseline and RAG paths additionally receive audience/goal/CTA because
    // those are part of their deployed prompt contracts.
    return targetMethod === "qlora"
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

  const activeMethod = METHODS.find((item) => item.id === method) ?? METHODS[0];
  const displayedSourceResult = compareMode ? comparison.find((entry) => entry.method === "personarag")?.result : result;

  return (
    <>
      <PageHeader
        title="Content generator"
        subtitle="Generate one draft or compare the deployed architectures from one shared brief."
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
            <fieldset>
              <legend className="mb-2 text-xs font-medium uppercase tracking-wide text-mute">
                Writing method
              </legend>
              <div className="space-y-2">
                {METHODS.map((item) => (
                  <label
                    key={item.id}
                    className={`block cursor-pointer rounded-xl border p-3 transition ${
                      method === item.id
                        ? "border-primary bg-primary/10"
                        : "border-line hover:border-primary/50"
                    }`}
                  >
                    <input
                      type="radio"
                      name="method"
                      value={item.id}
                      checked={method === item.id}
                      onChange={() => setMethod(item.id)}
                      disabled={compareMode}
                      className="sr-only"
                    />
                    <span className="block text-sm font-medium">{item.label}</span>
                    <span className="block text-xs text-mute">{item.model}</span>
                  </label>
                ))}
              </div>
            </fieldset>
            <label className={`block rounded-xl border p-3 transition ${
              compareMode ? "border-violet bg-violet/10" : "border-line hover:border-primary/50"
            }`}>
              <input
                type="checkbox"
                checked={compareMode}
                onChange={(event) => {
                  setCompareMode(event.target.checked);
                  setComparison([]);
                  setError(null);
                }}
                className="mr-2 accent-primary"
              />
              <span className="text-sm font-medium">Compare all deployed methods</span>
              <span className="mt-1 block text-xs leading-relaxed text-mute">
                Runs Base Llama, PersonaRAG, and direct QLoRA one at a time. This can take several minutes.
              </span>
            </label>
            <p className="text-xs leading-relaxed text-mute">
              {compareMode
                ? "The shared brief is held constant. Each method keeps its own production prompt contract; direct QLoRA receives only its training-aligned topic instruction."
                : activeMethod.detail}
            </p>
            {(method === "personarag" || compareMode) ? (
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
            ) : null}
          </Card>
        </div>

        {/* Center: brief and output */}
        <div className="min-w-0 space-y-5">
          <Card className="space-y-4">
            <Field label="Topic">
              <input
                className={inputClass}
                value={topic}
                onChange={(e) => setTopic(method === "qlora" && !compareMode ? qloraTopicOnly(e.target.value) : e.target.value)}
                placeholder="e.g. Hiring your first salesperson"
              />
            </Field>
            {method === "qlora" && !compareMode ? (
              <div className="rounded-xl border border-line bg-bg p-3 text-xs leading-relaxed text-mute">
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
            {(method !== "qlora" || compareMode) ? (
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
                <Field label="Call to action">
                  <input className={inputClass} value={cta} onChange={(e) => setCta(e.target.value)} />
                </Field>
              </>
            ) : null}
            <div className="flex flex-wrap items-center gap-3">
              <Button onClick={run} disabled={busy || missing.length > 0}>
                {busy ? (compareMode ? "Comparing..." : "Generating...") : compareMode ? "Compare methods" : result ? "Regenerate" : "Generate"}
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
              <h2 className="font-semibold">{compareMode ? "Comparison results" : "Draft"}</h2>
              {result && !compareMode ? (
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

            {compareMode && comparison.length ? (
              <LiveComparison entries={comparison} />
            ) : busy ? (
              <p className="animate-pulse text-sm text-mute" role="status">
                {compareMode
                  ? "Running each deployed method sequentially. The next method starts after the current one completes or fails."
                  : method === "personarag"
                  ? "Embedding your brief, retrieving sources and generating."
                  : method === "qlora"
                    ? "Generating with direct QLoRA."
                    : "Generating with Base Llama."} This can take a minute on a local model.
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
                  {result.metrics.wordCount} words - {result.method === "personarag" ? "PersonaRAG" : result.method === "qlora" ? "direct QLoRA" : "Base Llama"} - {result.model}
                  {result.latencyMs !== null ? ` - ${(result.latencyMs / 1000).toFixed(1)} s` : ""}
                </p>
                <div className="mt-4 rounded-xl border border-line bg-bg p-3">
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
                <div className="mt-4 rounded-xl border border-line bg-bg p-3">
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
                            <li key={`${issue.rule_id}-${issue.offset}-${index}`} className="rounded-lg border border-line p-2">
                              <span className="font-medium text-text">“{issue.matched_text}”</span>: {issue.message}
                              {issue.replacements.length ? <span> Suggested: {issue.replacements.join(", ")}</span> : null}
                            </li>
                          ))}
                        </ul>
                      ) : <p className="mt-2 text-xs text-mute">No screened issues. This is not a guarantee that the draft is error-free.</p>}
                    </div>
                  ) : null}
                </div>
              </>
            ) : (
              <EmptyState title={compareMode ? "Your comparison will appear here" : "Your draft will appear here"}>
                Fill in the brief and press {compareMode ? "Compare methods" : "Generate"}.
              </EmptyState>
            )}
          </Card>
        </div>

        {/* Right: active profile and sources */}
        <div className="space-y-5">
          <Card className="h-fit">
            <h2 className="mb-3 font-semibold">Active persona</h2>
            {persona ? (
              <div className="space-y-2">
                <div>
                  <p className="text-sm font-medium">{persona.name}</p>
                  <p className="text-xs text-mute">{persona.source}</p>
                </div>
                <p className="whitespace-pre-line text-xs leading-relaxed text-mute">{persona.summary}</p>
                {persona.mock ? <p className="text-xs text-warn">Sample persona profile (mock mode).</p> : null}
              </div>
            ) : personaError ? (
              <p className="text-sm text-mute">{personaError}</p>
            ) : (
              <p className="text-sm text-mute">Loading active persona profile…</p>
            )}
          </Card>

          <Card className="h-fit">
            <h2 className="mb-3 font-semibold">Retrieved sources</h2>
          {displayedSourceResult && displayedSourceResult.sources.length > 0 ? (
            <ul className="space-y-3">
              {displayedSourceResult.sources.map((s) => (
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
          ) : compareMode ? (
            <p className="text-sm text-mute">
              PersonaRAG sources for the shared brief appear in its comparison card after that method completes.
            </p>
          ) : result?.method === "qlora" ? (
            <p className="text-sm text-mute">
              direct QLoRA writes from the merged model. It does not retrieve Chroma sources.
            </p>
          ) : result?.method === "baseline" ? (
            <p className="text-sm text-mute">
              Base Llama writes from the user brief only. It does not use a persona profile or retrieve Chroma sources.
            </p>
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
      </div>
    </>
  );
}

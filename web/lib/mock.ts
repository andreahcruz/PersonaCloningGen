import type {
  EvaluationData,
  FormatId,
  GenerateRequest,
  GenerateResult,
  ModelInfo,
  PersonaProfile,
  GrammarCheckResult,
  ServiceStatus,
  Source,
  Statistics,
} from "./types";

// Sample data for demo mode only. The banner in the UI says so. Counts in MOCK_STATS are the
// real numbers from the 2026-09-18 full pipeline run; everything else is placeholder text.

export const MOCK_STATS: Statistics = {
  totalDocuments: 31443,
  searchableChunks: 38105,
  trainingExamples: 19961,
};

export const MOCK_MODELS: ModelInfo[] = [
  { name: "lemkin-clone", kind: "fine-tuned", available: true },
  { name: "llama3.1", kind: "base", available: true },
  { name: "llama3.2:3b", kind: "base", available: true },
  { name: "nomic-embed-text", kind: "embedding", available: true },
];

export const MOCK_PERSONA: PersonaProfile = {
  name: "Jason Lemkin",
  source: "Sample profile — Jason Lemkin operator voice",
  summary: "Sample persona profile for mock mode. Direct B2B SaaS operator tone with concrete examples and practical founder advice.",
  mock: true,
};

const SAMPLE: Source[] = [
  {
    id: "sample-1",
    sourceType: "blog",
    title: "Sample blog excerpt",
    score: 0.87,
    text: "[Sample text] Your first sales hire should be a builder who can close, not a manager. Hire for scrappiness, then layer process on top once you have repeatable wins.",
  },
  {
    id: "sample-2",
    sourceType: "linkedin",
    title: "Sample LinkedIn excerpt",
    score: 0.82,
    text: "[Sample text] Most founders wait too long to hire sales. If you have paying customers and you are still the only one selling, you are already late.",
  },
  {
    id: "sample-3",
    sourceType: "x",
    title: "Sample X excerpt",
    score: 0.76,
    text: "[Sample text] The best early AE has done the job at a similar-stage company and can show a real quota history.",
  },
  {
    id: "sample-4",
    sourceType: "youtube",
    title: "Sample YouTube excerpt",
    score: 0.71,
    text: "[Sample text] In the first year, what matters is learning what actually sells, so keep the hire close to you and review every deal together.",
  },
  {
    id: "sample-5",
    sourceType: "blog",
    title: "Sample blog excerpt",
    score: 0.66,
    text: "[Sample text] Do not hand the playbook to someone else until you have written it yourself and it has worked at least ten times.",
  },
];

export function mockSources(k: number): Source[] {
  return SAMPLE.slice(0, Math.max(1, Math.min(k, SAMPLE.length)));
}

export function mockSearch(query: string, type: string): Source[] {
  const q = query.trim().toLowerCase();
  return SAMPLE.filter(
    (s) => (type === "all" || s.sourceType === type) && (!q || s.text.toLowerCase().includes(q)),
  );
}

function cta(r: GenerateRequest, fallback: string): string {
  return r.cta && r.cta !== "none" ? r.cta : fallback;
}

const DRAFTS: Record<FormatId, (r: GenerateRequest) => string> = {
  linkedin_post: (r) =>
    `[Demo draft, not real model output]\n\n${r.topic}\n\nHere is the one thing I would tell ${r.audience}: ${r.goal}.\n\nStart small, measure what works, and repeat it.\n\n${cta(r, "")}`.trim(),
  blog_draft: (r) =>
    `[Demo draft, not real model output]\n\n# ${r.topic}\n\nWritten for ${r.audience}. Goal: ${r.goal}.\n\n## The short version\nA real draft appears here once the backend is connected.\n\n## What to do next\n${cta(r, "Pick one action this week.")}`,
  x_thread: (r) =>
    `[Demo draft, not real model output]\n\n1/ ${r.topic}\n\n2/ For ${r.audience}: ${r.goal}.\n\n3/ ${cta(r, "Follow for more.")}`,
  youtube_script: (r) =>
    `[Demo draft, not real model output]\n\nHOOK: ${r.topic}\n\nINTRO: Today, for ${r.audience}, we cover ${r.goal}.\n\nOUTRO: ${cta(r, "Thanks for watching.")}`,
};

export function mockGenerate(req: GenerateRequest): GenerateResult {
  const content = DRAFTS[req.format](req);
  const sources = req.method === "personarag" ? mockSources(req.k) : [];
  return {
    content,
    sources,
    model: req.model ?? (req.method === "qlora" ? "lemkin-qlora" : "llama3.1"),
    latencyMs: 1200,
    requestId: "demo",
    method: req.method,
    metrics: {
      wordCount: content.trim().split(/\s+/).filter(Boolean).length,
      characterCount: content.length,
      sentenceCount: Math.max(1, (content.match(/[.!?]+(?:\s|$)/g) ?? []).length),
      sourceCount: sources.length,
      sourceTypes: [...new Set(sources.map((source) => source.sourceType))],
      scope: "operational",
    },
    checks: {
      completion: { status: "pass", detail: "Ends with terminal punctuation.", value: null },
      formatCompliance: { status: "pass", detail: "Sample draft matches its selected mock format.", value: null },
      promptEcho: { status: "pass", detail: "No long exact instruction span was repeated.", value: 0 },
      unsupportedNumbers: sources.length
        ? { status: "pass", detail: "All detected numeric forms appear in returned sources.", value: 0 }
        : { status: "na", detail: "N/A: this method did not retrieve source evidence.", value: null },
      sourceCopying: sources.length
        ? { status: "pass", detail: "No long exact source span was copied.", value: 0 }
        : { status: "na", detail: "N/A: this method did not retrieve source evidence.", value: null },
      scope: "deterministic",
    },
    mock: true,
  };
}

export const MOCK_HEALTH: ServiceStatus[] = [
  { name: "FastAPI", status: "ok", detail: "Sample status" },
  { name: "Ollama", status: "ok", detail: "Sample status" },
  { name: "ChromaDB", status: "ok", detail: "Sample status" },
  { name: "Embedding model", status: "ok", detail: "nomic-embed-text (sample)" },
  { name: "Generation model", status: "degraded", detail: "Sample: model still loading" },
  { name: "Airflow", status: "unknown", detail: "Not checked" },
  { name: "MinIO", status: "down", detail: "Sample: unreachable" },
];

// Clearly labelled, compact sample of the versioned research manifest for mock mode.
export const EVALUATION: EvaluationData = {
  version: "sample-evidence-v1",
  title: "Sample evaluation evidence",
  description: "Mock-mode sample. It is not a regenerated evaluation run.",
  groups: [
    {
      id: "grammar",
      title: "Grammar screening (sample)",
      scope: "Research benchmark — not a model ranking",
      summary: "Sample summary of the published LanguageTool evaluation.",
      sample_size: "Sample data only",
      source: { branch: "sample", commit: "sample", report_url: "#" },
      metrics: [{ label: "Seeded mistakes detected", value: "19 / 40", detail: "Sample evidence." }],
      limitations: ["Mock mode does not load a live manifest."],
    },
  ],
};

export function mockGrammarCheck(text: string): GrammarCheckResult {
  const wordCount = text.trim().split(/\s+/).filter(Boolean).length;
  return { word_count: wordCount, issue_count: 0, issues_per_100_words: 0, issues: [] };
}

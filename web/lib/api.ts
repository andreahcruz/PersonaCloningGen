import { API_BASE, IS_MOCK } from "./config";
import {
  EVALUATION,
  MOCK_HEALTH,
  MOCK_MODELS,
  MOCK_STATS,
  mockGenerate,
  mockSearch,
} from "./mock";
import type {
  EvaluationData,
  FormatId,
  GenerateRequest,
  GenerateResult,
  ModelInfo,
  ServiceStatus,
  Source,
  SourceType,
  Statistics,
} from "./types";
import { FORMATS } from "./types";

// Every page talks to the backend only through this file. See docs/FRONTEND_API_CONTRACT.md for
// the endpoints. "Exists" means the backend on QLoRAFT has it today; "Proposed" means we still
// need it from the backend owners.

export class ApiError extends Error {
  status: number;
  hint?: string;
  constructor(message: string, status = 0, hint?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.hint = hint;
  }
}

/** Thrown for endpoints the backend does not have yet, so pages can show an empty state. */
export class NotAvailableError extends ApiError {
  constructor(endpoint: string) {
    super(
      `The backend does not provide ${endpoint} yet.`,
      404,
      "See docs/FRONTEND_API_CONTRACT.md for what the frontend expects.",
    );
    this.name = "NotAvailableError";
  }
}

const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError(
      "Could not reach the backend.",
      0,
      "Check that the backend is running and NEXT_PUBLIC_API_BASE points at it.",
    );
  }
  if (!res.ok) {
    if (res.status === 404) throw new NotAvailableError(path);
    let detail = `Request failed (${res.status})`;
    let hint: string | undefined;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
      if (typeof body?.hint === "string") hint = body.hint;
    } catch {
      /* body was not JSON */
    }
    throw new ApiError(detail, res.status, hint);
  }
  return (await res.json()) as T;
}

// ---- Formats (exists: GET /formats -> string[]) ----
export async function getFormats(): Promise<FormatId[]> {
  if (IS_MOCK) return FORMATS.map((f) => f.id);
  const ids = await request<string[]>("/formats");
  const known = new Set<string>(FORMATS.map((f) => f.id));
  return ids.filter((i): i is FormatId => known.has(i));
}

// ---- Models (proposed: GET /models) ----
export async function getModels(): Promise<{ models: ModelInfo[]; fromBackend: boolean }> {
  if (IS_MOCK) return { models: MOCK_MODELS, fromBackend: false };
  try {
    const body = await request<{ models: ModelInfo[] }>("/models");
    return { models: body.models, fromBackend: true };
  } catch (e) {
    if (e instanceof NotAvailableError) {
      // Fall back to the two names the project uses so the picker still works.
      return {
        models: [
          { name: "lemkin-clone", kind: "fine-tuned", available: true },
          { name: "llama3.1", kind: "base", available: true },
        ],
        fromBackend: false,
      };
    }
    throw e;
  }
}

// ---- Generate (exists: POST /generate -> { draft }; target adds sources, model, latency) ----
interface RawSource {
  id?: string;
  text?: string;
  score?: number | null;
  source_type?: string;
  title?: string;
  url?: string;
  date?: string;
}

interface RawGenerate {
  draft?: string;
  content?: string;
  sources?: RawSource[];
  model?: string;
  latency_ms?: number;
  request_id?: string;
}

const SOURCE_TYPES: SourceType[] = ["blog", "linkedin", "x", "youtube"];

function mapSource(s: RawSource, i: number): Source {
  const t = (s.source_type ?? "unknown").toLowerCase();
  return {
    id: s.id ?? `src-${i}`,
    text: s.text ?? "",
    score: typeof s.score === "number" ? s.score : null,
    sourceType: (SOURCE_TYPES as string[]).includes(t) ? (t as SourceType) : "unknown",
    title: s.title,
    url: s.url,
    date: s.date,
  };
}

export async function generate(req: GenerateRequest): Promise<GenerateResult> {
  if (IS_MOCK) {
    await wait(900);
    return mockGenerate(req);
  }
  const started = performance.now();
  const raw = await request<RawGenerate>("/generate", {
    method: "POST",
    body: JSON.stringify(req),
  });
  const content = raw.content ?? raw.draft ?? "";
  if (!content.trim()) {
    throw new ApiError("The model returned an empty draft.", 502, "Try again or pick another model.");
  }
  return {
    content,
    sources: (raw.sources ?? []).map(mapSource),
    model: raw.model ?? req.model ?? "default",
    latencyMs: raw.latency_ms ?? Math.round(performance.now() - started),
    requestId: raw.request_id,
    mock: false,
  };
}

// ---- Sources (proposed: GET /sources?q=&type=&limit=) ----
export async function searchSources(query: string, type: string): Promise<Source[]> {
  if (IS_MOCK) {
    await wait(250);
    return mockSearch(query, type);
  }
  const params = new URLSearchParams({ q: query, type, limit: "30" });
  const body = await request<{ items: RawSource[] }>(`/sources?${params}`);
  return body.items.map(mapSource);
}

// ---- Statistics (proposed: GET /statistics) ----
export async function getStatistics(): Promise<Statistics> {
  if (IS_MOCK) return MOCK_STATS;
  const b = await request<{
    total_documents: number;
    searchable_chunks: number;
    training_examples: number;
  }>("/statistics");
  return {
    totalDocuments: b.total_documents,
    searchableChunks: b.searchable_chunks,
    trainingExamples: b.training_examples,
  };
}

// ---- Health (exists: GET /health -> { status }; proposed: GET /health/dependencies) ----
export async function getHealth(): Promise<{ services: ServiceStatus[]; detailed: boolean }> {
  if (IS_MOCK) return { services: MOCK_HEALTH, detailed: true };
  try {
    const body = await request<{ services: ServiceStatus[] }>("/health/dependencies");
    return { services: body.services, detailed: true };
  } catch (e) {
    if (!(e instanceof NotAvailableError)) throw e;
    // Only the basic check exists today.
    await request<{ status: string }>("/health");
    return { services: [{ name: "FastAPI", status: "ok" }], detailed: false };
  }
}

// ---- Evaluation (proposed: GET /evaluation) ----
export async function getEvaluation(): Promise<EvaluationData> {
  if (IS_MOCK) return EVALUATION;
  try {
    return await request<EvaluationData>("/evaluation");
  } catch (e) {
    // The reported numbers live in the repo, so show them even before the endpoint exists.
    if (e instanceof NotAvailableError) return EVALUATION;
    throw e;
  }
}

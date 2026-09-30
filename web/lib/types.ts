export type FormatId =
  | "linkedin_post"
  | "blog_draft"
  | "x_thread"
  | "youtube_script";

export interface FormatOption {
  id: FormatId;
  label: string;
  blurb: string;
}

// The ids match the keys in formats/format_specs.yaml on the backend.
export const FORMATS: FormatOption[] = [
  { id: "linkedin_post", label: "LinkedIn post", blurb: "Short, punchy, one idea" },
  { id: "blog_draft", label: "Blog article", blurb: "Long form with sections" },
  { id: "x_thread", label: "X / Twitter thread", blurb: "Numbered, tweet-sized" },
  { id: "youtube_script", label: "YouTube script", blurb: "Spoken, with hooks" },
];

export type SourceType = "blog" | "linkedin" | "x" | "youtube" | "unknown";

export interface Source {
  id: string;
  text: string;
  /** Similarity in 0..1, or null when the backend does not report one. */
  score: number | null;
  sourceType: SourceType;
  title?: string;
  url?: string;
  date?: string;
}

export interface GenerateRequest {
  format: FormatId;
  topic: string;
  audience: string;
  goal: string;
  cta: string;
  k: number;
  model?: string;
  temperature?: number;
}

export interface GenerateResult {
  content: string;
  sources: Source[];
  model: string;
  /** Server-reported latency, else measured in the browser. */
  latencyMs: number | null;
  requestId?: string;
  /** True when the data is sample data, not real model output. */
  mock: boolean;
}

export interface ModelInfo {
  name: string;
  kind: "fine-tuned" | "base" | "embedding";
  available: boolean;
}

export type ServiceState = "ok" | "degraded" | "down" | "unknown";

export interface ServiceStatus {
  name: string;
  status: ServiceState;
  detail?: string;
}

export interface Statistics {
  totalDocuments: number;
  searchableChunks: number;
  trainingExamples: number;
  bySource?: Partial<Record<SourceType, number>>;
}

export interface EvalMetric {
  key: string;
  label: string;
  /** 0..1, or null for "pending". */
  value: number | null;
}

export interface EvalRun {
  id: string;
  label: string;
  note: string;
  prompts: number;
  metrics: EvalMetric[];
}

export interface EvalMatrixCell {
  system: "Base Llama" | "Base + RAG" | "Fine-tuned" | "Fine-tuned + RAG";
  available: boolean;
}

export interface EvaluationData {
  runs: EvalRun[];
  matrix: EvalMatrixCell[];
}

export interface HistoryItem {
  id: string;
  at: string;
  format: FormatId;
  topic: string;
  model: string;
  latencyMs: number | null;
  mock: boolean;
}

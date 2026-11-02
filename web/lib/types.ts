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
export type GenerationMethod = "baseline" | "personarag" | "qlora" | "model_f";

/** Readiness of the optional LightRAG + Relabel deployment path. */
export interface ModelFStatus {
  enabled: boolean;
  available: boolean;
  detail: string;
  snapshot_documents?: number;
  embedding_dimension?: number;
}

export interface PersonaProfile {
  name: string;
  source: string;
  summary: string;
  /** True only for the clearly labelled sample profile used in mock mode. */
  mock: boolean;
}

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
  method: GenerationMethod;
  model?: string;
  temperature?: number;
}

/**
 * Measurements returned for one finished request. They are operational
 * signals, not a score of factuality, writing quality, or persona voice.
 */
export interface DraftMetrics {
  wordCount: number;
  characterCount: number;
  sentenceCount: number;
  sourceCount: number;
  sourceTypes: SourceType[];
  scope: "operational";
}

export interface GenerateResult {
  content: string;
  sources: Source[];
  model: string;
  /** Server-reported latency, else measured in the browser. */
  latencyMs: number | null;
  requestId?: string;
  method: GenerationMethod;
  metrics: DraftMetrics;
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

export interface EvaluationMetric {
  label: string;
  value: string;
  detail: string;
}

export interface EvaluationSource {
  branch: string;
  commit: string;
  report_url: string;
}

export interface EvaluationComparison {
  label: string;
  columns: string[];
  rows: string[][];
}

export interface EvaluationGroup {
  id: string;
  title: string;
  scope: string;
  summary: string;
  sample_size: string;
  source: EvaluationSource;
  metrics: EvaluationMetric[];
  limitations: string[];
  comparison?: EvaluationComparison;
}

export interface EvaluationData {
  version: string;
  title: string;
  description: string;
  groups: EvaluationGroup[];
}

export interface GrammarCheckRequest {
  text: string;
}

export interface GrammarIssue {
  message: string;
  matched_text: string;
  offset: number;
  length: number;
  rule_id: string;
  replacements: string[];
}

export interface GrammarCheckResult {
  word_count: number;
  issue_count: number;
  issues_per_100_words: number;
  issues: GrammarIssue[];
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

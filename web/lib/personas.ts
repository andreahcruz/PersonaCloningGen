import type { PersonaId } from "./prefs";
import type { SourceType } from "./types";

export interface PersonaConfig {
  id: PersonaId;
  name: string;
  initials: string;
  tagline: string;
  /** "write" drafts content in the persona's voice. "ask" answers questions from sources only. */
  mode: "write" | "ask";
  modeLabel: string;
  /** What the knowledge base for this persona is made of. */
  corpus: string;
  sourceTypes: SourceType[];
  heroTitle: string;
  heroBody: string;
  cta: string;
  /** Honest status line for the live backend. */
  liveNote: string;
}

export const PERSONAS: Record<PersonaId, PersonaConfig> = {
  lemkin: {
    id: "lemkin",
    name: "Jason Lemkin",
    initials: "JL",
    tagline: "SaaS operator voice",
    mode: "write",
    modeLabel: "Writes drafts",
    corpus: "Blog, LinkedIn, X and YouTube writing",
    sourceTypes: ["blog", "linkedin", "x", "youtube"],
    heroTitle: "Write in the voice of a SaaS operator",
    heroBody:
      "Retrieve passages from Jason Lemkin's real writing and generate LinkedIn posts, blogs, X threads and YouTube scripts with RAG and QLoRA fine-tuning.",
    cta: "Start writing",
    liveNote: "Connected to the generation backend.",
  },
  musk: {
    id: "musk",
    name: "Elon Musk",
    initials: "EM",
    tagline: "Source-grounded answers",
    mode: "ask",
    modeLabel: "Answers from sources",
    corpus: "Public tweets and long-form interview transcripts",
    sourceTypes: ["x", "interview"],
    heroTitle: "Ask questions, get answers from the sources",
    heroBody:
      "Search a separate index of Elon Musk's public tweets and interview transcripts. Every answer cites the passages it used, and it never writes as him.",
    cta: "Ask a question",
    liveNote:
      "The retrieval index is built (feature/elon-musk-etl), but the API route that serves it is not available yet.",
  },
};

export const PERSONA_LIST: PersonaConfig[] = [PERSONAS.lemkin, PERSONAS.musk];

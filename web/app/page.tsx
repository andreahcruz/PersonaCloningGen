"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import HeroDemo from "@/components/HeroDemo";
import { getStatistics } from "@/lib/api";
import { useCountUp } from "@/lib/motion";
import { PERSONAS } from "@/lib/personas";
import { usePersonaId } from "@/lib/prefs";
import type { Statistics } from "@/lib/types";
import { Icon, buttonClass } from "@/components/ui";

const FEATURES = {
  lemkin: [
    { icon: "search", title: "Retrieval from real writing", body: "Drafts start from passages in Jason Lemkin's blog, LinkedIn, X and YouTube content." },
    { icon: "bolt", title: "Fine-tuned model", body: "A QLoRA-tuned Llama 3.1 8B model trained on his tone, rhythm and habits of argument." },
    { icon: "layers", title: "Four formats", body: "LinkedIn posts, blog articles, X threads and YouTube scripts from one brief." },
    { icon: "shield", title: "Visible evidence", body: "See which passages were retrieved, how closely they matched, and what the checks found." },
  ],
  musk: [
    { icon: "search", title: "Separate source index", body: "Public tweets and long-form interview transcripts, kept apart from the Lemkin index." },
    { icon: "doc", title: "Cited answers", body: "Answers point to the passages they used and say so when the sources are not enough." },
    { icon: "shield", title: "No impersonation", body: "It reports and interprets what the sources say. It does not write as Elon Musk." },
    { icon: "pulse", title: "Backend status", body: "The retrieval index is built. The API route that serves it is the next step." },
  ],
} as const;

const STEPS = {
  lemkin: [
    { title: "Describe the piece", body: "A topic, an audience and a goal. Or pick one of the suggested ideas." },
    { title: "Choose how it is written", body: "Base model, retrieval, fine-tuned, or all of them side by side." },
    { title: "Review the evidence", body: "Draft, format preview, rule-based checks and the sources used." },
  ],
  musk: [
    { title: "Ask a question", body: "In plain words, or start from an example." },
    { title: "Passages are retrieved", body: "The closest tweets and transcript chunks are found first." },
    { title: "Read a cited answer", body: "Each claim points to the source it came from." },
  ],
} as const;

function Counter({ label, value, note }: { label: string; value: number; note: string }) {
  const shown = useCountUp(value);
  return (
    <div className="px-8 py-8">
      <p className="text-4xl font-semibold tabular-nums tracking-tight sm:text-5xl">
        {new Intl.NumberFormat("en-US").format(shown)}
      </p>
      <p className="mt-2 text-sm font-medium">{label}</p>
      <p className="text-xs text-mute">{note}</p>
    </div>
  );
}

export default function HomePage() {
  const [personaId] = usePersonaId();
  const persona = PERSONAS[personaId];
  const [stats, setStats] = useState<Statistics | null>(null);

  useEffect(() => {
    let active = true;
    getStatistics()
      .then((value) => {
        if (active) setStats(value);
      })
      .catch(() => {
        /* the band simply stays hidden when the backend has no statistics route */
      });
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="space-y-24">
      {/* Hero */}
      <section className="relative overflow-hidden rounded-2xl border border-line bg-card px-6 py-14 shadow-soft sm:px-12 sm:py-16">
        <div aria-hidden className="bg-grid pointer-events-none absolute inset-0" />
        <div className="relative grid items-center gap-14 lg:grid-cols-[1.05fr_0.95fr]">
          <div className="rise">
            <p className="font-mono text-xs font-medium uppercase tracking-[0.14em] text-primary">
              {persona.name} / {persona.modeLabel}
            </p>
            <h1 key={personaId} className="font-display fade-in mt-5 text-4xl leading-[1.08] sm:text-6xl">
              {persona.heroTitle}
            </h1>
            <p className="mt-6 max-w-xl text-lg leading-relaxed text-mute">{persona.heroBody}</p>
            <div className="mt-9 flex flex-wrap gap-3">
              <Link href="/generate" className={buttonClass("primary", "px-6 py-3 text-base")}>
                {persona.cta}
                <Icon name="arrow" className="h-4 w-4" />
              </Link>
              <Link href="/evaluation" className={buttonClass("ghost", "px-6 py-3 text-base")}>
                See the evaluation
              </Link>
            </div>
          </div>
          <div className="rise" style={{ animationDelay: "120ms" }}>
            <HeroDemo persona={personaId} />
          </div>
        </div>
      </section>

      {/* Numbers */}
      {personaId === "lemkin" && stats ? (
        <section aria-label="The knowledge base in numbers" className="rise rounded-2xl border border-line bg-card shadow-soft">
          <div className="grid divide-y divide-line sm:grid-cols-3 sm:divide-x sm:divide-y-0">
            <Counter label="Documents" value={stats.totalDocuments} note="after ingestion filters" />
            <Counter label="Searchable chunks" value={stats.searchableChunks} note="embedded and retrievable" />
            <Counter label="Training examples" value={stats.trainingExamples} note="instruction and response rows" />
          </div>
        </section>
      ) : null}

      {/* Features */}
      <section aria-label="Features">
        <p className="font-mono text-[11px] font-medium uppercase tracking-[0.14em] text-primary">Capabilities</p>
        <h2 className="font-display mb-8 mt-2 text-2xl sm:text-3xl">What is inside</h2>
        <div className="grid gap-px overflow-hidden rounded-2xl border border-line bg-line sm:grid-cols-2 lg:grid-cols-4">
          {FEATURES[personaId].map((f) => (
            <div key={f.title} className="bg-card p-6 transition hover:bg-card-2">
              <span className="mb-5 grid h-10 w-10 place-items-center rounded-lg border border-line bg-card-2 text-primary">
                <Icon name={f.icon} className="h-5 w-5" />
              </span>
              <h3 className="font-semibold">{f.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-mute">{f.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section aria-label="How it works">
        <p className="font-mono text-[11px] font-medium uppercase tracking-[0.14em] text-primary">Workflow</p>
        <h2 className="font-display mb-8 mt-2 text-2xl sm:text-3xl">How it works</h2>
        <ol className="grid gap-4 md:grid-cols-3">
          {STEPS[personaId].map((s, i) => (
            <li key={s.title} className="rounded-2xl border border-line bg-card p-6">
              <span className="font-mono text-sm font-medium text-primary">{String(i + 1).padStart(2, "0")}</span>
              <p className="mt-3 font-semibold">{s.title}</p>
              <p className="mt-1.5 text-sm leading-relaxed text-mute">{s.body}</p>
            </li>
          ))}
        </ol>
      </section>

      {/* Closing call */}
      <section className="flex flex-wrap items-center justify-between gap-6 rounded-2xl bg-ink px-8 py-10 text-bg">
        <div>
          <h2 className="font-display text-2xl sm:text-3xl">Try it with your own brief</h2>
          <p className="mt-1.5 text-sm opacity-70">Demo mode uses clearly labelled sample data.</p>
        </div>
        <Link href="/generate" className={buttonClass("primary", "px-6 py-3 text-base")}>
          {persona.cta}
          <Icon name="arrow" className="h-4 w-4" />
        </Link>
      </section>
    </div>
  );
}

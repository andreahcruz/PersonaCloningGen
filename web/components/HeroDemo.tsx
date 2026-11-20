"use client";

import { useEffect, useState } from "react";
import { useReducedMotion } from "@/lib/motion";
import type { PersonaId } from "@/lib/prefs";

// An illustration of the flow, not real model output. The caption on the window says so.
const SCRIPT: Record<PersonaId, { label: string; input: string; chips: string[]; output: string; cite: boolean }> = {
  lemkin: {
    label: "Brief",
    input: "Hiring your first salesperson",
    chips: ["Blog", "LinkedIn", "X"],
    output:
      "Your first sales hire should be a builder who can close, not a manager. Hire for scrappiness first, then layer process on once you have repeatable wins.",
    cite: false,
  },
  musk: {
    label: "Question",
    input: "How does the company approach hard engineering problems?",
    chips: ["Interview", "X"],
    output:
      "Based on the retrieved passages, the answer is summarised here in plain words and each claim points back to the source it came from [Source 1] [Source 2].",
    cite: true,
  },
};

const TYPE_MS = 26;
const HOLD_TICKS = 90;

export default function HeroDemo({ persona }: { persona: PersonaId }) {
  const reduced = useReducedMotion();
  const script = SCRIPT[persona];
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (reduced) return;
    const id = setInterval(() => setTick((t) => t + 1), TYPE_MS);
    return () => clearInterval(id);
  }, [reduced]);

  const inputLen = script.input.length;
  const pauseTicks = 26;
  const outputLen = script.output.length;
  const cycle = inputLen + pauseTicks + outputLen + HOLD_TICKS;
  const t = reduced ? cycle - HOLD_TICKS / 2 : tick % cycle;

  const typedInput = script.input.slice(0, Math.min(t, inputLen));
  const searching = t >= inputLen && t < inputLen + pauseTicks;
  const outStart = inputLen + pauseTicks;
  const typedOutput = t >= outStart ? script.output.slice(0, t - outStart) : "";
  const showChips = t >= inputLen;
  const typing = t < inputLen;
  const writing = t >= outStart && t - outStart < outputLen;

  return (
    <div className="shadow-soft mx-auto w-full max-w-md rounded-2xl border border-line bg-card" aria-hidden>
      <div className="flex items-center gap-1.5 border-b border-line px-4 py-3">
        <span className="h-2.5 w-2.5 rounded-full bg-line" />
        <span className="h-2.5 w-2.5 rounded-full bg-line" />
        <span className="h-2.5 w-2.5 rounded-full bg-line" />
        <span className="ml-3 font-mono text-[11px] text-mute">illustration / not real output</span>
      </div>
      <div className="space-y-4 p-5 text-left">
        <div>
          <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-mute">{script.label}</p>
          <p className={`rounded-xl border border-line bg-card-2/60 px-3.5 py-2.5 text-sm ${typing ? "caret" : ""}`}>
            {typedInput || " "}
          </p>
        </div>

        <div className="flex min-h-7 flex-wrap items-center gap-2">
          {searching ? (
            <span className="flex items-center gap-2 text-xs text-mute">
              <span className="dot-pulse h-2 w-2 rounded-full bg-primary" />
              Retrieving passages
            </span>
          ) : null}
          {showChips && !searching
            ? script.chips.map((chip, i) => (
                <span
                  key={chip}
                  className="pop rounded-md bg-primary/12 px-2 py-0.5 text-xs font-medium text-primary"
                  style={{ animationDelay: `${i * 90}ms` }}
                >
                  {chip}
                </span>
              ))
            : null}
        </div>

        <div className="min-h-32 rounded-xl border border-line bg-card-2/60 p-4">
          <p className={`text-[15px] leading-relaxed ${writing ? "caret" : ""}`}>
            {typedOutput || " "}
          </p>
        </div>
      </div>
    </div>
  );
}

import type { ReactNode } from "react";
import type { ServiceState, SourceType } from "@/lib/types";

export function Card({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`rounded-2xl border border-line bg-card p-5 ${className}`}>{children}</div>
  );
}

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {subtitle ? <p className="mt-1 max-w-2xl text-sm text-mute">{subtitle}</p> : null}
      </div>
      {actions}
    </div>
  );
}

const BTN_BASE =
  "inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2 text-sm font-medium transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary disabled:cursor-not-allowed disabled:opacity-50";

const BTN_VARIANTS = {
  primary: "bg-primary text-white hover:bg-primary/85",
  violet: "bg-violet text-white hover:bg-violet/85",
  ghost: "border border-line bg-card-2 text-ink hover:border-primary/60",
} as const;

export function buttonClass(variant: keyof typeof BTN_VARIANTS = "primary", extra = ""): string {
  return `${BTN_BASE} ${BTN_VARIANTS[variant]} ${extra}`;
}

export function Button({
  variant = "primary",
  className = "",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: keyof typeof BTN_VARIANTS }) {
  return <button {...props} className={buttonClass(variant, className)} />;
}

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-mute">
        {label}
      </span>
      {children}
      {hint ? <span className="mt-1 block text-xs text-mute">{hint}</span> : null}
    </label>
  );
}

export const inputClass =
  "w-full rounded-xl border border-line bg-bg px-3 py-2 text-sm text-ink placeholder:text-mute/70 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary";

const STATE_STYLE: Record<ServiceState, { dot: string; text: string; label: string }> = {
  ok: { dot: "bg-ok", text: "text-ok", label: "Healthy" },
  degraded: { dot: "bg-warn", text: "text-warn", label: "Degraded" },
  down: { dot: "bg-bad", text: "text-bad", label: "Down" },
  unknown: { dot: "bg-mute", text: "text-mute", label: "Unknown" },
};

export function StatusBadge({ state }: { state: ServiceState }) {
  const s = STATE_STYLE[state];
  return (
    <span className={`inline-flex items-center gap-2 text-sm font-medium ${s.text}`}>
      <span className={`h-2.5 w-2.5 rounded-full ${s.dot}`} aria-hidden />
      {s.label}
    </span>
  );
}

const SOURCE_STYLE: Record<SourceType, string> = {
  blog: "bg-primary/15 text-primary",
  linkedin: "bg-violet/15 text-violet",
  x: "bg-ink/10 text-ink",
  youtube: "bg-bad/15 text-bad",
  unknown: "bg-mute/15 text-mute",
};

const SOURCE_LABEL: Record<SourceType, string> = {
  blog: "Blog",
  linkedin: "LinkedIn",
  x: "X",
  youtube: "YouTube",
  unknown: "Other",
};

export function SourceTag({ type }: { type: SourceType }) {
  return (
    <span className={`rounded-md px-2 py-0.5 text-xs font-medium ${SOURCE_STYLE[type]}`}>
      {SOURCE_LABEL[type]}
    </span>
  );
}

export function ScoreBar({ score }: { score: number | null }) {
  if (score === null) return <span className="text-xs text-mute">No score reported</span>;
  const pct = Math.round(Math.max(0, Math.min(1, score)) * 100);
  return (
    <div className="flex items-center gap-2" title={`Similarity ${score.toFixed(2)}`}>
      <div className="h-1.5 w-full rounded-full bg-line" role="presentation">
        <div className="h-1.5 rounded-full bg-primary" style={{ width: `${pct}%` }} />
      </div>
      <span className="w-9 text-right text-xs tabular-nums text-mute">{score.toFixed(2)}</span>
    </div>
  );
}

export function Notice({
  tone = "info",
  title,
  children,
}: {
  tone?: "info" | "warn" | "error";
  title?: string;
  children?: ReactNode;
}) {
  const tones = {
    info: "border-primary/40 bg-primary/10",
    warn: "border-warn/40 bg-warn/10",
    error: "border-bad/40 bg-bad/10",
  } as const;
  return (
    <div role={tone === "error" ? "alert" : "status"} className={`rounded-xl border p-3 text-sm ${tones[tone]}`}>
      {title ? <p className="font-medium">{title}</p> : null}
      {children ? <div className="text-mute">{children}</div> : null}
    </div>
  );
}

export function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <Card>
      <p className="text-xs font-medium uppercase tracking-wide text-mute">{label}</p>
      <p className="mt-2 text-3xl font-semibold tabular-nums">{value}</p>
      {note ? <p className="mt-1 text-xs text-mute">{note}</p> : null}
    </Card>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-2xl border border-dashed border-line p-8 text-center">
      <p className="font-medium">{title}</p>
      {children ? <div className="mx-auto mt-1 max-w-md text-sm text-mute">{children}</div> : null}
    </div>
  );
}

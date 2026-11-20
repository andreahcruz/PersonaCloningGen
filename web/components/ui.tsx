import type { CSSProperties, KeyboardEvent, ReactNode } from "react";
import type { ServiceState, SourceType } from "@/lib/types";

/* ---------- Icons (small inline set, no extra dependency) ---------- */

const ICON_PATHS: Record<string, ReactNode> = {
  sun: (
    <>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </>
  ),
  moon: <path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5Z" />,
  spark: <path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9L12 3ZM19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8L19 16Z" />,
  search: (
    <>
      <circle cx="11" cy="11" r="7" />
      <path d="m20 20-3.5-3.5" />
    </>
  ),
  copy: (
    <>
      <rect x="9" y="9" width="11" height="11" rx="2" />
      <path d="M5 15V5a2 2 0 0 1 2-2h10" />
    </>
  ),
  download: <path d="M12 3v12m0 0 4-4m-4 4-4-4M4 20h16" />,
  check: <path d="m5 12.5 4.5 4.5L19 7.5" />,
  arrow: <path d="M5 12h14m0 0-5-5m5 5-5 5" />,
  menu: <path d="M4 7h16M4 12h16M4 17h16" />,
  close: <path d="m6 6 12 12M18 6 6 18" />,
  doc: (
    <>
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5Z" />
      <path d="M14 3v5h5M9 13h6M9 17h6" />
    </>
  ),
  shield: <path d="M12 3l8 3v6c0 4.5-3.2 8-8 9-4.8-1-8-4.5-8-9V6l8-3Zm-3 9 2.2 2.2L15.5 10" />,
  chart: <path d="M4 20V10m6 10V4m6 16v-7m4 7H2" />,
  pulse: <path d="M3 12h4l2.5-6 5 12 2.5-6h4" />,
  info: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5M12 8v.01" />
    </>
  ),
  bolt: <path d="M13 2 4 14h7l-1 8 9-12h-7l1-8Z" />,
  layers: <path d="m12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5m-18 4 9 5 9-5" />,
  chat: <path d="M20 12a8 8 0 0 1-11.7 7.1L4 20l1-4.1A8 8 0 1 1 20 12Z" />,
  pen: <path d="M4 20h4L19 9a2.8 2.8 0 0 0-4-4L4 16v4Zm9.5-13.5 4 4" />,
};

export function Icon({
  name,
  className = "h-4 w-4",
}: {
  name: keyof typeof ICON_PATHS | string;
  className?: string;
}) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden
    >
      {ICON_PATHS[name] ?? null}
    </svg>
  );
}

/* ---------- Layout pieces ---------- */

export function Card({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`shadow-soft rounded-2xl border border-line bg-card p-5 ${className}`}>
      {children}
    </div>
  );
}

export function PageHeader({
  title,
  subtitle,
  actions,
  eyebrow,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  eyebrow?: string;
}) {
  return (
    <div className="mb-7 flex flex-wrap items-end justify-between gap-4">
      <div>
        {eyebrow ? (
          <p className="mb-2 font-mono text-[11px] font-medium uppercase tracking-[0.14em] text-primary">{eyebrow}</p>
        ) : null}
        <h1 className="font-display text-3xl sm:text-4xl">{title}</h1>
        {subtitle ? <p className="mt-2 max-w-2xl text-sm leading-relaxed text-mute">{subtitle}</p> : null}
      </div>
      {actions}
    </div>
  );
}

/* ---------- Buttons ---------- */

const BTN_BASE =
  "inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium transition active:scale-[0.98] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary disabled:cursor-not-allowed disabled:opacity-50 disabled:active:scale-100";

const BTN_VARIANTS = {
  primary:
    "bg-primary text-primary-ink shadow-soft hover:brightness-110",
  violet: "bg-violet text-primary-ink hover:brightness-110",
  ghost: "border border-line bg-card text-ink hover:border-primary/60 hover:bg-card-2",
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

/* ---------- Form pieces ---------- */

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
      <span className="mb-1.5 block text-xs font-semibold uppercase tracking-wider text-mute">
        {label}
      </span>
      {children}
      {hint ? <span className="mt-1 block text-xs text-mute">{hint}</span> : null}
    </label>
  );
}

export const inputClass =
  "w-full rounded-xl border border-line bg-card px-3.5 py-2.5 text-sm text-ink shadow-[inset_0_1px_0_rgba(0,0,0,0.02)] transition placeholder:text-mute/70 hover:border-primary/40 focus:border-primary focus:outline-none focus:ring-4 focus:ring-primary/15";

/** A row of mutually exclusive choices, shown as pills. */
export function Segmented<T extends string>({
  value,
  onChange,
  options,
  label,
}: {
  value: T;
  onChange: (next: T) => void;
  options: Array<{ id: T; label: string }>;
  label: string;
}) {
  return (
    <div role="group" aria-label={label} className="inline-flex rounded-xl border border-line bg-card-2 p-1">
      {options.map((option) => (
        <button
          key={option.id}
          type="button"
          aria-pressed={value === option.id}
          onClick={() => onChange(option.id)}
          className={`rounded-lg px-3 py-1.5 text-sm transition ${
            value === option.id
              ? "bg-card font-medium text-ink shadow-soft"
              : "text-mute hover:text-ink"
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

/* ---------- Status and feedback ---------- */

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
  blog: "bg-primary/12 text-primary",
  linkedin: "bg-violet/12 text-violet",
  x: "bg-ink/8 text-ink",
  youtube: "bg-bad/12 text-bad",
  interview: "bg-warn/14 text-warn",
  unknown: "bg-mute/15 text-mute",
};

const SOURCE_LABEL: Record<SourceType, string> = {
  blog: "Blog",
  linkedin: "LinkedIn",
  x: "X",
  youtube: "YouTube",
  interview: "Interview",
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
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-line" role="presentation">
        <div
          className="bg-accent-gradient h-1.5 rounded-full transition-[width] duration-700"
          style={{ width: `${pct}%` }}
        />
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
    info: "border-primary/30 bg-primary/8",
    warn: "border-warn/35 bg-warn/10",
    error: "border-bad/35 bg-bad/10",
  } as const;
  return (
    <div role={tone === "error" ? "alert" : "status"} className={`rounded-xl border p-3.5 text-sm ${tones[tone]}`}>
      {title ? <p className="font-medium">{title}</p> : null}
      {children ? <div className="text-mute">{children}</div> : null}
    </div>
  );
}

export function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <Card className="lift">
      <p className="text-xs font-semibold uppercase tracking-wider text-mute">{label}</p>
      <p className="mt-2 text-3xl font-semibold tabular-nums tracking-tight">{value}</p>
      {note ? <p className="mt-1 text-xs text-mute">{note}</p> : null}
    </Card>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-2xl border border-dashed border-line bg-card-2/50 p-8 text-center">
      <span className="mx-auto mb-3 grid h-10 w-10 place-items-center rounded-lg border border-line bg-card text-mute">
        <Icon name="doc" className="h-5 w-5" />
      </span>
      <p className="font-medium">{title}</p>
      {children ? <div className="mx-auto mt-1 max-w-md text-sm text-mute">{children}</div> : null}
    </div>
  );
}

/** Placeholder lines shown while a draft is being written. */
export function SkeletonLines({ lines = 5 }: { lines?: number }) {
  const widths = ["100%", "94%", "98%", "82%", "90%", "64%"];
  return (
    <div className="space-y-2.5" aria-hidden>
      {Array.from({ length: lines }, (_, i) => (
        <div key={i} className="shimmer h-3.5 rounded-md" style={{ width: widths[i % widths.length] }} />
      ))}
    </div>
  );
}

/* ---------- Interactive controls ---------- */

export interface TabItem<T extends string> {
  id: T;
  label: string;
  badge?: string | number;
  disabled?: boolean;
}

/** Accessible tabs: arrow keys move between them. */
export function Tabs<T extends string>({
  value,
  onChange,
  tabs,
  label,
}: {
  value: T;
  onChange: (next: T) => void;
  tabs: TabItem<T>[];
  label: string;
}) {
  const enabled = tabs.filter((t) => !t.disabled);
  function onKey(e: KeyboardEvent<HTMLDivElement>) {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const at = enabled.findIndex((t) => t.id === value);
    const next = enabled[(at + (e.key === "ArrowRight" ? 1 : -1) + enabled.length) % enabled.length];
    if (next) onChange(next.id);
  }
  return (
    <div role="tablist" aria-label={label} onKeyDown={onKey} className="flex gap-1 overflow-x-auto border-b border-line">
      {tabs.map((tab) => {
        const active = tab.id === value;
        return (
          <button
            key={tab.id}
            id={`tab-${tab.id}`}
            role="tab"
            type="button"
            aria-selected={active}
            aria-controls={`panel-${tab.id}`}
            tabIndex={active ? 0 : -1}
            disabled={tab.disabled}
            onClick={() => onChange(tab.id)}
            className={`relative -mb-px flex shrink-0 items-center gap-2 px-3.5 py-2.5 text-sm transition disabled:cursor-not-allowed disabled:opacity-40 ${
              active ? "font-medium text-ink" : "text-mute hover:text-ink"
            }`}
          >
            {tab.label}
            {tab.badge !== undefined ? (
              <span
                className={`rounded-full px-1.5 py-0.5 text-[11px] font-semibold tabular-nums ${
                  active ? "bg-primary/14 text-primary" : "bg-card-2 text-mute"
                }`}
              >
                {tab.badge}
              </span>
            ) : null}
            <span
              className={`bg-accent-gradient absolute inset-x-2 bottom-0 h-0.5 rounded-full transition-all duration-300 ${
                active ? "scale-x-100 opacity-100" : "scale-x-0 opacity-0"
              }`}
            />
          </button>
        );
      })}
    </div>
  );
}

export function Switch({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (next: boolean) => void;
  label: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className="switch shrink-0 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
    />
  );
}

/** Range slider whose filled part follows the value. */
export function RangeInput({
  value,
  min,
  max,
  onChange,
  label,
}: {
  value: number;
  min: number;
  max: number;
  onChange: (next: number) => void;
  label: string;
}) {
  const fill = ((value - min) / (max - min)) * 100;
  return (
    <input
      type="range"
      aria-label={label}
      min={min}
      max={max}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className="range w-full"
      style={{ "--fill": `${fill}%` } as CSSProperties}
    />
  );
}

/** A circular gauge, for example words written against the target length. */
export function Ring({
  value,
  max,
  label,
  sub,
}: {
  value: number;
  max: number;
  label: string;
  sub: string;
}) {
  const r = 26;
  const c = 2 * Math.PI * r;
  const ratio = Math.max(0, Math.min(1, value / max));
  const over = value > max * 1.25;
  return (
    <div className="flex items-center gap-3">
      <svg viewBox="0 0 64 64" className="h-16 w-16 -rotate-90" role="img" aria-label={`${label}: ${value} of about ${max}`}>
        <circle cx="32" cy="32" r={r} fill="none" strokeWidth="6" className="stroke-line" />
        <circle
          cx="32"
          cy="32"
          r={r}
          fill="none"
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - ratio)}
          className={`transition-[stroke-dashoffset] duration-1000 ${over ? "stroke-warn" : "stroke-primary"}`}
        />
      </svg>
      <div>
        <p className="text-2xl font-semibold tabular-nums leading-none">{value}</p>
        <p className="mt-1 text-xs text-mute">{sub}</p>
      </div>
    </div>
  );
}

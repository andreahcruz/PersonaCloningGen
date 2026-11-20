"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, type ReactNode } from "react";
import { IS_MOCK } from "@/lib/config";
import { PERSONA_LIST, PERSONAS } from "@/lib/personas";
import { usePersonaId, useTheme } from "@/lib/prefs";
import { Icon } from "@/components/ui";

const NAV = [
  { href: "/", label: "Home" },
  { href: "/generate", label: "Studio" },
  { href: "/evaluation", label: "Evaluation" },
  { href: "/health", label: "System health" },
  { href: "/about", label: "About" },
];

function isActive(pathname: string, href: string): boolean {
  return href === "/" ? pathname === "/" : pathname.startsWith(href);
}

function ThemeToggle() {
  const [theme, setTheme] = useTheme();
  const dark = theme === "dark";
  return (
    <button
      type="button"
      onClick={() => setTheme(dark ? "light" : "dark")}
      aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
      title={dark ? "Switch to light theme" : "Switch to dark theme"}
      className="relative grid h-9 w-9 place-items-center overflow-hidden rounded-xl border border-line bg-card text-mute transition hover:border-primary/60 hover:text-ink active:scale-95"
    >
      <span
        className={`absolute transition-all duration-300 ${dark ? "translate-y-0 rotate-0 opacity-100" : "translate-y-5 -rotate-90 opacity-0"}`}
      >
        <Icon name="moon" className="h-[18px] w-[18px]" />
      </span>
      <span
        className={`absolute transition-all duration-300 ${dark ? "-translate-y-5 rotate-90 opacity-0" : "translate-y-0 rotate-0 opacity-100"}`}
      >
        <Icon name="sun" className="h-[18px] w-[18px]" />
      </span>
    </button>
  );
}

function PersonaSwitcher() {
  const [persona, setPersona] = usePersonaId();
  return (
    <div
      role="group"
      aria-label="Persona"
      className="inline-flex rounded-xl border border-line bg-card-2 p-1"
    >
      {PERSONA_LIST.map((item) => {
        const active = persona === item.id;
        return (
          <button
            key={item.id}
            type="button"
            aria-pressed={active}
            onClick={() => setPersona(item.id)}
            title={`${item.name}: ${item.modeLabel}`}
            className={`flex items-center gap-2 rounded-lg px-2.5 py-1.5 text-sm transition ${
              active ? "bg-card font-medium text-ink shadow-soft" : "text-mute hover:text-ink"
            }`}
          >
            <span
              className={`grid h-5 w-5 place-items-center rounded-md text-[10px] font-bold ${
                active ? "bg-accent-gradient text-primary-ink" : "bg-line text-mute"
              }`}
              aria-hidden
            >
              {item.initials}
            </span>
            <span className="hidden sm:inline">{item.name}</span>
          </button>
        );
      })}
    </div>
  );
}

export default function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const [persona] = usePersonaId();
  const active = PERSONAS[persona];

  return (
    <div className="page-glow min-h-screen">
      <header className="sticky top-0 z-30 border-b border-line bg-card">
        <div className="mx-auto flex h-16 w-full max-w-7xl items-center gap-4 px-4 sm:px-6 lg:px-8">
          <Link href="/" className="flex shrink-0 items-center gap-2.5" onClick={() => setOpen(false)}>
            <span
              className="bg-accent-gradient grid h-9 w-9 place-items-center rounded-xl text-sm font-bold text-primary-ink shadow-soft"
              aria-hidden
            >
              LS
            </span>
            <span className="hidden leading-tight sm:block">
              <span className="block text-sm font-semibold">Lemkin Studio</span>
              <span className="block font-mono text-[11px] text-mute">persona content studio</span>
            </span>
          </Link>

          <nav aria-label="Main" className="ml-2 hidden items-center gap-1 lg:flex">
            {NAV.map((item) => {
              const current = isActive(pathname, item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  aria-current={current ? "page" : undefined}
                  className={`relative rounded-lg px-3 py-2 text-sm transition ${
                    current ? "font-medium text-ink" : "text-mute hover:text-ink"
                  }`}
                >
                  {item.label}
                  <span
                    className={`bg-accent-gradient absolute inset-x-3 -bottom-[13px] h-0.5 rounded-full transition-opacity ${
                      current ? "opacity-100" : "opacity-0"
                    }`}
                  />
                </Link>
              );
            })}
          </nav>

          <div className="ml-auto flex items-center gap-2">
            <span
              className={`hidden rounded-md border px-2 py-1 font-mono text-[11px] font-medium md:inline-block ${
                IS_MOCK ? "border-warn/40 text-warn" : "border-ok/40 text-ok"
              }`}
            >
              {IS_MOCK ? "Demo data" : "Live API"}
            </span>
            <PersonaSwitcher />
            <ThemeToggle />
            <button
              type="button"
              onClick={() => setOpen((value) => !value)}
              aria-label={open ? "Close menu" : "Open menu"}
              aria-expanded={open}
              className="grid h-9 w-9 place-items-center rounded-xl border border-line bg-card text-mute transition hover:text-ink lg:hidden"
            >
              <Icon name={open ? "close" : "menu"} className="h-[18px] w-[18px]" />
            </button>
          </div>
        </div>

        {open ? (
          <nav aria-label="Mobile" className="fade-in border-t border-line px-4 pb-3 pt-2 lg:hidden">
            {NAV.map((item) => {
              const current = isActive(pathname, item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => setOpen(false)}
                  aria-current={current ? "page" : undefined}
                  className={`block rounded-lg px-3 py-2.5 text-sm ${
                    current ? "bg-primary/10 font-medium text-primary" : "text-mute"
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>
        ) : null}
      </header>

      {IS_MOCK ? (
        <div className="border-b border-line bg-card-2 px-4 py-2 text-center text-xs text-mute">
          <span className="mr-2 inline-block h-1.5 w-1.5 rounded-full bg-warn align-middle" aria-hidden />
          Demo mode: text, sources and status are sample data, not real model output. Set
          NEXT_PUBLIC_API_MODE=live to use the backend.
        </div>
      ) : null}

      <div className="border-b border-line bg-card">
        <div className="mx-auto flex w-full max-w-7xl flex-wrap items-center gap-x-3 gap-y-1 px-4 py-2 text-xs text-mute sm:px-6 lg:px-8">
          <span className="font-mono font-medium text-ink">{active.name}</span>
          <span aria-hidden>·</span>
          <span>{active.modeLabel}</span>
          <span aria-hidden className="hidden sm:inline">·</span>
          <span className="hidden sm:inline">{active.corpus}</span>
        </div>
      </div>

      <main className="fade-in mx-auto w-full max-w-7xl px-4 py-9 sm:px-6 lg:px-8">{children}</main>

      <footer className="mt-6 border-t border-line bg-card">
        <div className="mx-auto flex w-full max-w-7xl flex-wrap items-center justify-between gap-2 px-4 py-6 text-xs text-mute sm:px-6 lg:px-8">
          <span>DATA 298B, Group 2.</span>
          <span>Drafts are model output: check facts before you publish.</span>
        </div>
      </footer>
    </div>
  );
}

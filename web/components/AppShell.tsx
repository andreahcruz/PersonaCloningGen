"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import { IS_MOCK } from "@/lib/config";

const NAV = [
  { href: "/", label: "Home" },
  { href: "/generate", label: "Generator" },
  { href: "/dashboard", label: "Dashboard" },
  { href: "/sources", label: "Sources" },
  { href: "/evaluation", label: "Evaluation" },
  { href: "/health", label: "System health" },
  { href: "/about", label: "About" },
];

function isActive(pathname: string, href: string): boolean {
  return href === "/" ? pathname === "/" : pathname.startsWith(href);
}

export default function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="min-h-screen md:flex">
      <aside className="border-b border-line bg-card md:sticky md:top-0 md:h-screen md:w-60 md:shrink-0 md:border-b-0 md:border-r">
        <div className="flex items-center gap-3 px-5 py-4">
          <span
            className="grid h-9 w-9 place-items-center rounded-xl bg-linear-to-br from-primary to-violet text-sm font-bold"
            aria-hidden
          >
            LS
          </span>
          <div>
            <p className="font-semibold leading-tight">Lemkin Studio</p>
            <p className="text-xs text-mute">AI persona content</p>
          </div>
        </div>
        <nav
          aria-label="Main"
          className="flex gap-1 overflow-x-auto px-3 pb-3 md:block md:space-y-1 md:overflow-visible md:pb-0"
        >
          {NAV.map((item) => {
            const active = isActive(pathname, item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={active ? "page" : undefined}
                className={`block whitespace-nowrap rounded-xl px-3 py-2 text-sm transition ${
                  active
                    ? "bg-primary/15 font-medium text-primary"
                    : "text-mute hover:bg-card-2 hover:text-ink"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="hidden px-5 py-4 md:absolute md:bottom-0 md:block">
          <span
            className={`rounded-md px-2 py-1 text-xs font-medium ${
              IS_MOCK ? "bg-warn/15 text-warn" : "bg-ok/15 text-ok"
            }`}
          >
            {IS_MOCK ? "Demo data" : "Live API"}
          </span>
        </div>
      </aside>

      <div className="min-w-0 flex-1">
        {IS_MOCK ? (
          <div className="border-b border-warn/30 bg-warn/10 px-4 py-2 text-center text-xs text-warn">
            Demo mode: text, sources and status are sample data, not real model output. Set
            NEXT_PUBLIC_API_MODE=live to use the backend.
          </div>
        ) : null}
        <main className="fade-in mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:px-8">{children}</main>
      </div>
    </div>
  );
}

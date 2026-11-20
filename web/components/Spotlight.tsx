import type { ReactNode } from "react";

/** A plain bordered card with a quiet hover state. */
export default function Spotlight({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`lift shadow-soft rounded-2xl border border-line bg-card ${className}`}>{children}</div>;
}

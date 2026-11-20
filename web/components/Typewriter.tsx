"use client";

import { useEffect, useState } from "react";
import { useReducedMotion } from "@/lib/motion";

/**
 * Reveals text a few characters at a time so a finished draft feels like it is being written.
 * Click the text to show everything at once. Give it key={text} so a new draft starts again.
 */
export default function Typewriter({ text, className = "" }: { text: string; className?: string }) {
  const reduced = useReducedMotion();
  const [count, setCount] = useState(0);
  const [skipped, setSkipped] = useState(false);

  useEffect(() => {
    if (reduced) return;
    const total = text.length;
    const step = Math.max(2, Math.ceil(total / 150));
    let shown = 0;
    const id = setInterval(() => {
      shown = Math.min(total, shown + step);
      setCount(shown);
      if (shown >= total) clearInterval(id);
    }, 16);
    return () => clearInterval(id);
  }, [text, reduced]);

  const done = reduced || skipped || count >= text.length;
  const visible = done ? text : text.slice(0, count);

  return (
    <div
      className={`${className} ${done ? "" : "caret cursor-pointer"}`}
      onClick={() => setSkipped(true)}
      title={done ? undefined : "Click to show the whole draft"}
    >
      {visible}
    </div>
  );
}

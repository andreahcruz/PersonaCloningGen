import type { HistoryItem } from "./types";

const KEY = "lemkin-studio-history";

// Browser-only convenience for the dashboard. Every access is guarded because storage can be blocked.
export function readHistory(): HistoryItem[] {
  try {
    const raw = window.localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as HistoryItem[]) : [];
  } catch {
    return [];
  }
}

export function addHistory(item: HistoryItem): void {
  try {
    const next = [item, ...readHistory()].slice(0, 10);
    window.localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    /* storage unavailable */
  }
}

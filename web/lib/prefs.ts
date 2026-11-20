"use client";

import { useSyncExternalStore } from "react";

// Small browser-preference store for the theme and the active persona.
// Both live on <html> as data attributes (so CSS can react) and in localStorage
// (so the choice survives a reload). Every storage access is guarded.

export type Theme = "light" | "dark";
export type PersonaId = "lemkin" | "musk";

const THEME_KEY = "lemkin-studio-theme";
const PERSONA_KEY = "lemkin-studio-persona";
const EVENT = "studio-prefs";

function write(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* storage unavailable */
  }
}

function subscribe(callback: () => void): () => void {
  window.addEventListener(EVENT, callback);
  window.addEventListener("storage", callback);
  return () => {
    window.removeEventListener(EVENT, callback);
    window.removeEventListener("storage", callback);
  };
}

/** Script run in <head> before paint, so there is no flash of the wrong theme. Light is the default. */
export const PREFS_INIT_SCRIPT = `(function(){try{var t=localStorage.getItem("${THEME_KEY}");var p=localStorage.getItem("${PERSONA_KEY}");var d=document.documentElement;d.setAttribute("data-theme",t==="dark"?"dark":"light");d.setAttribute("data-persona",p==="musk"?"musk":"lemkin");}catch(e){var d2=document.documentElement;d2.setAttribute("data-theme","light");d2.setAttribute("data-persona","lemkin");}})();`;

function currentTheme(): Theme {
  return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
}

function currentPersona(): PersonaId {
  return document.documentElement.getAttribute("data-persona") === "musk" ? "musk" : "lemkin";
}

export function useTheme(): [Theme, (next: Theme) => void] {
  const theme = useSyncExternalStore(subscribe, currentTheme, () => "light" as Theme);
  const set = (next: Theme) => {
    document.documentElement.setAttribute("data-theme", next);
    write(THEME_KEY, next);
    window.dispatchEvent(new Event(EVENT));
  };
  return [theme, set];
}

export function usePersonaId(): [PersonaId, (next: PersonaId) => void] {
  const persona = useSyncExternalStore(subscribe, currentPersona, () => "lemkin" as PersonaId);
  const set = (next: PersonaId) => {
    document.documentElement.setAttribute("data-persona", next);
    write(PERSONA_KEY, next);
    window.dispatchEvent(new Event(EVENT));
  };
  return [persona, set];
}

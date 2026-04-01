/**
 * Node + playwright-extra + stealth (Gemini-style stack), aligned with safer habits:
 * - Persistent profile: YOU sign in by hand the first time; no scripted passwords.
 * - Headed Chrome: same as normal browsing window.
 * - "See more": scroll into view + role/name locators (classes change often).
 *
 * Usage:
 *   node collect_activity_stealth.cjs "https://www.linkedin.com/in/HANDLE/recent-activity/all/" dump.html
 *   node collect_activity_stealth.cjs "URL" dump_first.html --first-page
 *
 * --first-page  Only expands/trims the initial viewport chunk — does NOT jump to
 *               document bottom (avoids loading the whole infinite feed). Use this
 *               to verify "See more" + HTML before a long run.
 *
 * --keep-open   After saving HTML, leave Chrome open and wait for Enter in this
 *               terminal before closing. Use this to scroll manually, SingleFile-save
 *               again, or simply pause. You must press Enter (closing the browser)
 *               before starting another run with the same LI_USER_DATA folder.
 *
 * Resume (scroll + exact tab URL):
 *   After each run we save .linkedin_collect_state.json next to your Chrome profile
 *   folder (LI_USER_DATA), so resume still works if you run the script from another
 *   directory. Override path with LI_STATE_FILE.
 *   The next run restores page.goto(savedPageUrl) + feed scroll (nested overflow), not
 *   only window.scrollY — LinkedIn’s activity feed usually scrolls inside a div.
 *   Old state files with only scrollY are applied to that same feed container. Use this
 *   after hitting LinkedIn’s “bottom”, manual pagination, or extra scrolling during --keep-open.
 *   Use --fresh to ignore/delete state and start from the top of the URL you pass.
 *   --first-page does not read/write resume state.
 *   Use the SAME CLI activity URL every time (e.g. always …/recent-activity/all/) so
 *   activityKey matches; otherwise resume is skipped.
 *
 * Multi-pass (one browser session, deeper feed, multiple HTML files):
 *   node collect_activity_stealth.cjs "URL" dump.html --passes 5
 *   Writes dump_pass01.html … dump_pass05.html after each batch of LI_ROUNDS scroll cycles.
 *   Resume state updates after every pass. Next terminal run continues from last scroll.
 *
 * Text-only (no giant HTML / no SingleFile):
 *   --extract-jsonl OUT.jsonl   After each pass, read post bodies from the live DOM and
 *                               append new rows (deduped by content hash). Small files.
 *   --no-html                   Skip page.content() saves (use with --extract-jsonl).
 *   --author "Name"             Fallback when the card has no detectable actor name.
 *   --prune-dom                 After extracting a post, remove that card from the DOM
 *                               (with --extract-jsonl) to keep the page lighter. Optional;
 *                               LinkedIn may rarely glitch — turn off if the feed breaks.
 *   --checkpoint-rounds N       With --extract-jsonl: save JSONL + resume every N scroll
 *                               rounds inside each pass (default 1 = every round). Use 0
 *                               or --no-checkpoint to only save at end of each pass.
 *
 * Performance / memory (optional):
 *   --block-media               Abort image + media requests (lighter network & decode).
 *   --full-scroll-every N       On the feed scroller: jump to bottom every N rounds;
 *                               other rounds use ~one viewport step (default 3). Set 0 to
 *                               never force bottom (incremental only). Env: LI_FULL_SCROLL_EVERY.
 *   --legacy-scroll             Old behavior: wheel + window.scrollTo(document.body height).
 *   --recycle-every-passes N    Close Chrome and reopen every N passes to flush JS heap
 *                               (resume still works from LI_STATE_FILE). 0 = one session
 *                               for all passes (default). Env: LI_RECYCLE_BROWSER_EVERY_PASSES.
 *
 * DOM / expand tuning (optional):
 *   --expand-scan-full-feed      Scan all “See more” buttons (slow on huge feeds). Default is
 *                               viewport-only + capped clicks (faster). Env: LI_EXPAND_FULL_FEED=1.
 *   --reload-dom-every-rounds N  After every N scroll rounds, page.goto + resume scroll to drop
 *                               document and trim DOM (0 = off). Env: LI_RELOAD_DOM_EVERY_ROUNDS.
 *   --reload-dom-every-passes N  Same between passes within a session (0 = off).
 *                               Env: LI_RELOAD_DOM_EVERY_PASSES.
 *   --max-rounds-per-pass N      Cap scroll rounds per pass (0 = use LI_ROUNDS). Shorter sessions.
 *                               Env: LI_MAX_ROUNDS_PER_PASS.
 *
 * Mobile web (lighter UI; may differ from desktop):
 *   --mobile                     Use a phone-sized viewport + mobile User-Agent (Playwright
 *                               “iPhone 13” profile). DOM/selectors often differ — if extract
 *                               fails, try desktop. For m.linkedin.com, pass that host in your
 *                               URL (LinkedIn may redirect). Env: LI_MOBILE=1.
 *
 * Optional env: LI_USER_DATA, LI_ROUNDS (default 18), LI_PAUSE_MIN, LI_PAUSE_MAX,
 *               LI_KEEP_OPEN=1, LI_STATE_FILE, LI_PASSES, LI_EXTRACT_JSONL, LI_AUTHOR,
 *               LI_PRUNE_DOM=1, LI_CHECKPOINT_ROUNDS (0 = off; unset = default 1 with extract),
 *               LI_BLOCK_MEDIA=1, LI_FULL_SCROLL_EVERY, LI_LEGACY_SCROLL=1,
 *               LI_RECYCLE_BROWSER_EVERY_PASSES, LI_EXPAND_FULL_FEED, LI_RELOAD_DOM_EVERY_ROUNDS,
 *               LI_RELOAD_DOM_EVERY_PASSES, LI_MAX_ROUNDS_PER_PASS, LI_MOBILE=1
 */

const crypto = require("crypto");
const { devices } = require("playwright");
const fs = require("fs/promises");
const path = require("path");
const readline = require("readline");

const extractPostsFromDom = require("./dom_extract_posts.js");
const extractSearchPostsFromDom = require("./dom_extract_search_posts.js");

const { chromium } = require("playwright-extra");
const StealthPlugin = require("puppeteer-extra-plugin-stealth");

chromium.use(StealthPlugin());

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * Same profile activity stream (ignore query string differences).
 */
function activityKeyFromUrl(u) {
  try {
    const { pathname, searchParams } = new URL(u);
    const p = pathname.replace(/\/+$/, "").toLowerCase() || "/";
    if (p === "/search/results/content") {
      const fromMember = searchParams.get("fromMember") || "";
      const keywords = searchParams.get("keywords") || "";
      const sortBy = searchParams.get("sortBy") || "";
      return `search:content?fromMember=${fromMember}&keywords=${keywords}&sortBy=${sortBy}`.toLowerCase();
    }
    const m = p.match(/^(\/in\/[^/]+\/recent-activity(?:\/[^/]*)?)$/);
    return m ? m[1] : p;
  } catch {
    return String(u || "").toLowerCase();
  }
}

function isSearchContentUrl(u) {
  try {
    const { pathname } = new URL(u);
    return pathname.replace(/\/+$/, "").toLowerCase() === "/search/results/content";
  } catch {
    return false;
  }
}

function selectDomExtractor(cliUrl) {
  return isSearchContentUrl(cliUrl) ? extractSearchPostsFromDom : extractPostsFromDom;
}

async function readResumeState(statePath) {
  try {
    const raw = await fs.readFile(statePath, "utf8");
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

async function writeResumeState(statePath, snap) {
  const payload = { ...snap, savedAt: new Date().toISOString() };
  await fs.writeFile(statePath, JSON.stringify(payload, null, 2), "utf8");
}

/** Older builds stored state in cwd; copy once if new profile-local file is missing. */
function contentFingerprint(text) {
  return crypto.createHash("sha256").update(String(text), "utf8").digest("hex").slice(0, 16);
}

/** path -> Set of content hashes; file read once per run (not after every pass). */
const extractDedupeCache = new Map();

function extractRowKey(r) {
  if (r && typeof r.urn === "string" && r.urn.trim()) return `urn:${r.urn.trim()}`;
  if (r && typeof r.url === "string" && r.url.trim()) return `url:${r.url.trim()}`;
  const c = r && r.content ? r.content : "";
  return `h:${contentFingerprint(c)}`;
}

async function appendExtractJsonl(jsonlPath, rows, authorFallback) {
  let seen = extractDedupeCache.get(jsonlPath);
  if (!seen) {
    seen = new Set();
    extractDedupeCache.set(jsonlPath, seen);
    try {
      const prev = await fs.readFile(jsonlPath, "utf8");
      for (const line of prev.split("\n")) {
        if (!line.trim()) continue;
        try {
          const o = JSON.parse(line);
          seen.add(extractRowKey(o));
        } catch {
          /* skip */
        }
      }
    } catch {
      /* no file */
    }
  }

  const lines = [];
  for (const r of rows) {
    const c = r.content || "";
    const k = extractRowKey(r);
    if (seen.has(k)) continue;
    seen.add(k);
    const row = {
      author: r.author || authorFallback,
      author_source: r.author ? "html" : "cli",
      content: c,
      format: r.format || "linkedin_post",
      char_count: r.char_count || c.length,
    };
    if (r.urn) row.urn = r.urn;
    if (r.url) row.url = r.url;
    if (r.published_text) row.published_text = r.published_text;
    lines.push(JSON.stringify(row));
  }

  if (lines.length) {
    await fs.appendFile(jsonlPath, lines.join("\n") + "\n", "utf8");
  }
  return lines.length;
}

async function migrateLegacyResumeState(statePath) {
  if (process.env.LI_STATE_FILE) return;
  try {
    await fs.access(statePath);
    return;
  } catch {
    /* missing */
  }
  const legacy = path.join(process.cwd(), ".linkedin_collect_state.json");
  try {
    await fs.access(legacy);
    await fs.copyFile(legacy, statePath);
    console.log("[collect] Copied old resume state from cwd →", statePath);
  } catch {
    /* no legacy */
  }
}

async function captureScrollState(page, cliUrl) {
  const data = await page.evaluate(() => {
    function findFeedScrollRoot() {
      const anchor =
        document.querySelector(".feed-shared-update-v2") ||
        document.querySelector("div[class*='feed-shared-update-v2']") ||
        document.querySelector("li.reusable-search__result-container") ||
        document.querySelector("main");
      if (!anchor) return document.scrollingElement || document.documentElement;
      // 1) Prefer an ancestor scroll container (fast), but choose the one
      // that is actually scrolled (largest scrollTop). This avoids falling
      // back to document/HTML when the inner scroller is present.
      let best = null;
      let bestScore = -1;
      let el = anchor;
      for (let d = 0; d < 50 && el; d++) {
        const cs = getComputedStyle(el);
        const oy = cs.overflowY;
        const of = cs.overflow;
        const tag = (el && el.tagName) ? String(el.tagName).toUpperCase() : "";
        const isScrollable =
          el.scrollHeight > el.clientHeight + 40 &&
          (oy === "auto" || oy === "scroll" || oy === "overlay" || of === "auto" || of === "scroll");
        // Prefer inner feed scrollers; document/body scrolling makes resume look like "from top".
        if (isScrollable && (tag === "HTML" || tag === "BODY")) {
          continue;
        }
        if (isScrollable) {
          const score = Number(el.scrollTop) || 0;
          if (score > bestScore) {
            bestScore = score;
            best = el;
          }
        }
        el = el.parentElement;
      }
      if (best) return best;

      // 2) Fallback: scan descendants under <main> for an inner scroller.
      // LinkedIn's feed scroller isn't always an ancestor of the first card.
      const start = anchor.closest && anchor.closest("main") ? anchor.closest("main") : anchor;
      const maxNodes = 2500;
      const walker = document.createTreeWalker(start, NodeFilter.SHOW_ELEMENT, null);
      let i = 0;
      let node = walker.currentNode;
      let best2 = null;
      let bestScore2 = -1;
      while (i < maxNodes && (node = walker.nextNode())) {
        i++;
        // Cheap filter first; avoids calling getComputedStyle for most nodes
        if (node.scrollHeight <= node.clientHeight + 40) continue;
        const cs = getComputedStyle(node);
        const oy = cs.overflowY;
        const of = cs.overflow;
        const tag = (node && node.tagName) ? String(node.tagName).toUpperCase() : "";
        const isScrollable =
          (oy === "auto" || oy === "scroll" || oy === "overlay" || of === "auto" || of === "scroll");
        if (isScrollable && (tag === "HTML" || tag === "BODY")) continue;
        if (isScrollable) {
          const score = Number(node.scrollTop) || 0;
          if (score > bestScore2) {
            bestScore2 = score;
            best2 = node;
          }
        }
      }
      return best2 || document.scrollingElement || document.documentElement;
    }
    // When our heuristics fail, we often fall back to HTML/BODY.
    // LinkedIn's nested feed scroller can be hard to detect via DOM shape;
    // if we get HTML/BODY, do a capped global scan for the scrollable
    // overflow container with the largest scrollTop.
    function globalScanScrollRoot() {
      const candidates = [];
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT, null);
      let node = walker.currentNode;
      let seen = 0;
      const maxNodes = 12000;
      while (seen < maxNodes && (node = walker.nextNode())) {
        seen++;
        const tag = node.tagName ? String(node.tagName).toUpperCase() : "";
        if (tag === "HTML" || tag === "BODY") continue;
        if (node.scrollHeight <= node.clientHeight + 40) continue;
        let cs;
        try {
          cs = getComputedStyle(node);
        } catch {
          continue;
        }
        const oy = cs.overflowY;
        const of = cs.overflow;
        const isScrollable =
          oy === "auto" || oy === "scroll" || oy === "overlay" || of === "auto" || of === "scroll";
        if (!isScrollable) continue;
        const score = Number(node.scrollTop) || 0;
        candidates.push({ node, score });
      }
      candidates.sort((a, b) => b.score - a.score);
      return candidates.length ? candidates[0].node : null;
    }

    let root = findFeedScrollRoot();
    const rootTag = root && root.tagName ? String(root.tagName).toUpperCase() : "";
    if (rootTag === "HTML" || rootTag === "BODY") {
      const scanned = globalScanScrollRoot();
      if (scanned) root = scanned;
    }
    let markerPrefix = "";
    try {
      const cards = Array.from(
        document.querySelectorAll("div.feed-shared-update-v2, article.feed-shared-update-v2")
      );
      let best = null;
      for (const card of cards) {
        // Filter: avoid obvious non-card variants.
        if (!card.classList) continue;
        const cls = Array.from(card.classList);
        const isCard = cls.includes("feed-shared-update-v2") ||
          cls.some(
            (c) =>
              c.startsWith("feed-shared-update-v2--") &&
              !c.startsWith("feed-shared-update-v2__")
          );
        if (!isCard) continue;

        const rect = card.getBoundingClientRect();
        if (rect.bottom < -80) continue;
        if (rect.top > window.innerHeight + 80) continue;

        const contentEl =
          card.querySelector("span.break-words") ||
          card.querySelector(".feed-shared-update-v2__description") ||
          card.querySelector(".update-components-text") ||
          card.querySelector(".feed-shared-text") ||
          card.querySelector(".feed-shared-update-v2__commentary");
        const rawText = contentEl ? contentEl.innerText : card.innerText;
        const text = String(rawText || "").replace(/\s+/g, " ").trim();
        if (!text || text.length < 30) continue;
        const prefix = text.slice(0, 60);
        if (!best || rect.top < best.rectTop) best = { rectTop: rect.top, prefix };
      }
      markerPrefix = best && best.prefix ? best.prefix.slice(0, 40) : "";
    } catch {
      /* ignore marker failures */
    }
    return {
      windowScrollY: window.scrollY || 0,
      docElScrollTop: document.documentElement.scrollTop || 0,
      rootScrollTop: root.scrollTop || 0,
      rootScrollHeight: root.scrollHeight,
      rootClientHeight: root.clientHeight,
      rootTag: root.tagName,
      rootId: root.id || "",
      rootClasses:
        typeof root.className === "string"
          ? root.className.split(/\s+/).filter(Boolean).slice(0, 10)
          : [],
      markerPrefix,
    };
  });

  const feedScroll = {
    scrollTop: data.rootScrollTop,
    tagName: data.rootTag,
    id: data.rootId,
    classTokens: data.rootClasses,
  };

  return {
    activityKey: activityKeyFromUrl(cliUrl),
    pageUrl: page.url(),
    scrollY: Math.max(
      data.windowScrollY,
      data.docElScrollTop,
      data.rootScrollTop
    ),
    windowScrollY: data.windowScrollY,
    feedScroll,
    markerPrefix: data.markerPrefix || "",
  };
}

/**
 * LinkedIn usually scrolls a nested overflow container; window.scrollTo alone stays at top.
 * We re-find that container after load and set scrollTop (repeatedly for slow hydration).
 */
async function applyResumeScroll(page, st) {
  const legacyY = Number(st.scrollY) || 0;
  const containerTop =
    st.feedScroll && typeof st.feedScroll.scrollTop === "number"
      ? st.feedScroll.scrollTop
      : legacyY;
  const winY =
    st.windowScrollY != null ? Number(st.windowScrollY) : legacyY;
  const markerPrefix = st.markerPrefix ? String(st.markerPrefix) : "";

  await page
    .waitForSelector("main, [class*='feed-shared-update']", {
      timeout: 25_000,
    })
    .catch(() => {});

  await sleep(1500);

  for (let attempt = 0; attempt < 4; attempt++) {
    await page.evaluate(
      ({ containerTop: ct, winY: wy }) => {
        function findFeedScrollRoot() {
          const anchor =
            document.querySelector(".feed-shared-update-v2") ||
            document.querySelector("div[class*='feed-shared-update-v2']") ||
            document.querySelector("li.reusable-search__result-container") ||
            document.querySelector("main");
          if (!anchor) return document.scrollingElement || document.documentElement;
          // 1) Prefer ancestor scroll container (fast), choose the one that is
          // currently scrolled (largest scrollTop).
          let best = null;
          let bestScore = -1;
          let el = anchor;
          for (let d = 0; d < 50 && el; d++) {
            const cs = getComputedStyle(el);
            const oy = cs.overflowY;
            const of = cs.overflow;
            const tag = (el && el.tagName) ? String(el.tagName).toUpperCase() : "";
            const isScrollable =
              el.scrollHeight > el.clientHeight + 40 &&
              (oy === "auto" || oy === "scroll" || oy === "overlay" || of === "auto" || of === "scroll");
            if (isScrollable && (tag === "HTML" || tag === "BODY")) continue;
            if (isScrollable) {
              const score = Number(el.scrollTop) || 0;
              if (score > bestScore) {
                bestScore = score;
                best = el;
              }
            }
            el = el.parentElement;
          }
          if (best) return best;

          // 2) Fallback: scan descendants (ensures we find inner feed scroller)
          const start = anchor.closest && anchor.closest("main") ? anchor.closest("main") : anchor;
          const maxNodes = 2500;
          const walker = document.createTreeWalker(start, NodeFilter.SHOW_ELEMENT, null);
          let i = 0;
          let node = walker.currentNode;
          let best2 = null;
          let bestScore2 = -1;
          while (i < maxNodes && (node = walker.nextNode())) {
            i++;
            if (node.scrollHeight <= node.clientHeight + 40) continue;
            const cs = getComputedStyle(node);
            const oy = cs.overflowY;
            const of = cs.overflow;
            const tag = (node && node.tagName) ? String(node.tagName).toUpperCase() : "";
            const isScrollable =
              oy === "auto" || oy === "scroll" || oy === "overlay" || of === "auto" || of === "scroll";
            if (isScrollable && (tag === "HTML" || tag === "BODY")) continue;
            if (isScrollable) {
              const score = Number(node.scrollTop) || 0;
              if (score > bestScore2) {
                bestScore2 = score;
                best2 = node;
              }
            }
          }
          return best2 || document.scrollingElement || document.documentElement;
        }
        function globalScanScrollRoot() {
          const candidates = [];
          const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT, null);
          let node = walker.currentNode;
          let seen = 0;
          const maxNodes = 12000;
          while (seen < maxNodes && (node = walker.nextNode())) {
            seen++;
            const tag = node.tagName ? String(node.tagName).toUpperCase() : "";
            if (tag === "HTML" || tag === "BODY") continue;
            if (node.scrollHeight <= node.clientHeight + 40) continue;
            let cs;
            try {
              cs = getComputedStyle(node);
            } catch {
              continue;
            }
            const oy = cs.overflowY;
            const of = cs.overflow;
            const isScrollable =
              oy === "auto" || oy === "scroll" || oy === "overlay" || of === "auto" || of === "scroll";
            if (!isScrollable) continue;
            const score = Number(node.scrollTop) || 0;
            candidates.push({ node, score });
          }
          candidates.sort((a, b) => b.score - a.score);
          return candidates.length ? candidates[0].node : null;
        }

        let root = findFeedScrollRoot();
        const rootTag = root && root.tagName ? String(root.tagName).toUpperCase() : "";
        if (rootTag === "HTML" || rootTag === "BODY") {
          const scanned = globalScanScrollRoot();
          if (scanned) root = scanned;
        }
        const max = Math.max(0, root.scrollHeight - root.clientHeight);
        root.scrollTop = Math.min(Math.max(0, ct), max);
        // If we found the inner feed scroller (not HTML), avoid fighting it
        // by forcing document scroll as well.
        const tag = root && root.tagName ? String(root.tagName).toUpperCase() : "";
        window.scrollTo(0, tag === "HTML" ? wy : 0);
      },
      { containerTop, winY }
    );
    await sleep(800 + attempt * 500);
  }

  // If our scrollTop/container heuristic is wrong (LinkedIn frequently is),
  // try to align by content marker instead of trusting scroll position.
  if (markerPrefix) {
    try {
      const ok = await page.evaluate((prefix) => {
        const cards = Array.from(
          document.querySelectorAll("div.feed-shared-update-v2, article.feed-shared-update-v2")
        );
        for (const card of cards) {
          if (!card || !card.classList) continue;
          const cls = Array.from(card.classList);
          const isCard = cls.includes("feed-shared-update-v2") ||
            cls.some(
              (c) =>
                c.startsWith("feed-shared-update-v2--") &&
                !c.startsWith("feed-shared-update-v2__")
            );
          if (!isCard) continue;

          const contentEl =
            card.querySelector("span.break-words") ||
            card.querySelector(".feed-shared-update-v2__description") ||
            card.querySelector(".update-components-text") ||
            card.querySelector(".feed-shared-text") ||
            card.querySelector(".feed-shared-update-v2__commentary");
          const rawText = contentEl ? contentEl.innerText : card.innerText;
          const text = String(rawText || "").replace(/\s+/g, " ").trim();
          if (!text || text.length < 30) continue;
          if (!text.includes(prefix)) continue;
          try {
            card.scrollIntoView({ block: "start", behavior: "instant" });
          } catch {
            try {
              card.scrollIntoView(true);
            } catch {
              /* ignore */
            }
          }
          return true;
        }
        return false;
      }, markerPrefix);
      if (ok) await sleep(1200);
    } catch {
      /* ignore marker alignment failures */
    }
  }
}

/**
 * Anchor-based resume: find the card whose extracted text contains `anchorPrefix`,
 * scroll it into view, and remove earlier cards from the DOM.
 * This is resilient when scrollTop restoration is unreliable due to LinkedIn feed
 * virtualization.
 */
async function resumeByAnchorPrefix(page, anchorPrefix, { maxSteps = 8 } = {}) {
  if (!anchorPrefix) return false;

  const found = await (async () => {
    for (let step = 0; step < maxSteps; step++) {
      const ok = await page.evaluate((prefix) => {
        function norm(s) {
          return String(s || "")
            .replace(/\s+/g, " ")
            .trim();
        }
        function isMainFeedCard(el) {
          if (!el || (el.tagName !== "DIV" && el.tagName !== "ARTICLE")) return false;
          const list = el.classList;
          if (!list) return false;
          for (let i = 0; i < list.length; i++) {
            const c = list[i];
            if (c === "feed-shared-update-v2") return true;
            if (
              c.startsWith("feed-shared-update-v2--") &&
              !c.startsWith("feed-shared-update-v2__")
            ) {
              return true;
            }
          }
          return false;
        }

        function extractContent(root) {
          const trySel = [
            () => root.querySelector("span.break-words"),
            () => root.querySelector(".feed-shared-update-v2__description"),
            () => root.querySelector(".update-components-text"),
            () => root.querySelector(".feed-shared-text"),
          ];
          for (let i = 0; i < trySel.length; i++) {
            const el = trySel[i]();
            if (!el) continue;
            const t = norm(el.innerText);
            if (t.length > 40) return t;
          }
          return norm(root.innerText);
        }

        const prefixNorm = norm(prefix);
        const innerH = window.innerHeight || 800;
        const roots = Array.from(
          document.querySelectorAll("div.feed-shared-update-v2, article.feed-shared-update-v2")
        ).filter((el) => isMainFeedCard(el));

        const visible = roots.filter((r) => {
          const rect = r.getBoundingClientRect();
          return rect.bottom >= -200 && rect.top <= innerH + 200;
        });
        if (!visible.length) return false;

        let match = null;
        let matchTop = 0;
        for (const root of visible) {
          const content = extractContent(root);
          if (!content) continue;
          // prefix is a normalized prefix from our earlier extraction
          if (content.startsWith(prefixNorm) || content.includes(prefixNorm)) {
            const rect = root.getBoundingClientRect();
            match = root;
            matchTop = rect.top;
            break;
          }
        }
        if (!match) return false;

        // Make sure it's visible.
        try {
          match.scrollIntoView({ block: "start", behavior: "instant" });
        } catch {
          try {
            match.scrollIntoView(true);
          } catch {
            /* ignore */
          }
        }

        // Prune earlier cards currently present in DOM.
        const all = roots;
        const allWithTop = all
          .map((r) => ({ r, t: r.getBoundingClientRect().top }))
          .sort((a, b) => a.t - b.t);
        for (let i = 0; i < allWithTop.length; i++) {
          if (allWithTop[i].t >= matchTop - 2) break;
          try {
            allWithTop[i].r.remove();
          } catch {
            /* ignore */
          }
        }
        return true;
      }, anchorPrefix);

      if (ok) return true;

      // Move a bit further so the anchor card hydrates.
      await nudgeScroll(page, 0.55);
      await sleep(650);
    }
    return false;
  })();

  return found;
}

/**
 * Restore last position for this author’s activity, or open cliUrl from the top.
 */
async function gotoActivityWithResume(page, cliUrl, { statePath, useResume, firstPage }) {
  if (firstPage || !useResume) {
    await page.goto(cliUrl, { waitUntil: "domcontentloaded", timeout: 120_000 });
    await sleep(3000);
    return;
  }

  const st = await readResumeState(statePath);
  const want = activityKeyFromUrl(cliUrl);
  if (st && st.pageUrl && st.activityKey === want) {
    const fs = st.feedScroll;
    const desc = fs
      ? `feed scrollTop≈${fs.scrollTop}`
      : `legacy scrollY≈${st.scrollY}`;
    console.log(`[collect] Resuming: ${desc} — ${String(st.pageUrl).slice(0, 72)}…`);
    try {
      await page.goto(st.pageUrl, { waitUntil: "domcontentloaded", timeout: 120_000 });
      await sleep(3000);
      await applyResumeScroll(page, st);
      if (st.anchorPrefix) {
        const ok = await resumeByAnchorPrefix(page, st.anchorPrefix, { maxSteps: 8 });
        console.log(`[collect] Anchor resume ${ok ? "matched" : "not found"}.`);
      }
      await sleep(1500);
      return;
    } catch (e) {
      console.warn("[collect] Resume goto failed, falling back to CLI URL:", e.message || e);
    }
  } else if (st && st.activityKey !== want) {
    console.log(
      "[collect] State file is for another activity URL; starting from top of your CLI URL."
    );
  } else if (!st || !st.pageUrl) {
    console.log("[collect] No resume state (or empty); opening CLI URL from the top.");
  }

  await page.goto(cliUrl, { waitUntil: "domcontentloaded", timeout: 120_000 });
  await sleep(3000);
}

async function feedScope(page) {
  const main = page.locator("main").first();
  try {
    if ((await main.count()) > 0) return main;
  } catch {
    /* use full page */
  }
  return page;
}

/**
 * One "See more" / "Show more" click. Viewport-only mode scans <main> buttons in document order
 * and clicks the first whose label matches — avoids walking hundreds of Playwright locators.
 */
async function clickOneSeeMore(
  page,
  {
    viewportOnly = true,
    maxCandidatesPerGroup = 24,
  } = {}
) {
  if (viewportOnly) {
    const handle = await page.evaluateHandle(() => {
      const root = document.querySelector("main") || document.body;
      const buttons = root.querySelectorAll("button");
      const vh = window.innerHeight;
      const vw = window.innerWidth;
      for (const b of buttons) {
        const label = (b.getAttribute("aria-label") || b.innerText || "").trim();
        if (!/see more|show more/i.test(label)) continue;
        const r = b.getBoundingClientRect();
        if (r.width < 2 || r.height < 2) continue;
        if (r.bottom < -40 || r.top > vh + 40 || r.right < 0 || r.left > vw + 20) continue;
        const cs = window.getComputedStyle(b);
        if (cs.visibility === "hidden" || cs.display === "none" || cs.opacity === "0") continue;
        return b;
      }
      return null;
    });
    try {
      const el = await handle.asElement();
      if (el) {
        try {
          // LinkedIn frequently re-renders the feed; the candidate button can become
          // unstable right after we find it. Treat "couldn't scroll into view"
          // as a non-fatal miss and keep expanding other buttons.
          await el.scrollIntoViewIfNeeded({ timeout: 4000 });
        } catch {
          return false;
        }
        await sleep(100 + Math.random() * 160);
        const ok = await el.click({ timeout: 3500 }).then(
          () => true,
          () => false
        );
        if (ok) return true;
      }
    } finally {
      await handle.dispose();
    }
  }

  const root = await feedScope(page);
  const cap = Math.max(4, Math.min(80, maxCandidatesPerGroup));
  const groups = [
    root.getByRole("button", { name: /see more/i }),
    root.getByRole("button", { name: /show more/i }),
    root.locator("button.feed-shared-inline-show-more-text__see-more-less-button"),
    root.locator("button").filter({ hasText: /^see more$/i }),
  ];

  for (const group of groups) {
    const n = await group.count().catch(() => 0);
    for (let j = 0; j < Math.min(n, cap); j++) {
      const el = group.nth(j);
      const vis = await el.isVisible({ timeout: 400 }).catch(() => false);
      if (!vis) continue;
      try {
        await el.scrollIntoViewIfNeeded({ timeout: 2000 });
      } catch {
        continue;
      }
      await sleep(100 + Math.random() * 140);
      const ok = await el
        .click({ timeout: 3500 })
        .then(() => true)
        .catch(() => false);
      if (ok) return true;
    }
  }
  return false;
}

/**
 * Keep clicking the next visible control until several passes find nothing.
 */
async function expandAllSeeMoreInView(
  page,
  {
    maxNoProgress = 3,
    maxClicks = 24,
    viewportOnly = true,
    maxCandidatesPerGroup = 24,
  } = {}
) {
  let noProgress = 0;
  let clicks = 0;
  const clickOpts = { viewportOnly, maxCandidatesPerGroup };
  while (noProgress < maxNoProgress && clicks < maxClicks) {
    const did = await clickOneSeeMore(page, clickOpts);
    if (did) {
      noProgress = 0;
      clicks++;
      await sleep(200 + Math.random() * 220);
    } else {
      noProgress++;
      await sleep(280);
    }
  }
  return clicks;
}

/**
 * Hard navigation to trim DOM while keeping scroll via saved resume state (same tab).
 */
async function reloadFeedPreserveScroll(page, cliUrl, statePath, useResume) {
  if (!useResume) {
    console.log("[collect] DOM reload: --no-resume; opening CLI URL fresh (scroll not restored).");
    await page.goto(cliUrl, { waitUntil: "domcontentloaded", timeout: 120_000 });
    await sleep(3000);
    return;
  }
  const snap = await captureScrollState(page, cliUrl);
  await writeResumeState(statePath, snap);
  const gotoUrl = snap.pageUrl || cliUrl;
  console.log("[collect] Reloading tab to drop bloated DOM (resume scroll after load)…");
  await page.goto(gotoUrl, { waitUntil: "domcontentloaded", timeout: 120_000 });
  await sleep(2500);
  await applyResumeScroll(page, snap);
  await sleep(1200);
}

/** Small viewport steps only — does not hit document bottom (no infinite load). */
async function nudgeScroll(page, fractionOfViewport) {
  const vp = page.viewportSize() || { height: 800 };
  const delta = Math.floor(vp.height * fractionOfViewport);
  await page.mouse.wheel(0, delta);
  await sleep(900 + Math.random() * 600);
}

/** Same feed-root discovery as applyResumeScroll — scroll inner scroller, not only window. */
async function scrollFeedRound(page, roundIndex, { fullScrollEvery = 3, legacyScroll = false } = {}) {
  if (legacyScroll) {
    const vp = page.viewportSize() || { height: 800 };
    await page.mouse.wheel(0, Math.floor(vp.height * 0.85));
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    return;
  }

  const viewportFraction = 0.82 + Math.random() * 0.18;
  await page.evaluate(
    ({ roundIndex: ri, fullScrollEvery: fse, viewportFraction: vf }) => {
      function findFeedScrollRoot() {
        const anchor =
          document.querySelector(".feed-shared-update-v2") ||
          document.querySelector("div[class*='feed-shared-update-v2']") ||
          document.querySelector("li.reusable-search__result-container") ||
          document.querySelector("main");
        if (!anchor) return document.scrollingElement || document.documentElement;
        let el = anchor;
        for (let d = 0; d < 50 && el; d++) {
          const cs = getComputedStyle(el);
          const oy = cs.overflowY;
          if (
            el.scrollHeight > el.clientHeight + 40 &&
            (oy === "auto" || oy === "scroll" || oy === "overlay")
          ) {
            return el;
          }
          el = el.parentElement;
        }
        return document.scrollingElement || document.documentElement;
      }
      const root = findFeedScrollRoot();
      const vp = window.innerHeight || 800;
      const delta = Math.floor(vp * vf);
      const max = Math.max(0, root.scrollHeight - root.clientHeight);
      const full = fse > 0 && (ri + 1) % fse === 0;
      if (full) {
        root.scrollTop = max;
      } else {
        root.scrollTop = Math.min(max, root.scrollTop + delta);
      }
    },
    { roundIndex, fullScrollEvery, viewportFraction }
  );
}

async function installMediaBlocker(page) {
  await page.route("**/*", (route) => {
    const t = route.request().resourceType();
    if (t === "image" || t === "media") return route.abort();
    route.continue();
  });
}

async function runFirstPageOnly(page, expandOpts = {}) {
  const eo = {
    maxNoProgress: expandOpts.maxNoProgress ?? 5,
    maxClicks: expandOpts.maxClicks ?? 40,
    viewportOnly: expandOpts.viewportOnly !== false,
    maxCandidatesPerGroup: expandOpts.maxCandidatesPerGroup ?? 24,
  };
  await sleep(2500);
  await expandAllSeeMoreInView(page, { ...eo, maxNoProgress: 6, maxClicks: Math.max(eo.maxClicks, 45) });

  await nudgeScroll(page, 0.45);
  await expandAllSeeMoreInView(page, eo);

  await nudgeScroll(page, 0.5);
  await expandAllSeeMoreInView(page, eo);

  await page.evaluate(() => window.scrollTo(0, 0));
  await sleep(800);
}

/**
 * Append deduped posts + refresh resume without waiting for end of pass.
 */
async function runExtractCheckpoint(page, o, roundIdx, totalRounds) {
  const extractor = selectDomExtractor(o.url);
  const rows = await page.evaluate(extractor, o.pruneDom);
  if (String(process.env.LI_DEBUG_EXTRACT || "") === "1") {
    const sample = (rows || [])
      .slice(0, 2)
      .map((r) => ({
        format: r.format,
        urn: r.urn,
        url: r.url,
        author: r.author,
        content_prefix: String(r.content || "").replace(/\s+/g, " ").slice(0, 80),
      }));
    console.log(
      `[collect][debug] extracted ${rows ? rows.length : 0} row(s) this checkpoint; sample=`,
      sample
    );
  }
  const n = await appendExtractJsonl(o.extractJsonl, rows, o.authorArg);
  const lastRow = rows && rows.length ? rows[rows.length - 1] : null;
  const anchorPrefix =
    lastRow && typeof lastRow.content === "string" && lastRow.content.trim()
      ? lastRow.content.trim().replace(/\s+/g, " ").slice(0, 120)
      : "";
  const tag = o.passLabel ? ` ${o.passLabel}` : "";
  console.log(
    `[collect] checkpoint${tag} round ${roundIdx}/${totalRounds}: +${n} new → ${o.extractJsonl}`
  );
  if (o.useResume) {
    const snap = await captureScrollState(page, o.url);
    if (anchorPrefix) snap.anchorPrefix = anchorPrefix;
    await writeResumeState(o.statePath, snap);
  }
  return { extracted: rows ? rows.length : 0, newRows: n };
}

async function scrollAndExpand(
  page,
  rounds,
  pauseMin,
  pauseMax,
  checkpointOpts,
  scrollOpts = {},
  extra = {}
) {
  const fullScrollEvery =
    scrollOpts.fullScrollEvery !== undefined ? scrollOpts.fullScrollEvery : 3;
  const legacyScroll = !!scrollOpts.legacyScroll;

  const expandOpts = extra.expandOpts || {};
  const eo = {
    maxNoProgress: expandOpts.maxNoProgress ?? 3,
    maxClicks: expandOpts.maxClicks ?? 24,
    viewportOnly: expandOpts.viewportOnly !== false,
    maxCandidatesPerGroup: expandOpts.maxCandidatesPerGroup ?? 24,
  };
  const reloadDomEveryRounds = Math.max(0, Number(extra.reloadDomEveryRounds) || 0);
  const reloadCtx = extra.reloadCtx;
  const maxRoundsPerPass = Math.max(0, Number(extra.maxRoundsPerPass) || 0);
  const effectiveRounds =
    maxRoundsPerPass > 0 ? Math.min(rounds, maxRoundsPerPass) : rounds;

  const stopAfterSame = Math.max(
    0,
    parseInt(process.env.LI_EARLY_STOP_SAME_EXTRACTED || "3", 10) || 0
  );
  let sameExtractedStreak = 0;
  let lastExtracted = null;

  for (let r = 0; r < effectiveRounds; r++) {
    if (
      r > 0 &&
      reloadDomEveryRounds > 0 &&
      r % reloadDomEveryRounds === 0 &&
      reloadCtx
    ) {
      await reloadFeedPreserveScroll(
        page,
        reloadCtx.cliUrl,
        reloadCtx.statePath,
        reloadCtx.useResume
      );
    }
    await expandAllSeeMoreInView(page, eo);
    await sleep(Math.floor((pauseMin + Math.random() * (pauseMax - pauseMin)) * 1000));
    await scrollFeedRound(page, r, { fullScrollEvery, legacyScroll });
    await sleep(Math.floor(800 + Math.random() * 800));

    if (
      checkpointOpts &&
      checkpointOpts.everyRounds > 0 &&
      checkpointOpts.extractJsonl
    ) {
      const rr = r + 1;
      if (rr % checkpointOpts.everyRounds === 0 || r === effectiveRounds - 1) {
        const snap = await runExtractCheckpoint(page, checkpointOpts, rr, effectiveRounds);
        if (stopAfterSame > 0 && snap && snap.newRows === 0) {
          if (lastExtracted != null && snap.extracted === lastExtracted) {
            sameExtractedStreak++;
          } else {
            sameExtractedStreak = 1;
          }
          lastExtracted = snap.extracted;
          if (sameExtractedStreak >= stopAfterSame) {
            const tag = checkpointOpts.passLabel ? ` ${checkpointOpts.passLabel}` : "";
            console.log(
              `[collect] early-stop${tag}: extracted stayed at ${snap.extracted} with +0 new for ${sameExtractedStreak} checkpoint(s); moving on.`
            );
            break;
          }
        } else if (snap && snap.newRows > 0) {
          sameExtractedStreak = 0;
          lastExtracted = snap.extracted;
        }
      }
    }
  }
}

async function waitForEnterBeforeClose() {
  if (!process.stdin.isTTY) {
    console.log(
      "[collect] stdin is not a TTY (--keep-open ignored for close delay); closing browser."
    );
    return;
  }
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  await new Promise((resolve) => {
    rl.question(
      "[collect] HTML saved. Chrome stays open — scroll, expand, or SingleFile-save if you want.\n" +
        "          Press Enter here when finished to close the browser (do this before the next run that uses the same profile folder).\n",
      () => {
        rl.close();
        resolve();
      }
    );
  });
}

function parseArgs() {
  const raw = process.argv.slice(2);
  let passes = Math.max(1, Number(process.env.LI_PASSES || "1"));
  const pi = raw.indexOf("--passes");
  if (pi !== -1 && raw[pi + 1] && !raw[pi + 1].startsWith("--")) {
    passes = Math.max(1, parseInt(raw[pi + 1], 10) || 1);
  }

  let extractJsonl = process.env.LI_EXTRACT_JSONL
    ? path.resolve(process.env.LI_EXTRACT_JSONL)
    : null;
  const ej = raw.indexOf("--extract-jsonl");
  if (ej !== -1 && raw[ej + 1] && !raw[ej + 1].startsWith("--")) {
    extractJsonl = path.resolve(raw[ej + 1]);
  }

  let authorArg = process.env.LI_AUTHOR || "Target Author";
  const ai = raw.indexOf("--author");
  if (ai !== -1 && raw[ai + 1] && !raw[ai + 1].startsWith("--")) {
    authorArg = raw[ai + 1];
  }

  const noHtml = raw.includes("--no-html");
  const pruneDom =
    raw.includes("--prune-dom") || String(process.env.LI_PRUNE_DOM || "") === "1";

  let checkpointRounds = 0;
  if (extractJsonl) {
    if (raw.includes("--no-checkpoint")) {
      checkpointRounds = 0;
    } else {
      const cri = raw.indexOf("--checkpoint-rounds");
      if (cri !== -1 && raw[cri + 1] && !raw[cri + 1].startsWith("--")) {
        const v = parseInt(raw[cri + 1], 10);
        checkpointRounds = Number.isNaN(v) ? 1 : Math.max(0, v);
      } else if (
        process.env.LI_CHECKPOINT_ROUNDS !== undefined &&
        String(process.env.LI_CHECKPOINT_ROUNDS) !== ""
      ) {
        const v = parseInt(String(process.env.LI_CHECKPOINT_ROUNDS), 10);
        checkpointRounds = Number.isNaN(v) ? 1 : Math.max(0, v);
      } else {
        checkpointRounds = 1;
      }
    }
  }

  const pos = [];
  for (let i = 0; i < raw.length; i++) {
    const a = raw[i];
    if (
      a === "--passes" ||
      a === "--extract-jsonl" ||
      a === "--author" ||
      a === "--checkpoint-rounds" ||
      a === "--full-scroll-every" ||
      a === "--recycle-every-passes" ||
      a === "--reload-dom-every-rounds" ||
      a === "--reload-dom-every-passes" ||
      a === "--max-rounds-per-pass"
    ) {
      i++;
      continue;
    }
    if (a.startsWith("--")) continue;
    pos.push(a);
  }

  const firstPage = raw.includes("--first-page");
  const keepOpen =
    raw.includes("--keep-open") || String(process.env.LI_KEEP_OPEN || "") === "1";
  const fresh = raw.includes("--fresh");
  const noResume = raw.includes("--no-resume");
  const mobile =
    raw.includes("--mobile") || String(process.env.LI_MOBILE || "") === "1";
  const blockMedia =
    raw.includes("--block-media") || String(process.env.LI_BLOCK_MEDIA || "") === "1";
  const legacyScroll =
    raw.includes("--legacy-scroll") || String(process.env.LI_LEGACY_SCROLL || "") === "1";

  let fullScrollEvery = 3;
  const fse = raw.indexOf("--full-scroll-every");
  if (fse !== -1 && raw[fse + 1] && !raw[fse + 1].startsWith("--")) {
    const v = parseInt(raw[fse + 1], 10);
    fullScrollEvery = Number.isNaN(v) ? 3 : Math.max(0, v);
  } else if (
    process.env.LI_FULL_SCROLL_EVERY !== undefined &&
    String(process.env.LI_FULL_SCROLL_EVERY) !== ""
  ) {
    const v = parseInt(String(process.env.LI_FULL_SCROLL_EVERY), 10);
    fullScrollEvery = Number.isNaN(v) ? 3 : Math.max(0, v);
  }

  let recycleEveryPasses = 0;
  const rep = raw.indexOf("--recycle-every-passes");
  if (rep !== -1 && raw[rep + 1] && !raw[rep + 1].startsWith("--")) {
    const v = parseInt(raw[rep + 1], 10);
    recycleEveryPasses = Number.isNaN(v) ? 0 : Math.max(0, v);
  } else if (
    process.env.LI_RECYCLE_BROWSER_EVERY_PASSES !== undefined &&
    String(process.env.LI_RECYCLE_BROWSER_EVERY_PASSES) !== ""
  ) {
    const v = parseInt(String(process.env.LI_RECYCLE_BROWSER_EVERY_PASSES), 10);
    recycleEveryPasses = Number.isNaN(v) ? 0 : Math.max(0, v);
  }

  const expandFullFeed =
    raw.includes("--expand-scan-full-feed") ||
    String(process.env.LI_EXPAND_FULL_FEED || "") === "1";
  const expandViewportOnly = !expandFullFeed;

  let reloadDomEveryRounds = 0;
  const rdr = raw.indexOf("--reload-dom-every-rounds");
  if (rdr !== -1 && raw[rdr + 1] && !raw[rdr + 1].startsWith("--")) {
    const v = parseInt(raw[rdr + 1], 10);
    reloadDomEveryRounds = Number.isNaN(v) ? 0 : Math.max(0, v);
  } else if (
    process.env.LI_RELOAD_DOM_EVERY_ROUNDS !== undefined &&
    String(process.env.LI_RELOAD_DOM_EVERY_ROUNDS) !== ""
  ) {
    const v = parseInt(String(process.env.LI_RELOAD_DOM_EVERY_ROUNDS), 10);
    reloadDomEveryRounds = Number.isNaN(v) ? 0 : Math.max(0, v);
  }

  let reloadDomEveryPasses = 0;
  const rdp = raw.indexOf("--reload-dom-every-passes");
  if (rdp !== -1 && raw[rdp + 1] && !raw[rdp + 1].startsWith("--")) {
    const v = parseInt(raw[rdp + 1], 10);
    reloadDomEveryPasses = Number.isNaN(v) ? 0 : Math.max(0, v);
  } else if (
    process.env.LI_RELOAD_DOM_EVERY_PASSES !== undefined &&
    String(process.env.LI_RELOAD_DOM_EVERY_PASSES) !== ""
  ) {
    const v = parseInt(String(process.env.LI_RELOAD_DOM_EVERY_PASSES), 10);
    reloadDomEveryPasses = Number.isNaN(v) ? 0 : Math.max(0, v);
  }

  let maxRoundsPerPass = 0;
  const mrp = raw.indexOf("--max-rounds-per-pass");
  if (mrp !== -1 && raw[mrp + 1] && !raw[mrp + 1].startsWith("--")) {
    const v = parseInt(raw[mrp + 1], 10);
    maxRoundsPerPass = Number.isNaN(v) ? 0 : Math.max(0, v);
  } else if (
    process.env.LI_MAX_ROUNDS_PER_PASS !== undefined &&
    String(process.env.LI_MAX_ROUNDS_PER_PASS) !== ""
  ) {
    const v = parseInt(String(process.env.LI_MAX_ROUNDS_PER_PASS), 10);
    maxRoundsPerPass = Number.isNaN(v) ? 0 : Math.max(0, v);
  }

  return {
    firstPage,
    keepOpen,
    fresh,
    noResume,
    passes,
    extractJsonl,
    noHtml,
    pruneDom,
    checkpointRounds,
    authorArg,
    blockMedia,
    legacyScroll,
    fullScrollEvery,
    recycleEveryPasses,
    expandViewportOnly,
    reloadDomEveryRounds,
    reloadDomEveryPasses,
    maxRoundsPerPass,
    mobile,
    url: pos[0],
    outPath: pos[1] || "dump.html",
  };
}

/** dump.html → dump_pass03.html */
function passOutPath(baseOut, passIndex, totalPasses) {
  const abs = path.resolve(baseOut);
  if (totalPasses <= 1) return abs;
  const dir = path.dirname(abs);
  const ext = path.extname(abs);
  const base = path.basename(abs, ext);
  const n = String(passIndex).padStart(2, "0");
  return path.join(dir, `${base}_pass${n}${ext}`);
}

function buildPersistentContextOpts(mobile) {
  const base = {
    headless: false,
    channel: "chrome",
    viewport: { width: 1280, height: 900 },
  };
  if (!mobile) return base;
  const phone = devices["iPhone 13"];
  return {
    ...base,
    userAgent: phone.userAgent,
    viewport: phone.viewport,
    deviceScaleFactor: phone.deviceScaleFactor,
    isMobile: phone.isMobile,
    hasTouch: phone.hasTouch,
  };
}

async function runFeedCollectionPasses(
  page,
  {
    passFrom,
    passTo,
    passCountTotal,
    rounds,
    pauseMin,
    pauseMax,
    extractJsonl,
    checkpointRounds,
    pruneDom,
    authorArg,
    useResume,
    statePath,
    url,
    noHtml,
    outPath,
    scrollOpts,
    expandOpts,
    reloadDomEveryRounds,
    reloadDomEveryPasses,
    maxRoundsPerPass,
    written,
  }
) {
  const extractor = selectDomExtractor(url);
  let lastAnchorPrefixInPass = "";
  for (let p = passFrom; p <= passTo; p++) {
    console.log(`[collect] Pass ${p}/${passCountTotal}: scrolling + expanding…`);
    const checkpointOpts =
      extractJsonl && checkpointRounds > 0
        ? {
            everyRounds: checkpointRounds,
            extractJsonl,
            pruneDom,
            authorArg,
            useResume,
            statePath,
            url,
            passLabel: `pass ${p}/${passCountTotal}`,
          }
        : null;
    lastAnchorPrefixInPass = "";
    await scrollAndExpand(page, rounds, pauseMin, pauseMax, checkpointOpts, scrollOpts, {
      expandOpts,
      reloadDomEveryRounds,
      reloadCtx: { cliUrl: url, statePath, useResume },
      maxRoundsPerPass,
    });
    if (extractJsonl && checkpointRounds === 0) {
      const rows = await page.evaluate(extractor, pruneDom);
      const n = await appendExtractJsonl(extractJsonl, rows, authorArg);
      const lastRow = rows && rows.length ? rows[rows.length - 1] : null;
      lastAnchorPrefixInPass =
        lastRow && typeof lastRow.content === "string" && lastRow.content.trim()
          ? lastRow.content.trim().replace(/\s+/g, " ").slice(0, 120)
          : "";
      console.log(
        `[collect] --extract-jsonl +${n} new (pass ${p}/${passCountTotal}) → ${extractJsonl}`
      );
    }
    if (!noHtml) {
      const dest = passOutPath(outPath, p, passCountTotal);
      const html = await page.content();
      await fs.writeFile(dest, html, "utf8");
      written.push(dest);
      console.log(`Wrote ${dest} (${html.length} bytes)`);
    }
    if (useResume && checkpointRounds === 0) {
      const snap = await captureScrollState(page, url);
      if (lastAnchorPrefixInPass) snap.anchorPrefix = lastAnchorPrefixInPass;
      await writeResumeState(statePath, snap);
      console.log(
        `[collect] Resume state updated scrollY=${snap.scrollY} (pass ${p}/${passCountTotal})`
      );
    }
    if (reloadDomEveryPasses > 0 && p % reloadDomEveryPasses === 0 && p < passTo) {
      await reloadFeedPreserveScroll(page, url, statePath, useResume);
    }
    if (p < passTo) {
      await sleep(1200 + Math.random() * 800);
    }
  }
}

async function main() {
  const {
    firstPage,
    keepOpen,
    fresh,
    noResume,
    passes,
    extractJsonl,
    noHtml,
    pruneDom,
    checkpointRounds,
    authorArg,
    blockMedia,
    legacyScroll,
    fullScrollEvery,
    recycleEveryPasses,
    expandViewportOnly,
    reloadDomEveryRounds,
    reloadDomEveryPasses,
    maxRoundsPerPass,
    mobile,
    url,
    outPath: outRel,
  } = parseArgs();
  const outPath = path.resolve(outRel);
  const userDataDir = path.resolve(process.env.LI_USER_DATA || "./user_data");
  const statePath = path.resolve(
    process.env.LI_STATE_FILE || path.join(userDataDir, ".linkedin_collect_state.json")
  );
  const rounds = Number(process.env.LI_ROUNDS || "18");
  const pauseMin = Number(process.env.LI_PAUSE_MIN || "2");
  const pauseMax = Number(process.env.LI_PAUSE_MAX || "4.5");
  const useResume = !firstPage && !noResume;
  const passCount = firstPage ? 1 : passes;

  const scrollOpts = { fullScrollEvery, legacyScroll };
  const expandOpts = { viewportOnly: expandViewportOnly };
  const contextOpts = buildPersistentContextOpts(mobile);

  if (noHtml && !extractJsonl) {
    console.error("--no-html requires --extract-jsonl <file.jsonl>");
    process.exit(2);
  }

  if (pruneDom && !extractJsonl) {
    console.error("--prune-dom only applies with --extract-jsonl (needs in-page extract).");
    process.exit(2);
  }

  if (!url || !url.includes("linkedin.com")) {
    console.error(
      'Usage: node collect_activity_stealth.cjs "https://www.linkedin.com/in/.../recent-activity/all/" [out.html] [--passes N] [--extract-jsonl posts.jsonl] [--no-html] [--prune-dom] [--checkpoint-rounds N] [--no-checkpoint] [--author "Name"] [--first-page] [--keep-open] [--fresh] [--no-resume] [--block-media] [--full-scroll-every N] [--legacy-scroll] [--recycle-every-passes N] [--expand-scan-full-feed] [--reload-dom-every-rounds N] [--reload-dom-every-passes N] [--max-rounds-per-pass N] [--mobile]'
    );
    console.error(
      "Optional env: LI_USER_DATA, LI_ROUNDS (default 18), LI_PAUSE_MIN, LI_PAUSE_MAX, LI_KEEP_OPEN=1, LI_STATE_FILE, LI_PASSES, LI_EXTRACT_JSONL, LI_AUTHOR, LI_PRUNE_DOM=1, LI_BLOCK_MEDIA, LI_FULL_SCROLL_EVERY, LI_LEGACY_SCROLL, LI_RECYCLE_BROWSER_EVERY_PASSES, LI_EXPAND_FULL_FEED, LI_RELOAD_DOM_EVERY_ROUNDS, LI_RELOAD_DOM_EVERY_PASSES, LI_MAX_ROUNDS_PER_PASS, LI_MOBILE=1"
    );
    process.exit(2);
  }

  if (firstPage && passes > 1) {
    console.log("[collect] --first-page ignores --passes; single snapshot only.");
  }

  if (fresh) {
    await fs.unlink(statePath).catch(() => {});
    console.log("[collect] --fresh: cleared resume state.");
  }

  await fs.mkdir(userDataDir, { recursive: true });
  await migrateLegacyResumeState(statePath);

  if (!firstPage) {
    console.log(
      `[collect] Resume file: ${statePath} (${useResume ? "will load/save" : "disabled (--no-resume)"})`
    );
  }
  if (pruneDom) {
    console.log("[collect] --prune-dom: extracted post cards will be removed from the page.");
  }
  if (blockMedia) {
    console.log("[collect] --block-media: blocking image + media requests.");
  }
  if (mobile) {
    console.log(
      "[collect] --mobile: iPhone 13 viewport + UA (touch flags on). Extractors target desktop cards; if you get 0 posts, try desktop or inspect mobile DOM."
    );
  }
  if (!legacyScroll) {
    const fullDesc =
      fullScrollEvery === 0 ? "never (incremental only)" : String(fullScrollEvery);
    console.log(
      `[collect] Incremental feed scroll (full jump every ${fullDesc} round(s)); --legacy-scroll for old behavior.`
    );
  } else {
    console.log("[collect] --legacy-scroll: wheel + document body bottom each round.");
  }
  if (!firstPage && recycleEveryPasses > 0) {
    console.log(
      `[collect] --recycle-every-passes ${recycleEveryPasses}: new browser session every ${recycleEveryPasses} pass(es).`
    );
  }
  if (expandViewportOnly) {
    console.log(
      "[collect] See more: viewport-first (fast); use --expand-scan-full-feed to scan entire DOM."
    );
  } else {
    console.log("[collect] --expand-scan-full-feed: full-feed button scan (slower on large pages).");
  }
  if (reloadDomEveryRounds > 0) {
    console.log(
      `[collect] --reload-dom-every-rounds ${reloadDomEveryRounds}: reload tab + resume scroll (drops DOM).`
    );
  }
  if (!firstPage && reloadDomEveryPasses > 0) {
    console.log(
      `[collect] --reload-dom-every-passes ${reloadDomEveryPasses}: reload + resume between passes.`
    );
  }
  if (!firstPage && maxRoundsPerPass > 0) {
    console.log(
      `[collect] --max-rounds-per-pass ${maxRoundsPerPass}: caps scroll rounds (LI_ROUNDS=${rounds}).`
    );
  }
  if (extractJsonl && checkpointRounds > 0) {
    console.log(
      `[collect] Checkpoints: JSONL + resume every ${checkpointRounds} scroll round(s) within each pass.`
    );
  } else if (extractJsonl) {
    console.log("[collect] Checkpoints off: JSONL + resume only at end of each pass.");
  }

  const written = [];

  function printCollectionSummary() {
    if (extractJsonl) {
      console.log(`[collect] Post text (JSONL): ${extractJsonl}`);
    }
    if (written.length) {
      console.log("Parse HTML dumps with:");
      for (const f of written) {
        console.log(
          `  python parse_linkedin_html.py "${f}" --jsonl --author "${authorArg}" -o "${path.basename(f, path.extname(f))}_posts.jsonl"`
        );
      }
    }
  }

  async function runKeepOpenTail(page) {
    await waitForEnterBeforeClose();
    let anchorPrefix = "";
    if (extractJsonl) {
      const extractor = selectDomExtractor(url);
      const rows = await page.evaluate(extractor, pruneDom);
      const n = await appendExtractJsonl(extractJsonl, rows, authorArg);
      const lastRow = rows && rows.length ? rows[rows.length - 1] : null;
      anchorPrefix =
        lastRow && typeof lastRow.content === "string" && lastRow.content.trim()
          ? lastRow.content.trim().replace(/\s+/g, " ").slice(0, 120)
          : "";
      if (n) {
        console.log(
          `[collect] --extract-jsonl +${n} after manual session → ${extractJsonl}`
        );
      }
    }
    if (useResume) {
      const snap = await captureScrollState(page, url);
      if (anchorPrefix) snap.anchorPrefix = anchorPrefix;
      await writeResumeState(statePath, snap);
      console.log(
        `[collect] Updated resume after manual session scrollY=${snap.scrollY}`
      );
    }
  }

  async function runAfterCollection(page) {
    printCollectionSummary();
    if (keepOpen) await runKeepOpenTail(page);
  }

  if (firstPage) {
    const context = await chromium.launchPersistentContext(userDataDir, contextOpts);
    let page = context.pages[0];
    if (!page) page = await context.newPage();
    if (blockMedia) await installMediaBlocker(page);
    try {
      await gotoActivityWithResume(page, url, { statePath, useResume, firstPage });

      console.log("[collect] --first-page: expanding visible posts only (no full-feed scroll).");
      await runFirstPageOnly(page, expandOpts);
      if (extractJsonl) {
        const extractor = selectDomExtractor(url);
        const rows = await page.evaluate(extractor, pruneDom);
        const n = await appendExtractJsonl(extractJsonl, rows, authorArg);
        console.log(`[collect] --extract-jsonl +${n} new posts → ${extractJsonl}`);
      }
      if (!noHtml) {
        const html = await page.content();
        await fs.writeFile(outPath, html, "utf8");
        written.push(outPath);
        console.log(`Wrote ${outPath} (${html.length} bytes)`);
      }

      await runAfterCollection(page);
    } finally {
      await context.close();
    }
    return;
  }

  console.log(`[collect] ${passCount} pass(es), ${rounds} scroll rounds each (LI_ROUNDS).`);

  if (recycleEveryPasses <= 0) {
    const context = await chromium.launchPersistentContext(userDataDir, contextOpts);
    let page = context.pages[0];
    if (!page) page = await context.newPage();
    if (blockMedia) await installMediaBlocker(page);
    try {
      await gotoActivityWithResume(page, url, { statePath, useResume, firstPage: false });
      await runFeedCollectionPasses(page, {
        passFrom: 1,
        passTo: passCount,
        passCountTotal: passCount,
        rounds,
        pauseMin,
        pauseMax,
        extractJsonl,
        checkpointRounds,
        pruneDom,
        authorArg,
        useResume,
        statePath,
        url,
        noHtml,
        outPath,
        scrollOpts,
        expandOpts,
        reloadDomEveryRounds,
        reloadDomEveryPasses,
        maxRoundsPerPass,
        written,
      });
      await runAfterCollection(page);
    } finally {
      await context.close();
    }
    return;
  }

  let passDone = 0;
  while (passDone < passCount) {
    const chunk = Math.min(recycleEveryPasses, passCount - passDone);
    const passFrom = passDone + 1;
    const passTo = passDone + chunk;
    const context = await chromium.launchPersistentContext(userDataDir, contextOpts);
    let page = context.pages[0];
    if (!page) page = await context.newPage();
    if (blockMedia) await installMediaBlocker(page);
    try {
      await gotoActivityWithResume(page, url, { statePath, useResume, firstPage: false });
      await runFeedCollectionPasses(page, {
        passFrom,
        passTo,
        passCountTotal: passCount,
        rounds,
        pauseMin,
        pauseMax,
        extractJsonl,
        checkpointRounds,
        pruneDom,
        authorArg,
        useResume,
        statePath,
        url,
        noHtml,
        outPath,
        scrollOpts,
        expandOpts,
        reloadDomEveryRounds,
        reloadDomEveryPasses,
        maxRoundsPerPass,
        written,
      });
    } finally {
      // When we recycle the browser context, we must persist scroll position
      // one last time; otherwise the next session may resume from a stale
      // checkpoint and appear "out of position".
      if (useResume) {
        try {
          const snap = await captureScrollState(page, url);
          await writeResumeState(statePath, snap);
          console.log(
            `[collect] Saved resume before recycle close scrollY=${snap.scrollY}`
          );
        } catch {
          /* ignore resume save failures during shutdown */
        }
      }
      await context.close();
    }
    passDone += chunk;
    if (passDone < passCount) {
      console.log(
        `[collect] Browser session closed after ${chunk} pass(es); reopening to flush memory…`
      );
      await sleep(2000 + Math.random() * 1000);
    }
  }

  printCollectionSummary();
  if (keepOpen) {
    const context = await chromium.launchPersistentContext(userDataDir, contextOpts);
    let page = context.pages[0];
    if (!page) page = await context.newPage();
    if (blockMedia) await installMediaBlocker(page);
    try {
      await gotoActivityWithResume(page, url, { statePath, useResume, firstPage: false });
      await runKeepOpenTail(page);
    } finally {
      await context.close();
    }
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

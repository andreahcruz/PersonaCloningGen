/**
 * X (Twitter) profile collector — process aligned with collect_activity_stealth.cjs habits:
 * - Persistent profile (default): you sign in by hand the first time; no scripted passwords.
 * - Headed Chrome by default when using persistent profile (same idea as LinkedIn script).
 * - "Show more" / "See more": viewport scan + role/label (classes change often).
 * - playwright-extra + puppeteer-extra-plugin-stealth (same stack as collect_activity_stealth.cjs).
 *   Stealth applies to launched browsers; --connectURL attaches to your Chrome (no extra patches).
 *
 * Usage:
 *   node x/scrape_x_profile.cjs --user HANDLE
 *   node x/scrape_x_profile.cjs --user HANDLE --first-page
 *
 * --first-page     Expand + extract visible tweets only; no long scroll to bottom.
 * --keep-open      After run, leave Chrome open until Enter (persistent / ephemeral only).
 *
 * Env (mirror LinkedIn-style LI_* but prefixed X_):
 *   X_USER_DATA          Chrome profile dir (default ./x_user_data)
 *   X_ROUNDS             Max scroll rounds per profile phase (0 = unlimited). Ignored for --search-chunks
 *                        (search uses --max-rounds-per-pass or unlimited); avoids 1-round search exits.
 *   X_PAUSE_MIN, X_PAUSE_MAX   Random pause seconds between rounds (default 3.5 / 8)
 *   X_PASSES             Repeat full cycle: timeline + with_replies (default 1)
 *   X_KEEP_OPEN=1        Same as --keep-open
 *   X_BLOCK_MEDIA=1      Same as --block-media
 *   X_PRUNE_DOM=1        Same as --prune-dom
 *   X_MAX_ARTICLES_BEFORE_PRUNE  With --prune-dom: trim oldest only when article count exceeds this (default 200)
 *   X_LEGACY_SCROLL=1    Same as --legacy-scroll
 *   X_RELOAD_DOM_EVERY_ROUNDS, X_RELOAD_DOM_EVERY_PASSES
 *   X_RECYCLE_BROWSER_EVERY_PASSES
 *   X_MAX_ROUNDS_PER_PASS
 *   X_EXPAND_FULL_FEED=1 Same as --expand-scan-full-feed
 *   X_MOBILE=1           Same as --mobile
 *   X_CHUNK_DAYS         With --search-chunks: days per search window (default 3)
 *   X_NO_NEW_LIMIT       Consecutive loops with no rows written (default 10 search, 2 profile)
 *   X_LOG_EVERY          Log every N scroll loops (default 1 = every loop)
 *   X_SCROLL_SHAKE=0     Disable wheel nudge + settle when a loop has no new tweets
 *   X_SCROLL_PACE        Wheel/scroll step scale (default 0.75); lower = gentler + longer post-scroll waits
 *   X_STRICT_NO_NEW_STREAK=0  With --search-only/--search-chunks: allow scroll position to reset no-new streak
 *   X_EARLY_STOP_STUCK_BOTTOM  Stop after N stuck loops at max scroll (default 2; 0 = off)
 *   X_NO_GPU_ARGS=1       Omit GPU-related Chrome flags (if VM/GPU issues)
 *
 * Alternate modes:
 *   --ephemeral          launch()+storageState (no persistent folder); default headless true
 *   --storageState path  Use with --ephemeral
 *   --connectURL URL     Attach to Chrome you started with --remote-debugging-port
 */

"use strict";

const fs = require("fs");
const path = require("path");
const readline = require("readline");
const { devices } = require("playwright");
const { chromium } = require("playwright-extra");
const StealthPlugin = require("puppeteer-extra-plugin-stealth");

chromium.use(StealthPlugin());

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * Match typical desktop Chrome. X’s Search index is picky: missing Accept-Language looks bot-like.
 * Never block service workers (see serviceWorkers: "allow" on every context): search uses SW for fetch.
 */
const BROWSER_EXTRA_HEADERS = {
  "Accept-Language": "en-US,en;q=0.9"
};

/** Prefer real GPU rendering over software raster (helps Canvas/WebGL look like normal Chrome). */
const CHROME_GPU_ARGS = ["--enable-gpu", "--disable-software-rasterizer", "--ignore-gpu-blocklist"];

/**
 * "Ghost bottom" recovery: scroll up slightly, then down, then wait so X's observer
 * can issue the next search/timeline cursor fetch (manual users pause naturally).
 */
function clampScrollPace(p) {
  const n = Number(p);
  if (!Number.isFinite(n) || n <= 0) return 1;
  return Math.min(1.5, Math.max(0.15, n));
}

/** Lower pace = gentler wheel deltas; waits scale by 1/pace so the loop stays human-slow. */
async function scrollShakeSettle(page, scrollPace = 1) {
  const p = clampScrollPace(scrollPace);
  const inv = 1 / p;
  await page.mouse.wheel(0, Math.round(-280 * p)).catch(() => {});
  await sleep(Math.floor((1200 + Math.random() * 900) * inv));
  await page.mouse.wheel(0, Math.round((320 + Math.floor(Math.random() * 180)) * p)).catch(() => {});
  await sleep(Math.floor((2800 + Math.random() * 4200) * inv));
}

function parseYmd(s) {
  const t = String(s || "").trim();
  if (!/^\d{4}-\d{2}-\d{2}$/.test(t)) return null;
  const [yy, mm, dd] = t.split("-").map((x) => parseInt(x, 10));
  const d = new Date(Date.UTC(yy, mm - 1, dd));
  if (!Number.isFinite(d.getTime())) return null;
  if (d.getUTCFullYear() !== yy || d.getUTCMonth() !== mm - 1 || d.getUTCDate() !== dd) return null;
  return d;
}

function fmtYmd(d) {
  const yy = d.getUTCFullYear();
  const mm = String(d.getUTCMonth() + 1).padStart(2, "0");
  const dd = String(d.getUTCDate()).padStart(2, "0");
  return `${yy}-${mm}-${dd}`;
}

function addDaysUtc(d, days) {
  const out = new Date(d.getTime());
  out.setUTCDate(out.getUTCDate() + days);
  return out;
}

function todayUtcYmd() {
  const now = new Date();
  return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
}

function buildSearchUrl({ query, live = true }) {
  const q = encodeURIComponent(String(query || ""));
  // f=live = "Latest" (reverse-chronological). Without it, X often shows "Top" — few tweets / wrong window.
  // Put f=live early; some builds ignore it if only at the very end.
  const f = live ? "f=live&" : "";
  return `https://x.com/search?${f}q=${q}&src=typed_query`;
}

/**
 * Search URLs request Latest, but X may still land on "Top". Force Latest tab + URL.
 */
async function ensureSearchLatestTab(page) {
  const u = page.url();
  if (!u.includes("/search")) return;

  try {
    if (!/([?&])f=live\b/.test(u)) {
      const join = u.includes("?") ? "&" : "?";
      await page.goto(`${u}${join}f=live`, { waitUntil: "domcontentloaded", timeout: 120_000 });
      await sleep(1200);
    }
  } catch {
    /* ignore */
  }

  const tab = page.getByRole("tab", { name: /^latest$/i }).first();
  const vis = await tab.isVisible({ timeout: 4000 }).catch(() => false);
  if (vis) {
    await tab.click({ timeout: 3000 }).catch(() => {});
    await sleep(1000);
    return;
  }

  await page.evaluate(() => {
    const tabs = Array.from(document.querySelectorAll('[role="tab"], a[href*="f=live"]'));
    for (const el of tabs) {
      const t = (el.innerText || el.getAttribute("aria-label") || "").trim();
      if (/^latest$/i.test(t)) {
        try {
          el.click();
          return;
        } catch {
          /* ignore */
        }
      }
    }
    const links = Array.from(document.querySelectorAll('a[href*="f=live"]'));
    for (const a of links) {
      try {
        a.click();
        return;
      } catch {
        /* ignore */
      }
    }
  });
  await sleep(800);
}

function getArg(name, fallback) {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx === -1) return fallback;
  const v = process.argv[idx + 1];
  if (!v || v.startsWith("--")) return fallback;
  return v;
}

function hasFlag(name) {
  return process.argv.includes(`--${name}`);
}

function wantsHelp() {
  return process.argv.includes("--help") || process.argv.includes("-h");
}

function intArg(name, fallback) {
  const v = getArg(name, undefined);
  if (v == null) return fallback;
  const n = Number.parseInt(String(v), 10);
  return Number.isFinite(n) ? n : fallback;
}

function floatArg(name, fallback) {
  const v = getArg(name, undefined);
  if (v == null) return fallback;
  const n = Number.parseFloat(String(v));
  return Number.isFinite(n) ? n : fallback;
}

function boolArg(name, fallback) {
  const v = getArg(name, fallback ? "true" : "false");
  return String(v).toLowerCase() === "true";
}

function envInt(name, fallback) {
  const s = process.env[name];
  if (s === undefined || s === "") return fallback;
  const n = Number.parseInt(String(s), 10);
  return Number.isFinite(n) ? n : fallback;
}

function envFloat(name, fallback) {
  const s = process.env[name];
  if (s === undefined || s === "") return fallback;
  const n = Number.parseFloat(String(s));
  return Number.isFinite(n) ? n : fallback;
}

function nowIso() {
  return new Date().toISOString();
}

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function appendJsonl(filePath, rows) {
  if (!rows.length) return 0;
  const body = rows.map((r) => JSON.stringify(r)).join("\n") + "\n";
  fs.appendFileSync(filePath, body, "utf8");
  return rows.length;
}

function loadExistingIds(jsonlPath) {
  const ids = new Set();
  if (!fs.existsSync(jsonlPath)) return ids;
  const txt = fs.readFileSync(jsonlPath, "utf8");
  for (const line of txt.split(/\r?\n/)) {
    const t = line.trim();
    if (!t) continue;
    try {
      const o = JSON.parse(t);
      if (o && typeof o.id === "string") ids.add(o.id);
    } catch {
      // ignore
    }
  }
  return ids;
}

async function waitForEnterBeforeClose(prompt) {
  if (!process.stdin.isTTY) {
    console.log("[x] stdin is not a TTY (--keep-open ignored); closing.");
    return;
  }
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  await new Promise((resolve) => {
    rl.question(prompt, () => {
      rl.close();
      resolve();
    });
  });
}

async function gotoAndStabilize(page, url) {
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 120_000 });
  await sleep(1500);
  if (String(url).includes("x.com/search") || page.url().includes("/search")) {
    await ensureSearchLatestTab(page);
  }
}

/** Open URL and wait long enough for tweet rows to hydrate (no scroll-position restore). */
async function gotoXPage(page, targetUrl) {
  await gotoAndStabilize(page, targetUrl);
  await sleep(1800 + Math.random() * 1400);
  await page
    .waitForFunction(
      () => document.querySelectorAll('a[href*="/status/"]').length > 0,
      { timeout: 60_000 }
    )
    .catch(() => {});
  await sleep(900 + Math.random() * 800);
}

/**
 * LinkedIn-style: one "Show more" click via viewport scan in main/body.
 */
async function clickOneShowMore(page, { viewportOnly = true } = {}) {
  if (viewportOnly) {
    const handle = await page.evaluateHandle(() => {
      const root = document.querySelector("main") || document.body;
      const candidates = root.querySelectorAll(
        'button, [role="button"], a'
      );
      const vh = window.innerHeight;
      const vw = window.innerWidth;
      for (const el of candidates) {
        const label = (
          el.getAttribute("aria-label") ||
          el.innerText ||
          ""
        ).trim();
        if (!/show more|see more/i.test(label)) continue;
        const r = el.getBoundingClientRect();
        if (r.width < 2 || r.height < 2) continue;
        if (r.bottom < -40 || r.top > vh + 40 || r.right < 0 || r.left > vw + 20) continue;
        const cs = window.getComputedStyle(el);
        if (cs.visibility === "hidden" || cs.display === "none" || Number(cs.opacity) === 0)
          continue;
        const art = el.closest("article");
        if (!art) continue;
        return el;
      }
      return null;
    });
    try {
      const el = await handle.asElement();
      if (el) {
        try {
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

  const root = page.locator("main").first();
  const n = await root.count().catch(() => 0);
  const scope = n > 0 ? root : page;
  const groups = [
    scope.getByRole("button", { name: /show more/i }),
    scope.getByRole("button", { name: /see more/i }),
    scope.locator("article a").filter({ hasText: /^Show more$/i })
  ];
  for (const group of groups) {
    const c = await group.count().catch(() => 0);
    for (let j = 0; j < Math.min(c, 24); j++) {
      const loc = group.nth(j);
      const vis = await loc.isVisible({ timeout: 400 }).catch(() => false);
      if (!vis) continue;
      try {
        await loc.scrollIntoViewIfNeeded({ timeout: 2000 });
      } catch {
        continue;
      }
      await sleep(100 + Math.random() * 140);
      const ok = await loc.click({ timeout: 3500 }).then(() => true, () => false);
      if (ok) return true;
    }
  }
  return false;
}

async function expandAllShowMoreInView(
  page,
  {
    maxNoProgress = 3,
    maxClicks = 24,
    viewportOnly = true
  } = {}
) {
  let noProgress = 0;
  let clicks = 0;
  while (noProgress < maxNoProgress && clicks < maxClicks) {
    const did = await clickOneShowMore(page, { viewportOnly });
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

async function extractVisibleTweets(page) {
  return page.evaluate(() => {
    const out = [];
    const articles = Array.from(document.querySelectorAll("article"));
    for (const art of articles) {
      const anchors = Array.from(art.querySelectorAll('a[href*="/status/"]'));
      const a =
        anchors.find((x) => /\/status\/\d+/.test(x.getAttribute("href") ?? "")) ?? anchors[0];
      const href = a?.getAttribute("href") ?? "";
      const m = href.match(/\/status\/(\d+)/);
      const id = m?.[1];
      if (!id) continue;
      const url = href.startsWith("http")
        ? href
        : `https://x.com${href.startsWith("/") ? href : `/${href}`}`;
      const created_at = art.querySelector("time")?.getAttribute("datetime") ?? undefined;
      const textEl = art.querySelector('[data-testid="tweetText"]');
      const text = (textEl && textEl.innerText && textEl.innerText.trim()) || undefined;
      const artText = (art && art.innerText) || "";
      const is_reply = /\bReplying to\b/i.test(artText);
      const is_repost_or_quote = /\bReposted\b/i.test(artText) || /\bRepost\b/i.test(artText);
      out.push({ id, url, created_at, text, is_reply, is_repost_or_quote });
    }
    const seen = new Set();
    return out.filter((t) => {
      if (seen.has(t.id)) return false;
      seen.add(t.id);
      return true;
    });
  });
}

async function extractVisibleTweetsWithRetry(page) {
  let batch = await extractVisibleTweets(page);
  if (batch.length === 0) {
    await sleep(2000 + Math.random() * 1500);
    batch = await extractVisibleTweets(page);
  }
  return batch;
}

async function pruneOldArticles(page, keepLast) {
  return page.evaluate((keep) => {
    const arts = Array.from(document.querySelectorAll("article"));
    if (arts.length <= keep) return 0;
    const removeCount = arts.length - keep;
    for (let i = 0; i < removeCount; i++) {
      try {
        arts[i].remove();
      } catch {
        // ignore
      }
    }
    return removeCount;
  }, keepLast);
}

async function nudgeScroll(page, fractionOfViewport, scrollPace = 1) {
  const p = clampScrollPace(scrollPace);
  const inv = 1 / p;
  const vp = page.viewportSize() || { height: 800 };
  const delta = Math.floor(vp.height * fractionOfViewport * p);
  await page.mouse.wheel(0, delta);
  await sleep(Math.floor((900 + Math.random() * 600) * inv));
}

async function countArticlesInDom(page) {
  return page.evaluate(() => document.querySelectorAll("article").length);
}

/**
 * Scroll the feed down using the same column X uses (primaryColumn / main), then wheel.
 * Returns pixel movement on the feed scroller + window (window.scrollY is often 0 on /search).
 */
async function scrollFeedDown(page, { legacyScroll = false, scrollPace = 1 } = {}) {
  const p = clampScrollPace(scrollPace);
  const vp = page.viewportSize() || { height: 800, width: 1100 };
  const delta = Math.floor(vp.height * (0.48 + Math.random() * 0.32) * p);

  if (legacyScroll) {
    await page.mouse.wheel(0, Math.floor(vp.height * 0.55 * p));
    await sleep(350 + Math.random() * 400);
    await page.mouse.wheel(0, Math.floor(vp.height * 0.38 * p));
    const m = await readFeedScrollMetrics(page);
    return {
      moved: true,
      scDelta: 0,
      winDelta: 0,
      nearBottom: m.scTop >= m.scMax - 40,
      metricsAfter: m
    };
  }

  const before = await readFeedScrollMetrics(page);

  await page
    .evaluate(
      (d) => {
        function isScrollable(el) {
          if (!el) return false;
          const cs = getComputedStyle(el);
          const oy = cs.overflowY;
          return (
            el.scrollHeight > el.clientHeight + 40 &&
            (oy === "auto" || oy === "scroll" || oy === "overlay")
          );
        }
        const order = [
          document.querySelector('[data-testid="primaryColumn"]'),
          document.querySelector('main[role="main"]'),
          document.querySelector("main")
        ];
        let el = null;
        for (const c of order) {
          if (c && isScrollable(c)) {
            el = c;
            break;
          }
        }
        const root = document.scrollingElement || document.documentElement;
        if (!el) {
          root.scrollTop = Math.min(
            Math.max(0, root.scrollHeight - window.innerHeight),
            root.scrollTop + d
          );
          return;
        }
        const max = Math.max(0, el.scrollHeight - el.clientHeight);
        el.scrollTop = Math.min(max, el.scrollTop + d);
      },
      delta
    )
    .catch(() => {});

  await page.mouse.wheel(0, delta).catch(() => {});
  await sleep(220 + Math.floor(Math.random() * 200));

  let after = await readFeedScrollMetrics(page);
  let scDelta = Math.abs(after.scTop - before.scTop);
  let winDelta = Math.abs(after.winY - before.winY);
  let moved = scDelta > 6 || winDelta > 6;

  if (!moved && before.scTop < before.scMax - 30) {
    for (let i = 0; i < 3; i++) {
      await page.mouse.wheel(0, Math.floor(vp.height * 0.42 * p)).catch(() => {});
      await sleep(200 + Math.random() * 150);
    }
    after = await readFeedScrollMetrics(page);
    scDelta = Math.abs(after.scTop - before.scTop);
    winDelta = Math.abs(after.winY - before.winY);
    moved = scDelta > 6 || winDelta > 6;
  }

  const nearBottom = after.scTop >= after.scMax - 48;
  return { moved, scDelta, winDelta, nearBottom, metricsAfter: after };
}

async function readFeedScrollMetrics(page) {
  return page.evaluate(() => {
    function isScrollable(el) {
      if (!el) return false;
      const cs = getComputedStyle(el);
      const oy = cs.overflowY;
      return (
        el.scrollHeight > el.clientHeight + 40 &&
        (oy === "auto" || oy === "scroll" || oy === "overlay")
      );
    }
    const winY = window.scrollY;
    const order = [
      document.querySelector('[data-testid="primaryColumn"]'),
      document.querySelector('main[role="main"]'),
      document.querySelector("main")
    ];
    for (const c of order) {
      if (c && isScrollable(c)) {
        const scTop = c.scrollTop;
        const scMax = Math.max(0, c.scrollHeight - c.clientHeight);
        return { winY, scTop, scMax };
      }
    }
    const root = document.scrollingElement || document.documentElement;
    const scTop = root.scrollTop;
    const scMax = Math.max(0, root.scrollHeight - window.innerHeight);
    return { winY, scTop, scMax };
  });
}

async function reloadDomGentle(page, targetUrl) {
  console.log("[x] Reloading tab to trim DOM…");
  await gotoXPage(page, targetUrl);
}

async function scrollToTopGentle(page, scrollPace = 1) {
  const p = clampScrollPace(scrollPace);
  const inv = 1 / p;
  for (let i = 0; i < 10; i++) {
    await page.mouse.wheel(0, -Math.floor((90 + Math.random() * 70) * p)).catch(() => {});
    await sleep(Math.floor((280 + Math.random() * 220) * inv));
  }
}

async function runFirstPageOnly(page, expandOpts) {
  const viewportOnly = expandOpts.viewportOnly !== false;
  const scrollPace = expandOpts.scrollPace != null ? expandOpts.scrollPace : 1;
  await sleep(2500);
  await expandAllShowMoreInView(page, {
    maxNoProgress: 6,
    maxClicks: 45,
    viewportOnly
  });
  await nudgeScroll(page, 0.45, scrollPace);
  await expandAllShowMoreInView(page, { maxNoProgress: 4, maxClicks: 30, viewportOnly });
  await nudgeScroll(page, 0.5, scrollPace);
  await expandAllShowMoreInView(page, { maxNoProgress: 3, maxClicks: 24, viewportOnly });
  await scrollToTopGentle(page, scrollPace);
  await sleep(900 + Math.random() * 500);
}

async function scrollAllTime(opts) {
  const {
    page,
    username,
    scrapedFrom,
    seenIds,
    maxTweets,
    noNewLimit,
    logEvery,
    baseDelayMs,
    maxBackoffS,
    acceptRow,
    writeRow,
    maxArticlesBeforePrune,
    keepLastArticles,
    errorCheckEvery,
    maxRounds,
    pruneDomAfterExtract,
    pauseMin,
    pauseMax,
    legacyScroll,
    reloadDomEveryRounds,
    expandViewportOnly,
    targetUrl,
    scrollShakeOnNoNew,
    scrollPace,
    earlyStopStuckBottom,
    stopOnBottom
  } = opts;

  const pace = clampScrollPace(scrollPace != null ? scrollPace : 1);
  const paceInv = 1 / pace;

  let totalWritten = 0;
  let noNewStreak = 0;
  let loops = 0;
  let backoffS = 0;
  let stuckBottomNoMove = 0;

  const net = { lastBadStatus: null, badCount: 0 };
  page.on("response", (resp) => {
    try {
      const status = resp.status();
      const url = resp.url();
      if ((status === 429 || status === 503) && (url.includes("graphql") || url.includes("/i/api/"))) {
        net.lastBadStatus = status;
        net.badCount += 1;
      }
    } catch {
      // ignore
    }
  });

  while (
    totalWritten < maxTweets &&
    noNewStreak < noNewLimit &&
    (maxRounds <= 0 || loops < maxRounds)
  ) {
    loops += 1;

    if (reloadDomEveryRounds > 0 && loops > 1 && (loops - 1) % reloadDomEveryRounds === 0 && targetUrl) {
      await reloadDomGentle(page, targetUrl);
    }

    let expanded = 0;
    try {
      expanded = await expandAllShowMoreInView(page, {
        maxNoProgress: 3,
        maxClicks: 24,
        viewportOnly: expandViewportOnly
      });
    } catch {
      expanded = 0;
    }

    await sleep(
      Math.floor((pauseMin + Math.random() * Math.max(0.01, pauseMax - pauseMin)) * 1000)
    );
    await sleep(600 + Math.floor(Math.random() * 900));

    const batch = await extractVisibleTweetsWithRetry(page);
    const batchCount = Array.isArray(batch) ? batch.length : 0;
    let newSeen = 0;
    let newWritten = 0;

    for (const t of batch) {
      const tid = t?.id;
      if (typeof tid !== "string") continue;
      if (seenIds.has(tid)) continue;
      seenIds.add(tid);
      newSeen += 1;

      const row = {
        id: tid,
        url: t.url,
        username,
        created_at: t.created_at,
        text: t.text,
        is_reply: !!t.is_reply,
        is_repost_or_quote: !!t.is_repost_or_quote,
        scraped_from: scrapedFrom,
        scraped_at: nowIso()
      };
      if (acceptRow(row)) {
        writeRow(row);
        newWritten += 1;
        totalWritten += 1;
      }
    }

    if (net.badCount) {
      backoffS = Math.min(maxBackoffS, Math.max(2, backoffS ? backoffS * 2 : 2));
      const jitter = 0.2 + Math.random() * 0.8;
      const planned = Math.min(maxBackoffS, backoffS + jitter);
      if (logEvery) {
        console.log(
          `[${scrapedFrom}] rate-limit status=${net.lastBadStatus} badResponses=${net.badCount} plannedBackoff=${planned.toFixed(1)}s`
        );
      }
      const t0 = Date.now();
      await page.waitForTimeout(Math.floor(planned * 1000));
      const actual = (Date.now() - t0) / 1000;
      net.badCount = 0;
      if (logEvery) console.log(`[${scrapedFrom}] backoffComplete actualBackoff=${actual.toFixed(1)}s`);
      continue;
    }

    const articleCount = await countArticlesInDom(page).catch(() => 0);
    let pruned = 0;
    if (pruneDomAfterExtract && articleCount > maxArticlesBeforePrune) {
      const keepTail = Math.max(50, keepLastArticles);
      pruned = await pruneOldArticles(page, keepTail).catch(() => 0);
    }

    const scrollOut = await scrollFeedDown(page, { legacyScroll, scrollPace: pace });

    // Optional: stop the moment we reach the feed bottom.
    if (stopOnBottom && scrollOut.nearBottom) {
      console.log(`[${scrapedFrom}] stop-on-bottom: nearBottom=true`);
      break;
    }

    await sleep(Math.floor((800 + Math.random() * 800) * paceInv));
    await page.waitForTimeout(Math.floor(baseDelayMs * paceInv));
    await page.waitForTimeout(Math.floor((500 + Math.floor(baseDelayMs * 0.5)) * paceInv));

    if (newSeen === 0 && scrollShakeOnNoNew) {
      await scrollShakeSettle(page, pace);
    }

    const feedMoved = scrollOut.moved;
    // Streak = loops with nothing appended to JSONL. Using newSeen reset was wrong: new IDs can
    // fail acceptRow (e.g. replies in originals phase) and never write, which pinned streak at 0 forever.
    if (newWritten > 0) noNewStreak = 0;
    else noNewStreak += 1;

    if (earlyStopStuckBottom > 0 && newSeen === 0 && scrollOut.nearBottom && !feedMoved) {
      stuckBottomNoMove += 1;
      if (stuckBottomNoMove >= earlyStopStuckBottom) {
        console.log(
          `[${scrapedFrom}] early stop: ${stuckBottomNoMove} loop(s) at max scroll with no movement and no new ids (end of chunk or stalled search)`
        );
        break;
      }
    } else {
      stuckBottomNoMove = 0;
    }

    if (logEvery && loops % logEvery === 0) {
      const m = scrollOut.metricsAfter || (await readFeedScrollMetrics(page).catch(() => null));
      const scTop = m?.scTop ?? "?";
      const scMax = m?.scMax ?? "?";
      console.log(
        `[${scrapedFrom}] loop=${loops} sessWrites=${totalWritten} idsInSet=${seenIds.size} newIds=${newSeen} savedLoop=${newWritten} vis=${batchCount} arts=${articleCount} pruned=${pruned} expand=${expanded} scΔ=${scrollOut.scDelta.toFixed(0)} winΔ=${scrollOut.winDelta.toFixed(0)} moved=${feedMoved} bottom≈${scrollOut.nearBottom} sc=${scTop}/${scMax} streak=${noNewStreak}`
      );
    }

    if (errorCheckEvery && loops % errorCheckEvery === 0) {
      const bodyText = await page.locator("body").innerText({ timeout: 2000 }).catch(() => "");
      if (/Something went wrong/i.test(bodyText) || /Try again/i.test(bodyText)) {
        await sleep(8000);
      }
    }

  }
}

function shouldUseGpuLaunchArgs() {
  return !(
    process.argv.includes("--no-gpu-args") ||
    process.argv.includes("--noGpuArgs") ||
    String(process.env.X_NO_GPU_ARGS || "").trim() === "1"
  );
}

function buildPersistentContextOpts(mobile) {
  const common = {
    extraHTTPHeaders: { ...BROWSER_EXTRA_HEADERS },
    // Required for X search: blocking SW breaks search timelines after a few pages.
    serviceWorkers: "allow"
  };
  const base = {
    headless: false,
    channel: "chrome",
    viewport: { width: 1280, height: 900 },
    locale: "en-US",
    ...common,
    ...(shouldUseGpuLaunchArgs() ? { args: [...CHROME_GPU_ARGS] } : {})
  };
  if (!mobile) return base;
  const phone = devices["iPhone 13"];
  return {
    ...base,
    locale: "en-US",
    userAgent: phone.userAgent,
    viewport: phone.viewport,
    deviceScaleFactor: phone.deviceScaleFactor,
    isMobile: phone.isMobile,
    hasTouch: phone.hasTouch
  };
}

async function installMediaBlockerOnContext(context) {
  await context.route("**/*", (route) => {
    const t = route.request().resourceType();
    // Blocking fonts can break X layout/hydration (blank timelines / stuck infinite scroll).
    // Keep --block-media to images+media; add an explicit --block-fonts if needed later.
    if (t === "image" || t === "media") return route.abort();
    return route.continue();
  });
}

/** Open a search URL with the same browser profile / headers as a normal run; no scrolling or scraping. */
async function runOpenSearchOnlyFlow(opts) {
  const {
    targetUrl,
    userDataDir,
    connectURL,
    ephemeral,
    headlessEphemeral,
    slowmo,
    channel,
    storageStatePath,
    blockMedia,
    mobile,
    keepOpen
  } = opts;

  console.log("[x] --open-search-only (no scroll / no scrape)\n", targetUrl);

  if (connectURL) {
    console.warn(
      "[x] CDP (--connectURL): remote debugging can reduce search trust vs launchPersistentContext; avoid CDP for heavy search scraping if you see shallow results."
    );
    const browser = await chromium.connectOverCDP(connectURL);
    const context = browser.contexts()[0];
    if (!context) throw new Error("connectOverCDP: no context");
    await context.setExtraHTTPHeaders({ ...BROWSER_EXTRA_HEADERS });
    const page = context.pages()[0] || (await context.newPage());
    try {
      await gotoXPage(page, targetUrl);
      if (keepOpen) {
        await waitForEnterBeforeClose("[x] Press Enter to disconnect CDP…\n");
      }
    } finally {
      await browser.close();
    }
    return;
  }

  if (ephemeral) {
    const launchOpts = { headless: headlessEphemeral, slowMo: slowmo, channel };
    if (shouldUseGpuLaunchArgs()) launchOpts.args = [...CHROME_GPU_ARGS];
    const browser = await chromium.launch(launchOpts);
    const ctxOpts = {
      locale: "en-US",
      extraHTTPHeaders: { ...BROWSER_EXTRA_HEADERS },
      serviceWorkers: "allow"
    };
    if (fs.existsSync(storageStatePath)) ctxOpts.storageState = storageStatePath;
    const context = await browser.newContext(ctxOpts);
    if (blockMedia) await installMediaBlockerOnContext(context);
    const page = await context.newPage();
    try {
      await gotoXPage(page, targetUrl);
      if (keepOpen) {
        await waitForEnterBeforeClose("[x] Press Enter to close the browser…\n");
      }
    } finally {
      await page.close().catch(() => {});
      await context.close();
      await browser.close();
    }
    return;
  }

  const contextOpts = buildPersistentContextOpts(mobile);
  const context = await chromium.launchPersistentContext(userDataDir, contextOpts);
  let page = context.pages()[0];
  if (!page) page = await context.newPage();
  if (blockMedia) await installMediaBlockerOnContext(context);
  try {
    await gotoXPage(page, targetUrl);
    if (keepOpen) {
      await waitForEnterBeforeClose(
        "[x] Press Enter to close Chrome (same as scrape:x with --keep-open)…\n"
      );
    }
  } finally {
    await context.close();
  }
}

async function main() {
  if (wantsHelp()) {
    console.log(`
Usage:
  node x/scrape_x_profile.cjs --user HANDLE [options]

LinkedIn-aligned flags:
  --user-data-dir DIR     (env X_USER_DATA, default ./x_user_data)
  --first-page
  --keep-open             (env X_KEEP_OPEN=1)
  --passes N              (env X_PASSES)
  --prune-dom             (env X_PRUNE_DOM=1) trim oldest articles only when DOM is long
  --max-articles-before-prune N  With --prune-dom: trim when article count exceeds N (default 200; env X_MAX_ARTICLES_BEFORE_PRUNE)
  --keepLastArticles N   Keep this many newest articles after a trim (default 100)
  --block-media           (env X_BLOCK_MEDIA=1)
  --legacy-scroll         (env X_LEGACY_SCROLL=1)
  --reload-dom-every-rounds N
  --reload-dom-every-passes N
  --max-rounds-per-pass N (env X_MAX_ROUNDS_PER_PASS; 0 = use X_ROUNDS; search-chunks: 0 = unlimited)
  --recycle-every-passes N
  --expand-scan-full-feed (env X_EXPAND_FULL_FEED=1)
  --mobile                (env X_MOBILE=1)
  --noNewLimit N          Stop after N loops with no JSONL writes (default 10 search / 2 profile; env X_NO_NEW_LIMIT)
  --logEvery N            Print progress every N loops (default 1 = every loop; env X_LOG_EVERY)
  --strict-no-new-streak   Streak counts only newSeen=0 (ignores scroll-based streak reset)
  --no-strict-no-new-streak  Re-enable scroll-based streak reset (default on: --search-only / --search-chunks)
  --scroll-pace N         Wheel delta scale vs baseline (default 0.75; env X_SCROLL_PACE; 1 = faster)
  --no-scroll-shake       Disable wheel up/down + settle when a loop finds no new tweets (env X_SCROLL_SHAKE=0)
  --early-stop-stuck-bottom N  Stop after N loops stuck at bottom (default 2; 0=off; env X_EARLY_STOP_STUCK_BOTTOM)
  --stop-on-bottom       Stop immediately when the feed reports nearBottom=true
  --originals-only       Only scrape originals (skip replies / with_replies)

Search chunking (older history workaround):
  --search-chunks          Scrape via search timelines instead of profile timeline
  --since YYYY-MM-DD       Oldest date (inclusive)
  --until YYYY-MM-DD       Newest date (exclusive-ish; used in query). Default: tomorrow (UTC)
  --chunk-days N           Days per window (default 3 with --search-chunks; env X_CHUNK_DAYS)
  --max-search-chunks N    Stop after N windows (0 = unlimited). For testing.
  --search-live true|false Use Latest (f=live). Default true
  --search-only            Only scrape search chunks (skip profile timeline)
  --open-search-only       Open search URL only: no scroll, no scrape (use with --search-query or --search-chunks+dates)
  --search-query \"...\"    Raw X search query for --open-search-only (env X_SEARCH_QUERY)

Env: X_ROUNDS (0=unlimited per phase), X_PAUSE_MIN (3.5), X_PAUSE_MAX (8), X_NO_NEW_LIMIT (2), X_LOG_EVERY (1)

Other:
  --no-gpu-args           Do not add --enable-gpu / --ignore-gpu-blocklist (env X_NO_GPU_ARGS=1)
  --ephemeral             launch()+storageState instead of persistent profile
  --storageState PATH
  --connectURL http://127.0.0.1:9222
  --headless true|false   (ephemeral default true; persistent stays headed false)
  --outDir data
`);
    return;
  }

  const username = String(getArg("user", "jasonlk")).replace(/^@/, "");
  const outDir = path.resolve(getArg("outDir", "data"));
  ensureDir(outDir);

  const originalsPath = path.join(outDir, `${username}_originals.jsonl`);
  const repliesPath = path.join(outDir, `${username}_replies.jsonl`);

  const connectURL = (getArg("connectURL", "") || "").trim();
  const ephemeral = hasFlag("ephemeral");
  const storageStatePath = path.resolve(getArg("storageState", "storageState.json"));

  const userDataDir = path.resolve(
    getArg("user-data-dir", "") || process.env.X_USER_DATA || "./x_user_data"
  );

  const firstPage = hasFlag("first-page") || hasFlag("firstPage");
  const openSearchOnly = hasFlag("open-search-only") || hasFlag("openSearchOnly");
  const keepOpen =
    hasFlag("keep-open") || hasFlag("keepOpen") || String(process.env.X_KEEP_OPEN || "") === "1";

  const passes = Math.max(
    1,
    intArg("passes", envInt("X_PASSES", 1))
  );
  const rounds = intArg("rounds", envInt("X_ROUNDS", 0));
  const pauseMin = envFloat("X_PAUSE_MIN", 3.5);
  const pauseMax = envFloat("X_PAUSE_MAX", 8);

  const maxTweets = intArg("maxTweets", 2_000_000_000);
  const logEvery = intArg("logEvery", envInt("X_LOG_EVERY", 1));
  const baseDelayMs = intArg("baseDelayMs", 600);
  const maxBackoffS = intArg("maxBackoffS", 60);
  const channelRaw = getArg("channel", "");
  const channel = channelRaw && channelRaw.trim() ? channelRaw.trim() : undefined;
  const slowmo = intArg("slowmo", 0);

  const maxArticlesBeforePrune = intArg(
    "max-articles-before-prune",
    intArg("maxArticlesBeforePrune", envInt("X_MAX_ARTICLES_BEFORE_PRUNE", 200))
  );
  const keepLastArticles = intArg("keepLastArticles", 100);
  const errorCheckEvery = intArg("errorCheckEvery", 25);

  const pruneDom = hasFlag("prune-dom") || hasFlag("pruneDom") || String(process.env.X_PRUNE_DOM || "") === "1";
  const blockMedia =
    hasFlag("block-media") || hasFlag("blockMedia") || String(process.env.X_BLOCK_MEDIA || "") === "1";

  const legacyScroll =
    hasFlag("legacy-scroll") || hasFlag("legacyScroll") || String(process.env.X_LEGACY_SCROLL || "") === "1";

  let reloadDomEveryRounds = intArg("reload-dom-every-rounds", 0);
  if (process.env.X_RELOAD_DOM_EVERY_ROUNDS) {
    reloadDomEveryRounds = envInt("X_RELOAD_DOM_EVERY_ROUNDS", reloadDomEveryRounds);
  }

  let reloadDomEveryPasses = intArg("reload-dom-every-passes", 0);
  if (process.env.X_RELOAD_DOM_EVERY_PASSES) {
    reloadDomEveryPasses = envInt("X_RELOAD_DOM_EVERY_PASSES", reloadDomEveryPasses);
  }

  let maxRoundsPerPass = intArg("max-rounds-per-pass", 0);
  if (process.env.X_MAX_ROUNDS_PER_PASS) {
    maxRoundsPerPass = envInt("X_MAX_ROUNDS_PER_PASS", maxRoundsPerPass);
  }

  let recycleEveryPasses = intArg("recycle-every-passes", 0);
  if (process.env.X_RECYCLE_BROWSER_EVERY_PASSES) {
    recycleEveryPasses = envInt("X_RECYCLE_BROWSER_EVERY_PASSES", recycleEveryPasses);
  }

  const expandFullFeed =
    hasFlag("expand-scan-full-feed") ||
    hasFlag("expandScanFullFeed") ||
    String(process.env.X_EXPAND_FULL_FEED || "") === "1";
  const expandViewportOnly = !expandFullFeed;

  const mobile = hasFlag("mobile") || String(process.env.X_MOBILE || "") === "1";

  const headlessEphemeral = boolArg("headless", true);

  const searchChunks =
    hasFlag("search-chunks") ||
    hasFlag("searchChunks") ||
    String(process.env.X_SEARCH_CHUNKS || "") === "1";
  const searchOnly =
    hasFlag("search-only") ||
    hasFlag("searchOnly") ||
    String(process.env.X_SEARCH_ONLY || "") === "1";
  const chunkDaysDefault = searchChunks
    ? envInt("X_CHUNK_DAYS", 3)
    : envInt("X_CHUNK_DAYS", 180);
  const chunkDays = Math.max(
    1,
    intArg("chunk-days", intArg("chunkDays", chunkDaysDefault))
  );
  const maxSearchChunks = Math.max(0, intArg("max-search-chunks", intArg("maxSearchChunks", 0)));
  const strictNoNewOptOut =
    hasFlag("no-strict-no-new-streak") ||
    hasFlag("noStrictNoNewStreak") ||
    String(process.env.X_STRICT_NO_NEW_STREAK || "").trim() === "0";
  const strictNoNewOptIn =
    hasFlag("strict-no-new-streak") ||
    hasFlag("strictNoNewStreak") ||
    String(process.env.X_STRICT_NO_NEW_STREAK || "").trim() === "1";
  const strictNoNewStreak =
    !strictNoNewOptOut &&
    (strictNoNewOptIn || searchOnly || searchChunks);
  const noNewLimit = intArg(
    "noNewLimit",
    envInt("X_NO_NEW_LIMIT", searchChunks || searchOnly ? 10 : 2)
  );
  const scrollShakeOnNoNew =
    !hasFlag("no-scroll-shake") &&
    !hasFlag("noScrollShake") &&
    String(process.env.X_SCROLL_SHAKE || "") !== "0";
  let scrollPace = floatArg("scroll-pace", floatArg("scrollPace", envFloat("X_SCROLL_PACE", 0.75)));
  if (!Number.isFinite(scrollPace) || scrollPace <= 0) scrollPace = 0.75;
  scrollPace = clampScrollPace(scrollPace);
  const earlyStopStuckBottom = intArg(
    "early-stop-stuck-bottom",
    intArg("earlyStopStuckBottom", envInt("X_EARLY_STOP_STUCK_BOTTOM", 2))
  );
  // Needs to be a flag (no value) like other boolean switches; boolArg requires `true|false`.
  const stopOnBottom = hasFlag("stop-on-bottom") || hasFlag("stopOnBottom");
  const originalsOnly =
    hasFlag("originals-only") ||
    hasFlag("originalsOnly") ||
    String(process.env.X_ORIGINALS_ONLY || "") === "1" ||
    // Default: when doing only search chunks, don't also scrape replies yet.
    (searchOnly && searchChunks);
  const searchLive = boolArg("search-live", boolArg("searchLive", true));
  const sinceArg = getArg("since", "") || process.env.X_SINCE || "";
  const untilArg = getArg("until", "") || process.env.X_UNTIL || "";
  const sinceDate = sinceArg ? parseYmd(sinceArg) : null;
  const untilDate = untilArg ? parseYmd(untilArg) : null;

  if (searchChunks && !sinceDate && !openSearchOnly) {
    console.error('[x] --search-chunks requires --since YYYY-MM-DD (or env X_SINCE).');
    process.exitCode = 2;
    return;
  }

  if (openSearchOnly) {
    const rawQ = (getArg("search-query", "") || getArg("searchQuery", "") || process.env.X_SEARCH_QUERY || "").trim();
    let targetUrl = null;
    if (rawQ) {
      targetUrl = buildSearchUrl({ query: rawQ, live: searchLive });
    } else if (searchChunks && sinceDate) {
      const end = untilDate || addDaysUtc(todayUtcYmd(), 1);
      if (end > sinceDate) {
        const windowEnd = end;
        const windowStartRaw = addDaysUtc(windowEnd, -chunkDays);
        const windowStart = windowStartRaw < sinceDate ? sinceDate : windowStartRaw;
        const q = `from:${username} since:${fmtYmd(windowStart)} until:${fmtYmd(windowEnd)} -filter:replies -filter:retweets`;
        targetUrl = buildSearchUrl({ query: q, live: searchLive });
      }
    }
    if (!targetUrl) {
      console.error(
        "[x] --open-search-only requires --search-query \"...\" OR (--search-chunks --since YYYY-MM-DD [--until YYYY-MM-DD])"
      );
      process.exitCode = 2;
      return;
    }
    await runOpenSearchOnlyFlow({
      targetUrl,
      userDataDir,
      connectURL,
      ephemeral,
      headlessEphemeral,
      slowmo,
      channel,
      storageStatePath,
      blockMedia,
      mobile,
      keepOpen
    });
    return;
  }

  const originalSeen = loadExistingIds(originalsPath);
  const replySeen = loadExistingIds(repliesPath);

  const timelineUrl = `https://x.com/${username}`;
  const repliesUrl = `https://x.com/${username}/with_replies`;

  async function runPhase(page, { scrapedFrom, url, seenIds, acceptRow, outPath }) {
    await gotoXPage(page, url);

    if (firstPage) {
      await runFirstPageOnly(page, { viewportOnly: expandViewportOnly, scrollPace });
      const batch = await extractVisibleTweetsWithRetry(page);
      let wrote = 0;
      for (const t of batch) {
        const tid = t?.id;
        if (typeof tid !== "string" || seenIds.has(tid)) continue;
        seenIds.add(tid);
        const row = {
          id: tid,
          url: t.url,
          username,
          created_at: t.created_at,
          text: t.text,
          is_reply: !!t.is_reply,
          is_repost_or_quote: !!t.is_repost_or_quote,
          scraped_from: scrapedFrom,
          scraped_at: nowIso()
        };
        if (acceptRow(row)) {
          appendJsonl(outPath, [row]);
          wrote++;
        }
      }
      console.log(`[${scrapedFrom}] --first-page wrote=${wrote}`);
      return;
    }

    // Global X_ROUNDS / --rounds caps profile scrolling. Search phases ignore X_ROUNDS (use unlimited
    // inner loops unless --max-rounds-per-pass N). Capped runs: pass --max-rounds-per-pass 1.
    let cap = maxRoundsPerPass > 0 ? maxRoundsPerPass : rounds > 0 ? rounds : 0;
    if (String(scrapedFrom).startsWith("search_")) {
      cap = maxRoundsPerPass > 0 ? maxRoundsPerPass : 0;
    }

    await scrollAllTime({
      page,
      username,
      scrapedFrom,
      seenIds,
      maxTweets,
      noNewLimit,
      logEvery,
      baseDelayMs,
      maxBackoffS,
      acceptRow,
      writeRow: (row) => appendJsonl(outPath, [row]),
      maxArticlesBeforePrune,
      keepLastArticles,
      errorCheckEvery,
      maxRounds: cap,
      pruneDomAfterExtract: pruneDom,
      pauseMin,
      pauseMax,
      legacyScroll,
      reloadDomEveryRounds,
      expandViewportOnly,
      targetUrl: url,
      scrollShakeOnNoNew,
      scrollPace,
      earlyStopStuckBottom,
      stopOnBottom
    });
  }

  async function runSearchChunks(context) {
    const start = sinceDate;
    const end = untilDate || addDaysUtc(todayUtcYmd(), 1);
    if (!start) return;
    if (end <= start) {
      console.error("[x] --until must be after --since.");
      process.exitCode = 2;
      return;
    }

    const spanDays = Math.max(0, (end.getTime() - start.getTime()) / 86400000);
    const estChunks = Math.max(1, Math.ceil(spanDays / chunkDays));
    console.log(
      `[x] Search chunking: since=${fmtYmd(start)} until=${fmtYmd(end)} chunkDays=${chunkDays} live=${searchLive} (~${estChunks} windows${maxSearchChunks > 0 ? `, cap=${maxSearchChunks}` : ""})`
    );

    let windowEnd = end;
    let chunkIndex = 0;
    while (windowEnd > start) {
      chunkIndex += 1;
      if (maxSearchChunks > 0 && chunkIndex > maxSearchChunks) {
        console.log(`[x] --max-search-chunks ${maxSearchChunks}: stopped after ${maxSearchChunks} window(s).`);
        break;
      }

      const windowStartRaw = addDaysUtc(windowEnd, -chunkDays);
      const windowStart = windowStartRaw < start ? start : windowStartRaw;
      const sinceStr = fmtYmd(windowStart);
      const untilStr = fmtYmd(windowEnd);

      console.log(`[x] Search chunk ${chunkIndex}${maxSearchChunks > 0 ? `/${maxSearchChunks}` : ""}: ${sinceStr} → ${untilStr}`);

      const qOriginals = `from:${username} since:${sinceStr} until:${untilStr} -filter:replies -filter:retweets`;
      const qReplies = `from:${username} since:${sinceStr} until:${untilStr} filter:replies`;

      const urlOriginals = buildSearchUrl({ query: qOriginals, live: searchLive });
      const urlReplies = buildSearchUrl({ query: qReplies, live: searchLive });

      // Fresh "window" per chunk so scroll/DOM state can't leak between date windows.
      const chunkPage = await context.newPage();
      try {
        console.log(`[x] Search originals @${username} ${sinceStr}..${untilStr}`);
        await runPhase(chunkPage, {
          scrapedFrom: `search_originals:${sinceStr}:${untilStr}`,
          url: urlOriginals,
          seenIds: originalSeen,
          acceptRow: (r) => !r.is_reply && !r.is_repost_or_quote,
          outPath: originalsPath
        });

        if (!originalsOnly) {
          console.log(`[x] Search replies @${username} ${sinceStr}..${untilStr}`);
          await runPhase(chunkPage, {
            scrapedFrom: `search_replies:${sinceStr}:${untilStr}`,
            url: urlReplies,
            seenIds: replySeen,
            acceptRow: (r) => !!r.is_reply,
            outPath: repliesPath
          });
        }
      } finally {
        await chunkPage.close().catch(() => {});
      }

      windowEnd = windowStart;
      const gapMs = 10_000 + Math.random() * 10_000;
      await sleep(gapMs);
    }
  }

  /** CDP or ephemeral launch: one phase per browser session (old behavior). */
  async function runPhaseStandalone({ scrapedFrom, url, seenIds, acceptRow, outPath }) {
    let browser;
    let context;
    let cdp = false;

    if (connectURL) {
      console.warn(
        "[x] CDP: connectOverCDP can lower search trust vs persistent launch; prefer default Chrome profile for deep search."
      );
      browser = await chromium.connectOverCDP(connectURL);
      cdp = true;
      context = browser.contexts()[0];
      if (!context) throw new Error("connectOverCDP: no context");
      console.log("[x] Connected over CDP:", connectURL);
      await context.setExtraHTTPHeaders({ ...BROWSER_EXTRA_HEADERS });
      console.log(
        "[x] CDP: set Accept-Language on context (ensure Chrome was not launched blocking service workers)"
      );
    } else {
      const launchOpts = { headless: headlessEphemeral, slowMo: slowmo, channel };
      if (shouldUseGpuLaunchArgs()) launchOpts.args = [...CHROME_GPU_ARGS];
      browser = await chromium.launch(launchOpts);
      const ctxOpts = {
        locale: "en-US",
        extraHTTPHeaders: { ...BROWSER_EXTRA_HEADERS },
        serviceWorkers: "allow"
      };
      if (fs.existsSync(storageStatePath)) ctxOpts.storageState = storageStatePath;
      context = await browser.newContext(ctxOpts);
    }

    if (blockMedia) await installMediaBlockerOnContext(context);
    const page = await context.newPage();

    try {
      await runPhase(page, { scrapedFrom, url, seenIds, acceptRow, outPath });
      if (keepOpen && !cdp && headlessEphemeral === false) {
        await waitForEnterBeforeClose(
          "[x] Press Enter to close the browser (before reusing the same profile folder)…\n"
        );
      }
    } finally {
      await page.close().catch(() => {});
      if (cdp) await browser.close();
      else {
        await context.close();
        await browser.close();
      }
    }
  }

  /** Persistent: one Chrome profile for all phases (LinkedIn-style). */
  async function runPersistentSession() {
    ensureDir(userDataDir);

    if (!firstPage) {
      console.log(`[x] Profile: ${userDataDir}`);
      const roundDesc = rounds > 0 ? `${rounds} rounds/phase` : "unlimited rounds/phase (X_ROUNDS=0)";
      console.log(`[x] ${passes} pass(es), ${roundDesc}; pause ${pauseMin}-${pauseMax}s`);
    }
    if (blockMedia) console.log("[x] --block-media on");
    if (pruneDom) console.log("[x] --prune-dom on");
    if (!legacyScroll) {
      console.log("[x] scroll: incremental wheel + nested scroller (no jump-to-bottom); --legacy-scroll for wheel-only");
    } else {
      console.log("[x] --legacy-scroll");
    }
    if (expandViewportOnly) {
      console.log("[x] Show more: viewport-first; --expand-scan-full-feed for whole DOM");
    } else {
      console.log("[x] --expand-scan-full-feed");
    }
    if (strictNoNewStreak) {
      console.log(
        "[x] --strict-no-new-streak is legacy; stop streak now uses saved rows only (newWritten). Use --noNewLimit."
      );
    }
    console.log(`[x] scroll pace: ${scrollPace} (post-scroll waits scale as 1/pace; --scroll-pace 1 = baseline)`);
    console.log(
      "[x] browser: locale=en-US, serviceWorkers=allow, Accept-Language=en-US,en;q=0.9 (search index)"
    );
    if (shouldUseGpuLaunchArgs()) {
      console.log("[x] GPU-ish Chrome flags enabled (Canvas/WebGL); --no-gpu-args or X_NO_GPU_ARGS=1 to disable");
    }
    if (earlyStopStuckBottom > 0) {
      console.log(
        `[x] early stop: after ${earlyStopStuckBottom} stuck loop(s) at feed bottom with no scroll delta (--early-stop-stuck-bottom 0 to disable)`
      );
    }
    if (scrollShakeOnNoNew) {
      console.log("[x] scroll shake on no-new: on (wheel up/down + settle; --no-scroll-shake to disable)");
    }

    const contextOpts = buildPersistentContextOpts(mobile);
    let passDone = 0;

    while (passDone < passes) {
      const chunk = recycleEveryPasses > 0 ? Math.min(recycleEveryPasses, passes - passDone) : passes - passDone;
      const passFrom = passDone + 1;
      const passTo = passDone + chunk;

      const context = await chromium.launchPersistentContext(userDataDir, contextOpts);
      let page = context.pages()[0];
      if (!page) page = await context.newPage();
      if (blockMedia) await installMediaBlockerOnContext(context);

      try {
        for (let p = passFrom; p <= passTo; p++) {
          console.log(`[x] Pass ${p}/${passes}`);

          if (!searchOnly) {
            console.log(`Scraping timeline @${username}`);
            await runPhase(page, {
              scrapedFrom: "timeline",
              url: timelineUrl,
              seenIds: originalSeen,
              acceptRow: (r) => !r.is_reply && !r.is_repost_or_quote,
              outPath: originalsPath
            });

            if (!originalsOnly) {
              console.log(`Scraping replies @${username}`);
              await runPhase(page, {
                scrapedFrom: "with_replies",
                url: repliesUrl,
                seenIds: replySeen,
                acceptRow: (r) => !!r.is_reply,
                outPath: repliesPath
              });
            }
          }

          if (searchChunks) {
            await runSearchChunks(context);
          }

          if (reloadDomEveryPasses > 0 && p % reloadDomEveryPasses === 0 && p < passTo) {
            await reloadDomGentle(page, timelineUrl);
          }

          if (p < passTo) await sleep(1200 + Math.random() * 800);
        }

        if (keepOpen && passTo === passes) {
          await waitForEnterBeforeClose(
            "[x] Press Enter to close Chrome (finish before next run with same X_USER_DATA)…\n"
          );
        }
      } finally {
        await context.close();
      }

      passDone += chunk;
      if (passDone < passes && recycleEveryPasses > 0) {
        console.log(`[x] Recycled browser after ${chunk} pass(es); reopening…`);
        await sleep(2000 + Math.random() * 1000);
      }
    }
  }

  if (connectURL || ephemeral) {
    for (let p = 1; p <= passes; p++) {
      console.log(`[x] Pass ${p}/${passes} (standalone browser)`);
      if (!searchOnly) {
        await runPhaseStandalone({
          scrapedFrom: "timeline",
          url: timelineUrl,
          seenIds: originalSeen,
          acceptRow: (r) => !r.is_reply && !r.is_repost_or_quote,
          outPath: originalsPath
        });
        if (!originalsOnly) {
          await runPhaseStandalone({
            scrapedFrom: "with_replies",
            url: repliesUrl,
            seenIds: replySeen,
            acceptRow: (r) => !!r.is_reply,
            outPath: repliesPath
          });
        }
      }
      if (searchChunks) {
        console.log(
          "[x] --search-chunks is implemented for the persistent profile flow. Re-run without --ephemeral/--connectURL to iterate date windows automatically."
        );
      }
    }
  } else {
    await runPersistentSession();
  }

  console.log("Wrote originals:", originalsPath);
  console.log("Wrote replies:  ", repliesPath);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});

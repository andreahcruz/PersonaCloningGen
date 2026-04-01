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
 * --fresh          Delete resume state and start from top.
 * --no-resume      Do not read/write resume state.
 *
 * Resume:
 *   State file defaults next to profile: X_USER_DATA/.x_collect_state.json
 *   Override: X_STATE_FILE or --state-file <path>
 *   Same --user and same phase key (timeline vs with_replies) must match for resume.
 *
 * Env (mirror LinkedIn-style LI_* but prefixed X_):
 *   X_USER_DATA          Chrome profile dir (default ./x_user_data)
 *   X_ROUNDS             Max scroll rounds per phase (0 = unlimited; only --noNewLimit stops)
 *   X_PAUSE_MIN, X_PAUSE_MAX   Random pause seconds between rounds (default 2 / 4.5)
 *   X_PASSES             Repeat full cycle: timeline + with_replies (default 1)
 *   X_STATE_FILE         Resume JSON path
 *   X_KEEP_OPEN=1        Same as --keep-open
 *   X_BLOCK_MEDIA=1      Same as --block-media
 *   X_PRUNE_DOM=1        Same as --prune-dom
 *   X_FULL_SCROLL_EVERY  Same as --full-scroll-every
 *   X_LEGACY_SCROLL=1    Same as --legacy-scroll
 *   X_RELOAD_DOM_EVERY_ROUNDS, X_RELOAD_DOM_EVERY_PASSES
 *   X_RECYCLE_BROWSER_EVERY_PASSES
 *   X_MAX_ROUNDS_PER_PASS
 *   X_EXPAND_FULL_FEED=1 Same as --expand-scan-full-feed
 *   X_MOBILE=1           Same as --mobile
 *
 * Alternate modes:
 *   --ephemeral          launch()+storageState (no persistent folder); default headless true
 *   --storageState path  Use with --ephemeral
 *   --connectURL URL     Attach to Chrome you started with --remote-debugging-port
 */

"use strict";

const fs = require("fs");
const fsPromises = require("fs/promises");
const path = require("path");
const readline = require("readline");
const { devices } = require("playwright");
const { chromium } = require("playwright-extra");
const StealthPlugin = require("puppeteer-extra-plugin-stealth");

chromium.use(StealthPlugin());

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

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

async function readState(statePath) {
  try {
    const raw = await fsPromises.readFile(statePath, "utf8");
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

async function writeState(statePath, snap) {
  ensureDir(path.dirname(statePath));
  const payload = { ...snap, savedAt: nowIso() };
  await fsPromises.writeFile(statePath, JSON.stringify(payload, null, 2), "utf8");
}

function activityKey(username, scrapedFrom) {
  return `x:${username}:${scrapedFrom}`;
}

async function migrateLegacyResumeState(statePath) {
  if (process.env.X_STATE_FILE) return;
  try {
    await fsPromises.access(statePath);
    return;
  } catch {
    /* missing */
  }
  const legacy = path.join(process.cwd(), ".x_collect_state.json");
  try {
    await fsPromises.access(legacy);
    await fsPromises.copyFile(legacy, statePath);
    console.log("[x] Copied old resume state from cwd →", statePath);
  } catch {
    /* no legacy */
  }
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
}

/** Capture scroll + URL for X (window scroll; no LinkedIn nested scroller). */
async function captureXScrollState(page, actKey) {
  const scrollY = await page.evaluate(() => window.scrollY).catch(() => 0);
  return {
    activityKey: actKey,
    pageUrl: page.url(),
    scrollY,
    anchorTweetId: "",
    anchorPrefix: ""
  };
}

async function applyResumeXScroll(page, st) {
  const y = Number(st.scrollY) || 0;
  await sleep(800);
  for (let attempt = 0; attempt < 4; attempt++) {
    await page.evaluate((yy) => window.scrollTo(0, yy), y);
    await sleep(500 + attempt * 400);
  }
}

async function gotoXWithResume(page, targetUrl, { statePath, useResume, actKey, firstPage }) {
  if (firstPage || !useResume) {
    await gotoAndStabilize(page, targetUrl);
    await sleep(1500);
    return;
  }

  const st = await readState(statePath);
  if (st && st.pageUrl && st.activityKey === actKey) {
    console.log(`[x] Resuming scrollY≈${st.scrollY} — ${String(st.pageUrl).slice(0, 72)}…`);
    try {
      await page.goto(st.pageUrl, { waitUntil: "domcontentloaded", timeout: 120_000 });
      await sleep(2500);
      await applyResumeXScroll(page, st);
      if (st.anchorTweetId) {
        const ok = await resumeByAnchorId(page, String(st.anchorTweetId), { maxSteps: 25 });
        console.log(`[x] Anchor id resume ${ok ? "matched" : "not found"}.`);
      }
      await sleep(1000);
      return;
    } catch (e) {
      console.warn("[x] Resume goto failed, falling back:", e.message || e);
    }
  } else if (st && st.activityKey !== actKey) {
    console.log("[x] State file is for another phase; opening target URL from top.");
  }

  await gotoAndStabilize(page, targetUrl);
  await sleep(1500);
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
      const url = href.startsWith("http") ? href : `https://x.com${href}`;
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

async function removeArticlesByIds(page, ids) {
  if (!ids.length) return 0;
  const slice = ids.slice(0, 200);
  return page.evaluate((wantedIds) => {
    const wanted = new Set(wantedIds);
    let removed = 0;
    const arts = Array.from(document.querySelectorAll("article"));
    for (const art of arts) {
      const a = art.querySelector('a[href*="/status/"]');
      const href = a?.getAttribute("href") ?? "";
      const m = href.match(/\/status\/(\d+)/);
      const id = m?.[1];
      if (id && wanted.has(id)) {
        try {
          art.remove();
          removed++;
        } catch {
          // ignore
        }
      }
    }
    return removed;
  }, slice);
}

async function resumeByAnchorId(page, anchorId, { maxSteps = 35 } = {}) {
  if (!anchorId) return false;
  for (let i = 0; i < maxSteps; i++) {
    const found = await page.evaluate((aid) => {
      const arts = Array.from(document.querySelectorAll("article"));
      let matchIdx = -1;
      for (let j = 0; j < arts.length; j++) {
        const a = arts[j].querySelector('a[href*="/status/"]');
        const href = a?.getAttribute("href") ?? "";
        const m = href.match(/\/status\/(\d+)/);
        const id = m?.[1];
        if (id && id === aid) {
          matchIdx = j;
          break;
        }
      }
      if (matchIdx === -1) return false;
      try {
        arts[matchIdx].scrollIntoView({ block: "start", behavior: "instant" });
      } catch {}
      for (let k = 0; k < matchIdx; k++) {
        try {
          arts[k].remove();
        } catch {}
      }
      return true;
    }, anchorId);
    if (found) {
      await sleep(800);
      return true;
    }
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.keyboard.press("PageDown").catch(() => {});
    await sleep(900);
  }
  return false;
}

async function nudgeScroll(page, fractionOfViewport) {
  const vp = page.viewportSize() || { height: 800 };
  const delta = Math.floor(vp.height * fractionOfViewport);
  await page.mouse.wheel(0, delta);
  await sleep(900 + Math.random() * 600);
}

async function scrollXRound(page, roundIndex, { fullScrollEvery = 3, legacyScroll = false } = {}) {
  if (legacyScroll) {
    const vp = page.viewportSize() || { height: 800 };
    await page.mouse.wheel(0, Math.floor(vp.height * 0.85));
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    return;
  }
  const viewportFraction = 0.82 + Math.random() * 0.18;
  const full = fullScrollEvery > 0 && (roundIndex + 1) % fullScrollEvery === 0;
  if (full) {
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  } else {
    const vp = page.viewportSize() || { height: 800 };
    await page.mouse.wheel(0, Math.floor(vp.height * viewportFraction));
  }
}

async function reloadDomPreserveScroll(page, targetUrl, statePath, actKey, useResume) {
  if (!useResume) {
    await gotoAndStabilize(page, targetUrl);
    return;
  }
  const base = await captureXScrollState(page, actKey);
  const snap = await readState(statePath);
  if (snap && snap.anchorTweetId) base.anchorTweetId = snap.anchorTweetId;
  await writeState(statePath, base);
  const gotoUrl = base.pageUrl || targetUrl;
  console.log("[x] Reloading tab to trim DOM (resume after load)…");
  await page.goto(gotoUrl, { waitUntil: "domcontentloaded", timeout: 120_000 });
  await sleep(2500);
  await applyResumeXScroll(page, base);
  if (base.anchorTweetId) {
    await resumeByAnchorId(page, String(base.anchorTweetId), { maxSteps: 20 });
  }
  await sleep(800);
}

async function runFirstPageOnly(page, expandOpts) {
  const viewportOnly = expandOpts.viewportOnly !== false;
  await sleep(2500);
  await expandAllShowMoreInView(page, {
    maxNoProgress: 6,
    maxClicks: 45,
    viewportOnly
  });
  await nudgeScroll(page, 0.45);
  await expandAllShowMoreInView(page, { maxNoProgress: 4, maxClicks: 30, viewportOnly });
  await nudgeScroll(page, 0.5);
  await expandAllShowMoreInView(page, { maxNoProgress: 3, maxClicks: 24, viewportOnly });
  await page.evaluate(() => window.scrollTo(0, 0));
  await sleep(800);
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
    pruneEvery,
    keepLastArticles,
    errorCheckEvery,
    maxRounds,
    checkpointEveryRounds,
    statePath,
    actKey,
    pageUrlForState,
    pruneDomAfterExtract,
    pauseMin,
    pauseMax,
    fullScrollEvery,
    legacyScroll,
    reloadDomEveryRounds,
    expandViewportOnly,
    targetUrl
  } = opts;

  let totalWritten = 0;
  let noNewStreak = 0;
  let loops = 0;
  let backoffS = 0;
  let lastAnchorId = "";
  let lastTextPrefix = "";

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

    if (
      reloadDomEveryRounds > 0 &&
      loops > 1 &&
      (loops - 1) % reloadDomEveryRounds === 0 &&
      statePath &&
      targetUrl
    ) {
      await reloadDomPreserveScroll(page, targetUrl, statePath, actKey, !!statePath);
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

    const batch = await extractVisibleTweets(page);
    let newSeen = 0;
    let newWritten = 0;
    const processedIds = [];

    for (const t of batch) {
      const tid = t?.id;
      if (typeof tid !== "string") continue;
      if (seenIds.has(tid)) continue;
      seenIds.add(tid);
      newSeen += 1;
      processedIds.push(tid);
      lastAnchorId = tid;
      const tx = t.text || "";
      lastTextPrefix = tx.replace(/\s+/g, " ").trim().slice(0, 120);

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

    if (newSeen === 0) noNewStreak += 1;
    else noNewStreak = 0;

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

    let prunedById = 0;
    if (pruneDomAfterExtract && processedIds.length) {
      prunedById = await removeArticlesByIds(page, processedIds).catch(() => 0);
    }

    const prevScrollY = await page.evaluate(() => window.scrollY).catch(() => 0);
    await scrollXRound(page, loops - 1, { fullScrollEvery, legacyScroll });
    await sleep(800 + Math.random() * 800);
    await page.waitForTimeout(baseDelayMs);
    await page.waitForTimeout(500 + Math.floor(baseDelayMs * 0.5));

    let pruned = 0;
    if (pruneEvery && loops % pruneEvery === 0) {
      pruned = await pruneOldArticles(page, keepLastArticles).catch(() => 0);
    }

    if (statePath && checkpointEveryRounds > 0 && loops % checkpointEveryRounds === 0) {
      const snap = {
        activityKey: actKey,
        pageUrl: page.url() || pageUrlForState,
        scrollY: await page.evaluate(() => window.scrollY).catch(() => 0),
        anchorTweetId: lastAnchorId,
        anchorPrefix: lastTextPrefix,
        seenCount: seenIds.size,
        writtenThisRun: totalWritten
      };
      try {
        await writeState(statePath, snap);
      } catch {
        // ignore
      }
    }

    if (logEvery && loops % logEvery === 0) {
      const scrollY = await page.evaluate(() => window.scrollY).catch(() => null);
      console.log(
        `[${scrapedFrom}] loops=${loops} written=${totalWritten} seen=${seenIds.size} newSeen=${newSeen} newWritten=${newWritten} expanded=${expanded} pruned=${pruned} prunedById=${prunedById} noNewStreak=${noNewStreak} scrollY=${scrollY} prevScrollY=${prevScrollY}`
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

function buildPersistentContextOpts(mobile) {
  const base = {
    headless: false,
    channel: "chrome",
    viewport: { width: 1280, height: 900 }
  };
  if (!mobile) return base;
  const phone = devices["iPhone 13"];
  return {
    ...base,
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
    if (t === "image" || t === "media" || t === "font") return route.abort();
    return route.continue();
  });
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
  --fresh / --no-resume
  --state-file PATH       (env X_STATE_FILE; default <user-data>/.x_collect_state.json)
  --passes N              (env X_PASSES)
  --checkpoint-rounds N
  --no-checkpoint
  --prune-dom             (env X_PRUNE_DOM=1)
  --block-media           (env X_BLOCK_MEDIA=1)
  --full-scroll-every N   (env X_FULL_SCROLL_EVERY)
  --legacy-scroll         (env X_LEGACY_SCROLL=1)
  --reload-dom-every-rounds N
  --reload-dom-every-passes N
  --max-rounds-per-pass N (env X_MAX_ROUNDS_PER_PASS; 0 = use X_ROUNDS)
  --recycle-every-passes N
  --expand-scan-full-feed (env X_EXPAND_FULL_FEED=1)
  --mobile                (env X_MOBILE=1)

Env: X_ROUNDS (0=unlimited per phase), X_PAUSE_MIN (2), X_PAUSE_MAX (4.5)

Other:
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

  const defaultStateInProfile = path.join(userDataDir, ".x_collect_state.json");
  const statePath = path.resolve(
    getArg("state-file", "") ||
      process.env.X_STATE_FILE ||
      getArg("stateFile", "") ||
      defaultStateInProfile
  );

  const firstPage = hasFlag("first-page") || hasFlag("firstPage");
  const noResume = hasFlag("no-resume") || hasFlag("noResume");
  const fresh = hasFlag("fresh");
  const keepOpen =
    hasFlag("keep-open") || hasFlag("keepOpen") || String(process.env.X_KEEP_OPEN || "") === "1";

  const passes = Math.max(
    1,
    intArg("passes", envInt("X_PASSES", 1))
  );
  const rounds = intArg("rounds", envInt("X_ROUNDS", 0));
  const pauseMin = envFloat("X_PAUSE_MIN", 2);
  const pauseMax = envFloat("X_PAUSE_MAX", 4.5);

  const noNewLimit = intArg("noNewLimit", 15);
  const maxTweets = intArg("maxTweets", 2_000_000_000);
  const logEvery = intArg("logEvery", 10);
  const baseDelayMs = intArg("baseDelayMs", 250);
  const maxBackoffS = intArg("maxBackoffS", 60);
  const channelRaw = getArg("channel", "");
  const channel = channelRaw && channelRaw.trim() ? channelRaw.trim() : undefined;
  const slowmo = intArg("slowmo", 0);

  const pruneEvery = intArg("pruneEvery", 5);
  const keepLastArticles = intArg("keepLastArticles", 100);
  const errorCheckEvery = intArg("errorCheckEvery", 25);

  let checkpointRounds = intArg("checkpoint-rounds", 5);
  if (hasFlag("no-checkpoint") || hasFlag("noCheckpoint")) checkpointRounds = 0;
  if (getArg("checkpointRounds", "") !== "") {
    checkpointRounds = intArg("checkpointRounds", checkpointRounds);
  }

  const pruneDom = hasFlag("prune-dom") || hasFlag("pruneDom") || String(process.env.X_PRUNE_DOM || "") === "1";
  const blockMedia =
    hasFlag("block-media") || hasFlag("blockMedia") || String(process.env.X_BLOCK_MEDIA || "") === "1";

  let fullScrollEvery = intArg("full-scroll-every", 3);
  if (process.env.X_FULL_SCROLL_EVERY !== undefined && process.env.X_FULL_SCROLL_EVERY !== "") {
    fullScrollEvery = envInt("X_FULL_SCROLL_EVERY", fullScrollEvery);
  }
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

  const useResume = !firstPage && !noResume;
  const headlessEphemeral = boolArg("headless", true);

  if (fresh && fs.existsSync(statePath)) {
    try {
      fs.unlinkSync(statePath);
      console.log("[x] --fresh cleared state:", statePath);
    } catch {
      // ignore
    }
  }

  const originalSeen = loadExistingIds(originalsPath);
  const replySeen = loadExistingIds(repliesPath);

  const timelineUrl = `https://x.com/${username}`;
  const repliesUrl = `https://x.com/${username}/with_replies`;

  async function runPhase(page, { scrapedFrom, url, seenIds, acceptRow, outPath }) {
    const actKey = activityKey(username, scrapedFrom);
    await gotoXWithResume(page, url, { statePath, useResume, actKey, firstPage });

    if (firstPage) {
      await runFirstPageOnly(page, { viewportOnly: expandViewportOnly });
      const batch = await extractVisibleTweets(page);
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
      if (useResume) {
        const snap = await captureXScrollState(page, actKey);
        const last = batch.length ? batch[batch.length - 1] : null;
        if (last?.id) snap.anchorTweetId = last.id;
        await writeState(statePath, snap);
      }
      return;
    }

    const cap =
      maxRoundsPerPass > 0 ? maxRoundsPerPass : rounds > 0 ? rounds : 0;

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
      pruneEvery,
      keepLastArticles,
      errorCheckEvery,
      maxRounds: cap,
      checkpointEveryRounds: useResume ? checkpointRounds : 0,
      statePath: useResume ? statePath : null,
      actKey,
      pageUrlForState: url,
      pruneDomAfterExtract: pruneDom,
      pauseMin,
      pauseMax,
      fullScrollEvery,
      legacyScroll,
      reloadDomEveryRounds,
      expandViewportOnly,
      targetUrl: url
    });
  }

  /** CDP or ephemeral launch: one phase per browser session (old behavior). */
  async function runPhaseStandalone({ scrapedFrom, url, seenIds, acceptRow, outPath }) {
    const actKey = activityKey(username, scrapedFrom);
    let browser;
    let context;
    let cdp = false;

    if (connectURL) {
      browser = await chromium.connectOverCDP(connectURL);
      cdp = true;
      context = browser.contexts()[0];
      if (!context) throw new Error("connectOverCDP: no context");
      console.log("[x] Connected over CDP:", connectURL);
    } else {
      browser = await chromium.launch({ headless: headlessEphemeral, slowMo: slowmo, channel });
      context = await browser.newContext(
        fs.existsSync(storageStatePath) ? { storageState: storageStatePath } : undefined
      );
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
    await migrateLegacyResumeState(statePath);

    if (!firstPage) {
      console.log(
        `[x] Resume file: ${statePath} (${useResume ? "will load/save" : "disabled"})`
      );
      console.log(`[x] Profile: ${userDataDir}`);
      const roundDesc = rounds > 0 ? `${rounds} rounds/phase` : "unlimited rounds/phase (X_ROUNDS=0)";
      console.log(`[x] ${passes} pass(es), ${roundDesc}; pause ${pauseMin}-${pauseMax}s`);
    }
    if (blockMedia) console.log("[x] --block-media on");
    if (pruneDom) console.log("[x] --prune-dom on");
    if (!legacyScroll) {
      console.log(
        `[x] scroll: full jump every ${fullScrollEvery || "never"} round(s); --legacy-scroll for wheel+body`
      );
    } else {
      console.log("[x] --legacy-scroll");
    }
    if (expandViewportOnly) {
      console.log("[x] Show more: viewport-first; --expand-scan-full-feed for whole DOM");
    } else {
      console.log("[x] --expand-scan-full-feed");
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

          console.log(`Scraping timeline @${username}`);
          await runPhase(page, {
            scrapedFrom: "timeline",
            url: timelineUrl,
            seenIds: originalSeen,
            acceptRow: (r) => !r.is_reply && !r.is_repost_or_quote,
            outPath: originalsPath
          });

          console.log(`Scraping replies @${username}`);
          await runPhase(page, {
            scrapedFrom: "with_replies",
            url: repliesUrl,
            seenIds: replySeen,
            acceptRow: (r) => !!r.is_reply,
            outPath: repliesPath
          });

          if (reloadDomEveryPasses > 0 && p % reloadDomEveryPasses === 0 && p < passTo) {
            const actKey = activityKey(username, "timeline");
            await reloadDomPreserveScroll(page, timelineUrl, statePath, actKey, useResume);
          }

          if (p < passTo) await sleep(1200 + Math.random() * 800);
        }

        if (keepOpen && passTo === passes) {
          await waitForEnterBeforeClose(
            "[x] Press Enter to close Chrome (finish before next run with same X_USER_DATA)…\n"
          );
        }
      } finally {
        if (useResume) {
          try {
            const snap = await captureXScrollState(
              page,
              activityKey(username, "with_replies")
            );
            const st = await readState(statePath);
            if (st?.anchorTweetId) snap.anchorTweetId = st.anchorTweetId;
            await writeState(statePath, snap);
          } catch {
            // ignore
          }
        }
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
      await runPhaseStandalone({
        scrapedFrom: "timeline",
        url: timelineUrl,
        seenIds: originalSeen,
        acceptRow: (r) => !r.is_reply && !r.is_repost_or_quote,
        outPath: originalsPath
      });
      await runPhaseStandalone({
        scrapedFrom: "with_replies",
        url: repliesUrl,
        seenIds: replySeen,
        acceptRow: (r) => !!r.is_reply,
        outPath: repliesPath
      });
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

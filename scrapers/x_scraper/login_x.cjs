"use strict";

const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright-extra");
const StealthPlugin = require("puppeteer-extra-plugin-stealth");

chromium.use(StealthPlugin());

function getArg(name, fallback) {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx === -1) return fallback;
  const v = process.argv[idx + 1];
  if (!v || v.startsWith("--")) return fallback;
  return v;
}

function boolArg(name, fallback) {
  const v = String(getArg(name, fallback ? "true" : "false")).toLowerCase();
  return v === "true";
}

async function main() {
  const storageState = path.resolve(getArg("storageState", "storageState.json"));
  const headless = boolArg("headless", false);
  const channelRaw = getArg("channel", "");
  const channel = channelRaw && channelRaw.trim() ? channelRaw.trim() : undefined;
  const connectURL = (getArg("connectURL", "") || "").trim();

  fs.mkdirSync(path.dirname(storageState), { recursive: true });

  if (connectURL) {
    const browser = await chromium.connectOverCDP(connectURL);
    const context = browser.contexts()[0];
    if (!context) {
      throw new Error("No browser context. Start Chrome with --remote-debugging-port first.");
    }
    const page = context.pages()[0] || (await context.newPage());
    await page.goto("https://x.com/home", { waitUntil: "domcontentloaded" });
    await page.waitForSelector('[data-testid="SideNav_AccountSwitcher_Button"]', {
      timeout: 10 * 60 * 1000
    });
    await context.storageState({ path: storageState });
    await browser.close();
    console.log(`Saved storage state from connected Chrome to: ${storageState}`);
    return;
  }

  const browser = await chromium.launch({ headless, channel });
  const context = await browser.newContext();
  const page = await context.newPage();

  await page.goto("https://x.com/i/flow/login", { waitUntil: "domcontentloaded" });

  await page.waitForSelector('[data-testid="SideNav_AccountSwitcher_Button"]', {
    timeout: 10 * 60 * 1000
  });

  await context.storageState({ path: storageState });
  await browser.close();

  console.log(`Saved storage state to: ${storageState}`);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});

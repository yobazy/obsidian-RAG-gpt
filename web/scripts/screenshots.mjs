// Screenshots of the running app (point it at the fixture vault, never real notes).
//   node scripts/screenshots.mjs [outDir] [baseUrl] [question]
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { chromium, devices } from "playwright";

const outDir = process.argv[2] ?? "screenshots";
const base = process.argv[3] ?? "http://127.0.0.1:5190";
const question = process.argv[4] ?? "What hydration did I settle on for the weekend loaf, and why?";
mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch();
const shots = [];

async function run(name, context) {
  const page = await context.newPage();
  await page.goto(base);
  await page.getByText(/Searching \d+ notes/).waitFor({ timeout: 60_000 });
  const shot = async (label, opts = {}) => {
    const path = join(outDir, `${name}-${label}.png`);
    await page.screenshot({ path, ...opts });
    shots.push(path);
  };
  await shot("1-empty");

  await page.getByLabel("Your question").fill(question);
  await page.getByRole("button", { name: "Ask" }).click();
  await page.locator(".timing").waitFor({ timeout: 180_000 });
  await page.getByRole("button", { name: "Source 1" }).first().click().catch(() => {});
  await page.waitForTimeout(600);
  // Phone: viewport shots, since the sticky ask box would float mid-page in a full-page capture.
  if (name === "phone") {
    await page.evaluate(() => window.scrollTo(0, 0));
    await shot("2-answer");
    await page.locator(".sources").scrollIntoViewIfNeeded();
    await shot("2b-sources");
  } else {
    await shot("2-answer", { fullPage: true });
  }

  await page.getByRole("button", { name: "Preview" }).first().click();
  await page.locator(".drawer-title").waitFor();
  await shot("3-note-preview");
  await page.close();
}

await run("desktop", await browser.newContext({ viewport: { width: 1440, height: 960 }, colorScheme: "dark" }));
await run("phone", await browser.newContext({ ...devices["iPhone 13"], colorScheme: "dark" }));
await browser.close();
console.log(shots.join("\n"));

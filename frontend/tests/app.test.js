const test = require("node:test");
const assert = require("node:assert");

const { buildRequestPayload, validatePayload } = require("../app.js");

test("buildRequestPayload trims fields and defaults cta/k", () => {
  const payload = buildRequestPayload({
    format: "blog_draft",
    topic: "  Hiring a VP Sales  ",
    audience: " B2B founders ",
    goal: " share a takeaway ",
    cta: "   ",
    k: "",
  });
  assert.strictEqual(payload.format, "blog_draft");
  assert.strictEqual(payload.topic, "Hiring a VP Sales");
  assert.strictEqual(payload.audience, "B2B founders");
  assert.strictEqual(payload.goal, "share a takeaway");
  assert.strictEqual(payload.cta, "none");
  assert.strictEqual(payload.k, 8);
});

test("buildRequestPayload keeps a valid k and cta", () => {
  const payload = buildRequestPayload({
    format: "x_thread",
    topic: "t",
    audience: "a",
    goal: "g",
    cta: "invite comments",
    k: "12",
  });
  assert.strictEqual(payload.cta, "invite comments");
  assert.strictEqual(payload.k, 12);
});

test("validatePayload flags missing required fields", () => {
  const errors = validatePayload({ topic: "", audience: "", goal: "" });
  assert.strictEqual(errors.length, 3);
});

test("validatePayload passes when required fields present", () => {
  const errors = validatePayload({ topic: "x", audience: "y", goal: "z" });
  assert.strictEqual(errors.length, 0);
});

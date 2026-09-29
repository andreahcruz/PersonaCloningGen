// API_BASE can be overridden at runtime (e.g. window.API_BASE = "http://localhost:8001" for local dev
// without the nginx proxy). Default "/api" relies on nginx.conf proxying to the backend service.
const API_BASE = (typeof window !== "undefined" && window.API_BASE) || "/api";

function buildRequestPayload(form) {
  const k = Number(form.k);
  return {
    format: form.format,
    topic: (form.topic || "").trim(),
    audience: (form.audience || "").trim(),
    goal: (form.goal || "").trim(),
    cta: (form.cta || "").trim() || "none",
    k: Number.isFinite(k) && k > 0 ? k : 8,
  };
}

function validatePayload(payload) {
  const errors = [];
  if (!payload.topic) errors.push("Topic is required.");
  if (!payload.audience) errors.push("Audience is required.");
  if (!payload.goal) errors.push("Goal is required.");
  return errors;
}

async function fetchFormats() {
  const res = await fetch(`${API_BASE}/formats`);
  if (!res.ok) throw new Error(`Failed to load formats (${res.status})`);
  return res.json();
}

async function generateDraft(payload) {
  const res = await fetch(`${API_BASE}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("brief-form");
    const formatSelect = document.getElementById("format");
    const output = document.getElementById("output");
    const errorBox = document.getElementById("error");
    const submitBtn = document.getElementById("submit-btn");

    fetchFormats()
      .then((formats) => {
        formatSelect.innerHTML = formats
          .map((f) => `<option value="${f}">${f}</option>`)
          .join("");
      })
      .catch(() => {
        formatSelect.innerHTML = '<option value="linkedin_post">linkedin_post</option>';
        errorBox.textContent = "Could not reach the backend to load formats — is it running?";
      });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      errorBox.textContent = "";
      output.textContent = "";

      const payload = buildRequestPayload({
        format: formatSelect.value,
        topic: document.getElementById("topic").value,
        audience: document.getElementById("audience").value,
        goal: document.getElementById("goal").value,
        cta: document.getElementById("cta").value,
        k: document.getElementById("k").value,
      });

      const errors = validatePayload(payload);
      if (errors.length) {
        errorBox.textContent = errors.join(" ");
        return;
      }

      submitBtn.disabled = true;
      submitBtn.textContent = "Generating...";
      output.textContent = "Embedding query, retrieving chunks, generating...";

      try {
        const { draft } = await generateDraft(payload);
        output.textContent = draft;
      } catch (err) {
        errorBox.textContent = err.message;
        output.textContent = "";
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = "Generate";
      }
    });
  });
}

if (typeof module !== "undefined") {
  module.exports = { buildRequestPayload, validatePayload };
}

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { GenerateRequest } from "@/lib/types";

const REQ: GenerateRequest = {
  format: "linkedin_post",
  topic: "Hiring",
  audience: "Founders",
  goal: "One takeaway",
  cta: "none",
  k: 5,
  method: "personarag",
  model: "lemkin-clone",
  temperature: 0.7,
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

// config.ts reads env vars when it is first imported, so reset modules between mode changes.
async function loadApi(mode: "mock" | "live") {
  vi.resetModules();
  vi.stubEnv("NEXT_PUBLIC_API_MODE", mode);
  vi.stubEnv("NEXT_PUBLIC_API_BASE", "http://backend.test");
  return await import("@/lib/api");
}

describe("live mode", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("maps the current backend response ({ draft }) to a result", async () => {
    const api = await loadApi("live");
    fetchMock.mockResolvedValueOnce(jsonResponse({ draft: "Hello world" }));

    const res = await api.generate(REQ);

    expect(fetchMock).toHaveBeenCalledWith(
      "http://backend.test/generate",
      expect.objectContaining({ method: "POST" }),
    );
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toMatchObject({
      format: "linkedin_post",
      topic: "Hiring",
      k: 5,
    });
    expect(res.content).toBe("Hello world");
    expect(res.sources).toEqual([]);
    expect(res.mock).toBe(false);
    expect(res.latencyMs).not.toBeNull();
  });

  it("maps the target response with sources, model and latency", async () => {
    const api = await loadApi("live");
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        content: "Draft",
        model: "lemkin-clone",
        latency_ms: 4820,
        request_id: "abc",
        sources: [
          { text: "Passage", score: 0.87, source_type: "Blog" },
          { text: "Other", source_type: "podcast" },
        ],
      }),
    );

    const res = await api.generate(REQ);

    expect(res.latencyMs).toBe(4820);
    expect(res.requestId).toBe("abc");
    expect(res.sources[0]).toMatchObject({ sourceType: "blog", score: 0.87 });
    expect(res.sources[1]).toMatchObject({ sourceType: "unknown", score: null });
    expect(res.metrics).toMatchObject({ wordCount: 1, sourceCount: 2, scope: "operational" });
  });

  it("preserves server-side operational metrics for a live comparison", async () => {
    const api = await loadApi("live");
    fetchMock.mockResolvedValueOnce(jsonResponse({
      draft: "One sentence.",
      method: "personarag",
      metrics: {
        word_count: 2,
        character_count: 13,
        sentence_count: 1,
        source_count: 1,
        source_types: ["blog"],
        scope: "operational",
      },
      sources: [{ id: "source-1", text: "Evidence", source_type: "blog" }],
    }));

    const res = await api.generate(REQ);

    expect(res.metrics).toEqual({
      wordCount: 2,
      characterCount: 13,
      sentenceCount: 1,
      sourceCount: 1,
      sourceTypes: ["blog"],
      scope: "operational",
    });
  });

  it("rejects an empty draft with a hint", async () => {
    const api = await loadApi("live");
    fetchMock.mockResolvedValueOnce(jsonResponse({ draft: "   " }));

    await expect(api.generate(REQ)).rejects.toMatchObject({ name: "ApiError", hint: expect.any(String) });
  });

  it("surfaces the backend detail message on errors", async () => {
    const api = await loadApi("live");
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: "Generation failed: model not found" }, 502));

    await expect(api.generate(REQ)).rejects.toMatchObject({
      message: "Generation failed: model not found",
      status: 502,
    });
  });

  it("explains an unreachable backend", async () => {
    const api = await loadApi("live");
    fetchMock.mockRejectedValueOnce(new TypeError("fetch failed"));

    await expect(api.generate(REQ)).rejects.toMatchObject({
      message: "Could not reach the backend.",
      hint: expect.stringContaining("NEXT_PUBLIC_API_BASE"),
    });
  });

  it("falls back to the basic health check when dependencies are not available", async () => {
    const api = await loadApi("live");
    fetchMock
      .mockResolvedValueOnce(new Response("not found", { status: 404 }))
      .mockResolvedValueOnce(jsonResponse({ status: "ok" }));

    const res = await api.getHealth();

    expect(res.detailed).toBe(false);
    expect(res.services).toEqual([{ name: "FastAPI", status: "ok" }]);
  });

  it("reports a missing sources endpoint as NotAvailableError", async () => {
    const api = await loadApi("live");
    fetchMock.mockResolvedValueOnce(new Response("nope", { status: 404 }));

    await expect(api.searchSources("hiring", "all")).rejects.toBeInstanceOf(api.NotAvailableError);
  });

  it("keeps only formats the UI knows about", async () => {
    const api = await loadApi("live");
    fetchMock.mockResolvedValueOnce(jsonResponse(["linkedin_post", "blog_draft", "mystery"]));

    expect(await api.getFormats()).toEqual(["linkedin_post", "blog_draft"]);
  });

  it("loads the active persona profile", async () => {
    const api = await loadApi("live");
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        name: "Jason Lemkin",
        source: "Jason Lemkin — operator voice",
        summary: "Direct operator advice.",
      }),
    );

    await expect(api.getPersona()).resolves.toEqual({
      name: "Jason Lemkin",
      source: "Jason Lemkin — operator voice",
      summary: "Direct operator advice.",
      mock: false,
    });
    expect(fetchMock).toHaveBeenCalledWith("http://backend.test/persona", expect.any(Object));
  });

  it("surfaces a missing persona endpoint without blocking other API calls", async () => {
    const api = await loadApi("live");
    fetchMock.mockResolvedValueOnce(new Response("not found", { status: 404 }));

    await expect(api.getPersona()).rejects.toBeInstanceOf(api.NotAvailableError);
  });

  it("offers default models when /models does not exist", async () => {
    const api = await loadApi("live");
    fetchMock.mockResolvedValueOnce(new Response("nope", { status: 404 }));

    const res = await api.getModels();

    expect(res.fromBackend).toBe(false);
    expect(res.models.map((m) => m.name)).toContain("lemkin-clone");
  });

  it("loads Model F availability without treating an old backend as an error", async () => {
    const api = await loadApi("live");
    fetchMock.mockResolvedValueOnce(jsonResponse({
      enabled: true,
      available: true,
      detail: "2713 catalogued snapshot documents, 768-dimension embeddings",
      snapshot_documents: 2713,
      embedding_dimension: 768,
    }));

    await expect(api.getModelFStatus()).resolves.toMatchObject({
      available: true,
      snapshot_documents: 2713,
      embedding_dimension: 768,
    });
    expect(fetchMock).toHaveBeenCalledWith("http://backend.test/model-f/status", expect.any(Object));
  });

  it("hides Model F safely when an older backend has no status endpoint", async () => {
    const api = await loadApi("live");
    fetchMock.mockResolvedValueOnce(new Response("not found", { status: 404 }));

    await expect(api.getModelFStatus()).resolves.toMatchObject({ available: false, enabled: false });
  });

  it("loads the versioned evaluation manifest", async () => {
    const api = await loadApi("live");
    fetchMock.mockResolvedValueOnce(jsonResponse({
      version: "evidence-v1",
      title: "Evaluation evidence",
      description: "Precomputed.",
      groups: [{
        id: "rag-v2",
        title: "RAG-v2",
        scope: "research-only",
        summary: "Separate index.",
        sample_size: "21,193 chunks",
        source: { branch: "vb-puk-34", commit: "abc", report_url: "https://example.test/report" },
        metrics: [],
        limitations: ["Not live candidate data."],
      }],
    }));

    const res = await api.getEvaluation();

    expect(res.version).toBe("evidence-v1");
    expect(res.groups[0].scope).toBe("research-only");
  });

  it("runs grammar only when the caller explicitly requests it", async () => {
    const api = await loadApi("live");
    fetchMock.mockResolvedValueOnce(jsonResponse({
      word_count: 4,
      issue_count: 1,
      issues_per_100_words: 25,
      issues: [{ message: "Possible agreement error", matched_text: "is", offset: 0, length: 2, rule_id: "AGREEMENT", replacements: ["are"] }],
    }));

    await expect(api.checkGrammar("is founders ready")).resolves.toMatchObject({ issue_count: 1 });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://backend.test/quality/grammar",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("surfaces an unavailable grammar service", async () => {
    const api = await loadApi("live");
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: "Grammar service unavailable. Try again shortly." }, 503));

    await expect(api.checkGrammar("Draft")).rejects.toMatchObject({ status: 503 });
  });
});

describe("mock mode", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("never touches the network and labels output as mock", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const api = await loadApi("mock");

    const res = await api.generate(REQ);

    expect(fetchMock).not.toHaveBeenCalled();
    expect(res.mock).toBe(true);
    expect(res.content).toContain("Demo draft");
    expect(res.sources.length).toBeLessThanOrEqual(REQ.k);
  });

  it("returns a clearly labelled sample persona in mock mode", async () => {
    const api = await loadApi("mock");

    const persona = await api.getPersona();

    expect(persona.mock).toBe(true);
    expect(persona.source).toContain("Sample profile");
  });

  it("filters sample sources by type and query", async () => {
    const api = await loadApi("mock");

    const blog = await api.searchSources("", "blog");
    expect(blog.length).toBeGreaterThan(0);
    expect(blog.every((s) => s.sourceType === "blog")).toBe(true);

    const none = await api.searchSources("zzzz-no-match", "all");
    expect(none).toEqual([]);
  });
});

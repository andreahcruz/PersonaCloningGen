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

  it("offers default models when /models does not exist", async () => {
    const api = await loadApi("live");
    fetchMock.mockResolvedValueOnce(new Response("nope", { status: 404 }));

    const res = await api.getModels();

    expect(res.fromBackend).toBe(false);
    expect(res.models.map((m) => m.name)).toContain("lemkin-clone");
  });

  it("still shows the reported evaluation run when /evaluation does not exist", async () => {
    const api = await loadApi("live");
    fetchMock.mockResolvedValueOnce(new Response("nope", { status: 404 }));

    const res = await api.getEvaluation();

    expect(res.runs.length).toBeGreaterThan(0);
    expect(res.matrix.every((c) => !c.available)).toBe(true);
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

  it("filters sample sources by type and query", async () => {
    const api = await loadApi("mock");

    const blog = await api.searchSources("", "blog");
    expect(blog.length).toBeGreaterThan(0);
    expect(blog.every((s) => s.sourceType === "blog")).toBe(true);

    const none = await api.searchSources("zzzz-no-match", "all");
    expect(none).toEqual([]);
  });
});

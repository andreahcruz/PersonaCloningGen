export const runtime = "nodejs";

const UPSTREAM = process.env.API_INTERNAL_URL ?? "http://backend:8000";

/** Keep the optional LightRAG service private behind the existing web relay. */
export async function GET() {
  try {
    const response = await fetch(`${UPSTREAM}/model-f/status`, {
      cache: "no-store",
      signal: AbortSignal.timeout(10_000),
    });
    return new Response(response.body, {
      status: response.status,
      headers: { "content-type": response.headers.get("content-type") ?? "application/json" },
    });
  } catch (error) {
    return Response.json(
      { detail: `Model F status relay failed: ${error instanceof Error ? error.message : "unknown error"}` },
      { status: 502 },
    );
  }
}

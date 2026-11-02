export const runtime = "nodejs";

const UPSTREAM = process.env.API_INTERNAL_URL ?? "http://backend:8000";

/** Server-side relay keeps the FastAPI service private from browser clients. */
export async function GET() {
  try {
    const response = await fetch(`${UPSTREAM}/evaluation`, {
      cache: "no-store",
      signal: AbortSignal.timeout(10_000),
    });
    return new Response(response.body, {
      status: response.status,
      headers: { "content-type": response.headers.get("content-type") ?? "application/json" },
    });
  } catch (error) {
    return Response.json(
      { detail: `Evaluation relay failed: ${error instanceof Error ? error.message : "unknown error"}` },
      { status: 502 },
    );
  }
}

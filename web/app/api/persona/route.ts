export const runtime = "nodejs";

const UPSTREAM = process.env.API_INTERNAL_URL ?? "http://backend:8000";

/**
 * Keep the backend private: the browser uses the same `/api` boundary as the
 * generation request, while this server-side route fetches the active profile.
 */
export async function GET() {
  try {
    const response = await fetch(`${UPSTREAM}/persona`, {
      cache: "no-store",
      signal: AbortSignal.timeout(10_000),
    });

    return new Response(response.body, {
      status: response.status,
      headers: { "content-type": response.headers.get("content-type") ?? "application/json" },
    });
  } catch (error) {
    return Response.json(
      { detail: `Persona profile relay failed: ${error instanceof Error ? error.message : "unknown error"}` },
      { status: 502 },
    );
  }
}

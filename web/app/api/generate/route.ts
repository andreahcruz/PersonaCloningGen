import { NextRequest } from "next/server";

export const runtime = "nodejs";
// CPU inference can take several minutes.  The generic Next rewrite closes the
// upstream socket early, so generation uses this explicit, long-lived relay.
export const maxDuration = 600;

const UPSTREAM = process.env.API_INTERNAL_URL ?? "http://backend:8000";

export async function POST(request: NextRequest) {
  try {
    const response = await fetch(`${UPSTREAM}/generate`, {
      method: "POST",
      headers: { "content-type": request.headers.get("content-type") ?? "application/json" },
      body: await request.text(),
      cache: "no-store",
      signal: AbortSignal.timeout(600_000),
    });

    return new Response(response.body, {
      status: response.status,
      headers: { "content-type": response.headers.get("content-type") ?? "application/json" },
    });
  } catch (error) {
    return Response.json(
      { detail: `Generation relay failed: ${error instanceof Error ? error.message : "unknown error"}` },
      { status: 502 },
    );
  }
}

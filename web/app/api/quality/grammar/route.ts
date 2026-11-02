import { NextRequest } from "next/server";

export const runtime = "nodejs";

const UPSTREAM = process.env.API_INTERNAL_URL ?? "http://backend:8000";

/** The browser talks only to Next; LanguageTool stays on Docker's private network. */
export async function POST(request: NextRequest) {
  try {
    const response = await fetch(`${UPSTREAM}/quality/grammar`, {
      method: "POST",
      headers: { "content-type": request.headers.get("content-type") ?? "application/json" },
      body: await request.text(),
      cache: "no-store",
      signal: AbortSignal.timeout(20_000),
    });
    return new Response(response.body, {
      status: response.status,
      headers: { "content-type": response.headers.get("content-type") ?? "application/json" },
    });
  } catch (error) {
    return Response.json(
      { detail: `Grammar relay failed: ${error instanceof Error ? error.message : "unknown error"}` },
      { status: 502 },
    );
  }
}

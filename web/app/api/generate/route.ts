import { request as httpRequest } from "node:http";
import { NextRequest } from "next/server";

export const runtime = "nodejs";
// CPU inference can take several minutes.  The generic Next rewrite closes the
// upstream socket early, so generation uses this explicit, long-lived relay.
export const maxDuration = 600;

const UPSTREAM = process.env.API_INTERNAL_URL ?? "http://backend:8000";
const GENERATION_TIMEOUT_MS = 600_000;

/**
 * Node's built-in fetch has a five-minute response-header timeout. Generation
 * does not send headers until CPU Ollama has completed the draft, so Model F
 * can legitimately exceed that limit. Use node:http here with our explicit
 * ten-minute socket timeout instead.
 */
function relayGeneration(body: string, contentType: string): Promise<Response> {
  const url = new URL(`${UPSTREAM}/generate`);

  return new Promise((resolve, reject) => {
    const upstream = httpRequest({
      protocol: url.protocol,
      hostname: url.hostname,
      port: url.port,
      path: `${url.pathname}${url.search}`,
      method: "POST",
      headers: { "content-type": contentType },
    }, (response) => {
      const chunks: Buffer[] = [];
      response.on("data", (chunk: Buffer) => chunks.push(chunk));
      response.on("end", () => {
        resolve(new Response(Buffer.concat(chunks), {
          status: response.statusCode ?? 502,
          headers: { "content-type": response.headers["content-type"] ?? "application/json" },
        }));
      });
      response.on("error", reject);
    });

    upstream.setTimeout(GENERATION_TIMEOUT_MS, () => {
      upstream.destroy(new Error("Generation relay timed out after 10 minutes."));
    });
    upstream.on("error", reject);
    upstream.end(body);
  });
}

export async function POST(request: NextRequest) {
  try {
    const response = await relayGeneration(
      await request.text(),
      request.headers.get("content-type") ?? "application/json",
    );

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

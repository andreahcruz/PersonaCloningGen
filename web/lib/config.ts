// "mock" (default) serves clearly labelled sample data so the UI works without a backend.
// "live" calls the FastAPI backend. Set both in web/.env.local, see .env.example.
export const API_MODE: "mock" | "live" =
  process.env.NEXT_PUBLIC_API_MODE === "live" ? "live" : "mock";

// Default "/api" assumes a reverse proxy that strips the prefix, as in the nginx.conf on
// Pukhraj's frontend branch. For local dev without a proxy use http://localhost:8001, which is
// the backend port in docker-compose.
export const API_BASE: string = (process.env.NEXT_PUBLIC_API_BASE ?? "/api").replace(/\/$/, "");

export const IS_MOCK = API_MODE === "mock";

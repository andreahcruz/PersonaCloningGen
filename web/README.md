# Lemkin Studio (web)

The new frontend for the Jason Lemkin persona project. It replaces the Streamlit app
(`user_interface.py`) and is built for Linear issue **PUK-5**.

Stack: Next.js 16 (App Router), React 19, Tailwind CSS 4, Recharts. Node 22.

## Run it

```bash
cd web
npm install
npm run dev        # http://localhost:3000
```

By default the app runs in **demo mode**. It shows clearly labelled sample data, so you can see and
review the design with no backend. A banner on every page says so.

To use the real backend, copy `.env.example` to `.env.local` and set:

```
NEXT_PUBLIC_API_MODE=live
NEXT_PUBLIC_API_BASE=http://localhost:8001
```

Port 8001 is the backend published by `docker-compose.yml`. Restart `npm run dev` after changing
env files.

## Look and feel

- Light theme by default (cool white and graphite, IBM Plex type, one blue accent) with a dark theme button in the top
  bar. Elon Musk mode uses a deep green accent instead.
  The choice is remembered in the browser.
- A persona switch (Jason Lemkin or Elon Musk) in the top bar changes the colours, the home page and the Studio.
  Lemkin writes drafts. Elon Musk is a source-grounded "ask" mode that cites tweets and interview transcripts and
  does not write as him. The Musk screen shows sample data in demo mode and says "not connected yet" in live
  mode, because the API route (`POST /ask`, see the contract) is not built.
- Studio: tap-to-fill idea chips and recent briefs, a brief-readiness meter, a typewriter reveal, a real elapsed
  timer while a draft is written, and tabs for Draft, Preview (post, thread, article or script layout), Checks and
  Sources. Ctrl or Cmd + Enter generates.
- Colours are CSS variables in `app/globals.css`. The theme and persona live in `lib/prefs.ts`, the persona
  settings in `lib/personas.ts`, animation helpers in `lib/motion.ts`. Motion is turned off for visitors who ask
  their system for reduced motion.

## Pages

| Route | What it does |
|---|---|
| `/` | Home and feature overview |
| `/generate` | Main page: format, brief, model, retrieved chunks, temperature, draft, copy and download as Markdown, retrieved sources |
| `/dashboard` | Corpus counts, key service status, recent drafts from this browser |
| `/sources` | Search and filter the knowledge base |
| `/evaluation` | Gold-rubric chart and the four-system comparison table |
| `/health` | Status of API, Ollama, Chroma, models, Airflow and MinIO |
| `/about` | Project explanation and team |

## How it connects to the backend

All backend calls live in `lib/api.ts`. Which endpoints exist today, and which are still needed, is in
[`../docs/FRONTEND_API_CONTRACT.md`](../docs/FRONTEND_API_CONTRACT.md). Where an endpoint is missing
the page shows an explanation instead of failing. Nothing on screen is made up in live mode.

## Checks

```bash
npm run lint
npm test           # API layer tests (vitest)
npm run build
```

## Not done yet

- Temperature is sent to the API but `generate.py` does not use it yet.
- Sources, models, statistics, dependency health and evaluation need the endpoints in the contract.
- The four-system evaluation comparison needs results from PUK-13.
- No Docker service yet. Add one once the team agrees which frontend to ship.

# Vrushabh Bodarya: contributions (DATA 298B, Team 2)

Project: AI-driven persona cloning for B2B content generation (Jason Lemkin / SaaStr). RAG plus a QLoRA fine-tuned Llama 3.1 8B.

This page lists the work I did, how I did it, and which branch holds it. The work is still on separate branches and has not been merged into `QLoRAFT` yet. Linear issue IDs are in brackets.

## Branches I pushed

| Branch | Linear | What it holds | Commits |
|---|---|---|---|
| `vb-puk-5` | PUK-5 | Lemkin Studio frontend (Next.js) | `f5bd9c8` |
| `vb-puk-17` | PUK-17 | Error handling and logging | `12e26b2`, `61c5ca0` |
| `vb` | PUK-11 | Model improvements document | `5f72d43` |

## Work I did

### PUK-5: Frontend migration (Streamlit to Next.js)
- Built the Lemkin Studio web app in `web/`, to replace the Streamlit UI.
- Added an API layer that works in two modes: demo mode (no backend needed) and live mode (calls the FastAPI service).
- Added tests for the API layer and wrote the API contract so the frontend and backend teammates agree on request and response shapes.
- I only changed frontend files. Backend changes are proposed to the backend owners, not edited by me.

### PUK-17: Error handling and logging
- Added clear, typed errors with a hint for the person running it, for pipeline, retrieval, model and deployment failures.
- Added retries, output validation and a health check.
- Fixed Airflow DAG problems. One was a bug where only 1,000 documents were read from the persona store (pagination).
- Fixed a false "malformed rows" warning in `mark_training_ready`.
- Added tests for the new error paths.

### PUK-11: Model improvements document
- Wrote the model improvements and design changes document, including the gibberish-export bug and how it was fixed.

### Running and checking the pipeline
- Ran the full Docker and Airflow pipeline end to end on my laptop (31,443 documents, 38,105 chunks) and wrote the run plan from what worked and what did not.

### Course deliverables (not stored in this repo)
- Workbook 1 (Team 2 workbook) and the Demo 1 slide deck. I checked every number in the deck against the experiment records.

## Linear cards assigned to me where the code was written by teammates

These cards are assigned to me in Linear. The implementation commits are by Kevin Gao, so I list them here only for the record and do not claim the code.

| Linear card | Where the code is |
|---|---|
| Persona extraction pipeline and persona pivot (patio11 to Jason Lemkin) | Kevin Gao and team commits on the integration branches |
| DATA298 presentation notebook (Data Notebook v1) | `docs/DATA298_PRESENTATION_NOTEBOOK.ipynb`, commit by Kevin Gao |
| Data scraping: X/Twitter and YouTube transcript collectors | `scrape_transcripts.py` and the scraper branch, commits by Kevin Gao |

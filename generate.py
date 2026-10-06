#!/usr/bin/env python3
"""
Generate PG-style content from a format (linkedin_post, blog_draft), content brief,
and RAG retrieval. Uses Ollama /api/embed and /api/generate.

  python generate.py --format linkedin_post --topic "..." --audience "..." --goal "..." --cta "..." --k 8 --out outputs/pg_linkedin_001.md
  python generate.py --format blog_draft --topic "..." --audience "..." --goal "..." --k 10 --out outputs/pg_blog_001.md
  python generate.py --format x_thread --topic "..." --audience "..." --goal "..." --k 8 --out outputs/thread_001.md
  python generate.py --format youtube_script --topic "..." --audience "..." --goal "..." --k 10 --out outputs/script_001.md
"""
import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

import requests
import yaml
import chromadb
from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT, Settings
from chromadb.errors import NotFoundError

_SJ = Path(__file__).resolve().parent / "spark_jobs"
if str(_SJ) not in sys.path:
    sys.path.insert(0, str(_SJ))
from corpus_footer_scrub import strip_footer_noise
from output_checks import looks_degenerate


OLLAMA_BASE_DEFAULT = "http://localhost:11434"
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "lemkin_content")
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_GEN_MODEL = "llama3.1"
# CPU-only demo instances need a compact RAG prompt: enough evidence for grounding,
# without several minutes of prompt evaluation before any draft is produced.
CONTEXT_MAX_CHARS = 2400

# Ollama /api/generate can exceed 5 min on cold model load + long prompts (Docker → host).
_GEN_TIMEOUT = float(os.environ.get("OLLAMA_GENERATE_TIMEOUT", "900"))
_EMBED_TIMEOUT = float(os.environ.get("OLLAMA_EMBED_TIMEOUT", "60"))
# Extra attempts (not total) for transient Ollama failures; the delay doubles each try.
_EMBED_RETRIES = int(os.environ.get("OLLAMA_EMBED_RETRIES", "2"))
_GEN_RETRIES = int(os.environ.get("OLLAMA_GENERATE_RETRIES", "1"))
_RETRY_BASE_SEC = float(os.environ.get("OLLAMA_RETRY_BASE_SEC", "1.5"))

logger = logging.getLogger("lemkin.rag")


def configure_logging() -> None:
    """Send ``lemkin.*`` logs to stderr once (level from ``LOG_LEVEL``, default INFO).

    Used by the CLI and Streamlit. In Docker the stream lands in ``docker compose logs``.
    """
    log = logging.getLogger("lemkin")
    if getattr(log, "_lemkin_configured", False):
        return
    log.setLevel(getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO))
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    log.addHandler(handler)
    log.propagate = False
    log._lemkin_configured = True  # type: ignore[attr-defined]


# ── Errors ────────────────────────────────────────────────────────────────
# Expected failures carry a ``hint`` saying how to fix them, so the CLI and Streamlit
# can show something actionable instead of a raw traceback. They subclass RuntimeError
# so existing ``except RuntimeError`` / ``except Exception`` handlers keep working.
class LemkinError(RuntimeError):
    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


class ConfigError(LemkinError):
    """Missing or unreadable local files (format specs, persona profile)."""


class OllamaError(LemkinError):
    """Ollama returned an error response."""


class OllamaUnavailableError(OllamaError):
    """Ollama is unreachable or timed out."""


class ModelNotFoundError(OllamaError):
    """The requested Ollama model is not pulled / registered."""


class EmbeddingError(LemkinError):
    """Ollama answered but the embedding payload was unusable."""


class RetrievalError(LemkinError):
    """Chroma is unreachable, empty, or incompatible with the query embedding."""


class GenerationError(LemkinError):
    """The model returned nothing usable (empty or degenerate output)."""


def load_format_spec(path: Path, format_name: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            specs = yaml.safe_load(f)
    except OSError as e:
        raise ConfigError(f"Cannot read format specs {path}: {e}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"Format specs {path} is not valid YAML: {e}") from e
    if not isinstance(specs, dict) or format_name not in specs:
        available = list(specs.keys()) if isinstance(specs, dict) else []
        raise ConfigError(f"Unknown format: {format_name}. Available: {available}")
    return specs[format_name]


def load_persona(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            profile = json.load(f)
    except OSError as e:
        raise ConfigError(
            f"Cannot read persona profile {path}: {e}",
            hint="Run the Airflow DAG through extract_persona, or copy persona_profile.json into ./data.",
        ) from e
    except json.JSONDecodeError as e:
        raise ConfigError(
            f"Persona profile {path} is not valid JSON: {e}",
            hint="Re-run the Airflow DAG task extract_persona to regenerate it.",
        ) from e
    if not isinstance(profile, dict):
        raise ConfigError(f"Persona profile {path} must be a JSON object.")
    return profile


def summarize_persona(profile: dict) -> str:
    vocab = profile.get("vocabulary_tendencies", {})
    structure = profile.get("structure_patterns", {})
    framing = profile.get("favorite_framing", {})
    themes = profile.get("favorite_themes", {})
    meta = profile.get("meta", {})
    top_words = ", ".join(vocab.get("top_50_content_words", [])[:15])
    starters = ", ".join(structure.get("common_sentence_starters_bigrams", [])[:8])
    framing_phrases = ", ".join(framing.get("favorite_framing_phrases", [])[:10])
    theme_list = themes.get("top_content_words_and_bigrams", [])[:10]
    theme_examples = ", ".join(str(t) for t in theme_list) if theme_list else "B2B SaaS, GTM, fundraising"
    source_hint = meta.get("source", "Jason Lemkin — operator voice")
    lines = [
        f"You are writing in the persona of {source_hint}.",
        "Emulate: direct, experienced B2B / SaaS operator tone; concrete metrics and examples.",
        f"Vocabulary tendencies: {top_words}.",
        f"Sentence openers to echo (sparingly): {starters}.",
        f"Framing phrases: {framing_phrases}.",
        f"Themes to lean on: {theme_examples}.",
        "Favor clarity and specificity over generic hype.",
    ]
    return "\n".join(lines)


def format_spec_to_prompt(spec: dict, heading: str = "Format requirements") -> str:
    lines = [heading]
    lines.append(f"Output type: {spec.get('output_type', 'markdown')}.")
    if "length_target_words" in spec:
        lo, hi = spec["length_target_words"][0], spec["length_target_words"][1]
        lines.append(f"Target length: {lo}–{hi} words.")
    if "structure" in spec:
        lines.append("Structure:")
        for item in spec["structure"]:
            if isinstance(item, dict):
                for k, v in item.items():
                    lines.append(f"  - {k}: {v}")
            else:
                lines.append(f"  - {item}")
    if "banned" in spec:
        lines.append("Avoid: " + "; ".join(spec["banned"]))
    if "style" in spec:
        lines.append("Style: " + "; ".join(spec["style"]))
    return "\n".join(lines)


def _error_detail(resp: requests.Response) -> str:
    """Best-effort text of an Ollama error body (``{"error": "..."}`` or raw text)."""
    try:
        return str(resp.json().get("error") or resp.text)[:300]
    except (ValueError, AttributeError):
        return (resp.text or "")[:300]


def _ollama_post(
    path: str,
    payload: dict,
    base_url: str,
    model: str,
    timeout: float,
    retries: int,
    what: str,
) -> requests.Response:
    """POST to Ollama, mapping failures to typed errors and retrying transient ones."""
    url = f"{base_url.rstrip('/')}{path}"
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        cause: Exception | None = None
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
        except requests.exceptions.Timeout as e:
            cause = e
            err: LemkinError = OllamaUnavailableError(
                f"Ollama {what} request to {base_url} timed out after {timeout:.0f}s.",
                hint=(
                    "The model may still be loading (cold start) or the prompt is very long. "
                    "Raise OLLAMA_GENERATE_TIMEOUT / OLLAMA_EMBED_TIMEOUT, or retry once it is loaded."
                ),
            )
            # A timed-out generation has already waited up to _GEN_TIMEOUT; do not repeat it silently.
            transient = what == "embed"
        except requests.exceptions.ConnectionError as e:
            cause = e
            err = OllamaUnavailableError(
                f"Cannot connect to Ollama at {base_url}.",
                hint=(
                    "Start it with `ollama serve`. From Docker the URL must be "
                    "http://host.docker.internal:11434 (OLLAMA_BASE); on the host use "
                    "http://localhost:11434."
                ),
            )
            transient = True
        else:
            if resp.status_code < 400:
                return resp
            detail = _error_detail(resp)
            if resp.status_code == 404 and "not found" in detail.lower():
                raise ModelNotFoundError(
                    f"Ollama model '{model}' not found ({detail}).",
                    hint=(
                        f"Run `ollama pull {model}`. For the fine-tune, run "
                        "`python -m host_finetune.register_ollama` and check `ollama list`."
                    ),
                )
            err = OllamaError(
                f"Ollama {what} returned HTTP {resp.status_code} from {url}: {detail}",
                hint="Check the Ollama server log. HTTP 5xx is retried automatically.",
            )
            transient = resp.status_code >= 500
        if not transient or attempt == attempts:
            raise err from cause
        delay = _RETRY_BASE_SEC * 2 ** (attempt - 1)
        logger.warning(
            "Ollama %s attempt %d/%d failed (%s); retrying in %.1fs",
            what, attempt, attempts, err, delay,
        )
        time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def ollama_embed(text: str, base_url: str, model: str) -> list[float]:
    resp = _ollama_post(
        "/api/embed", {"model": model, "input": text},
        base_url, model, _EMBED_TIMEOUT, _EMBED_RETRIES, "embed",
    )
    try:
        vec = resp.json()["embeddings"][0]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise EmbeddingError(
            f"Ollama returned an unusable /api/embed response for model '{model}'.",
            hint="Make sure the embed model is an embedding model, e.g. `ollama pull nomic-embed-text`.",
        ) from e
    if not vec:
        raise EmbeddingError(f"Ollama returned an empty embedding for model '{model}'.")
    return vec


def ollama_generate(
    prompt: str,
    base_url: str,
    model: str,
    *,
    max_tokens: int | None = None,
    options: dict[str, object] | None = None,
    keep_alive: int | str | None = None,
) -> str:
    payload: dict[str, object] = {"model": model, "prompt": prompt, "stream": False}
    request_options = dict(options or {})
    if max_tokens is not None:
        request_options["num_predict"] = max_tokens
    if request_options:
        payload["options"] = request_options
    # Ollama unloads a model after a few minutes unless the request says otherwise.
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    resp = _ollama_post(
        "/api/generate", payload,
        base_url, model, _GEN_TIMEOUT, _GEN_RETRIES, "generate",
    )
    try:
        return (resp.json().get("response") or "").strip()
    except (ValueError, AttributeError) as e:
        raise GenerationError(
            f"Ollama returned invalid JSON from /api/generate for model '{model}'.",
            hint="Check the Ollama server log; restarting `ollama serve` usually clears this.",
        ) from e


def ollama_chat(
    user_message: str,
    base_url: str,
    model: str,
    *,
    options: dict[str, object] | None = None,
    keep_alive: int | str | None = None,
) -> str:
    """Generate one assistant reply through Ollama's chat endpoint.

    This deliberately sends *only* a user message.  The Relabel QLoRA was
    evaluated with Llama 3.1's own chat template and no application-provided
    system prompt; adding one changes the training distribution.
    """
    payload: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": user_message}],
        "stream": False,
    }
    if options:
        payload["options"] = dict(options)
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    resp = _ollama_post(
        "/api/chat", payload,
        base_url, model, _GEN_TIMEOUT, _GEN_RETRIES, "chat",
    )
    try:
        return ((resp.json().get("message") or {}).get("content") or "").strip()
    except (ValueError, AttributeError) as e:
        raise GenerationError(
            f"Ollama returned invalid JSON from /api/chat for model '{model}'.",
            hint="Check the Ollama server log; restarting `ollama serve` usually clears this.",
        ) from e


def validate_draft(draft: str, gen_model: str) -> str:
    """Reject empty or collapsed model output instead of showing it as a result.

    Kept out of ``ollama_generate`` on purpose: eval_rag also uses that as a judge call
    and wants to score bad outputs, not raise on them.
    """
    text = (draft or "").strip()
    if not text:
        raise GenerationError(
            f"Model '{gen_model}' returned an empty response.",
            hint=(
                "Try again, retrieve fewer chunks, or pick another model. For a fine-tuned "
                "model run `python -m host_finetune.diagnose_pipeline`."
            ),
        )
    if looks_degenerate(text):
        logger.warning("degenerate output from %s: %r", gen_model, text[:200])
        raise GenerationError(
            f"Model '{gen_model}' produced degenerate output (a token repeated over and over).",
            hint=(
                "This matches the known broken-GGUF failure. Re-export with "
                "`python -m host_finetune.merge_and_export`, re-register with "
                "`python -m host_finetune.register_ollama`, or use the stock `llama3.1` model. "
                "`python -m host_finetune.diagnose_pipeline` localises the fault."
            ),
        )
    return text


def _chroma_target() -> str:
    if os.environ.get("CHROMA_USE_HTTP", "").lower() in ("1", "true", "yes"):
        return f"{os.environ.get('CHROMA_HOST', 'localhost')}:{os.environ.get('CHROMA_PORT', '8000')}"
    return "local persistent index"


def _chroma_connect_error(err: Exception) -> RetrievalError:
    msg = str(err)
    target = _chroma_target()
    if "tenant" in msg.lower():
        return RetrievalError(
            f"Chroma at {target} rejected the tenant/database: {msg}",
            hint=(
                "The Chroma server image and pip `chromadb` must be the same version (pinned to 1.5.5), "
                "and a chroma-data volume from an older layout must be recreated: "
                "`docker compose down`, `docker volume rm <project>_chroma-data`, `docker compose up -d`, "
                "then re-run load_to_chroma in Airflow. See README 'Chroma default_tenant'."
            ),
        )
    return RetrievalError(
        f"Cannot connect to Chroma ({target}): {msg}",
        hint=(
            "Check `docker compose ps chroma`. CHROMA_HOST/CHROMA_PORT are chroma:8000 inside Docker "
            "and localhost:8000 from the host."
        ),
    )


def _get_collection(index_path: Path):
    """Open the Chroma collection or raise ``RetrievalError`` with a fix hint."""
    try:
        if os.environ.get("CHROMA_USE_HTTP", "").lower() in ("1", "true", "yes"):
            client = chromadb.HttpClient(
                host=os.environ.get("CHROMA_HOST", "localhost"),
                port=int(os.environ.get("CHROMA_PORT", "8000")),
                tenant=os.environ.get("CHROMA_TENANT", DEFAULT_TENANT),
                database=os.environ.get("CHROMA_DATABASE", DEFAULT_DATABASE),
            )
        else:
            client = chromadb.PersistentClient(
                path=str(index_path), settings=Settings(anonymized_telemetry=False)
            )
        return client.get_collection(COLLECTION_NAME)
    except NotFoundError as err:
        try:
            existing = [c.name for c in client.list_collections()]
        except Exception:
            existing = []
        hint = (
            f"Chroma has no collection {COLLECTION_NAME!r}. "
            "In Airflow, run the DAG through **load_to_chroma** (MinIO → Chroma). "
            "If you removed the **chroma-data** Docker volume or upgraded Chroma, reload once."
        )
        if existing:
            hint += f" Existing collections: {existing}."
        raise RetrievalError(f"Collection {COLLECTION_NAME!r} does not exist.", hint=hint) from err
    except Exception as err:
        raise _chroma_connect_error(err) from err


def retrieve(index_path: Path, query_embedding: list[float], k: int, embed_model: str) -> list[dict]:
    collection = _get_collection(index_path)
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas"],
        )
    except Exception as err:
        if "dimension" in str(err).lower():
            raise RetrievalError(
                f"Query embedding ({len(query_embedding)} dims) does not match the collection: {err}",
                hint=(
                    f"The collection was built with a different embedding model than '{embed_model}'. "
                    "Use the same model as the Spark job (nomic-embed-text) or rebuild the collection."
                ),
            ) from err
        raise RetrievalError(
            f"Chroma query failed: {err}",
            hint="Check `docker compose logs chroma`.",
        ) from err
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    return [{"chunk_text": d, **m} for d, m in zip(docs, metas)]


def _ollama_has_model(tags: set[str], model: str) -> bool:
    return model in tags or (":" not in model and f"{model}:latest" in tags)


def check_dependencies(
    ollama_base: str,
    embed_model: str,
    gen_model: str,
    persona_path: Path,
    format_specs_path: Path,
    index_path: Path,
) -> list[tuple[str, bool, str]]:
    """Preflight for a deployment: ``[(check, ok, detail)]``. Never raises.

    Lets Streamlit and the CLI report "Chroma is empty" or "model not pulled" up front
    instead of failing in the middle of a generation.
    """
    results: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        results.append((name, ok, detail))
        (logger.info if ok else logger.warning)(
            "preflight %s: %s (%s)", name, "ok" if ok else "FAILED", detail
        )

    add("format specs", format_specs_path.is_file(), str(format_specs_path))
    add("persona profile", persona_path.is_file(), str(persona_path))

    tags: set[str] | None = None
    try:
        r = requests.get(f"{ollama_base.rstrip('/')}/api/tags", timeout=5)
        r.raise_for_status()
        tags = {(m.get("name") or "") for m in r.json().get("models") or []}
        add("ollama", True, f"{len(tags)} models at {ollama_base}")
    except Exception as e:  # noqa: BLE001 - report, never raise
        add("ollama", False, f"cannot reach {ollama_base}: {e}")
    if tags is not None:
        for label, model in (("embed model", embed_model), ("generate model", gen_model)):
            ok = _ollama_has_model(tags, model)
            add(label, ok, model if ok else f"'{model}' not in `ollama list`")

    try:
        count = _get_collection(index_path).count()
        add("chroma collection", count > 0, f"{COLLECTION_NAME}: {count} chunks")
    except LemkinError as e:
        add("chroma collection", False, str(e))
    return results


def format_context(chunks: list[dict], max_chars: int = CONTEXT_MAX_CHARS) -> str:
    lines = ["Reference excerpts from the Jason Lemkin corpus (tone and ideas only; do not copy verbatim):"]
    total = 0
    for i, ch in enumerate(chunks):
        text = strip_footer_noise(str(ch.get("chunk_text", "")).strip())
        if not text:
            continue
        doc_id = ch.get("doc_id", "?")
        header = f"[Chunk {i+1} — doc_id={doc_id}]"
        if total + len(header) + len(text) + 2 > max_chars:
            remaining = max_chars - total - len(header) - 5
            if remaining <= 0:
                break
            text = text[:remaining] + "..."
        lines.append(header)
        lines.append(text)
        total += len(header) + len(text) + 2
        if total >= max_chars:
            break
    return "\n".join(lines)


def build_prompt(
    format_spec: dict,
    persona_summary: str,
    topic: str,
    audience: str,
    goal: str,
    cta: str,
    context_block: str,
) -> str:
    """Single user prompt for Ollama: matches the fine-tune ``instruction`` prefix.

    Training rows look like
    ``Write in the style of Jason Lemkin about: <title>`` — keep that opening so
    the Modelfile ``### Instruction:`` block aligns with SFT.
    """
    title = topic.strip() or "the topic below"
    instruction_head = f"Write in the style of Jason Lemkin about: {title}."
    brief = "\n".join([
        "Content brief:",
        f"- Audience: {audience.strip()}",
        f"- Goal: {goal.strip()}",
        f"- Call to action: {cta.strip() or 'none'}",
    ])
    format_block = format_spec_to_prompt(format_spec)
    voice = "Voice and persona (emulate this tone):\n" + persona_summary.strip()
    closing = (
        "Write the draft now: follow the format requirements and content brief. "
        "Use the reference excerpts for tone and ideas only; do not copy long passages verbatim. "
        "Do not include link-recirculation blocks (e.g. 'Related Posts'), raw tweet embeds, or newsletter footers. "
        "Output only the draft in markdown, with no preamble or meta-commentary."
    )
    return "\n\n".join(
        [
            instruction_head,
            voice,
            format_block,
            brief,
            context_block.strip(),
            closing,
        ]
    )


def run_rag_pipeline(
    *,
    format_name: str,
    topic: str,
    audience: str,
    goal: str,
    cta: str,
    k: int,
    index_path: Path,
    persona_path: Path,
    format_specs_path: Path,
    ollama_base: str,
    embed_model: str,
    gen_model: str,
) -> str:
    """Embed → retrieve → prompt → generate → validate. Shared by the CLI and Streamlit.

    Every run gets a short request id so its log lines can be grepped together, e.g.
    ``docker compose logs streamlit | grep a1b2c3d4``.
    """
    request_id = uuid.uuid4().hex[:8]
    started = time.perf_counter()
    logger.info(
        "[%s] generate start format=%s gen_model=%s embed_model=%s k=%d topic=%.80r",
        request_id, format_name, gen_model, embed_model, k, topic,
    )
    try:
        format_spec = load_format_spec(format_specs_path, format_name)
        persona_summary = summarize_persona(load_persona(persona_path))

        t = time.perf_counter()
        query_embed = ollama_embed(f"{topic}. {goal}. Audience: {audience}.", ollama_base, embed_model)
        logger.info("[%s] embedded query in %.2fs (dim=%d)", request_id, time.perf_counter() - t, len(query_embed))

        t = time.perf_counter()
        chunks = retrieve(index_path, query_embed, k, embed_model)
        if not chunks:
            raise RetrievalError(
                f"Chroma collection {COLLECTION_NAME!r} returned no chunks.",
                hint="The collection is empty. In Airflow, run the DAG through load_to_chroma.",
            )
        logger.info("[%s] retrieved %d chunks in %.2fs", request_id, len(chunks), time.perf_counter() - t)

        prompt = build_prompt(
            format_spec, persona_summary, topic, audience, goal, cta,
            format_context(chunks, max_chars=CONTEXT_MAX_CHARS),
        )
        logger.info("[%s] generating with %s (prompt=%d chars)", request_id, gen_model, len(prompt))
        t = time.perf_counter()
        draft = validate_draft(ollama_generate(prompt, ollama_base, gen_model), gen_model)
        logger.info("[%s] generated %d chars in %.2fs", request_id, len(draft), time.perf_counter() - t)
    except LemkinError as e:
        logger.error(
            "[%s] generation failed after %.2fs (%s): %s | hint: %s",
            request_id, time.perf_counter() - started, type(e).__name__, e, e.hint or "-",
        )
        raise
    except Exception:
        logger.exception("[%s] unexpected error during generation", request_id)
        raise
    logger.info("[%s] done in %.2fs", request_id, time.perf_counter() - started)
    return draft


def main():
    configure_logging()
    p = argparse.ArgumentParser(description="Generate Lemkin-style B2B content via RAG + Ollama.")
    p.add_argument("--format", required=True, help="Format name (e.g. linkedin_post, blog_draft)")
    p.add_argument("--topic", required=True, help="Topic or title")
    p.add_argument("--audience", required=True, help="Target audience")
    p.add_argument("--goal", required=True, help="Goal of the piece")
    p.add_argument("--cta", default="none", help="Call to action (default: none)")
    p.add_argument("--k", type=int, default=8, help="Number of chunks to retrieve (default: 8)")
    p.add_argument("--out", required=True, help="Output markdown file path")
    p.add_argument("--index", default="data/index", help="Chroma index directory (default: data/index)")
    p.add_argument("--persona", default="data/persona_profile.json", help="Persona JSON path")
    p.add_argument("--format-specs", default="formats/format_specs.yaml", help="Format specs YAML path")
    p.add_argument("--ollama-base", default=OLLAMA_BASE_DEFAULT, help="Ollama base URL")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Ollama embed model")
    p.add_argument("--gen-model", default=DEFAULT_GEN_MODEL, help="Ollama generate model")
    args = p.parse_args()

    base = Path(".")
    format_specs_path = base / args.format_specs
    persona_path = base / args.persona
    index_path = base / args.index
    out_path = base / args.out

    if not format_specs_path.exists():
        print(f"Error: format specs not found: {format_specs_path}", file=sys.stderr)
        sys.exit(1)
    if not persona_path.exists():
        print(f"Error: persona profile not found: {persona_path}", file=sys.stderr)
        sys.exit(1)
    use_http = os.environ.get("CHROMA_USE_HTTP", "").lower() in ("1", "true", "yes")
    if not use_http and not index_path.exists():
        print(f"Error: index not found: {index_path}. Run build_index.py first.", file=sys.stderr)
        sys.exit(1)

    try:
        draft = run_rag_pipeline(
            format_name=args.format,
            topic=args.topic,
            audience=args.audience,
            goal=args.goal,
            cta=args.cta,
            k=args.k,
            index_path=index_path,
            persona_path=persona_path,
            format_specs_path=format_specs_path,
            ollama_base=args.ollama_base,
            embed_model=args.embed_model,
            gen_model=args.gen_model,
        )
    except LemkinError as e:
        print(f"Error: {e}", file=sys.stderr)
        if e.hint:
            print(f"Hint: {e.hint}", file=sys.stderr)
        sys.exit(1)

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(draft, encoding="utf-8")
    except OSError as e:
        print(f"Error: cannot write {out_path}: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

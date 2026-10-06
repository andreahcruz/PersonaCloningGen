"""lemkin-relabel medium grid. One template per medium; only the topic changes.

Gold topics are the first 15 titles from the experiment gold file.
Casual topics are ordinary SaaS subjects, not article headlines.
Rules for LinkedIn and X come before the topic so the model does not copy them.
"""
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "openai_ft" / "lemkin_gold_eval.jsonl"
OUT = Path(__file__).with_name("medium_grid_out.json")
MODEL = "lemkin-relabel"

TEMPLATES = {
    "blog": (
        "Write a blog post in the style of Jason Lemkin about {topic}. "
        "Write the post itself in about 250 words, with no title line.",
        500,
        1.1,
    ),
    "talk": (
        "Write a spoken talk in the style of Jason Lemkin about {topic}. "
        "Use short spoken sentences to founders, about 160 words, with no title and no headings.",
        400,
        1.15,
    ),
    "linkedin": (
        "About 80 words. The first line is a claim, not a headline. No hashtags. "
        "Write a LinkedIn post in the style of Jason Lemkin about {topic}.",
        320,
        1.1,
    ),
    "x": (
        "Write one X post in the style of Jason Lemkin. "
        "Exactly 3 sentences, under 60 words, no headline, and no hashtags. "
        "Topic: {topic}",
        120,
        1.1,
    ),
}

CASUAL = [
    "usage based pricing",
    "hiring a first AE",
    "churn that shows up in month three",
    "firing a VP who is nice but missing the number",
    "board meetings that eat the whole monday",
    "free pilots that never turn into paid",
    "reps missing quota two quarters in a row",
    "whether a bridge round is a bad sign",
    "buyers who only want the AI feature",
    "onboarding that drags on for six weeks",
    "selling against a bigger suite",
    "the founder still running every demo",
    "retention slipping after the first renewal",
    "annual deals versus monthly",
    "a sales team that will not prospect",
]


def gold_topics(n: int = 15) -> list[str]:
    topics = []
    with GOLD.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            topics.append(json.loads(line)["topic"])
            if len(topics) == n:
                break
    if len(topics) != n:
        raise RuntimeError(f"expected {n} gold topics, found {len(topics)}")
    return topics


def generate(prompt: str, budget: int, penalty: float) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 42,
            "top_p": 0.9,
            "repeat_penalty": penalty,
            "num_ctx": 4096,
            "num_predict": budget,
        },
    }
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    answer = ((data.get("message") or {}).get("content") or "").strip()
    return {
        "answer": answer,
        "words": len(answer.split()),
        "tokens": data.get("eval_count"),
        "stop": data.get("done_reason"),
    }


def main() -> None:
    sets = [("gold", gold_topics()), ("casual", CASUAL)]
    rows = []
    for kind, topics in sets:
        for index, topic in enumerate(topics, start=1):
            for medium, (template, budget, penalty) in TEMPLATES.items():
                prompt = template.format(topic=topic)
                print(f"{kind} {index:02d} {medium}", flush=True)
                got = generate(prompt, budget, penalty)
                rows.append({
                    "kind": kind,
                    "topic": topic,
                    "medium": medium,
                    "prompt": prompt,
                    "num_predict": budget,
                    "repeat_penalty": penalty,
                    **got,
                })
                print(f"  words={got['words']} stop={got['stop']}", flush=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT} n={len(rows)}")


if __name__ == "__main__":
    main()

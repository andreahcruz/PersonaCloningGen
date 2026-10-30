import json
from pathlib import Path

raw = Path(__file__).resolve().parents[1] / "raw"
for name in ("generation_naive_parent_ids.json", "generation_mix_parent_ids.json"):
    payload = json.loads((raw / name).read_text(encoding="utf-8"))
    print("FILE", name)
    for trace in payload["traces"]:
        answer = trace.get("answer") or ""
        rejected = trace.get("validation_rejected") or []
        reasons = sorted({item.get("reason") for item in rejected if isinstance(item, dict)})
        print(
            trace["id"],
            trace["medium"],
            "verified",
            len(trace.get("verified_claims") or []),
            "rejected",
            len(rejected),
            reasons,
            "anneka",
            "anneka" in answer.casefold(),
            "5.15",
            "5.15" in answer,
        )
        if trace["medium"] == "blog":
            print(answer[:500].replace("\n", " "))
            print("---")

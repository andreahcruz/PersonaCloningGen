"""Measure what the LanguageTool profile catches, on the 40 ordinary grammar mistakes of eval_tools.grammar_audit.

Run it from the repository root, with Java 17+ and `pip install language_tool_python` available:

    python docs/evaluation/examples/data/grammar_lt/audit_languagetool.py

It compares three checkers on the same sentences: the rule based check in eval_tools.grammar, LanguageTool with
its default rules, and LanguageTool with the profile in eval_tools.grammar_lt. It prints the numbers and writes
`coverage_audit.json` next to this file. About 4 minutes, almost all of it spent counting how often each word
appears in 1,200 real posts (the vocabulary used to avoid calling Lemkin's own words typos).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))

from eval_tools import grammar, grammar_audit, grammar_lt  # noqa: E402


def main() -> None:
    texts = grammar.reference_texts(str(ROOT / "data"), 400, 0)
    vocabulary = grammar_lt.build_vocabulary(t for medium_texts in texts.values() for t in medium_texts)
    with grammar_lt.Checker(vocabulary) as checker:
        profile = grammar_audit.coverage(check=checker.check_text)
        default = grammar_audit.coverage(check=lambda s: checker.tool.check(grammar.clean_for_checking(s)))
    rule_based = grammar_audit.coverage()
    for name, result in (("rule based eval_tools.grammar", rule_based), ("LanguageTool, default rules", default), ("LanguageTool, our profile", profile)):
        print(f"{name:32s} ordinary mistakes caught {result['broad_caught']}/{result['broad_total']} ({result['broad_share']:.0%}); "
              f"rule-targeted sentences {result['rule_targeted_caught']}/{result['rule_targeted_total']}")
    (HERE / "coverage_audit.json").write_bytes(json.dumps({"rule_based": rule_based, "languagetool_default": default, "languagetool_profile": profile}, indent=1).encode("utf-8"))


if __name__ == "__main__":
    main()

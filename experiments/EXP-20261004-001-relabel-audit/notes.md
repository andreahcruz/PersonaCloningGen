# Failed read-only audit

No inputs or training artifacts were modified. JSONL reading failed because splitlines treated a Unicode separator inside a JSON string as a new record. This attempt is retained with FAILED status. The physical-newline reader fix and regression test precede EXP-20261004-002. No metrics were produced by this attempt.

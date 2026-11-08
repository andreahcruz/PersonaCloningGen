# Incomplete TASK-002 exposure-review attempt

This CPU-only audit exited 1 because the new log reader decoded the historical UTF-16 training log as UTF-8 with replacement characters. The completion-evidence check therefore returned false. The numeric dataset membership, literal-text match counts, candidate checks, and retrieval parity were unaffected. Do not use this attempt's negative completion inference as a finding about training.

The failure is an implementation defect in this new audit, not a defect in the frozen training run. Preserve this attempt's manifest, raw outputs, and metrics. The corrected reader detects UTF-16 BOMs and has regression tests for UTF-16, UTF-8 BOM, and UTF-8. The user approved a separate corrected run EXP-20261108-003-gold-exposure-review; it supersedes this attempt for the completion inference without replacing these artifacts.

No model was trained, evaluated, or loaded, and no frozen source, split, index, model, or historical experiment was modified.

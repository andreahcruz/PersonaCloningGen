# Unified rescore

Question: on the saved drafts, which models pass completion, copying, format, and applicable grounding gates, and how do the separate persona, perspective, writing, task, and diagnostic scores look?

Held fixed: saved answer text, prompts, splits, and adapters. No model was loaded and no source experiment summary was overwritten.

Writing quality and BERTScore were not recomputed because those judge and embedding outputs are not in the saved traces. Persona utility stays empty until writing quality is calibrated. Perspective fidelity is framing-rate distance against held-out Lemkin posts, not claim entailment.

# EXP-20261001-009 — Repaired QLoRA

Question: after the fit keeps newlines and attaches a colon lead-in to the next chunk of the same document, does a rank-16 QLoRA on that balanced file still train on the 5070 Ti?

Held fixed: rank 16, alpha 32, learning rate 1e-4, 2 epochs, batch 1, gradient accumulation 16, AdamW 8-bit, VRAM cap 0.85. Dataset `host_finetune/data/dataset_fit512_keepbreaks_balanced.jsonl`. Split `experiments/EXP-20261001-008-repaired-balanced-split`. Adapter `host_finetune/output/lemkin_lora_repaired`. The earlier adapters were not overwritten.

## Result

Best validation loss 1.5311 at step 3134. Train loss 1.954. Runtime about 3.9 hours. This validation set is the repaired split, so 1.5311 is not comparable to 1.5753 from the previous balanced file. Per-medium validation losses were not saved; the extra forward pass ran out of memory and training kept `eval_loss` only, as in the earlier runs.

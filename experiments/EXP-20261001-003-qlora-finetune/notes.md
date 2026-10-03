# EXP-20261001-003 — QLoRA fine-tune

Training run on `dataset_from_cleaned_sources_fit512.jsonl` with the medium-stratified split from EXP-20261001-002. Two epochs. Optimizer is `adamw_8bit` so Adam state stays on the 5070 Ti. The first launch used batch 2 and hit CUDA out of memory at step 35. The restart uses batch 1 and gradient accumulation 16, so the effective batch stays 16, with the process capped at 85% of the card.

Checkpoints: a resume checkpoint every 400 steps (three kept), plus a kept adapter whenever `eval_loss`, a per-medium loss, or the macro average of blog / LinkedIn / X / talk loss sets a new low. The average-best folder is the model to deploy.

The previous `host_finetune/output/lemkin_lora` adapter is not overwritten. This run writes `host_finetune/output/lemkin_lora_cleaned`.

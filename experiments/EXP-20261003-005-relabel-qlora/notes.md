# Relabel QLoRA

Question: if the only change from the repaired rank-16 run is the EXP-20261003-004 training file, do early `<|eot_id|>` stops and mid-sentence stops fall?

Held fixed: Llama 3.1 8B 4-bit NF4, LoRA rank 16, alpha 32, learning rate 1e-4, seed 42, 2 epochs, batch 1, gradient accumulation 16, sequence length 512, AdamW 8-bit, VRAM cap 0.85. The repaired adapter at `host_finetune/output/lemkin_lora_repaired` is not overwritten. This run writes `host_finetune/output/lemkin_lora_relabel`.

Dataset sha256 `1231835ecc5dcb608fefa0357d325affb2afbdc3aeade2a63b1dc42b2920897b`. Split manifest sha256 `6e31aedf33bbc81488641e3a2d0d284199bd7c3e724d97d27e801a37010ea7fb`. Train 21,377, validation 2,588, test held out 2,580.

The first launch exited before the first step. Unsloth printed a message the Windows cp1252 console could not encode. The relaunch sets `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8`. Hyperparameters are unchanged.

The relaunch exited 0 after 12,167 seconds (about 3.4 hours). Final train loss 1.797. Best eval loss 1.3316 at step 2,674, the last step. Eval loss fell at every checkpoint: 1.4016, 1.3645, 1.3421, 1.3410, 1.3342, 1.3317, 1.3316. The per-medium training eval ran out of memory at each checkpoint and was skipped; training continued on eval loss only. That is the same skip the repaired run recorded. This eval loss is not comparable to the repaired adapter’s 1.5311, because the validation rows and their prompts are different.

The saved adapter is `host_finetune/output/lemkin_lora_relabel`. Generation comparison against the repaired adapter has not been run. Decision metrics are the share of answers that emit `<|eot_id|>` before half of `max_new_tokens`, the mid-sentence stop rate, and the Lemkin voice scores in `host_finetune/eval_rag.py`. BLEU and ROUGE are logged and do not decide the run.

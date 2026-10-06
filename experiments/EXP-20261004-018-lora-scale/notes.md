# EXP-20261004-018 — LoRA scale curve

Question: can a lower inference-time LoRA contribution keep Lemkin voice and stop the unsupported quantities that full-strength Relabel adds?

Held fixed: the EXP-017 prompts, accepted claims, seeds, decoding, and both gates. Repair was not run. The adapter was not retrained or merged.

Change: `LoraLayer.set_scale` on the loaded 4-bit model, then restored to the trained scale after every sample. Scales 0.75, 0.50, and 0.25 were generated. Scale 1.00 is the saved Relabel draw. The two base blogs were redrawn with the adapter off so voice could be scored; both matched the saved diagnostic word counts.

Result: no intermediate scale is deployable. Copying stayed clean. Numeric failures were 4, 4, 3, and 1 cells at 1.00, 0.75, 0.50, and 0.25. The 0.25 drafts that passed are closer to base voice than to Relabel, and gold_001 blog at 0.25 still invented quantities and scored farther from Lemkin than base. The 8-cell sanity and the full 120 were not started.

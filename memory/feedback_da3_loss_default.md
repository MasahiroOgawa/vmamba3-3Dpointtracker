---
name: Default loss form is the original DA3 paper loss, never Kendall-Gal
description: Always use the original DA3 aleatoric form `c·|err| − λ·log(c)` as the default; only opt into Kendall-Gal log-scale when the user explicitly asks.
type: feedback
---

The original DA3 paper uses the aleatoric form `c·|err| − λ·log(c)`. Kendall-Gal log-scale was experimentally introduced in §15.59.2 (commit `6ace4ee`) and reverted in §15.59.3 (commit `d9b71fe revert(loss): default to original DA3 loss`). Never re-introduce Kendall-Gal as a default.

**Why:** §15.59.8 was launched without `--no-kendall-gal` because the CLI in `train_super.py` had Kendall-Gal as an opt-out default even after the d9b71fe revert (only the dataclass default was updated, the CLI override `cfg.weights.use_kendall_gal = not args.no_kendall_gal` flipped it back). The whole §15.59.8 narrative (negative L_M, "memorisation collapse") was an artefact of the Kendall-Gal log-scale unbounded-below behaviour, not the architecture. Re-running with the original DA3 loss gave a qualitatively different (positive, bounded-below) loss curve. The user's standing instruction `feedback_stay_close_to_da3_paper.md` makes this even more important: deviations from the DA3 paper setup need measured wins, not silent CLI defaults.

**How to apply:**
- The CLI in `train_super.py` now exposes `--use-kendall-gal` as an opt-in flag (default: legacy DA3 form). Do NOT pass `--use-kendall-gal` unless the user explicitly asks for the Kendall-Gal experiment.
- When orchestrators (e.g., `scripts/multi_scene_distill.py`, `scripts/scene_overfit_perlayer_init.py`) call `train_super`, they MUST NOT pass `--use-kendall-gal`. They should rely on the default.
- The dataclass `DA3LossWeights.use_kendall_gal` default in `src/mamba3_attn/train/da3_loss.py` is `False` — keep it that way.
- If you ever consider switching the default back to Kendall-Gal for any reason, stop and ask the user first.

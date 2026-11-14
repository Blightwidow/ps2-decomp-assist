# AGENTS.md

Fine-tune Gemma 4 E2B (MLX, Mac only) to assist PS2 matching decompilation for sly1-decomp and sly2-decomp. Plan: `docs/PLAN.md`. Eval results and gotchas: `docs/EVAL.md`.

## Hard rules

- **Disc-derived data never enters the conversation or any hosted service.** Asm, records, splits, candidates, diffs and `data/t2/*` come from the user's disc images (sly1 `LEGAL.md`). Print counts, lengths and logs only. Use synthetic fixtures in tests. Never print candidate text, diffs or asm. One user-approved exception (2026-11-14): asm of 20 validation functions went to Haiku for `eval/haiku_bench.py`; any further hosted use needs fresh approval.
- **One heavy MLX job at a time** (48 GB Mac). Two concurrent jobs saturated RAM once. After the user kills something, wait for their go-ahead. Run heavy jobs through `scripts/guarded_run.py` (kills on low available RAM, exit code 99).
- Create venvs with `uv ... --link-mode copy` (`uv clean` empties linked venvs).
- Test set stays frozen until the final model. Evaluate on the validation sample.
- Quality gates after every edit: `ruff format`, `ruff check`, `mypy`, then `pytest -q tests`. Spell names out, no abbreviations. Stage files by name; never `git add .`.

## Throughput lessons (measured)

- **Generation batch size sets throughput.** `Generator.sample` batches the samples of one prompt (`batch_size` 8). With n=8 per prompt the eval ran about 1.2 samples/s. With n=2 per prompt (failure mining) batches held only 2 sequences and it ran about 0.4 samples/s, so 2,000 functions took about 3 h instead of the estimated 1 h. When sampling few candidates per prompt, batch across prompts or raise n and subsample. Estimate time from the actual batch shape, not from an earlier run with a different one.
- A temperature of 0.8 with the pilot model sometimes runs to the 512-token cap, which slows a batch further.
- Verification (compile plus compare) runs about 14 candidates/s with 4 workers; it is not the bottleneck.
- A job that prints only at the end (`run_eval`) shows an empty log until it finishes. Wait in the background with a pid-based loop and report when it exits; do not make the user ask.

## Model gotchas

- Chat template with `enable_thinking=False`; prompt and train formats must match.
- The converted base model must list `eos_token_id` `[1, 106, 50]` (includes `<turn|>`). With only `<eos>` it never stops and every metric is 0. `models/gemma-4-e2b-base-bf16` is patched.
- mlx-lm `iters` counts micro-batches; an optimizer update happens every `grad_accumulation_steps`. `clear_cache_threshold` is in bytes. The config has no `model` key, pass `--model` on the CLI.
- Use `-it` as the default model (A/B vs base within noise on 150 functions, lower validation loss).

## Waiters

Do not `pgrep -f` a pattern that appears in your own `sh -c` command line (it matches itself and loops forever). Capture the pid first, or wait on a file marker.

## Layout

`decomp_assist/` inference CLI (target, loop); `scripts/decomp-assist` launcher; `.claude/skills/decomp-assist/` skill; usage in `docs/USAGE.md`. `pipeline/` extraction, splits, SFT and repair data, mining. `eval/` harness and persistent verifier workers (`eval/workers/`). `configs/` LoRA configs. `runs/` logs and adapters (ignored). `data/` ignored. `docs/` plan, eval, splits.

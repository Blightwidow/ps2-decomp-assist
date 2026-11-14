# Stage 4: evaluation harness and baselines

## Harness

- `pipeline/verify.py` splices a candidate function into its original source file; persistent per-repo workers (`eval/workers/`) compile it with the repo's own toolchain and compare with retail using the repo's own code (sly1 `tools/check.py`; sly2 `compile_unit` + `compare`, per-address flags, whole-file symbol suffix patch).
- Rule gate: banned-pattern scan (inline asm, volatile, register pins, INCLUDE_ASM, m2c leftovers), Hungarian-prefix check (sly1), sly2 `check_source.py`. **Only EXACT and rule-clean counts.**
- Metrics (`eval/metrics.py`): unbiased pass@k, size buckets (<=100/<=200/<=400 B), similarity (difflib ratio for sly1, 1 - score/max(instr) for sly2).
- CLI: `python -m eval.run_eval --method ground_truth|retrieval|zeroshot|fewshot|model --run-id ID [--sample 150 --n 8 --temperature 0.6 --adapter DIR]`. Results in `eval/results/` (git-ignored; candidates contain disc-derived text and stay local).

## Ground-truth gate

The original C of every record must verify EXACT and rule-clean. Results: sly1 all pass except 2 train; sly2 4/386 test, 3/140 validation, 26/3,018 train fail. 35 records removed from the splits after assignment (see `docs/SPLITS.md`). The faster upstream `compare.py` was adopted afterwards; 100/100 sly2 test records re-checked EXACT.

## Baselines

150 test functions (75 per game, deduped slice, <=400 B, rule-clean EXACT). Gemma 4 E2B-it bf16, thinking off, T=0.6, n=8.

| Method | EXACT@8 | Compiles@1 | Mean best similarity |
|---|---:|---:|---:|
| Retrieval (nearest train function, n=4) | 1.3% (@4) | 6.7% | 0.072 |
| Zero-shot | 1.3% | 26% | 0.089 |
| Few-shot (2 nearest train shots) | **3.3%** | 35% | 0.193 |

Few-shot by size: <=100 B 7.6% EXACT@8 (compiles@1 52%); <=200 B 0%; <=400 B 0%. By game: sly1 2.7%, sly2 4.0%.

Caveats: 150 functions is noisy (3.3% = 5 functions). m2c baseline skipped: its output carries `M2C_` placeholders, which the rule gate rejects by design. Full-build confirmation (`build.py` / `make elf` with candidate spliced) is not implemented; deferred to Stage 8, needed for >=5% of RFT samples.

## S-targets (re-set)

- S1 rule-clean EXACT@8: pilot go/no-go >= 10% (3x best baseline); full SFT >= 15%.
- S2 compiles@1: pilot >= 70%; final >= 90%.
- S3 mean best similarity: > 0.193; target >= 0.45.

## Stage 5: pilot go/no-go (2026-11-13)

Pilot: `-it` (`mlx-community/gemma-4-e2b-it-bf16`), 1,000 stratified train examples (`pipeline/sft_data.py`, deduped slice, rule-clean), LoRA r=32 scale 2 on all attention + MLP linears, 35 layers, lr 1e-4 cosine, 3 epochs (750 optimizer steps; batch 1 x accumulation 4: batch 4 peaked above available RAM), completion-only loss. Config `configs/pilot_lora.yaml`, run under `scripts/guarded_run.py` (memory watchdog). Validation loss 1.48 -> 0.257, still falling.

Evaluated on the 150-function validation sample (seed 0, deduped, 75 per game), n=8, T=0.6:

| Method | EXACT@1 | EXACT@8 | Compiles@1 | Mean best similarity |
|---|---:|---:|---:|---:|
| Few-shot base (validation) | 1.2% | 1.3% | 38% | 0.159 |
| Pilot `-it` adapter | 8.3% | **14.7%** | 62.7% | 0.488 |

Pilot by size, EXACT@8: <=100 B 22.9%, <=200 B 10.6%, <=400 B 3.0%. By game: sly1 16.0%, sly2 13.3%. Rule-clean rate 100%.

**Verdict: GO.** S1 pilot target (>=10% EXACT@8, >=3x baseline) met; S2 pilot target (compiles@1 >=70%) missed (62.7%, mostly 200-400 B). The test set was not used.

Open: base-vs-`-it` A/B not run (needs the converted base at `models/gemma-4-e2b-base-bf16`, one more ~1 h job); continue with `-it`. Caveat: 150 functions, so ±3 points of noise.

## Stage 6: full SFT-1 (2026-11-14)

Recipe: LoRA r=32, scale 2.0, batch 1 x 4 accumulation, lr 1e-4 cosine, 16,509 micro-batches (~3 epochs of 5,503 examples), max 2,048 tokens, prompt masked. Config `configs/full_lora.yaml`. The `-it` run was resumed once from its step-3000 checkpoint after a watchdog kill (`configs/full_resume_lora.yaml`). No single-axis sweep run: the first recipe met the target.

Same 150-function validation sample, n=8, T=0.6:

| Model | EXACT@1 | EXACT@4 | EXACT@8 | Compiles@1 | Mean best similarity | Final val loss |
|---|---:|---:|---:|---:|---:|---:|
| Few-shot base | - | - | 1.3% | 38% | 0.159 | - |
| Pilot `-it` (1k examples) | 8.3% | 11.6% | 14.7% | 62.7% | 0.488 | - |
| **Full SFT `-it`** | 19.8% | 26.0% | **29.3%** | 68.7% | 0.616 | 0.181 |
| Full SFT base | 17.2% | 23.5% | 28.0% | 70.0% | 0.602 | 0.230 |

By size (EXACT@8 / compiles@1):

| Size | `-it` | base |
|---|---|---|
| <=100 B (70) | 48.6% / 87.1% | 47.1% / 87.1% |
| <=200 B (47) | 19.1% / 63.8% | 17.0% / 74.5% |
| <=400 B (33) | 3.0% / 36.4% | 3.0% / 27.3% |

By game, EXACT@8: `-it` sly1 25.3%, sly2 33.3%; base sly1 26.7%, sly2 29.3%. Rule-clean rate 100% for both. Generation ~1,010-1,030 s each.

**A/B verdict: `-it` continues.** Differences (1.3 points EXACT@8, 1.3 points compiles@1 the other way) are inside the +-3 point noise of 150 functions. `-it` has the lower validation loss (0.181 vs 0.230) and the higher EXACT@1/@4 and similarity, so it stays the default.

**Targets:** S1 full-SFT target (>=15% EXACT@8) met (29.3%, about 22x the few-shot baseline). S2 final target (>=90% compiles@1) not met (68.7%); the gap is concentrated in 200-400 B functions (compiles 36%). That motivates Stage 7 (repair) and Stage 8 (RFT). The test set was not used.

Gotcha: the converted base model's `config.json` and `generation_config.json` list only `<eos>` (id 1) as stop token, while `-it` lists `[1, 106, 50]` (including `<turn|>`). Using base with the chat template then never stops and scores 0%. `models/gemma-4-e2b-base-bf16` was patched to the `-it` list; the first run is kept as `eval/results/full_sft1_base_val_noeos.json`.

## Stage 7: T2 repair, SFT-2 (2026-11-14)

**Recipe.** Continue from SFT-1 (`runs/full_sft1b`) on `-it`: 4,768 repair examples (3,170 model failures mined from 2,000 unseen train functions with the pilot adapter, 2,239 synthetic mutants, synthetic share about 41%) plus 2,500 T1 replay examples (7,268 train). LoRA r=32, lr 5.0e-5 cosine, batch 1 x 4 accumulation, 7,268 micro-batches, about 2.5 h. Validation loss on the mixed set (half T2, half T1): 0.188 at start, 0.131 at the end (noisy between, 0.135 to 0.204).

**T1 regression** (150 validation functions, n=8, T=0.6):

| Model | EXACT@1 | @4 | @8 | Compiles@1 | Mean best sim |
|---|---:|---:|---:|---:|---:|
| SFT-1 `-it` | 19.8% | 26.0% | 29.3% | 68.7% | 0.616 |
| SFT-2 | 21.5% | 27.0% | 30.0% | 66.7% | 0.583 |

By size (EXACT@8 / compiles@1): <=100 B 52.9% / 84.3% (SFT-1 48.6% / 87.1%), <=200 B 17.0% / 61.7% (19.1% / 63.8%), <=400 B 0.0% / 36.4% (3.0% / 36.4%). EXACT@8 is flat within noise; compiles@1 and best similarity are a little lower, mostly on larger functions. No clear regression, no T1 gain.

**S4 repair eval** (`eval/repair_eval.py`): SFT-1's own validation candidates are re-verified with instruction diffs; for each function with no exact candidate (106 of 150) one failed candidate goes into the repair prompt and 8 repairs are sampled.

| | Result |
|---|---:|
| Solved by SFT-1 alone (EXACT@8) | 44 (29.3%) |
| Repair EXACT@1 / @4 / @8 on the 106 unsolved | 4.6% / 7.9% / 9.4% |
| Repair compiles@1 | 58.5% |
| Mean best similarity of repairs | 0.463 |
| Combined solved (T1@8 plus one repair round) | 36.0% (+6.7 points, 10 functions) |

**Verdict.** The S4 target (>=30% rule-clean EXACT@4 on repairs) is not met on the unsolved set. The set is hard: SFT-1 failed all 8 samples on each function. Repair still adds 6.7 points when run after T1, at the cost of one more verifier round and 8 more samples. 41% of repairs fail to compile, which is the main drag.

**Next.** Stage 8 (RFT) targets compile rate and larger functions. Ideas for repair: lower the synthetic share, add real near-misses, and add compiler-feedback rounds. A repair loop in the CLI should use SFT-2 only for functions T1 did not solve.

## Stage 8: RFT sizing pilot, stopped at a documented plateau (2026-11-14)

**Pilot** (`pipeline/rft_mining.py`): SFT-1 sampled on 300 deduped rule-clean train functions (100 per size bucket), n=8, T=0.8. Kept rule-clean EXACT candidates; "novel" means different from the ground-truth text beyond whitespace.

| Size | Functions | With an exact sample | With a novel exact sample |
|---|---:|---:|---:|
| <=100 B | 100 | 82 | 60 |
| <=200 B | 100 | 59 | 50 |
| <=400 B | 100 | 7 | 5 |

400 records kept over 148 functions, 313 novel. About 1 h for 300 functions.

**Why stopped.** SFT-1 trained on every rule-clean train function, so the exact rates are largely memorisation (82% / 59% / 7% against 29% on validation). The 200-400 B bucket, where compiles@1 and EXACT lag most, yields almost nothing (7 of 100). Novel exact samples were not inspected (disc-derived text) and are probably small rewrites. Mining all 4,019 functions would cost about 13 h and mostly add small-function variants. A short retrain on the 313 novel records was not run: too few to measure a gain on 150 validation functions.

**Verdict.** No measured RFT gain; plateau documented. SFT-1 stays the T1 default; SFT-2 is the repair model for functions SFT-1 fails. Open directions if work resumes: data for 200-400 B functions (real near-misses, splitting), lower synthetic share in T2, compile-error feedback rounds. Pilot records stay in git-ignored `data/rft/rft_pilot.jsonl`. The full-build (`make elf`) confirmation was never run; EXACT remains a proxy until it is.

## Haiku benchmark (2026-11-14, user-approved exception)

Zero-shot Claude Haiku (headless `claude -p --model haiku`, no tools, same system and user prompt and verifier) on 20 validation functions (10 per game, a subset of the 150-function sample), n=8. The asm and context of those 20 functions were sent to the hosted API with the user's explicit approval; no ground truth and no train examples were sent. `eval/haiku_bench.py`.

| Model (same 20 functions) | EXACT@1 | @4 | @8 | Compiles@1 | Mean best sim |
|---|---:|---:|---:|---:|---:|
| Haiku zero-shot | 36.3% | 51.7% | 55.0% | 75% | 0.804 |
| SFT-1 `-it` | 22.5% | 31.4% | 40.0% | 85% | 0.615 |
| SFT-2 | 16.3% | 22.1% | 25.0% | 80% | 0.498 |

**Read.** Haiku beats the local model on matches (about 15 points EXACT@8 over SFT-1) while compiling less often at @1 (75% vs 85%). The 20-function sample is easier than the 150 (SFT-1 gets 40% here vs 29.3%) and the noise is about +-10 points, so treat it as a rough ordering only. Zero-shot Haiku was not given few-shot examples. The local model's case is therefore offline use, privacy (disc data never leaves the machine) and cost, not accuracy.

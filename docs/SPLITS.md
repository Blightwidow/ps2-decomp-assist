# Stage 3: splits (frozen 2026-11-13)

Code: `pipeline/similarity.py`, `pipeline/splits.py`, `pipeline/build_splits.py`, `pipeline/unmatched_sly2.py` (+ tests). Output: `data/splits/` (git-ignored, disc-derived). Re-run: `python -m pipeline.build_splits` (deterministic, seed 0).

## Pipeline

1. **Eligible** = rule-clean and <= 2048 tokens (prompt + completion, Gemma 4 tokenizer): 6,887 of 8,106 (excluded: sly1 268 over limit + 5 rule-dirty; sly2 219 over limit + 727 rule-dirty; kept in `excluded.jsonl`).
2. **Duplicate links** (9,175): identical masked asm (registers/immediates/labels/symbols masked), identical normalized C tokens, asm 4-shingle Jaccard >= 0.8. A C-token Jaccard link was tried and dropped: normalized C shingles are too common to index (zero links).
3. **Split unit = source group**, not duplicate cluster: sly1 = real file (`files.tsv`) or 64 KiB window; sly2 = module directory x 64 KiB address block. (Merging duplicate links into components chained 4,407 of 6,887 records into one component; groups alone peak at 419.) Groups > 60 records are train-only.
4. **Duplicates only remove records**: a small cross-split duplicate cluster drops the lower-priority side (train < validation < test); a boilerplate cluster (> 25 members: stubs, getters, thunks) is removed from validation/test. 467 records dropped (319 train, 97 test, 51 validation).
5. **Leak audit**: no source group and no duplicate cluster spans splits; no identical masked asm or identical normalized C across splits (0 problems). Audit runs on every build and aborts on failure.

## Sizes

| Split | Records | sly1 | sly2 | deduped slice (no trivial, first of its asm) | median tokens |
|---|---:|---:|---:|---:|---:|
| train | 5503 | 2511 | 2992 | 4019 | 644 |
| validation | 293 | 156 | 137 | 270 | 697 |
| test | 589 | 207 | 382 | 523 | 628 |

Size buckets (<=100 / <=200 / <=400 B): train 53% / 34% / 14%; test 54% / 30% / 16%. Validation is small (293): treat val metrics as noisy, use rule-clean EXACT@k with several seeds.

## Extra sets

- `cross_game_sly1_test.jsonl` (2573 sly1 records with **no** sly2 near-duplicate): for the one cross-game run (train sly2 only, test sly1).
- `unmatched_sly2.jsonl` (1,731 asm-only functions, 3..100 instructions, SDK/blocker/handwritten excluded): the S6 set. No reference C and empty context (headers-only context to be decided at Stage 4). Any rule-clean verified EXACT here is real, uncontaminated value.
- **No chronological test set**: history starts 2026-10-28 and most functions land in the first two days (see `docs/EXTRACTION.md`).

## Freeze

- Test ids sha256: `5b5c192374fded9a918800e902a97979dcdbfb6a4daf023cf081c7993452ed89`
- Validation ids sha256: `517e955e2693bb45d4a9fa7b3cf311372f70a98f93c937c04d68f34b0f388730`
- Never tune on test. Changing extraction, thresholds, seed or repo SHAs changes these hashes: that requires a new version of this document.

## Known limitations

- Group-level splitting keeps whole files together, so test skews toward small groups (<= 60 records); large source files appear only in train.
- Cross-game lineage leaks are removed only where assembly/C are near-identical (Jaccard 0.8); weaker structural similarity can remain.

Ground-truth failures (35 records whose original C does not reproduce EXACT/rule-clean through the harness) are removed **after** assignment, so the frozen group assignment does not shift. Test hash above is post-removal (re-frozen 2026-11-13).

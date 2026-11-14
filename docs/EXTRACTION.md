# Stage 2: extraction results (2026-11-13)

Code: `pipeline/` (asm_parse, cpp_parse, context, records, sly1, sly2, matched_dates, extract, token_stats); tests in `tests/` use **synthetic** fixtures only. Run: `python -m pipeline.extract sly1|sly2` (prints counts only), output `data/records/<game>.jsonl` (git-ignored, disc-derived: never read into a hosted assistant).

## Counts (<= 400 B, game code, SDK dropped)

| | sly1 | sly2 |
|---|---:|---:|
| Records extracted | 3,355 | 4,751 |
| Rule-clean (`inline_asm`, `volatile`, register pin, `INCLUDE_ASM`, m2c leftovers) | 3,350 | 4,024 |
| Groups (file / address window / module) | 96 | 249 |
| Skipped: too large (> 400 B) | 569 | 203 |
| Skipped: SDK | n/a (lib/libm not read) | 337 |
| Skipped: no known symbol / static helper | 145 | - |
| Skipped: asm size mismatch | 0 | 59 |
| Skipped: inline-helper / blocker | - | 69 |
| Skipped: no definition found / ambiguous / no asm | - | 49 / 1 / 10 |

Sly2 rule hits are almost all `m2c_leftover` (sources not yet cleaned up upstream); they stay in the data flagged `rule_clean=false` and are excluded from headline metrics.

## Token lengths (Gemma 4 tokenizer, v1 prompt format)

| | median | p90 | max | > 2048 |
|---|---:|---:|---:|---:|
| sly1 | 789 | 1,921 | 7,569 | 269 |
| sly2 | 527 | 1,380 | 3,046 | 220 |

~12.5 tokens per asm line (higher than the plan's 8-12 assumption). v1 drops records > 2048 tokens (489 total). Estimate: ~7.6k records x ~650 tokens x 3 epochs = ~15M tokens ~ 5 h at the measured ~850 tok/s.

## Verification (Stage 2 exit criterion)

- sly1: all 3,355 records re-checked with the repo's `tools/check.py` (8 parallel workers): **3,353 EXACT**, 2 DIFFER (`func_001EBEB0`, `func_00135F48`): data offset differences; they match in the full build (`tools/build.py` -> `MATCH`, report 100%) because globals they reference get retail addresses only when linked. Kept, flagged by name in `data/records/sly1_verify_failures.txt`.
- sly2: not re-checked per record (its per-function verifier needs the "lab" setup, Stage 4). Provenance = `src/MANIFEST.tsv` (matched) + asm address and size agreement (59 mismatches dropped) + full `make elf` PASS.
- Standalone compile of (context + generated function) is a Stage 4 harness concern.

## Design decisions made here

- **sly1 is address-only**: readable names from `config/names/**/*.tsv` are reversed to `func_XXXXXXXX` / `D_XXXXXXXX` in C and context (asm is already address-only). Names never mix games.
- **sly2 keeps repo names** in the records; `pipeline.sly2.function_aliases()` gives name -> `func_XXXXXXXX` for the address-only ablation (functions only; data symbols not aliased yet).
- **Context** = declarations only: header chain + own file's declarations, selected by identifier reachability from the function body, struct bodies trimmed to members the function uses (others replaced by `/* ... */`), forward declarations dropped when a definition exists, 4,000-char cap (deepest chunks dropped first), comments stripped, `names.h` skipped. Median context 436 chars (sly1). Known limitation: header content is read at HEAD (a struct may have gained fields after the function matched), mitigated by member trimming.
- **Groups for splitting**: sly1 = real file from `config/us/files.tsv`, else 64 KiB address window; sly2 = module directory (some are huge, e.g. `new_100008`: Stage 3 subdivides by address block).
- **Dates** (`matched_date`): first commit adding the name. History starts 2026-10-28 and most functions land in the first days (3,005 sly1 functions on 2026-10-29), so a chronological split is weak. But both repos were created **after** Gemma 4's release (tech report arXiv 2607), so these repos cannot be in its pretraining; memorization risk is mainly TheOnlyZac/sly1 (older, public).

## Environment note

All venvs were rebuilt with `--link-mode copy` after packages vanished from them (suspected `uv clean`): project `.venv` (pinned in `requirements.lock.txt`), `data/raw/sly1-decomp/venv`. sly2's `.venv` uses Homebrew Python 3.15 and was unaffected.

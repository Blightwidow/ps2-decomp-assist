# Upstream rules relevant to dataset and eval (Stage 1)

Read from the pinned repos (SHAs in `data/MANIFEST.json`). Quotes are paraphrased; see the cited files.

## Verifiers

| Repo | Command | Output | Notes |
|------|---------|--------|-------|
| sly1 | `python3 tools/check.py FILE.cpp FUNC` | `EXACT` / `DIFFER` / `ERROR` + diff | **~0.44 s** per call on this Mac. Full gate: `python3 tools/build.py` -> `MATCH` (20 s). |
| sly2 | `scripts/matching/try.py <addr> <src>` | `score N` (0 = match) + diff, then rule check + link check | Needs the "lab" registry `data/matching/functions.tsv` and `work/matching/split` (spimdisasm split). **Not set up yet**: how to bootstrap it is unclear; do it in Stage 4. Alternatives: `make unit`, `make report` (objdiff), `scripts/matching/judge.sh`. Single compile ~0.4-1.2 s (`scripts/build/compile_unit.py`). Full gate: `make elf` -> `PASS: reconstructed image matches retail` (~70 s warm). |

Both AGENTS.md: single-function EXACT is not a match; only the full build is.

## Rule checks (EXACT counts only if rule-clean)

- sly2: `python3 scripts/matching/check_source.py FILE...` prints `file:line: kind: what`, exit 1 on any hit. `banned` = inline asm, register pins, `volatile` on ordinary data (`record.py scan`); `cleanup` = raw `M2C_UNK`/`M2C_FIELD` offsets, `/* Error` comments, m2c temp names, tabs, trailing whitespace, padding locals (scratch/junk/dummy/unused/spill).
- sly1 (AGENTS.md "Matching rules" L84-208, "Rules" L242-268): plain C++, **no inline asm in functions, no register pins, no artificial barriers**; `volatile` only for hardware registers / data the original had volatile; `do { } while (0)` only as the macro form the original had; no names from other decompilations (Hungarian prefixes `psw`, `plo`, `grf` are a red flag); defining globals is the lead's step. No dedicated checker: Stage 4 builds one from these rules, reusing the sly2 `banned` scan.

## Other rules

- sly1 LEGAL.md: do not post disc-generated files anywhere, including a chat. Our setup logs go to the scratchpad; extracted records/asm stay local, and are never read into hosted-assistant context.
- Contribution mechanics (coordination issue, 500 unfinished-function cap) apply only if contributing upstream. AI-disclosure/attribution: ignored by decision.

## Environment facts (Stage 1)

- sly1: toolchain from `AngheloAlf/sce_ps2_sdk_24` + `wibo-macos` (x86_64, via Rosetta) + brew `mips-linux-gnu-binutils`. `tools/setup.py` recognised the US disc; PAL missing (not needed).
- sly2: `./setup.sh` needs the extracted `SCUS_973.16` (`scripts/disc_extract.py`), not the ISO. It exits 1 only for missing Ghidra (`analyzeHeadless`, name-recovery only). The SDK range needs Sony's 2.9-ee-991111b at `work/toolchains/ee-gcc2.9-991111b`: not downloaded by setup; symlinked to the sly1 mirror's compiler and `make elf` then passes. (SDK code is out of dataset scope, but the full gate needs it.)
- The sly2 setup created `.venv` with Python 3.15 (works).

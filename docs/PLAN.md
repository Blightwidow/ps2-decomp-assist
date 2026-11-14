# Fine-tuning Gemma 4 E2B for PS2 decompilation (Sly 1 / Sly 2)

Status: v2 (v1 reviewed by an Opus subagent; review applied, see Section 13). Date: 2026-11-13.

## 1. Goal

Fine-tune a small local model (Gemma 4 E2B, "2B effective params" class) as a cheap first-pass **matching-decompilation assistant** for:

- <https://github.com/OpenThievius/sly1-decomp> (Sly Cooper 1, NTSC-U `SCUS_971.98`, AGPL-3.0)
- <https://github.com/OpenThievius/sly2-decomp> (Sly 2, NTSC-U `SCUS_973.16`, GPL-3.0)

"Matching" = C/C++ that, built with the original SN GCC 2.95.3 (MIPS R5900), reproduces retail bytes. The repos give a **deterministic verifier**, so the plan is built around it: the verifier is the eval oracle, the data filter, and the reward for rejection-sampling fine-tuning (RFT).

### Scope of v1 (deliberately narrow)

- **Game code only.** SDK / library / compiler-runtime code is **out** (Sony gcc 2.9x ranges, `src/lib`, `src/libm`, sly2 SDK range in `config/us/sony_gcc29.txt`). One compiler per game.
- Functions **<= 400 bytes** (the repos' "Haiku tier"), sequences <= 2k tokens. Larger buckets only after v1 shows signal.
- Out: `vu`, `system`, `handwritten`, `mmi` blockers; original-asm functions; C that needs shared inline-asm helpers.
- Out entirely (cut after review): GGUF/Ollama export, PAL augmentation, naming/struct task (T3), DPO/GRPO, MinHash, cross-game ablations beyond one run.

### Success criteria (provisional; re-set from Stage 4 baselines)

| ID | Criterion | Target |
|----|-----------|--------|
| S1 | **Rule-clean** EXACT@8 on held-out functions <= 400 B (unbiased estimator, n >= 8 samples) | **>= 3x best baseline (few-shot 3.3%) i.e. >= 10% at pilot go/no-go; full SFT target >= 15%**; reference point: sly1 AGENTS.md reports Haiku retries matched ~7% (28/420) |
| S2 | Compiles@1 on held-out | >= 90% (baseline best 35%; pilot go/no-go >= 70%) |
| S3 | Mean fuzzy similarity over **all** held-out functions (alignment-based, objdiff-style; EXACT counted 100) | > 0.193 (best baseline mean best-of-8 similarity on the 150 slice); target >= 0.45 |
| S4 | Repair: given candidate + diff, rule-clean EXACT@4 on real near-misses and on model-generated failures | >= 30% |
| S5 | Local inference <= 30 s per candidate on the Mac | met |
| S6 | Value on **currently unmatched** sly2 functions (<= 400 B): any rule-clean verified EXACT is uncontaminated real value | reported separately |

## 2. Findings from repo recon (reviewed against live repos)

| Fact | Consequence |
|------|-------------|
| Repos contain **no asm, executables or disc data**. Asm is regenerated from the user's own disc (sly1: `tools/setup_asm.py`; sly2: `./setup.sh` + `make elf`, splat). | Dataset build needs own disc images. |
| Matched C++ lives in `src/`. **sly1 `src/main/XXXXXXXX.cpp` are 16 KiB address windows**, not real source files. Real file/class structure: `config/us/files.tsv`, `classes.tsv` (sly1). sly2: `src/<module>/`, one function per file or whole file, `src/MANIFEST.tsv`. | Split and context must use real file/class structure, not file names. |
| Counts (drift): sly1 US 4,791/4,807 done (4,482 exact C, 311 original asm, 12 C+inline asm); sly2 US 5,485/6,059 matched (767 original-asm blockers, 151 inline-asm helpers). | After dropping SDK/size/blockers, expect a few thousand <= 400 B game-code pairs. Measure at Stage 2. |
| **Verifiers differ per repo.** sly1: `python3 tools/check.py FILE.cpp FUNC` -> `EXACT` / `DIFFER` / `ERROR` + side-by-side disasm, **no match %**. sly2: `scripts/matching/try.py <address> <source.cpp>` -> `score N` (0 = match) + diff; masks relocated fields, then runs a link check and `check_source.py` (rules) at score 0. Both AGENTS.md: single-function EXACT is not a match; sly1 needs `tools/build.py` -> `MATCH`, sly2 needs `make elf`. | Per-repo adapter -> common result schema; EXACT is a proxy; confirm with a full build on samples (Section 5). |
| Rule checkers exist (sly2 `check_source.py`; sly1 AGENTS.md rules: no inline asm outside shared helpers, no `volatile`/fake-prototype/`do{}while(0)` tricks, no names from other decomps, Hungarian-prefix red flags `psw`, `plo`, `grf`). | EXACT only counts when **rule-clean**. RFT would otherwise reward hacks. |
| **macOS is supported by both repos**: sly2 `try.py` hardcodes `wibo-macos`; sly2 setup fetches macOS arm64 binutils; sly1 TOOLCHAIN.md says wibo or Wine on macOS. | R1 is Medium. Run each repo's own setup first; fall back to Wine/Docker only if it fails. |
| sly1 names: address names (`func_0011A338`, `D_...`) plus readable-name macros (`config/names`, `include/names.h`). sly2 names: richer, from `config/recovered`. | Name policy per game (4.4). |
| Sly 2 `config/sly1/` data comes from **TheOnlyZac/sly1** (CC0), a third-party decomp, **not** OpenThievius/sly1-decomp. Sly 2 states it copied no code/names from sly1-decomp. Games still share code lineage at binary level. | Cross-game near-duplicates remain a leak risk. |
| sly2 `nonmatching/MANIFEST.tsv` scores near misses; it is not a matched-function list. Matched list = `progress/us/report.json` + `src/`. | Use the report for matched functions. |
| PAL builds reuse the same C: after normalization PAL asm is a near-duplicate. | No PAL augmentation; dedupe PAL into US. |
| Licensing: tooling AGPL/GPL, but `src/` reconstructs Sony/Sucker Punch code. sly1 LEGAL.md: no third-party decomp names/code, no prototype data, and **do not post files generated from your own disc anywhere, including a chat**. | Dataset/weights private; no disc-derived content in any hosted service (Section 10). |

Hardware: Apple Silicon arm64, 48 GB, Python 3.15 (too new for most ML wheels; use `uv` venv with 3.11/3.12). Working dir empty.

## 3. Model choice

- Base: **Gemma 4 E2B** (user-confirmed; not Gemma 3n). Repo id, license/terms, total vs effective parameter count, context length, architecture quirks (per-layer embeddings, towers, chat template) and MLX/PEFT support are **unverified**. Stage 0 records them in `docs/MODEL_NOTES.md` before any number here (token caps, LoRA targets, "fits in bf16") is trusted. Memory fit depends on **total**, not effective, params. Fallback if Gemma 4 tooling unusable: Gemma 3n E2B.
- Base vs `-it`: pick via one small A/B at the pilot (Stage 5).
- Method: LoRA on language-model linear layers; vision/audio towers frozen/ignored. Not full FT, not QLoRA.

## 4. Data pipeline

### 4.1 Tasks

| Task | Input | Output | Source |
|------|-------|--------|--------|
| **T1 asm->C++** (primary) | declarations context + target asm | function in repo style | matched `src/` x regenerated asm |
| **T2 repair** (added only if pilot passes) | context + asm + candidate C + verifier diff | corrected C | real near-misses (`nonmatching/`, later-matched) + synthetic mutations + model-generated failures |

### 4.2 Extraction

1. Clone both repos at pinned SHAs under `data/raw/` (git-ignored); record in `data/MANIFEST.json`.
2. Discs in git-ignored paths; run each repo's setup. **Gate:** US build reproduces retail (`MATCH` / `make elf` ok). Otherwise stop.
3. Matched functions = `progress/us/report.json` + `src/`; keep only game-code, <= 400 B, non-blocker (use the repos' tags from `tools/todo.py` / MANIFEST / `classes.tsv`).
4. Record: `{game, func_id, real_file, class, module, size, tags, compiler, asm, c, context, matched_at_commit, matched_at_date}`. `matched_at_date` from git history feeds the time-based test set.
5. Re-verify each pair at build time: adapter returns EXACT **and** rule-clean; sample (>= 5%, and all test items) also passes the full build check.

### 4.3 Filtering

Drop SDK/lib code, blockers, original asm, inline-asm-helper C, anything failing re-verification or the rule check. Exact duplicates and trivial getters/thunks are tagged (`trivial`), not silently dropped, so eval can report a "deduped, >= N instructions" slice.

### 4.4 Normalization

- Asm: strip splat noise/addresses; local labels `.L1..`; one canonical format; per-repo adapters + one parser, unit-tested on golden files that are **synthetic** (hand-written snippets, not disc-derived).
- Names: **sly1 = address-only names** for train and prompt (readable-name macros stripped), and outputs are filtered for banned names / Hungarian prefixes. sly2: named vs address-only is a one-run ablation (address-only mirrors the real use case on unnamed functions; also avoids TheOnlyZac-derived names leaking into sly1 outputs). Never mix sly2 names into sly1 prompts.
- Context = **declarations only**: structs, externs, callee prototypes, typedefs from `include/*.h` needed by the function. **Never** other matched function bodies. Headers gained fields because other functions got matched: pin headers at the commit where the target function matched, or restrict to fields the target actually uses. Cap ~1k tokens. Candidate is spliced into the real file only when verifying.
- Compiler in each record/prompt (one compiler per game in v1).

### 4.5 Prompt format

```
<system> Matching decompiler: PS2 EE (MIPS R5900), SN GCC 2.95.3, C++. Output only the function. </system>
<user>
GAME: sly2  FUNC: func_001B5F28  SIZE: 0x68  COMPILER: sn-gcc-2.95.3
CONTEXT: <declarations>
ASM: <normalized asm>
</user>
<assistant> <function> </assistant>
```

T2 adds `CANDIDATE:` and `DIFF:` (truncated). Loss on assistant tokens only. Measure tokens/instruction with the Gemma 4 tokenizer in Stage 2; v1 cap 2k tokens total.

### 4.6 Synthetic repair data (only after pilot go)

Mutation operators from AGENT_WORKFLOW.md's "what moves the bytes" list (flip compare operands, if/fall-through form, `if` <-> `?:`, statement order, temporaries, signed/unsigned counters, load splitting, widths). Mutate -> compile -> keep if DIFFER and rule-clean -> `(mutated C, diff) -> original C`. <= 8 per function, synthetic <= 70% of T2. Evaluate T2 on real near-misses and **model-generated** failures, not on synthetic.

### 4.7 Splits (leakage control)

- Unit = **real source file / class** (`files.tsv`, `classes.tsv`), or address blocks separated by gaps; never 16 KiB window files, never single functions.
- Near-dup clustering: normalized-asm shingle Jaccard (registers/immediates/addresses masked) + C token shingles; connected components stay on one side. Covers PAL/US, Sly1<->Sly2 lineage, repeats, thunks.
- train ~85 / val ~5 / test ~10, stratified by size bucket.
- **Extra test sets:** (a) *time-based*: functions matched after Gemma 4's training cutoff (public repos + TheOnlyZac/sly1 may be memorized); (b) *unmatched sly2* functions <= 400 B (S6), reported apart: the matched set is the easier part of the game, remaining work is harder; (c) one *cross-game* run (train sly2, test sly1) after dedupe.
- Freeze test sets at Stage 3; tune only on val. Memorization probe after training (verbatim reproduction of train neighbours).

## 5. Evaluation harness (built before training)

- `eval/adapters/{sly1,sly2}.py` wrap `check.py` / `try.py` and emit a common record `{compiles, exact, score, similarity, rule_clean, diff, runtime}`. `exact` = verifier EXACT/score 0; `rule_clean` = repo rule checker passes (sly2 `check_source.py`; sly1 an equivalent built from AGENTS.md rules: inline asm, `volatile`, fake prototypes, barrier tricks, banned names). Headline metrics count only `exact AND rule_clean`.
- EXACT is a **proxy**: on every test item and a >= 5% sample of RFT/synthetic items also run the full build (`tools/build.py` -> `MATCH` / `make elf`) with the candidate spliced in.
- Similarity: alignment-based fuzzy % (objdiff-style) per function, EXACT = 100, mean over **all** functions (not only non-EXACT), size-mismatch handled by alignment, not zeroed.
- pass@k: unbiased estimator with n >= k samples; temperature chosen on val; report overall plus "deduped, >= N instructions" slice; per size bucket, per tag, per game.
- **Baselines (Stage 4, before any SFT):** (a) **retrieval**: nearest train asm (normalized) -> copy its C with symbols renamed; (b) **few-shot** base Gemma 4 E2B with the k nearest train neighbours as examples; (c) base zero-shot; (d) m2c output. Frontier-API baseline **not used** (disc-derived text, Section 10).
- Sanity: ground-truth C = 100% exact and rule-clean on the test set; known-bad mutation = DIFFER; a banned construct = not rule-clean. Fix the harness before anything else.
- Results: `eval/results/<run_id>.json` with pinned seeds, repo SHAs, toolchain hashes.

## 6. Training plan

Framework: mlx-lm LoRA on the Mac if Gemma 4 E2B supported; else HF+PEFT/Unsloth (see R4).

| Item | Start | Notes |
|------|-------|-------|
| Precision | bf16 | verify fit vs total params |
| LoRA targets | LM attention + MLP linears | |
| Rank/alpha | r=32 / 64 | **only sweep axis in v1: rank or LR, not both** |
| LR | 1e-4 cosine | |
| Epochs | 2-3, early-stop on val rule-clean EXACT@4 | loss != match |
| Seq len | <= 2k | |
| Loss mask | completion only | |

**Compute honesty:** rough estimate ~6-8k ex x ~1-2k tokens x 3 epochs = tens of M tokens per run; wall time unknown. Stage 0/1 **measure** train tokens/s and compile latency, then recompute this schedule. RFT cost = (#train fns x samples) generations + same number of compiles; size it from measurements, cap samples/function (e.g. 8) and use only functions the pilot shows within reach.

Stage order: baselines -> **pilot** (1k-example T1 LoRA) -> **go/no-go** -> full SFT-1 (T1) -> (if go) SFT-2 (T1 + T2) -> RFT (keep rule-clean EXACT samples not already in data; rerun SFT) -> eval each step.

**Go/no-go rule (pilot):** proceed to T2/RFT only if the pilot beats the best baseline on val rule-clean EXACT@8 by a clear margin. Otherwise stop and write up (the answer is then: retrieval/few-shot is enough, or model too small).

Ablations (small, one each, after the pilot): names vs address-only (sly2), context on/off, base vs `-it`, one cross-game run.

## 7. Inference & workflow

- Fused or adapter model run locally via MLX (or the chosen framework).
- CLI `decomp-assist FUNC`: prompt from local checkout -> n candidates -> adapter verify (+ rule check) -> best + diff; refine loop up to R rounds (T2 format) if T2 shipped. Only verified, rule-clean output is surfaced; never commit unverified output; full build confirms before any commit.
- Harness integration (decision 2026-11-14): the model is not an agent (trained on one single-turn task, no tool calling). A stronger agent (Claude Code, opencode) calls it as a tool: ship a small MCP server and/or a skill that wraps `decomp-assist` (input: game + function; output: verified rule-clean C++ or the closest candidate with its diff and similarity). The verifier loop stays inside the CLI; the wrapper only shells out. Optional later: `mlx_lm.server` backend.
- Portability (decision 2026-11-14, last step of Stage 9, best effort; v1 stays Mac-only): (1) export the final model for non-Apple runtimes: fuse the adapter into the bf16 model, convert to GGUF for llama.cpp, and/or convert the adapter to PEFT format; **first verify Gemma 4 support** in those tools, and re-run the validation sample on the exported model to confirm EXACT@8 is within noise of the MLX result; (2) check the verifier toolchain on Linux (Linux `wibo` build, binutils, repo `setup.sh`) and document what works; (3) keep the MCP/skill wrapper runtime-agnostic (it only shells out to `decomp-assist`). Windows: WSL2 supported if the Linux check passes (the GGUF model runs under llama.cpp there); native Windows untested (the upstream build tooling assumes a POSIX shell). Document unsupported platforms plainly.
- AI-disclosure/attribution conventions upstream: out of scope for this plan (decision: ignore).

## 8. Infra & repo layout

```
e2b-ps2/
  docs/PLAN.md  docs/MODEL_NOTES.md (S0)  docs/UPSTREAM_RULES.md (S1: rule checkers, tags, names rules)
  data/raw|records|sft/       git-ignored
  pipeline/                   extract, normalize, filter, dedupe, split, mutate
  eval/adapters/ eval/results/
  train/ runs/ tests/         (tests use synthetic fixtures only)
```

- `uv` venv (3.11/3.12), pinned requirements. `git init`; `.gitignore` for data, runs, discs, toolchains, weights.
- Toolchain: each repo's own setup first (wibo-macos path). Fallbacks only on failure: Wine/Rosetta, `linux/amd64` container, remote Linux x86 (would receive disc-derived data: needs explicit user approval, Section 10).
- Secrets in macOS Keychain. Quality gates per change: format -> lint -> type check -> pytest. Parser/normalizer/splitter/adapters/metrics unit-tested on synthetic fixtures.

## 9. Stages

| # | Stage | Deliverables | Exit criteria | Est. | Status |
|---|-------|--------------|---------------|------|--------|
| 0 | Decisions & env | `docs/MODEL_NOTES.md` (Gemma 4 E2B facts); `uv` venv; framework smoke test (generate + 10-step LoRA); measured train tokens/s | smoke test passes; versions pinned; throughput known | 0.5-1 d | **Complete** (2026-11-13; ~850 train tok/s, see MODEL_NOTES) |
| 1 | Recon & toolchain | repos pinned; discs in place; both US builds `MATCH`; `docs/UPSTREAM_RULES.md` (rule checkers, tag sources); measured compile latency | ground-truth rebuild OK; latency known; schedule recomputed | 1-3 d (R1) | **Complete** (2026-11-13). Both US builds match retail; check ~0.44 s (sly1), compile 0.4-1.2 s (sly2). Open: sly2 per-function verifier lab setup -> Stage 4 |
| 2 | Extract & normalize | records JSONL; normalizer + declarations-only context extractor + tests; token histogram; counts after filters | each record re-verifies exact + rule-clean | 2-3 d | **Complete** (2026-11-13). 3,355 sly1 + 4,751 sly2 records; sly1 3,353/3,355 EXACT via check.py; sly2 provenance via manifest + full build; see `docs/EXTRACTION.md` |
| 3 | Dedupe, splits, freeze | clusters, splits (file/class), time-based + unmatched-sly2 + cross-game sets; leak audit | no cluster crosses splits; test hashes frozen | 1 d | **Complete** (2026-11-13). train 5,531 / val 296 / test 593; audit clean; no time-based set (history too short); see `docs/SPLITS.md` |
| 4 | Harness + baselines | adapters, rule gating, full-build check, fuzzy metric; retrieval, few-shot, zero-shot, m2c baselines | ground truth 100%; baselines recorded; S-targets re-set | 2-3 d | **Complete** (2026-11-13). See `docs/EVAL.md`. Best baseline: few-shot EXACT@8 3.3%. m2c skipped; full-build confirmation deferred to Stage 8 |
| 5 | **Pilot + go/no-go** | 1k-ex T1 LoRA, base vs `-it` A/B | pass/fail vs best baseline recorded | 1-2 d | **Complete** (2026-11-13). GO: val EXACT@8 14.7% vs few-shot 1.3%; compiles@1 62.7% (<70% target). Base-vs-`-it` A/B deferred, `-it` used. See `docs/EVAL.md` |
| 6 | Full SFT-1 | adapter, one-axis sweep | beats baselines on val; no collapse | 2-4 d | Complete |
| 7 | T2 synthetic + SFT-2 (if go) | mutation pipeline; mixed SFT | S4 on real + model-generated failures; no T1 regression | 3-4 d | Complete (S4 not met: repair 9.4% EXACT@8 on unsolved, combined 36.0% vs 29.3%; see EVAL.md) |
| 8 | RFT (if go) | verified samples; retrain | measured gain or documented plateau | 2-3 d, resized by measurements | Complete (2026-11-14): stopped at documented plateau after sizing pilot; see EVAL.md |
| 9 | Inference CLI + harness tool | `decomp-assist`, MCP server and/or skill wrapping it, README, S6 run on unmatched sly2, portability export + Linux toolchain check | S5 met | 1-2 d | In Progress: CLI, skill and usage doc done (2026-11-14); S6 run, portability export and Linux toolchain check open. Harness tool is a skill, not MCP (user decision) |

Stage 0 independent of 1-3; Stage 4 starts once Stage 1 builds.

## 10. Legal, policy and privacy

- Underlying code is Sony/Sucker Punch property; decomp repos publish reconstructions for preservation/interoperability. Training derivatives have unsettled status -> dataset, adapters and weights stay **private/local**; no public upload.
- sly1 LEGAL.md: files generated from your own disc must not be posted anywhere, "including a chat". Therefore: **no disc-derived content (asm, golden files, debug dumps, extracted records, decomp-side C when generated from disc asm) in any hosted service or hosted coding-assistant context**, including this Claude session. Tests/fixtures are synthetic. Frontier-API baselines are excluded. Renting a remote GPU/VM is a hosted service: only with explicit user approval of what data goes there.
- Name hygiene: sly1 outputs must not contain names from other decomps (incl. TheOnlyZac/sly1) or Sly 2; filter outputs; sly1 prompts address-only (4.4).
- Discs: user's own legitimate copies; never committed.
- Training-data licensing (Sly1 AGPL, Sly2 GPL, third-party CC0/MIT/CC-BY-SA in `THIRD_PARTY_NOTICES.md`): private use sidesteps; revisit before any release.
- Upstream contribution mechanics (coordination issue, 500-function cap): follow if contributing. AI-disclosure/attribution handling: ignored by decision.

## 11. Risks

| ID | Risk | L | Mitigation |
|----|------|---|------------|
| R1 | SN GCC on macOS arm64 slow/fails | **Resolved** (wibo-macos under Rosetta works; ~0.4 s/check) | repos support `wibo-macos`; run their setup first; measure latency; fallbacks in Section 8 |
| R2 | Leakage (PAL/US, lineage, 16 KiB windows, memorized public repos) | High | file/class splits, shingle clustering, time-based + unmatched sets, memorization probe |
| R3 | RFT/eval rewards rule-breaking EXACT | High | rule-clean gate everywhere; full build on samples |
| R4 | Gemma 4 E2B tooling immature (MLX/PEFT) | Med-High | Stage 0 smoke test; fallback PEFT/Unsloth; last resort Gemma 3n |
| R5 | Tiny dataset; overfit | Med | LoRA, early-stop on EXACT@k, cluster splits |
| R6 | Retrieval/few-shot beats the fine-tune | Med | baselines first; pilot go/no-go; stop honestly |
| R7 | Compute/RFT cost underestimated | Med | measure throughput/latency first; v1 <= 400 B; cap samples |
| R8 | Synthetic mutations unrepresentative | Med | cap share; eval on real + model-generated failures |
| R9 | Repos move under us | Med | pinned SHAs, idempotent extraction, versioned manifest |
| R10 | Selection bias (matched = easy) | Med | report on unmatched sly2 set separately |
| R11 | Policy/legal (disc-derived data leaves machine) | Med | Section 10; synthetic fixtures; no hosted assistants on data |
| R12 | Python 3.15 wheels | Low | resolved: `uv` Python 3.12 venv works |
| R13 | **Disk: 13 GB free (99% full)** after model download; discs, split asm, toolchains, records, base-model copy and runs need more | High | free space or move `HF_HOME`/`data`/`runs` to external storage before Stage 1 |

## 12. Open questions

1. Disc images for NTSC-U Sly 1 and Sly 2 available locally?
2. Base or `-it`: decide at pilot. Exact Gemma 4 E2B repo id/license: Stage 0.
3. ~~Mac only or rented GPU~~: resolved, **Mac only** (MLX).
4. Name policy for sly2 (named vs address-only): ablation at pilot, default address-only.
5. Time/compute budget (Mac days)?

## 13. Review log (v1 -> v2)

Applied from the Opus review: verifier facts corrected (sly2 `try.py`, sly1 `check.py`, full-build confirmation); rule-clean gate on EXACT/RFT; R1 lowered (macOS supported); SDK/lib dropped (user decision); TheOnlyZac provenance fixed; splits by real file/class with declarations-only context; time-based and unmatched-sly2 test sets; PAL augmentation dropped; retrieval + few-shot baselines; pilot go/no-go; fuzzy similarity over all functions; unbiased pass@k + dedupe slice; v1 scoped <= 400 B / <= 2k tokens with compute re-estimated from measurements; name contamination policy; no disc-derived data in hosted services; cuts (GGUF, T3, DPO/GRPO, MinHash, extra ablations). Not applied by decision: AI-disclosure/attribution handling upstream.

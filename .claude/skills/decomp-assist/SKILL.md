---
name: decomp-assist
description: Decompile a small unmatched Sly Cooper function (sly1, sly2 or sly3, at most 400 bytes) with the local fine-tuned model. Runs sample, verify, repair against the repo's own compiler and returns verified rule-clean C++ or the closest attempt with its instruction diff. Use for a function in nonmatching/, or any function whose source file already holds a stub definition.
---

# decomp-assist

A local model (Gemma 4 E2B, LoRA) proposes C++ for one function; the game repo's own verifier scores every proposal. It solves roughly 30% of small functions at first sight (about 36% with repair). Treat the result as a lead, not an answer.

## Run

`scripts/decomp-assist` lives in the e2b-ps2 checkout, three folders above this skill's folder. Run it with `--json`:

```
scripts/decomp-assist --game sly2 --function func_XXXXXXXX --json
scripts/decomp-assist --game sly1 --function func_XXXXXXXX --source nonmatching/func_XXXXXXXX.cpp --json
scripts/decomp-assist --game sly3 --function func_XXXXXXXX --source nonmatching/func_XXXXXXXX.cpp --json
```

- `--source` is a path relative to the game repo. It must already hold one definition of the function (the staged `nonmatching/` attempt, or the shared file it lives in). sly2 finds it from the manifests; sly1 and sly3 always need it.
- `--repository PATH` points at another checkout of the game repo; the default is `data/raw/<game>-decomp` in the e2b-ps2 checkout.
- Defaults: 8 samples per round, 2 repair rounds; adapters come from Hugging Face on first run (`--adapter` and `--repair-adapter` take local folders). A run on a small function takes seconds to a few minutes. One run at a time: it loads the model into memory.
- The definition already in `--source` is tried first, and matches without loading the model when it is already exact. Put an m2c draft there (unknown `?` types as `s32`) to give the model a starting point: it also joins the first round, so repair can start from it. `--no-seed` skips it.
- Exit code 0 means `exact`, 1 means `closest` or `none`, 2 means the target could not be built (the message says why).

## Read the result

Output fields: `status`, `code`, `similarity`, `diff`, `rounds_used`, `candidates_tried`, `notes`.

- `exact`: the code is byte-identical to retail and rule-clean. Write it into the source file, then run the repo's full build; the function counts as matched only when that build reproduces retail.
- `closest`: `diff` shows target against ours, instruction by instruction. Use it as the starting point for a manual or `tsearch` pass, and keep the attempt in `nonmatching/` with its verdict, as the repo's AGENTS.md describes. `notes` says when the closest attempt does not compile.
- `none`: the model produced no code. Retry with `--temperature 0.8` once, then work the function by hand.

## Done when

The function holds a verified rule-clean `exact` result that the repo's full build confirms, or the closest attempt and its diff are recorded in `nonmatching/`.

## Data handling

`code` and `diff` derive from the disc image. Keep them inside the checkout and this session. Posting them to issues, PRs or any hosted chat conflicts with the game repo's LEGAL.md.

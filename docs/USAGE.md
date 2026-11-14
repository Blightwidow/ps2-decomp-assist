# Using decomp-assist

Local assistant for matching decompilation of small functions (at most 400 bytes) in sly1-decomp and sly2-decomp. Mac with Apple silicon, MLX. The game repos must be checked out and set up (their `setup.sh`), by default under `data/raw/`.

## CLI

```
scripts/decomp-assist --game sly2 --function func_XXXXXXXX
scripts/decomp-assist --game sly1 --function func_XXXXXXXX --source nonmatching/func_XXXXXXXX.cpp
```

Flow: build the prompt from the checkout (asm plus declarations context) -> sample 8 candidates (T1 adapter, downloaded from Hugging Face on first run; `--adapter runs/full_sft1b` for a local one) -> verify each with the repo's compiler, comparer and rule checks -> if none is a rule-clean EXACT, repair the closest failure with the SFT-2 adapter (repair adapter, `--repair-adapter runs/sft2` for a local one) for up to 2 rounds, feeding it the instruction diff or compiler error. Options: `--n`, `--rounds`, `--temperature`, `--repository`, `--json`, `--adapter`, `--repair-adapter`.

Only `status: exact` is a verified match, and it still needs the repo's full build before a commit. `closest` carries the diff and similarity. Exit code 0 exact, 1 closest or none, 2 target error.

`--source` must name a file that already holds one definition of the function (a staged `nonmatching/` attempt or the shared file). sly2 resolves it from `nonmatching/MANIFEST.tsv` or `src/MANIFEST.tsv`.

## Skill

`.claude/skills/decomp-assist/SKILL.md` lets Claude Code (or another harness that reads skills) call the CLI as a tool. Project skills load when the session runs in this checkout; to use it from a game repo, symlink the folder into that repo's `.claude/skills/` or into `~/.claude/skills/`. The model is not an agent, so the harness agent decides what to do with the result.

## Expectations

Validation sample (150 functions): 29.3% rule-clean EXACT@8 on the first round, about 36% with one repair round, compiles@1 about 68%. Almost nothing above 200 bytes. See `docs/EVAL.md`.

## Data

`code` and `diff` in the output derive from the disc image. Keep them in the checkout and the local session; the game repos' LEGAL.md forbids posting disc-generated files, hosted chats included.

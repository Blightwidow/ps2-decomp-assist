---
license: apache-2.0
library_name: mlx
base_model: mlx-community/gemma-4-e2b-it-bf16
pipeline_tag: text-generation
tags:
  - mlx
  - lora
  - decompilation
  - ps2
  - mips
---

# ps2-decomp-assist (Gemma 4 E2B, LoRA)

Two LoRA adapters (rank 32) for `mlx-community/gemma-4-e2b-it-bf16` that turn MIPS R5900 (PS2 EE) assembly into C++ for **small functions (at most 400 bytes)**. Used by the [ps2-decomp-assist](https://github.com/Blightwidow/ps2-decomp-assist) CLI, which verifies every candidate with the decompilation project's own compiler and comparer. Raw model output is a proposal, not a match.

| Folder | Task | Prompt | Output |
|---|---|---|---|
| `t1/` | translate | game, function id, size, declarations context, assembly | the function |
| `repair/` | repair | the same prompt plus a failed `CANDIDATE:` and its `DIFF:` (instruction diff or compiler error) | the corrected function |

`repair/` continues from `t1/`; it was trained on a mix of repair and translation examples.

`gguf/` holds the same two models fused into the bf16 base and exported for llama.cpp (Q8_0, 4.9 GB each): `gguf/ps2-decomp-assist-t1-Q8_0.gguf` and `gguf/ps2-decomp-assist-repair-Q8_0.gguf`. On 150 validation functions the T1 Q8_0 export scores 28.7% EXACT@8 (MLX adapter: 29.3%, inside the noise). Q4_K_M lost about 8 points and is not published. Serve with `llama-server --jinja --chat-template-kwargs '{"enable_thinking": false}'`; commands and results in the repository's `docs/GGUF.md`.

## Use

```
git clone https://github.com/Blightwidow/ps2-decomp-assist && cd ps2-decomp-assist
scripts/decomp-assist --game sly2 --function func_XXXXXXXX
```

The CLI downloads these adapters on first run. Without the CLI: `mlx_lm.generate --model mlx-community/gemma-4-e2b-it-bf16 --adapter-path <folder>` with the chat template, thinking off. Prompt formats are in the repository (`pipeline/token_stats.py`, `pipeline/repair_data.py`).

## Training

LoRA on all attention and MLP projections, mask-prompt (loss on the function only), bf16, 2,048-token sequences. `t1/`: all rule-clean training functions of the two games (5,503 examples), about 3 epochs in one run that was resumed once after an interruption. `repair/`: 4,768 repair examples (3,170 mined model failures, 2,239 synthetic mutations of correct code) plus 2,500 translation examples. Validation loss 0.181 (`t1/`) and 0.131 on the mixed set (`repair/`).

Trained and evaluated only on **Sly Cooper and the Thievius Raccoonus** and **Sly 2** (NTSC-U), compiled with SN GCC 2.95.3. Other games and compilers are untested.

## Evaluation

150 held-out validation functions, 8 samples each, exact byte match plus rule-clean:

| | EXACT@1 | EXACT@4 | EXACT@8 | Compiles@1 |
|---|---:|---:|---:|---:|
| `t1/` | 19.8% | 26.0% | 29.3% | 68.7% |
| `repair/` as a translator | 21.5% | 27.0% | 30.0% | 66.7% |
| `t1/` then one `repair/` round on its failures | – | – | 36.0% | – |

By size (EXACT@8): up to 100 B 48.6%, up to 200 B 19.1%, up to 400 B 3.0%. Noise on 150 functions is about 3 points. On a 20-function sample zero-shot Claude Haiku matched 55% against 40% for `t1/`. Full results: the repository's `docs/EVAL.md`.

## Limits

- Weak above 200 bytes; about a third of candidates do not compile.
- The model memorises the functions it trained on; scores on training functions are not a measure of skill.
- Needs the game project's verifier for any claim of a match, and the project's full build before a commit.
- Output derives from the disc image you supply. Keep it local and follow the game project's rules on posting generated material.

## Data

The adapters are trained on pairs of assembly generated from discs the authors own and the authors' own decompiled source. No disc data, assembly or dataset is distributed here.

Unofficial fan tooling. Not affiliated with Sony Interactive Entertainment, Sucker Punch Productions or Google.

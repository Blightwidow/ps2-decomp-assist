# ps2-decomp-assist

A small local model that proposes C++ for **small PS2 functions** (MIPS R5900 assembly, at most 400 bytes) and checks each proposal with the decompilation project's own compiler and comparer. Only a candidate that rebuilds byte-identical and passes the project's rule checks is reported as a match.

It is a fine-tuned [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B) (LoRA adapters, MLX, Apple silicon). It was trained and measured on **Sly Cooper and the Thievius Raccoonus** and **Sly 2** (NTSC-U, SN GCC 2.95.3). Other PS2 games are untested; transfer is most plausible for games built with the same compiler.

## Results

150 held-out validation functions from the two Sly games, 8 samples per function, rule-clean exact matches only:

| | EXACT@1 | EXACT@8 | Compiles@1 |
|---|---:|---:|---:|
| Few-shot retrieval baseline | – | 1.3% | 38% |
| T1 adapter | 19.8% | 29.3% | 68.7% |
| T1 plus one repair round | – | 36.0% | – |

Functions up to 100 bytes reach 48.6% EXACT@8; almost nothing matches above 200 bytes. A 20-function comparison put zero-shot Claude Haiku ahead on matches (55% EXACT@8 against 40%), so the case for this model is local use and keeping disc data on your machine, not accuracy. Details, caveats and noise levels: [docs/EVAL.md](docs/EVAL.md).

## Install

Apple silicon Mac, Python 3.12 or newer, [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/Blightwidow/ps2-decomp-assist
cd ps2-decomp-assist
uv venv --link-mode copy
uv pip install --link-mode copy -e .
```

You also need your own checkout of the game's decompilation repository, set up with its own tooling (disc image, compiler, assembly), under `data/raw/<game>-decomp` or anywhere with `--repository`. Nothing here ships game data or a compiler.

## Use

```
scripts/decomp-assist --game sly2 --function func_XXXXXXXX
scripts/decomp-assist --game sly1 --function func_XXXXXXXX --source nonmatching/func_XXXXXXXX.cpp
```

The adapters download from Hugging Face on first run ([blightwidow/ps2-decomp-assist-gemma4-e2b-lora](https://huggingface.co/blightwidow/ps2-decomp-assist-gemma4-e2b-lora)). Use `--adapter` and `--repair-adapter` for local folders. See [docs/USAGE.md](docs/USAGE.md).

Other runtimes: the adapters export to GGUF for llama.cpp; Q8_0 matches the MLX results ([docs/GGUF.md](docs/GGUF.md)).

A Claude Code skill in `.claude/skills/decomp-assist/` lets an agent call the CLI as a tool.

## Reproduce

[docs/REPRODUCE.md](docs/REPRODUCE.md) rebuilds the dataset and both adapters from your own disc images. The dataset itself is never published.

## Data and legal

Training pairs were built from assembly generated from discs the authors own and from the authors' own decompiled source. No disc data, assembly, dataset, compiler or SDK file is in this repository or in the adapters' repository; `tests/test_no_disc_material.py` guards the repository against it. The model's output (code and instruction diffs) derives from your disc: keep it local and follow the game repository's rules on posting generated material.

Unofficial fan tooling. Not affiliated with Sony Interactive Entertainment, Sucker Punch Productions or Google.

## License

Code and adapters: [Apache-2.0](LICENSE). Third-party notices: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

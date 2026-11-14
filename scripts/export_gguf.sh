#!/bin/sh
# Fuse a LoRA adapter into the bf16 base and export GGUF (bf16 and Q8_0).
#   scripts/export_gguf.sh NAME ADAPTER_DIR      e.g. scripts/export_gguf.sh t1 runs/full_sft1b
# Needs llama.cpp (brew install llama.cpp) and its source for the converter in toolchains/llama.cpp
# with a venv at toolchains/gguf-venv (see docs/GGUF.md). Output: models/gguf/ps2-decomp-assist-NAME-{bf16,Q8_0}.gguf
set -eu
name=$1
adapter=$2
here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here"
export HF_HOME="$here/models/hf"
.venv/bin/python -m mlx_lm fuse --model mlx-community/gemma-4-e2b-it-bf16 --adapter-path "$adapter" --save-path "models/fused-$name"
mkdir -p models/gguf
toolchains/gguf-venv/bin/python toolchains/llama.cpp/convert_hf_to_gguf.py "models/fused-$name" \
  --outfile "models/gguf/ps2-decomp-assist-$name-bf16.gguf" --outtype bf16
llama-quantize "models/gguf/ps2-decomp-assist-$name-bf16.gguf" "models/gguf/ps2-decomp-assist-$name-Q8_0.gguf" Q8_0

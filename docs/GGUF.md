# GGUF export (llama.cpp)

The adapters are MLX LoRA. For other runtimes, fuse each adapter into `mlx-community/gemma-4-e2b-it-bf16` and convert to GGUF. llama.cpp supports Gemma 4 (`conversion/gemma.py`).

## Export

```
brew install llama.cpp
git clone --depth 1 https://github.com/ggml-org/llama.cpp toolchains/llama.cpp
uv venv toolchains/gguf-venv --link-mode copy
uv pip install --link-mode copy --python toolchains/gguf-venv/bin/python -r toolchains/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt "transformers==5.22.2"
scripts/export_gguf.sh t1 runs/full_sft1b
scripts/export_gguf.sh repair runs/sft2
```

The `transformers` pin matters: the fused tokenizer config is written by transformers 5.x and the version llama.cpp's requirements pull in (4.57) cannot read it.

## Run

One `llama-server` per model; the CLI talks to them:

```
llama-server -m ps2-decomp-assist-t1-Q8_0.gguf --jinja --chat-template-kwargs '{"enable_thinking": false}' -ngl 99 -c 20480 -np 8 --port 8099
llama-server -m ps2-decomp-assist-repair-Q8_0.gguf --jinja --chat-template-kwargs '{"enable_thinking": false}' -ngl 99 -c 20480 -np 8 --port 8098
scripts/decomp-assist --game sly2 --function func_XXXXXXXX --server-url http://localhost:8099 --repair-server-url http://localhost:8098
```

`--chat-template-kwargs` with thinking off is required: the models were trained on the thinking-off template.

## Fidelity (150 validation functions, n=8, T=0.6, rule-clean EXACT)

| Model | EXACT@1 | @4 | @8 | Compiles@1 | Mean best similarity |
|---|---:|---:|---:|---:|---:|
| MLX SFT-1 (reference) | 19.8% | 26.0% | 29.3% | 68.7% | 0.616 |
| GGUF bf16 | 20.7% | 27.3% | 30.0% | 66.0% | 0.598 |
| GGUF Q8_0 | 20.8% | 26.2% | 28.7% | 70.7% | 0.610 |
| GGUF Q4_K_M | 16.1% | 19.9% | 22.0% | 64.7% | 0.564 |

bf16 and Q8_0 are within the +-3 point noise of MLX. Q4_K_M loses about 8 points of EXACT@8: use Q8_0 or bf16. Noise: the sample is 150 functions. The repair model was converted and quantized the same way; its Q8_0 passed an end-to-end CLI run but has no separate validation run.

Run `python -m eval.run_eval --method model --server-url http://localhost:8099 --run-id ID --split validation` to re-measure an export.

## Not yet checked

The verifier toolchain on Linux and Windows (WSL2): the GGUF models run there, but each game repository's compiler and compare tools were only run on macOS.

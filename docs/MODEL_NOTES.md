# Gemma 4 E2B: verified notes (Stage 0, 2026-11-13)

Sources: HF API + model card for `google/gemma-4-E2B`, config.json, local smoke tests.

## Checkpoints

| Repo | Use |
|------|-----|
| `google/gemma-4-E2B` | base (pretrained) |
| `google/gemma-4-E2B-it` | instruction-tuned |
| `mlx-community/gemma-4-e2b-it-bf16` | MLX bf16 of `-it`; **used for smoke tests** (9.5 GB on disk) |
| `mlx-community/gemma-4-e2b-it-4bit` | MLX 4-bit (not planned; QLoRA out of scope) |

Base-vs-`-it` A/B (Stage 5) needs a base checkpoint in MLX format: convert `google/gemma-4-E2B` with `mlx_lm convert` (needs ~10 GB download + ~10 GB output; see disk note).

## Facts (model card / config)

- License: **Apache 2.0** (plan's licensing worry about base-model terms is resolved).
- Params: 5.12B total (bf16, 10.2 GB safetensors), 2.3B "effective" (Per-Layer Embeddings). mlx-lm reports 4.63B (LM + embeddings counted its own way).
- Text: 35 layers, hidden 1536, MLP 6144, 8 heads, 1 KV head, head_dim 256 (global 512), hybrid attention (sliding window 512, every 5th layer global), `num_kv_shared_layers=20`, vocab 262,144, logit softcap 30, tied embeddings.
- Context: 128K. Multimodal (vision ~150M, audio ~300M params): unused; LoRA hits only the language model in mlx-lm (`gemma4_text`).
- Reasoning: thinking mode built in.

## Tooling (pinned in `requirements.lock.txt`)

- Python 3.12 via `uv` venv `.venv`; `mlx==0.32.4`, `mlx-lm==0.32.2` (has `gemma4`, `gemma4_text`), `transformers==5.22.2`.
- `HF_HOME=$PWD/models/hf` keeps the cache in the project (git-ignored).

## Smoke test results (Apple Silicon, 48 GB)

| Test | Result |
|------|--------|
| Generate (bf16, 120 tokens) | works; ~54 tok/s generation, peak 9.3 GB |
| LoRA 10 iters, short toy data | works; loss 3.96 -> 0.28 (toy) |
| LoRA 20 iters, batch 4 x ~1.46k tokens, rank 8, 16 layers, grad-checkpoint, `--mask-prompt` | works; ~1.7 s/example, **~850 tokens/s full-sequence training throughput** (wall, incl. load) |
| Adapter save (26 MB for 16 layers r8) + load for generation | works |
| Footgun | If a sequence is truncated so no assistant tokens remain, prompt masking gives zero target tokens -> loss `nan`. Pre-filter by length (Stage 2) instead of relying on truncation. |

Throughput implication: 7k examples x ~1.5k tokens x 3 epochs ~ 31M tokens ~ 10 h per run (rank 8; recompute with real rank/layers/token counts at pilot).

## Chat template: thinking must be pinned

Default template inserts `<|think|>` into the system turn; generation then starts with `<|channel>thought ...` (observed). Training via `mlx_lm lora` applies the template **without** `enable_thinking=False`.
`apply_chat_template(..., enable_thinking=False)` removes `<|think|>`.

Decision: train and serve with the **same** non-thinking format (no `<|think|>`, answer directly). Stage 2 must either pre-render text with `enable_thinking=False` (use the `text` dataset field) or confirm mlx-lm passes the flag, and Stage 5 must test generation parity (no thought channel emitted).

## Open items

- Confirm exact LoRA targets/keys in mlx-lm for `gemma4_text` and which layers `--num-layers` selects (smoke test used last 16 of 35).
- Base-model MLX conversion for the A/B.
- **Disk**: 13 GB free after the model download (volume 99% full). Data, runs and a base-model copy will not fit comfortably. Free space or point `HF_HOME`/`data`/`runs` at external storage before Stage 1 (toolchains + discs + split asm need more).

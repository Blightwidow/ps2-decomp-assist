# Reproducing the dataset and adapters

Needs: Apple silicon Mac with 48 GB (smaller works with shorter sequences), your own NTSC-U disc images, and both game repositories set up with their own tooling so that `asm/` (sly1) or `build/splat/asm/` (sly2) exists and the repos' verifiers run. Place or symlink the repositories at `data/raw/sly1-decomp` and `data/raw/sly2-decomp`. Everything under `data/` and `runs/` is git-ignored and disc-derived; never publish it.

Environment: `uv venv --link-mode copy && uv pip install --link-mode copy -r requirements.lock.txt`. Run heavy MLX jobs one at a time, through `scripts/guarded_run.py` (it kills a job when available RAM runs low).

1. **Extract** matched functions with their asm and context: `python -m pipeline.extract sly1` and `sly2`.
2. **Splits** (by source file or class, deduplicated slice, validation and test sets): `python -m pipeline.build_splits`. Details in [SPLITS.md](SPLITS.md). The test set stays frozen until the final model.
3. **Tier 1 data**: `python -m pipeline.sft_data full ...` (see the module docstring).
4. **SFT-1**: `python -m mlx_lm lora --model mlx-community/gemma-4-e2b-it-bf16 --config configs/full_lora.yaml --adapter-path runs/full_sft1` (the shipped `t1/` adapter is this run, resumed once from a checkpoint with `configs/full_resume_lora.yaml` after an interruption).
5. **Evaluate**: `python -m eval.run_eval --method model --run-id ID --split validation --adapter runs/full_sft1b --sample 150 --n 8`.
6. **Tier 2 repair data**: `python -m pipeline.failure_mining`, `python -m pipeline.mutation_mining`, then `python -m pipeline.repair_data sft2 ...`.
7. **SFT-2**: `python -m mlx_lm lora --config configs/sft2_lora.yaml --adapter-path runs/sft2 --resume-adapter-file runs/sft2/sft1_final.safetensors`.
8. **Repair eval**: `python -m eval.repair_eval --run-id ID --candidates eval/results/RUN.candidates.jsonl --adapter runs/sft2`.

Results and gotchas (EOS tokens for the base model, `iters` counting micro-batches, generation batch shape) are in [EVAL.md](EVAL.md).

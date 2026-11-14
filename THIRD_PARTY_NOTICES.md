# Third-party notices

- **Gemma 4 E2B** (Google), Apache-2.0: base model. The adapters in this project are fine-tuned on `mlx-community/gemma-4-e2b-it-bf16`, a bf16 MLX conversion of the instruction-tuned release. License: https://ai.google.dev/gemma/docs/gemma_4_license
- **mlx**, **mlx-lm** (Apple), MIT: training and inference.
- **transformers**, **huggingface-hub** (Hugging Face), Apache-2.0: tokenizer and adapter download.
- **tree-sitter**, **tree-sitter-cpp**, MIT: C++ parsing for context building.
- **rabbitizer**, MIT: instruction disassembly for diffs, installed in each game repository's own environment.
- **Training source**: the decompiled C and C++ of OpenThievius/sly1-decomp and OpenThievius/sly2-decomp, written by the authors of this project.
- Verification runs each game repository's own tools, installed separately. This repository includes none of them, no compiler and no SDK files.

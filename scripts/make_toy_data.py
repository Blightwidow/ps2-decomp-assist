"""Synthetic toy SFT data for the Stage 0 smoke test. No game/disc-derived content."""

import json
import random
import sys
from pathlib import Path

output_directory = Path(sys.argv[1])
approximate_instruction_count = int(sys.argv[2])
example_count = int(sys.argv[3])
output_directory.mkdir(parents=True, exist_ok=True)
random_generator = random.Random(0)

system_prompt = "Matching decompiler: PS2 EE (MIPS R5900), SN GCC 2.95.3, C++. Output only the function."


def make_example(index: int) -> dict[str, object]:
    field_offset = random_generator.choice([0x4, 0x8, 0x10, 0x708])
    filler = "\n".join(
        f"    addiu $t{index % 8}, $t{(index + step) % 8}, {step}" for step in range(approximate_instruction_count)
    )
    assembly = f"func_{index:08X}:\n    lw $v0, {hex(field_offset)}($a0)\n{filler}\n    jr $ra\n    nop"
    c_code = f"int func_{index:08X}(struct Obj* object) {{\n    return object->field_{field_offset:x};\n}}"
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"ASM:\n{assembly}"},
            {"role": "assistant", "content": c_code},
        ]
    }


for split_name, count in (("train", example_count), ("valid", max(4, example_count // 10))):
    with open(output_directory / f"{split_name}.jsonl", "w") as handle:
        handle.writelines(json.dumps(make_example(index)) + "\n" for index in range(count))

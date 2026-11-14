from eval.repair_eval import build_repair_messages
from pipeline.repair_data import REPAIR_SYSTEM_PROMPT

ROW = {"game": "sly1", "function_id": "f1", "size_bytes": 8, "context": "", "assembly": "nop", "c_text": "void f() {}"}


def test_repair_prompt_has_the_training_system_prompt_and_candidate() -> None:
    messages = build_repair_messages(ROW, "bad", {"compiled": True, "diff": "a | b"})
    assert messages[0]["content"] == REPAIR_SYSTEM_PROMPT and "CANDIDATE:\nbad\nDIFF:\na | b" in messages[1]["content"]


def test_repair_prompt_never_contains_the_ground_truth() -> None:
    messages = build_repair_messages(ROW, "bad", {"compiled": True, "diff": "a | b"})
    assert "void f() {}" not in messages[1]["content"]


def test_compile_failure_prompt_carries_the_compiler_note() -> None:
    messages = build_repair_messages(ROW, "bad", {"compiled": False, "note": "error: boom"})
    assert "DIFF:\nCOMPILE ERROR: error: boom" in messages[1]["content"]

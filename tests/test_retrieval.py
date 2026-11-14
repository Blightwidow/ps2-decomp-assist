from eval.retrieval import Retriever, assembly_symbols, retarget

NEIGHBOR = {
    "game": "sly1",
    "function_id": "func_00100000",
    "definition_name": "func_00100000",
    "assembly": "lw $v0, 0x8($a0)\njal func_00100100\nnop\nlui $a1, %hi(D_00200000)\njr $ra\nnop",
    "c_text": "int func_00100000(Obj* a) { return func_00100100(a->f8, D_00200000); }",
}
TARGET = {
    "game": "sly1",
    "function_id": "func_00100500",
    "definition_name": "func_00100500",
    "assembly": "lw $v0, 0x8($a0)\njal func_00100900\nnop\nlui $a1, %hi(D_00300000)\njr $ra\nnop",
    "c_text": "",
}


def test_symbols_in_order_of_first_appearance() -> None:
    assert assembly_symbols(NEIGHBOR["assembly"]) == ["func_00100100", "D_00200000"]


def test_retarget_renames_function_and_symbols() -> None:
    text = retarget(NEIGHBOR, TARGET)
    assert "func_00100500(Obj* a)" in text and "func_00100900(a->f8, D_00300000)" in text


def test_nearest_returns_most_similar_training_row() -> None:
    other = {**NEIGHBOR, "function_id": "func_X", "assembly": "\n".join(f"addu $a{i % 4}, $a1, $a2" for i in range(20))}
    retriever = Retriever([other, NEIGHBOR])
    assert retriever.nearest(TARGET)[0]["function_id"] == "func_00100000"

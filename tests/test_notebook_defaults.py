from __future__ import annotations

import json
from pathlib import Path


def test_game_notebook_has_safe_default_switches_and_valid_code():
    notebook_path = Path(__file__).parents[1] / "notebooks" / "game_testing_notebook.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    compile(code, str(notebook_path), "exec")
    assert "RUN_API_CHECK = False" in code
    assert "RUN_OFFLINE_BENCHMARK = True" in code
    assert "RUN_LIVE_GAME = True" in code
    assert "BLOCK_LIVE_ON_BENCHMARK_FAILURE = False" in code
    assert "PRELOAD_SELECTED_STRATEGIES = True" in code
    assert "PROMPT_FOR_CREDENTIALS = False" in code
    assert "RUN_BEST_SINGLE_ALL_CATEGORIES = True" in code
    assert "RUN_BEST_BY_CATEGORY = True" in code
    assert "Benchmark timings below are warm timings" in code
    assert "ARCHITECTURES = {" in code
    assert "BEST_SINGLE_ARCHITECTURE_KEY" in code
    assert "BEST_BY_CATEGORY_KEYS" in code
    assert "CATEGORY_BENCHMARKS = {" in code
    assert "CATEGORY_PROBES = CATEGORY_BENCHMARKS" in code
    assert "live_plans = make_live_plans()" in code
    assert "TUTORIAL_SET = [" in code
    assert "all_correct = all" in code
    assert "return all_correct and max_seconds <= 20.0 and not has_fallbacks" in code
    assert "Benchmark gate failed. No live game started." in code
    assert "continuing final run" in code
    assert '"kind": "gemma_tool_rag_quant"' in code
    assert '"kind": "mixed_quantized_rag"' in code
    assert '"kind": "qwen_tool_council_quant"' in code
    assert "class DataRoutedFrankenStrategy" in code
    mixed_block = code.split("def mixed_quantized_routed_rag():", 1)[1].split(
        "def gemma_e2b_rag_council_e4b_judge():", 1
    )[0]
    assert "model_id=model_id" not in mixed_block
    assert "self._models = {}" in code
    assert "news_or_report_gemma_rag" in code
    assert "RoutedStrategy" in code
    assert "loaded devices:" in code

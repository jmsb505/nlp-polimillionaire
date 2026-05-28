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
    assert "RUN_LIVE_GAME = False" in code
    assert "RUN_OFFLINE_BENCHMARK = True" in code
    assert "MODELS_TO_RUN = [" in code
    assert '"run": True' not in code
    assert '"kind": "gemma_quantized"' in code
    assert '"kind": "mixed_quantized"' in code
    assert "RoutedStrategy" in code
    assert "loaded-model speed check:" in code

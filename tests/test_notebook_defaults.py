from __future__ import annotations

import csv
import json
from pathlib import Path


def _normalized(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").splitlines()).strip() + "\n"


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
    assert "RUN_OFFLINE_BENCHMARK = False" in code
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
    assert "Set RUN_OFFLINE_BENCHMARK=True before a live 4-bit run." not in code
    assert "Benchmark did not pass; continuing because benchmark gating is disabled." in code
    assert "RAG warm-up failed. No live game started for this plan." in code
    assert "RAG warm-up warning: expected option 2; continuing live run." not in code
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


def test_speech_game_notebook_uses_speech_runner_and_valid_code():
    notebook_path = Path(__file__).parents[1] / "notebooks" / "game_testing_speech_notebook.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    compile(code, str(notebook_path), "exec")
    assert "SpeechGameRunner" in code
    assert "transcriber=speech_transcribe" in code
    assert "SPEECH_AUDIO_FETCH_DELAY_SECONDS" in code
    assert 'SPEECH_WHISPER_DEVICE = "cpu"' in code
    assert "SPEECH_REPLAY_OK = run_speech_replay_check()" in code
    assert "BLOCK_LIVE_ON_BENCHMARK_FAILURE = True" in code
    assert "speech_{clean_name" in code
    assert "GameRunner(client" not in code


def test_submission_notebook_keeps_standalone_harness_and_shared_builders():
    root = Path(__file__).parents[1]
    submission = json.loads((root / "notebooks" / "submission_notebook.ipynb").read_text(encoding="utf-8"))
    game = json.loads((root / "notebooks" / "game_testing_notebook.ipynb").read_text(encoding="utf-8"))

    embedded_helper = ""
    for cell in submission["cells"]:
        source = "".join(cell.get("source", []))
        if "class BaseStrategy" in source and "class GemmaLLM" in source and "class GameRunner" in source:
            embedded_helper = source
            break
    assert embedded_helper
    assert "class BaseStrategy" in embedded_helper
    assert "class GemmaLLM" in embedded_helper
    assert "class GameRunner" in embedded_helper
    assert "quora.com" in embedded_helper
    assert "brainly.com" in embedded_helper

    matching_cells = [
        (14, 20),  # architecture builders
        (16, 22),  # strategy factory
        (18, 24),  # benchmark runner
    ]
    for game_index, submission_index in matching_cells:
        game_source = "".join(game["cells"][game_index].get("source", []))
        submission_source = "".join(submission["cells"][submission_index].get("source", []))
        assert _normalized(game_source) == _normalized(submission_source)


def test_clean_submission_notebook_is_self_contained_and_simpler():
    notebook_path = Path(__file__).parents[1] / "notebooks" / "submission_notebook_clean.ipynb"
    original_path = Path(__file__).parents[1] / "notebooks" / "submission_notebook.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    original = json.loads(original_path.read_text(encoding="utf-8"))

    assert len(notebook["cells"]) < len(original["cells"])
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell.get("cell_type") == "code")

    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    for index, cell in enumerate(code_cells):
        compile("".join(cell.get("source", [])), f"{notebook_path}:code_cell_{index}", "exec")

    code = "\n\n".join("".join(cell.get("source", [])) for cell in code_cells)
    assert "RUN_LIVE_GAME = True" in code
    assert "RUN_OFFLINE_BENCHMARK = False" in code
    assert "RUN_BEST_BY_CATEGORY = True" in code
    assert "COMPETITION_IDS = [0, 1, 2, 3, 4, 5]" in code
    assert "CLEAR_MEMORY_AFTER_EACH_MODEL = True" in code
    assert "HF_HOME" in code
    assert "HUGGINGFACE_HUB_CACHE" in code
    assert "run_results = run_all_categories()" in code
    assert "show_results(run_results)" in code
    assert "SpeechGameRunner" not in code

    hidden_code_cells = [
        cell for cell in code_cells
        if cell.get("metadata", {}).get("jupyter", {}).get("source_hidden")
    ]
    assert len(hidden_code_cells) >= 3

    namespace: dict[str, object] = {}
    exec("".join(hidden_code_cells[0].get("source", [])), namespace)
    exec("".join(hidden_code_cells[1].get("source", [])), namespace)
    Question = namespace["Question"]
    AnswerOption = namespace["AnswerOption"]
    CalculatorStrategy = namespace["CalculatorStrategy"]
    RAGStrategy = namespace["RAGStrategy"]
    FakeLLM = namespace["FakeLLM"]
    route_question = namespace["route_question"]

    bayes_question = Question(
        1,
        "Suppose 4% of the population have a certain disease. A laboratory blood test gives a positive reading for 95% of people who have the disease and for 5% of people who do not have the disease. If a person tests positive, what is the probability the person has the disease?",
        [AnswerOption(0, "0.086"), AnswerOption(1, "0.442"), AnswerOption(2, "0.558"), AnswerOption(3, "0.038")],
    )
    assert CalculatorStrategy().answer(bayes_question).option_id == 1

    haruspex_question = Question(
        2,
        "What term describes a kind of sign interpreted by a haruspex and not an augur, involving coniectura rather than observatio?",
        [AnswerOption(0, "Miraculum"), AnswerOption(1, "Prodigium"), AnswerOption(2, "Portentum"), AnswerOption(3, "Ostentum")],
    )
    decision = route_question(haruspex_question)
    assert decision.route == "rag"
    assert decision.reason == "factual"

    bias_question = Question(
        3,
        "A scientist investigated the effect of workplace stress on heart disease in humans. Men of various ages were divided into two groups based on whether they described their work as very stressful or not very stressful. During the one year investigation the scientist monitored the heart health of each man. What was the bias in this investigation?",
        [AnswerOption(0, "The investigation only lasted one year."), AnswerOption(1, "The age of the participants varied."), AnswerOption(2, "The only organ studied was the heart."), AnswerOption(3, "The investigation tested only men.")],
    )
    assert CalculatorStrategy().answer(bias_question).option_id == 3

    dilemma_question = Question(
        4,
        "What condition must hold true for the payoffs in a Prisoner's Dilemma to ensure mutual cooperation is superior to mutual defection?",
        [AnswerOption(0, "R > T > S > P"), AnswerOption(1, "T > R > P > S"), AnswerOption(2, "R > P and T > R and P > S"), AnswerOption(3, "R > P > T > S")],
    )
    rag = RAGStrategy(
        llm=FakeLLM(['{"option_id": 0, "confidence": 1.0, "reason": "bad"}']),
        retriever=lambda query, cfg, llm: ("Prisoner dilemma evidence", [], 0.0),
    )
    assert rag.answer(dilemma_question).option_id == 2


def test_minimal_submission_notebook_is_trimmed_and_functional():
    notebook_path = Path(__file__).parents[1] / "notebooks" / "submission_notebook_minimal.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell.get("cell_type") == "code")
    assert max(
        len("".join(cell.get("source", [])).splitlines())
        for cell in notebook["cells"]
    ) <= 700

    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    for index, cell in enumerate(code_cells):
        compile("".join(cell.get("source", [])), f"{notebook_path}:code_cell_{index}", "exec")

    code = "\n\n".join("".join(cell.get("source", [])) for cell in code_cells)
    assert "RUN_LIVE_GAME = True" in code
    assert "RUN_LOCAL_BENCHMARK = True" in code
    assert "USE_LOCAL_BENCHMARK_SELECTION = True" in code
    assert "RUN_OFFLINE_BENCHMARK = False" in code
    assert "RUN_BEST_BY_CATEGORY = True" in code
    assert "COMPETITION_IDS = [0, 1, 2, 3, 4, 5]" in code
    assert "CLEAR_MEMORY_AFTER_EACH_MODEL = True" in code
    assert "CLEAR_MEMORY_AFTER_EACH_ARCHITECTURE = True" in code
    assert "SHOW_MODEL_DEVICES = False" in code
    assert "run_results = run_all_categories()" in code
    assert "summary_df = show_results(run_results)" in code
    assert "Local benchmark" in code or "Local benchmark" in "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "LangChainAgenticStrategy" not in code
    assert "DataRoutedFrankenStrategy" not in code
    assert "data-routed" not in code.lower()
    assert "franken" not in code.lower()
    assert "SpeechGameRunner" not in code
    assert "class RandomStrategy" not in code
    assert "class HeuristicStrategy" not in code
    assert "@dataclass" not in code
    assert "class Question" not in code
    assert "class AnswerPrediction" not in code
    assert "class BaseStrategy" not in code

    for expected in [
        '"gemma_e2b_two_agent_quant_council"',
        '"qwen_two_agent_quant_council"',
        '"mixed_gemma_qwen_routed_rag"',
        '"gemma_e2b_routed_rag"',
        '"label": "Gemma E2B"',
        '"label": "Gemma E2B 4-bit"',
        '"label": "Qwen 2B thinking"',
        '"label": "Gemma + Qwen 4-bit + RAG"',
        '"label": "Category router"',
    ]:
        assert expected in code

    namespace: dict[str, object] = {}
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code" and cell.get("metadata", {}).get("jupyter", {}).get("source_hidden"):
            exec("".join(cell.get("source", [])), namespace)

    route_question = namespace["route_question"]
    rule_answer = namespace["rule_answer"]
    normalize_result = namespace["normalize_result"]

    result = normalize_result({"correct": True, "gameOver": False, "earnedAmount": 4000, "currentLevel": 8})
    assert result["game_over"] is False
    assert result["earned_amount"] == 4000
    assert result["current_level"] == 8

    assert "from ddgs import DDGS" in code
    assert '"duckduckgo-search"' not in code
    assert "generation_token_ids" in code
    assert 'kwargs["pad_token_id"]' in code
    assert "set_verbosity_error()" in code

    bayes_question = {
        "id": 1,
        "text": "Suppose 4% of the population have a certain disease. A laboratory blood test gives a positive reading for 95% of people who have the disease and for 5% of people who do not have the disease. If a person tests positive, what is the probability the person has the disease?",
        "options": [{"id": 0, "text": "0.086"}, {"id": 1, "text": "0.442"}, {"id": 2, "text": "0.558"}, {"id": 3, "text": "0.038"}],
    }
    assert rule_answer(bayes_question)["option_id"] == 1

    haruspex_question = {
        "id": 2,
        "text": "What term describes a kind of sign interpreted by a haruspex and not an augur, involving coniectura rather than observatio?",
        "options": [{"id": 0, "text": "Miraculum"}, {"id": 1, "text": "Prodigium"}, {"id": 2, "text": "Portentum"}, {"id": 3, "text": "Ostentum"}],
    }
    decision = route_question(haruspex_question)
    assert decision["route"] == "rag"
    assert decision["reason"] == "factual"

    zoo_tv_question = {
        "id": 5,
        "text": "What was the primary goal of U2's music during the Zoo TV tour in 1992-1993?",
        "options": [{"id": 0, "text": "To explore electronic dance music"}, {"id": 1, "text": "To promote their live performances"}, {"id": 2, "text": "To support the release of a new album"}, {"id": 3, "text": "To challenge television in daily life"}],
    }
    assert route_question(zoo_tv_question)["route"] == "rag"

    dated_news_question = {
        "id": 6,
        "text": "According to the article published on 2026-05-14, which company conducted the experiment?",
        "options": [{"id": 0, "text": "Google"}, {"id": 1, "text": "JP Morgan"}, {"id": 2, "text": "Walmart"}, {"id": 3, "text": "Emergence AI"}],
    }
    assert route_question(dated_news_question)["route"] == "rag"

    bias_question = {
        "id": 3,
        "text": "A scientist investigated the effect of workplace stress on heart disease in humans. Men of various ages were divided into two groups based on whether they described their work as very stressful or not very stressful. During the one year investigation the scientist monitored the heart health of each man. What was the bias in this investigation?",
        "options": [{"id": 0, "text": "The investigation only lasted one year."}, {"id": 1, "text": "The age of the participants varied."}, {"id": 2, "text": "The only organ studied was the heart."}, {"id": 3, "text": "The investigation tested only men."}],
    }
    assert rule_answer(bias_question)["option_id"] == 3

    dilemma_question = {
        "id": 4,
        "text": "What condition must hold true for the payoffs in a Prisoner's Dilemma to ensure mutual cooperation is superior to mutual defection?",
        "options": [{"id": 0, "text": "R > T > S > P"}, {"id": 1, "text": "T > R > P > S"}, {"id": 2, "text": "R > P and T > R and P > S"}, {"id": 3, "text": "R > P > T > S"}],
    }
    assert rule_answer(dilemma_question)["option_id"] == 2

    square_root_question = {
        "id": 7,
        "text": "What is the least possible positive integer value of n such that sqrt{18 n} is an integer?",
        "options": [{"id": 0, "text": "8"}, {"id": 1, "text": "3"}, {"id": 2, "text": "2"}, {"id": 3, "text": "10"}],
    }
    assert rule_answer(square_root_question)["option_id"] == 2

    shadow_question = {
        "id": 8,
        "text": "The BEST way to make a shadow on a living room wall is to",
        "options": [{"id": 0, "text": "stand between a light and the wall"}, {"id": 1, "text": "hold a lamp near the floor"}, {"id": 2, "text": "sit close to the wall near a window"}, {"id": 3, "text": "turn off the lights in the room"}],
    }
    assert rule_answer(shadow_question)["option_id"] == 0

    electrical_question = {
        "id": 9,
        "text": "Which is an example of a form of electrical energy?",
        "options": [{"id": 0, "text": "sound waves"}, {"id": 1, "text": "windmill"}, {"id": 2, "text": "radio waves"}, {"id": 3, "text": "lightning"}],
    }
    assert rule_answer(electrical_question)["option_id"] == 3


def test_local_benchmark_csv_has_required_shape():
    root = Path(__file__).parents[1]
    csv_path = root / "data" / "local_benchmark" / "question_bank.csv"
    assert csv_path.exists()

    required = {
        "bank_id",
        "category_id",
        "category",
        "source_log",
        "question",
        "option_0",
        "option_1",
        "option_2",
        "option_3",
        "gold_option_id",
        "topic_tag",
        "why_included",
    }
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert rows
    assert required.issubset(rows[0])

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
        assert row["question"].strip()
        assert int(row["gold_option_id"]) in {0, 1, 2, 3}
        for option_index in range(4):
            assert row[f"option_{option_index}"].strip()

    assert len(counts) == 6
    assert all(count >= 5 for count in counts.values())


def test_minimal_submission_setup_names_are_clean():
    root = Path(__file__).parents[1]
    notebook_path = root / "notebooks" / "submission_notebook_minimal.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    settings_source = "".join(notebook["cells"][7].get("source", []))

    namespace: dict[str, object] = {"os": __import__("os"), "pd": __import__("pandas")}
    exec(settings_source, namespace)
    architectures = namespace["ARCHITECTURES"]

    for key, item in architectures.items():
        label = item["label"]
        assert "franken" not in label.lower()
        assert "data-routed" not in label.lower()
        if "4-bit" in label:
            assert item.get("quantized") is True, key

    expected_labels = {
        "Heuristic baseline",
        "Gemma E2B",
        "Gemma E4B",
        "Qwen 2B",
        "Qwen 2B thinking",
        "Gemma E2B 4-bit",
        "Gemma E4B 4-bit",
        "Qwen 2B 4-bit",
        "Gemma E2B 4-bit + RAG",
        "Gemma E4B 4-bit + RAG",
        "Qwen 2B 4-bit + RAG",
        "Gemma 4-bit council",
        "Qwen 4-bit council",
        "Gemma + Qwen council",
        "Gemma + Qwen 4-bit council",
        "Gemma + Qwen 4-bit + RAG",
        "Category router",
    }
    assert {item["label"] for item in architectures.values()} == expected_labels


def test_minimal_submission_benchmark_selection_works_with_fake_rows():
    root = Path(__file__).parents[1]
    notebook_path = root / "notebooks" / "submission_notebook_minimal.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    namespace: dict[str, object] = {}
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code" and cell.get("metadata", {}).get("jupyter", {}).get("source_hidden"):
            exec("".join(cell.get("source", [])), namespace)

    pd = namespace["pd"]
    select_benchmark_setups = namespace["select_benchmark_setups"]
    fake_results = pd.DataFrame(
        [
            {"setup_key": "a", "setup": "Setup A", "category_id": 0, "category": "entertainment", "correct": True, "seconds": 2.0, "fallback": False},
            {"setup_key": "a", "setup": "Setup A", "category_id": 1, "category": "history", "correct": False, "seconds": 1.0, "fallback": False},
            {"setup_key": "b", "setup": "Setup B", "category_id": 0, "category": "entertainment", "correct": False, "seconds": 1.0, "fallback": False},
            {"setup_key": "b", "setup": "Setup B", "category_id": 1, "category": "history", "correct": True, "seconds": 1.0, "fallback": False},
            {"setup_key": "b", "setup": "Setup B", "category_id": 1, "category": "history", "correct": True, "seconds": 1.1, "fallback": False},
        ]
    )

    selected_by_category, selected_overall = select_benchmark_setups(fake_results)
    assert selected_by_category[0] == "a"
    assert selected_by_category[1] == "b"
    assert selected_overall == "b"


def test_minimal_submission_live_functions_do_not_read_benchmark_bank():
    root = Path(__file__).parents[1]
    notebook_path = root / "notebooks" / "submission_notebook_minimal.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    answer_source = "".join(notebook["cells"][15].get("source", []))
    runner_source = "".join(notebook["cells"][18].get("source", []))

    assert "load_question_bank" not in answer_source
    assert "load_question_bank" not in runner_source
    assert "QUESTION_BANK_ROWS" not in answer_source
    assert "QUESTION_BANK_ROWS" not in runner_source

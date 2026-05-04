# Repo Structure

This is the repo in plain English.

## Root Files

- `README.md` explains how to run the notebook and what the project needs.
- `WORKFLOWS.md` explains the common things we edit, like heuristics, Gemma prompts, and live runs.
- `REPO_STRUCTURE.md` is this quick map of the repo.
- `requirements.txt` lists the Python packages needed by the notebook and tests.
- `pytest.ini` tells pytest where the tests are.
- `.gitignore` keeps caches, model files, logs, env files, and exported outputs out of git.

## `notebooks/`

- `notebooks/` contains the notebooks we actually run.
- `notebooks/sample_notebook.ipynb` tests selected local models on one sample question.
- `notebooks/game_testing_notebook.ipynb` runs selected local models through the real game API one after another.

## `src/`

- `src/` contains our small helper package so the notebook does not become a giant code dump.

## `src/polimillionaire/`

- `src/polimillionaire/` contains the few helper modules we wrote for the assignment.
- `src/polimillionaire/__init__.py` makes the main helpers easy to import.
- `src/polimillionaire/types.py` defines the shared question, option, and answer dataclasses.
- `src/polimillionaire/strategies.py` contains model loading, prompt building, output parsing, and answer strategies.
- `src/polimillionaire/runner.py` contains the live game runner, safe delay, fallback logic, JSONL logging, and result summaries.

## `millionaire_client/`

- `millionaire_client/` is the provided API client, kept unchanged.
- `millionaire_client/__init__.py` exposes the provided client package.
- `millionaire_client/base.py` handles the low-level HTTP requests.
- `millionaire_client/client.py` is the main client object used by the notebook.
- `millionaire_client/auth.py` handles login and authentication.
- `millionaire_client/game.py` handles game sessions, questions, and answer submission.
- `millionaire_client/competitions.py` lists and reads competition data.
- `millionaire_client/leaderboard.py` reads leaderboard data.
- `millionaire_client/models.py` defines the client-side response models.
- `millionaire_client/exceptions.py` defines the client errors.

## `tests/`

- `tests/` contains small checks for the helper modules.
- `tests/conftest.py` defines shared test fixtures.
- `tests/test_strategy_contract.py` checks that strategies return valid answers.
- `tests/test_gemma_output_parser.py` checks that Gemma output parsing handles good and bad model text.
- `tests/test_game_runner.py` checks fallback behavior, live-run logging, and submission-error handling.
- `tests/test_evaluation.py` checks the result summary helper.
- `tests/test_gemma_smoke.py` is an optional real Gemma test that only runs when explicitly enabled.

## `results/`

- `results/` stores outputs from runs.
- `results/runs/` stores JSONL logs from notebook and live game runs.
- `results/runs/heuristic_notebook.jsonl` is an old heuristic run log.
- `results/runs/gemma_notebook.jsonl` is an old Gemma notebook run log.
- `results/runs/gemma_20260503_125406.jsonl` is the successful Gemma live run log used for analysis.

## `data/`

- `data/` stores local model/cache files and is mostly ignored by git.
- `data/hf_home/` is the local Hugging Face cache where Gemma weights can live.

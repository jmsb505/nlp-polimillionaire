# Workflows

Keep it simple: run the notebook, edit helpers only when needed.

## Main Flow

1. Open `notebooks/sample_notebook.ipynb`.
2. Run setup.
3. Pick models in `MODELS_TO_TEST`.
4. Run the selected local models on the sample question.
5. Open `notebooks/game_testing_notebook.ipynb`.
6. Pick models and competition.
7. Login and run the selected models one after another.
8. Read `results/runs/`.

## Add A New Heuristic

Edit `src/polimillionaire/strategies.py`.

Add a class like this:

```python
class MyHeuristicStrategy(BaseStrategy):
    name = "my_heuristic"

    def answer(self, question: Question) -> AnswerPrediction:
        option = question.first_option()
        return AnswerPrediction(
            option_id=option.id,
            answer_text=option.text,
            confidence=0.4,
            reasoning="Rule selected this option.",
            metadata={"strategy": self.name},
        )
```

Then import it in the game notebook if you want to use it live.

Run:

```powershell
python -m pytest
```

## Improve The Current Heuristic

Change `HeuristicStrategy.answer()` in `strategies.py`.

Useful small ideas:

- ignore stopwords;
- treat `NOT` questions differently;
- reward exact phrase overlap;
- penalize extreme wording like `only` or `never`.

Keep it fast.

## Change Model Prompting

Edit `build_prompt()` for Gemma or `build_qwen_prompt()` for Qwen thinking in `strategies.py`.

Keep the prompt short:

- one answer only;
- one-line JSON;
- no long reasoning for Gemma; Qwen is the thinking experiment.

After editing:

1. run tests;
2. run the model test notebook;
3. run the game notebook only if `fallback: False`.

## Try The Council

In a notebook model list, set `run=True` for `Gemma 4 E2B Council`.

It runs one loaded Gemma model three times with sampling and different seeds. A majority answer is used directly. The deterministic judge is only used if the votes do not have a majority.

Use the sample notebook first. In a live game it makes four generations per question, so latency matters.

For a different model perspective, choose `Gemma + Qwen Mixed Council (4-bit)` in the game notebook. Install `bitsandbytes` in that notebook kernel first:

```python
%pip install -U bitsandbytes
```

It uses the class NF4/double-quantization recipe so Gemma and short non-thinking Qwen can fit together more reliably. It is another `MODELS_TO_RUN` choice, not a separate pipeline. Turn on `RUN_OFFLINE_BENCHMARK` before running this option live.

To compare it fairly, run `Gemma 4 E2B (4-bit)` and `Gemma + Qwen Mixed Council (4-bit)` with `RUN_LIVE_GAME = False`. Only this quantized mixed option restricts its judge to candidate answers and falls back to the Gemma vote instead of comparing model confidence scores. Existing council choices keep their earlier behavior.

## Change Live Game Logic

Edit `src/polimillionaire/runner.py`.

This file handles:

- API question conversion;
- strategy timeout fallback;
- safe delay;
- JSONL logs;
- result summaries.

## Read Logs

Logs are JSONL files:

```text
results/runs/
```

The notebook uses `load_jsonl()` and `summarize_attempts()`.

## Before Submission

- fill group info;
- add video link;
- add academic integrity statement;
- run notebook cleanly;
- export to HTML.

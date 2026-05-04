# Workflows

Keep it simple: run the notebook, edit helpers only when needed.

## Main Flow

1. Open `notebooks/sample_notebook.ipynb`.
2. Run setup.
3. Pick models in `MODELS_TO_TEST`.
4. Run the selected local models on the sample question.
5. Open `notebooks/game_testing_notebook.ipynb`.
6. Pick models and competition.
7. Login and run one live game.
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

## Change Gemma Prompting

Edit `build_prompt()` in `strategies.py`.

Keep the prompt short:

- one answer only;
- one-line JSON;
- no long reasoning.

After editing:

1. run tests;
2. run the model test notebook;
3. run the game notebook only if `fallback: False`.

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

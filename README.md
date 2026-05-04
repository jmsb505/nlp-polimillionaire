# PoliMillionaire NLP Assignment

The notebooks are the main interface.

Use:

```text
notebooks/sample_notebook.ipynb
```

for local model tests on one sample question.

Use:

```text
notebooks/game_testing_notebook.ipynb
```

for real PoliMillionaire game runs.

The helper code lives in `src/polimillionaire/`.

## Helper Files

```text
src/polimillionaire/types.py       shared dataclasses
src/polimillionaire/strategies.py  model loading, prompting, parsing, strategies
src/polimillionaire/runner.py      live game runner, logs, summaries
```

The provided `millionaire_client/` package is unchanged.

## Install

Install in the same environment used by the notebook kernel:

```powershell
pip install -r requirements.txt
```

Gemma 4 needs `transformers>=5.7.0`.

## Credentials

Do not save credentials in the repo.

```powershell
$env:POLIMILLIONAIRE_USERNAME="your_username"
$env:POLIMILLIONAIRE_PASSWORD="your_password"
```

## Logs

Live game logs are saved in:

```text
results/runs/
```

## Tests

```powershell
python -m pytest
```

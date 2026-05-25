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
src/polimillionaire/strategies.py  Gemma/Qwen loading, prompting, parsing, strategies
src/polimillionaire/runner.py      live game runner, logs, summaries
```

The `millionaire_client/` package comes from the supplied API client ZIP and now includes text and speech mode support.

## Install

Install in the same environment used by the notebook kernel:

```powershell
pip install -r requirements.txt
```

Gemma 4 and Qwen3.5 need `transformers>=5.7.0`.

The notebooks can test `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`, and `Qwen/Qwen3.5-2B` locally.

The game notebook also has an optional 4-bit Gemma + Qwen mixed council based on the class quantization recipe.

For the 4-bit council only, install:

```python
%pip install -U bitsandbytes
```

Then restart the notebook kernel. The game notebook is set up to compare 4-bit Gemma alone with the 4-bit mixed council offline before a live run.

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

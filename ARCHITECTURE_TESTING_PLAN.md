# Architecture Testing Plan

We are not locking the final setup yet.

The old raw single-model tests are not the main comparison anymore because they do not use the newer tools, routing, RAG, and council logic. The notebook now compares architectures that include the current harness.

## Exploratory Live Testing Mode

```python
RUN_LIVE_GAME = True
BLOCK_LIVE_ON_BENCHMARK_FAILURE = False
```

This runs every current architecture across all six live categories. The benchmark still runs first so we have controlled reference scores, but a failed benchmark does not stop the live exploratory run.

## Architectures To Compare

| Architecture | Purpose |
| --- | --- |
| Tools + heuristic fallback | Fast floor baseline for calculators and parser checks. |
| Gemma E2B 4-bit tools + routed RAG | Single local Gemma with the current tool/RAG harness. |
| Qwen 3.5 2B 4-bit tools + routed RAG | Single local Qwen with the same harness. |
| Gemma E2B 4-bit tools + RAG council | Gemma council with tools and retrieval. |
| Qwen 3.5 2B 4-bit tools + RAG council | Qwen council with tools and retrieval. |
| Data-routed Gemma/Qwen/tools franken | Router picks tools, Gemma, Qwen, or RAG by question type. |
| Gemma E4B 4-bit tools + routed RAG | Heavier single-model test. |
| Gemma + Qwen 4-bit mixed routed RAG | Mixed-model test for memory and accuracy. |

## Why 4-bit By Default

The machine has limited VRAM for repeated local tests. The serious comparison rows use 4-bit quantization because it gives us enough room to test routing and councils without constantly offloading to CPU or disk.

Full precision single-model rows stay in the notebook as legacy/manual checks, but they are not part of the default architecture grid.

## Selection Criteria After Exploration

For each category, pick the architecture with:

- highest live category accuracy;
- good benchmark category accuracy;
- no fallback on category probes;
- warm max latency under 20 seconds;
- no CPU/disk/meta offload;
- acceptable live API behavior once tested on that category.

Only after the exploratory live run should we choose what to change or which category-specific architectures to test again.

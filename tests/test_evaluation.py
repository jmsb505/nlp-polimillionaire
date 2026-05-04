from __future__ import annotations

from polimillionaire.runner import summarize_attempts


def test_summarize_attempts_counts_accuracy():
    summary = summarize_attempts(
        [
            {"elapsed_seconds": 1.0, "result": {"correct": True, "timed_out": False}},
            {"elapsed_seconds": 3.0, "result": {"correct": False, "timed_out": True}},
        ]
    )
    assert summary["total"] == 2
    assert summary["correct"] == 1
    assert summary["accuracy"] == 0.5
    assert summary["timed_out"] == 1
    assert summary["avg_elapsed_seconds"] == 2.0

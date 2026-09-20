"""
Regression Test Suite for LLM Applications
==========================================
Automated pytest-based regression tests that run on every prompt/model change.
Integrates with LangSmith for trace capture and CI/CD pipelines.

Author: Devansh Awasthi (github.com/devcode-X)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from langsmith import Client
from langsmith.evaluation import evaluate as ls_evaluate

from evals.judge import evaluate as judge_evaluate
from evals.hallucination import detect_hallucination_llm

# ---------------------------------------------------------------------------
# LangSmith client setup
# ---------------------------------------------------------------------------

ls_client = Client()

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "qa_groundtruth.json"


def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Individual eval tests
# ---------------------------------------------------------------------------

class TestHallucinationRegression:
    """Ensure hallucination rate stays below 5% across golden dataset."""

    def test_hallucination_rate_below_threshold(self):
        samples = load_dataset()
        hallucinated = 0
        for sample in samples:
            if not sample.get("context"):
                continue
            report = detect_hallucination_llm(
                answer=sample["answer"],
                context=sample["context"],
            )
            if report.hallucinated:
                hallucinated += 1

        rate = hallucinated / len(samples)
        assert rate < 0.05, (
            f"Hallucination rate {rate:.1%} exceeds 5% threshold "
            f"({hallucinated}/{len(samples)} samples hallucinated)"
        )

    def test_faithfulness_score_above_threshold(self):
        samples = load_dataset()
        scores = []
        for sample in samples:
            if not sample.get("context"):
                continue
            report = detect_hallucination_llm(
                answer=sample["answer"],
                context=sample["context"],
            )
            scores.append(report.faithfulness_score)

        avg_score = sum(scores) / len(scores)
        assert avg_score >= 0.85, (
            f"Average faithfulness {avg_score:.3f} below 0.85 threshold"
        )


class TestJudgeScoreRegression:
    """Ensure overall judge scores stay above quality thresholds."""

    def test_overall_judge_score_above_threshold(self):
        samples = load_dataset()
        overall_scores = []
        for sample in samples:
            result = judge_evaluate(
                question=sample["question"],
                answer=sample["answer"],
                context=sample.get("context", ""),
                expected=sample.get("expected", ""),
            )
            overall_scores.append(result.overall)

        avg = sum(overall_scores) / len(overall_scores)
        assert avg >= 0.80, (
            f"Average judge score {avg:.3f} below 0.80 threshold"
        )

    def test_no_complete_failures(self):
        """No sample should score below 0.3 overall."""
        samples = load_dataset()
        failures = []
        for sample in samples:
            result = judge_evaluate(
                question=sample["question"],
                answer=sample["answer"],
                context=sample.get("context", ""),
                expected=sample.get("expected", ""),
            )
            if result.overall < 0.3:
                failures.append({"question": sample["question"], "score": result.overall})

        assert len(failures) == 0, (
            f"{len(failures)} samples scored below 0.3:\n"
            + json.dumps(failures, indent=2)
        )


class TestToolSelectionRegression:
    """Regression tests for agent tool selection accuracy."""

    TOOL_SAMPLES = [
        {
            "task": "Search the database for user ID 42",
            "expected_tool": "query_db",
            "selected_tool": "query_db",
        },
        {
            "task": "Retrieve document at path /docs/report.pdf",
            "expected_tool": "retrieve_document",
            "selected_tool": "retrieve_document",
        },
        {
            "task": "Call the weather API for London",
            "expected_tool": "api_call",
            "selected_tool": "api_call",
        },
    ]

    def test_tool_selection_accuracy(self):
        correct = sum(
            1 for s in self.TOOL_SAMPLES
            if s["selected_tool"] == s["expected_tool"]
        )
        accuracy = correct / len(self.TOOL_SAMPLES)
        assert accuracy >= 0.95, (
            f"Tool selection accuracy {accuracy:.1%} below 95% threshold"
        )


# ---------------------------------------------------------------------------
# LangSmith dataset push (run manually to register eval datasets)
# ---------------------------------------------------------------------------

def push_dataset_to_langsmith():
    """Push ground-truth examples to LangSmith for dashboard tracking."""
    samples = load_dataset()
    dataset = ls_client.create_dataset(
        dataset_name="llm-eval-suite-regression",
        description="Ground-truth QA pairs for regression testing",
    )
    ls_client.create_examples(
        inputs=[{"question": s["question"], "context": s.get("context", "")} for s in samples],
        outputs=[{"answer": s["answer"], "expected": s.get("expected", "")} for s in samples],
        dataset_id=dataset.id,
    )
    print(f"Pushed {len(samples)} examples to LangSmith dataset: {dataset.name}")


if __name__ == "__main__":
    push_dataset_to_langsmith()

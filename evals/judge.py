"""
LLM-as-a-Judge Evaluator
========================
Uses GPT-4o as a judge to score LLM outputs across multiple dimensions:
  - Faithfulness (hallucination detection)
  - Relevancy
  - Tool selection accuracy
  - Task completion

Author: Devansh Awasthi (github.com/devcode-X)
"""

from __future__ import annotations

import json
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI()

# ---------------------------------------------------------------------------
# Pydantic schemas for structured judge output
# ---------------------------------------------------------------------------

class JudgeScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Score between 0 and 1")
    verdict: Literal["pass", "fail", "borderline"]
    reasoning: str = Field(..., description="Step-by-step reasoning for the score")


class EvalResult(BaseModel):
    faithfulness: JudgeScore
    relevancy: JudgeScore
    completeness: JudgeScore
    overall: float
    hallucinated: bool


# ---------------------------------------------------------------------------
# Judge prompt templates
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = """\
You are an expert evaluator assessing whether an AI-generated answer is faithful to the provided context.

CONTEXT:
{context}

QUESTION:
{question}

AI ANSWER:
{answer}

Evaluate faithfulness: Does the answer contain ONLY information supported by the context?
Score 1.0 = fully grounded, 0.0 = completely hallucinated.

Respond in JSON with keys: score (float 0-1), verdict (pass/fail/borderline), reasoning (string).
"""

RELEVANCY_PROMPT = """\
You are an expert evaluator assessing whether an AI answer is relevant to the question.

QUESTION:
{question}

AI ANSWER:
{answer}

Score 1.0 = perfectly relevant, 0.0 = completely off-topic.
Respond in JSON with keys: score (float 0-1), verdict (pass/fail/borderline), reasoning (string).
"""

COMPLETENESS_PROMPT = """\
You are an expert evaluator. Given the question and expected answer, assess completeness of the AI answer.

QUESTION:
{question}

EXPECTED ANSWER (ground truth):
{expected}

AI ANSWER:
{answer}

Score 1.0 = covers all key points, 0.0 = misses everything.
Respond in JSON with keys: score (float 0-1), verdict (pass/fail/borderline), reasoning (string).
"""


# ---------------------------------------------------------------------------
# Core judge functions
# ---------------------------------------------------------------------------

def _call_judge(prompt: str) -> JudgeScore:
    """Call GPT-4o with a structured output parse."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = json.loads(response.choices[0].message.content)
    return JudgeScore(**raw)


def evaluate(
    question: str,
    answer: str,
    context: str = "",
    expected: str = "",
    faithfulness_threshold: float = 0.7,
) -> EvalResult:
    """
    Run full LLM-as-a-Judge evaluation on a single (question, answer) pair.

    Args:
        question: The user question or task.
        answer:   The LLM / agent response to evaluate.
        context:  Retrieved context (for faithfulness / hallucination check).
        expected: Ground-truth expected answer (for completeness).
        faithfulness_threshold: Minimum score to consider answer non-hallucinated.

    Returns:
        EvalResult with per-dimension scores and hallucination flag.
    """
    faithfulness = _call_judge(
        FAITHFULNESS_PROMPT.format(context=context, question=question, answer=answer)
    ) if context else JudgeScore(score=1.0, verdict="pass", reasoning="No context provided — skipped.")

    relevancy = _call_judge(
        RELEVANCY_PROMPT.format(question=question, answer=answer)
    )

    completeness = _call_judge(
        COMPLETENESS_PROMPT.format(question=question, expected=expected, answer=answer)
    ) if expected else JudgeScore(score=1.0, verdict="pass", reasoning="No ground truth provided — skipped.")

    overall = round(
        (faithfulness.score + relevancy.score + completeness.score) / 3, 4
    )

    return EvalResult(
        faithfulness=faithfulness,
        relevancy=relevancy,
        completeness=completeness,
        overall=overall,
        hallucinated=faithfulness.score < faithfulness_threshold,
    )


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def batch_evaluate(samples: list[dict]) -> list[dict]:
    """
    Evaluate a list of samples. Each sample should have:
      question, answer, context (optional), expected (optional).

    Returns list of result dicts with scores appended.
    """
    results = []
    for i, sample in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] Evaluating: {sample['question'][:60]}...")
        result = evaluate(
            question=sample["question"],
            answer=sample["answer"],
            context=sample.get("context", ""),
            expected=sample.get("expected", ""),
        )
        results.append({**sample, "eval": result.model_dump()})
    return results


if __name__ == "__main__":
    # Quick smoke test
    sample = {
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris, which is also the largest city.",
        "context": "France is a country in Western Europe. Its capital city is Paris.",
        "expected": "Paris",
    }
    result = evaluate(**{k: sample[k] for k in ["question", "answer", "context", "expected"]})
    print(result.model_dump_json(indent=2))

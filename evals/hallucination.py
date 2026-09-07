"""
Hallucination Detection
=======================
Reference-based hallucination detection for RAG and agent pipelines.
Supports both LLM-judge and embedding-similarity approaches.

Author: Devansh Awasthi (github.com/devcode-X)
"""

from __future__ import annotations

import numpy as np
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()


class HallucinationReport(BaseModel):
    hallucinated: bool
    confidence: float          # 0-1, how confident we are in the verdict
    unsupported_claims: list[str]
    supported_claims: list[str]
    faithfulness_score: float


def _get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    return client.embeddings.create(input=text, model=model).data[0].embedding


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


def detect_hallucination_embedding(
    answer: str,
    context: str,
    threshold: float = 0.75,
) -> HallucinationReport:
    """
    Fast embedding-similarity hallucination check.
    Uses cosine similarity between answer and context embeddings.
    Low similarity → likely hallucination.
    """
    answer_emb = _get_embedding(answer)
    context_emb = _get_embedding(context)
    similarity = _cosine_similarity(answer_emb, context_emb)

    hallucinated = similarity < threshold
    return HallucinationReport(
        hallucinated=hallucinated,
        confidence=abs(similarity - threshold) + 0.5,
        unsupported_claims=[] if not hallucinated else ["Embedding similarity below threshold"],
        supported_claims=[] if hallucinated else ["Embedding similarity above threshold"],
        faithfulness_score=round(similarity, 4),
    )


def detect_hallucination_llm(
    answer: str,
    context: str,
) -> HallucinationReport:
    """
    Detailed LLM-based hallucination detection.
    Decomposes the answer into atomic claims and checks each against context.
    """
    prompt = f"""You are a hallucination detection expert.

CONTEXT (the only source of truth):
{context}

AI ANSWER TO EVALUATE:
{answer}

Task:
1. Break the answer into individual atomic claims.
2. For each claim, determine if it is SUPPORTED or UNSUPPORTED by the context.
3. Return a JSON object with:
   - "supported_claims": list of strings
   - "unsupported_claims": list of strings
   - "faithfulness_score": float (0-1, fraction of claims supported)
   - "hallucinated": bool (true if any unsupported claims exist)
   - "confidence": float (0-1, your confidence in this verdict)

Return ONLY valid JSON."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )

    import json
    raw = json.loads(response.choices[0].message.content)
    return HallucinationReport(**raw)


def run_hallucination_eval(
    samples: list[dict],
    method: str = "llm",
) -> dict:
    """
    Run hallucination eval on a batch of samples.

    Args:
        samples: List of {"answer": str, "context": str} dicts.
        method:  "llm" (accurate) or "embedding" (fast).

    Returns:
        Summary dict with hallucination rate and per-sample reports.
    """
    reports = []
    for s in samples:
        if method == "embedding":
            report = detect_hallucination_embedding(s["answer"], s["context"])
        else:
            report = detect_hallucination_llm(s["answer"], s["context"])
        reports.append(report.model_dump())

    hallucination_rate = sum(r["hallucinated"] for r in reports) / len(reports)
    avg_faithfulness = sum(r["faithfulness_score"] for r in reports) / len(reports)

    return {
        "total_samples": len(reports),
        "hallucination_rate": round(hallucination_rate, 4),
        "avg_faithfulness_score": round(avg_faithfulness, 4),
        "reports": reports,
    }

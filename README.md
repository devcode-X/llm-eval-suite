# LLM Eval Suite 🧪

> **Agentic Reliability & Hallucination Detection Framework**  
> Structured LLM evaluations with LLM-as-a-Judge scoring, automated regression testing, and LangSmith observability.

---

## Overview

A comprehensive evaluation framework for LLM applications and AI agents, built around three core pillars:

1. **Correctness & Hallucination Detection** — Reference-based evaluation comparing model outputs against ground-truth context
2. **LLM-as-a-Judge** — GPT-4o-powered scoring rubrics for tool selection, structured output quality, and task completion
3. **Regression Testing Pipeline** — Automated eval runs to catch regressions when prompts, models, or agent workflows change

---

## Features

- ✅ Ground-truth dataset builder (JSON / CSV)
- ✅ LLM-as-a-Judge evaluator (hallucination, faithfulness, relevancy, tool accuracy)
- ✅ LangSmith integration for run tracing and eval dashboards
- ✅ Precision / Recall / F1 metrics per evaluation dimension
- ✅ Automated regression tests via `pytest`
- ✅ Human-in-the-loop annotation support
- ✅ CLI runner for CI/CD integration

---

## Project Structure

```
llm-eval-suite/
├── evals/
│   ├── judge.py            # LLM-as-a-Judge evaluator
│   ├── hallucination.py    # Reference-based hallucination detection
│   ├── tool_accuracy.py    # Tool selection + structured output eval
│   └── regression.py       # Automated regression test suite
├── datasets/
│   ├── qa_groundtruth.json # QA ground-truth examples
│   └── agent_traces.json   # Agent execution trace samples
├── scripts/
│   ├── build_dataset.py    # Dataset builder utility
│   └── run_evals.py        # CLI eval runner
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/devcode-X/llm-eval-suite.git
cd llm-eval-suite
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
export LANGCHAIN_API_KEY=ls-...
export LANGCHAIN_TRACING_V2=true

# Run full evaluation suite
python scripts/run_evals.py --dataset datasets/qa_groundtruth.json

# Run regression tests
pytest evals/regression.py -v
```

---

## Evaluation Metrics

| Metric | Method | Target |
|---|---|---|
| Hallucination Rate | Reference-based LLM judge | < 5% |
| Answer Faithfulness | RAGAS-style context grounding | > 0.90 |
| Tool Selection Accuracy | Ground-truth trace comparison | > 0.95 |
| Task Completion Rate | End-to-end agent eval | > 0.88 |
| Consistency (cross-run) | Multi-sample variance check | σ < 0.05 |

---

## Tech Stack

`Python` · `OpenAI API` · `LangSmith` · `LangChain` · `Pandas` · `pytest` · `Pydantic`

---

## Author

**Devansh Awasthi** · [GitHub](https://github.com/devcode-X) · [LinkedIn](https://www.linkedin.com/in/devansh-awasthi-a828102b6/)

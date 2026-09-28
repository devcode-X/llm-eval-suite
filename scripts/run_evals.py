"""
Eval Suite CLI Runner
=====================
Run the full evaluation pipeline from the command line.

Usage:
    python scripts/run_evals.py --dataset datasets/qa_groundtruth.json
    python scripts/run_evals.py --dataset datasets/qa_groundtruth.json --method embedding
    python scripts/run_evals.py --dataset datasets/qa_groundtruth.json --output results.json

Author: Devansh Awasthi (github.com/devcode-X)
"""

import argparse
import json
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.judge import batch_evaluate
from evals.hallucination import run_hallucination_eval


def main():
    parser = argparse.ArgumentParser(description="LLM Eval Suite CLI Runner")
    parser.add_argument("--dataset", required=True, help="Path to ground-truth JSON dataset")
    parser.add_argument("--method", default="llm", choices=["llm", "embedding"],
                        help="Hallucination detection method (default: llm)")
    parser.add_argument("--output", default=None, help="Output JSON file for results (optional)")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM judge evaluation")
    parser.add_argument("--skip-hallucination", action="store_true", help="Skip hallucination evaluation")
    args = parser.parse_args()

    # Load dataset
    with open(args.dataset) as f:
        samples = json.load(f)
    print(f"\n📂 Loaded {len(samples)} samples from {args.dataset}\n")

    results = {"dataset": args.dataset, "n_samples": len(samples)}

    # --- LLM Judge Evaluation ---
    if not args.skip_judge:
        print("🧑‍⚖️  Running LLM-as-a-Judge evaluation...")
        judge_results = batch_evaluate(samples)
        avg_overall = sum(r["eval"]["overall"] for r in judge_results) / len(judge_results)
        results["judge"] = {
            "avg_overall_score": round(avg_overall, 4),
            "per_sample": judge_results,
        }
        print(f"   ✅ Avg judge score: {avg_overall:.4f}\n")

    # --- Hallucination Evaluation ---
    if not args.skip_hallucination:
        print(f"🔍 Running hallucination detection (method={args.method})...")
        hal_samples = [s for s in samples if s.get("context")]
        if hal_samples:
            hal_report = run_hallucination_eval(hal_samples, method=args.method)
            results["hallucination"] = hal_report
            print(f"   ✅ Hallucination rate: {hal_report['hallucination_rate']:.1%}")
            print(f"   ✅ Avg faithfulness:   {hal_report['avg_faithfulness_score']:.4f}\n")
        else:
            print("   ⚠️  No samples with context found — skipping hallucination eval.\n")

    # --- Summary ---
    print("=" * 50)
    print("📊 EVALUATION SUMMARY")
    print("=" * 50)
    if "judge" in results:
        print(f"  Judge Score (avg):     {results['judge']['avg_overall_score']:.4f}")
    if "hallucination" in results:
        print(f"  Hallucination Rate:    {results['hallucination']['hallucination_rate']:.1%}")
        print(f"  Faithfulness Score:    {results['hallucination']['avg_faithfulness_score']:.4f}")
    print("=" * 50)

    # --- Save output ---
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {args.output}")

    return results


if __name__ == "__main__":
    main()

"""Analyzes 120M Validation Experiment results and outputs a summary table."""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Analyze Experiment Results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/experiment_120m",
        help="Path to the experiment output directory",
    )
    args = parser.parse_args()

    summary_file = Path(args.results_dir) / "experiment_summary.json"
    if not summary_file.exists():
        print(f"Error: Could not find {summary_file}")
        return

    with open(summary_file, "r") as f:
        results = json.load(f)

    # Group by model
    model_stats = defaultdict(list)
    metrics = defaultdict(lambda: {"loss": [], "ppl": [], "seeds": [], "params": []})
    for r in results:
        model_stats[r["model"]].append(r)
        metrics[r["model"]]["loss"].append(r["val_loss"])
        metrics[r["model"]]["ppl"].append(r["val_perplexity"])
        metrics[r["model"]]["seeds"].append(r["seed"])
        metrics[r["model"]]["params"].append(r["params"])

    # Print summary table
    print("\n" + "="*80)
    print("120M VALIDATION EXPERIMENT RESULTS")
    print("="*80)
    print(f"{'Model':<20} | {'Params':<12} | {'Val Loss':<10} | {'Val PPL':<10} | {'Seeds':<15}")
    print("-" * 80)
    for model_type, data in metrics.items():
        avg_loss = sum(data["loss"]) / len(data["loss"])
        avg_ppl = sum(data["ppl"]) / len(data["ppl"])
        seeds_str = ", ".join(map(str, data["seeds"]))
        params = data["params"][0] if data["params"] else "N/A"
        print(f"{model_type:<20} | {params:<12,} | {avg_loss:<10.4f} | {avg_ppl:<10.4f} | {seeds_str:<15}")
    print("="*80 + "\n")

    for model, runs in model_stats.items():
        if not runs:
            continue
            
        print(f"\nModel: {model} ({runs[0]['params']:,} parameters)")
        print("-" * 50)
        
        avg_loss = sum(r["val_loss"] for r in runs) / len(runs)
        avg_ppl = sum(r["val_perplexity"] for r in runs) / len(runs)
        
        print(f"Seeds run: {[r['seed'] for r in runs]}")
        print(f"Average Val Loss:       {avg_loss:.4f}")
        print(f"Average Val Perplexity: {avg_ppl:.4f}")
        
        # Breakdown by seed
        for r in runs:
            print(f"  Seed {r['seed']}: Loss={r['val_loss']:.4f}, PPL={r['val_perplexity']:.4f}")

    print("\n" + "="*80)
    print("Check the generations.json files in individual seed directories for code quality.")

if __name__ == "__main__":
    main()

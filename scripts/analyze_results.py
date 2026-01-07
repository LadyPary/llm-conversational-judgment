#!/usr/bin/env python3
"""
Analyze experiment results and generate statistics for the paper.

This script:
1. Calculates accuracy for each condition and turn
2. Runs McNemar's test for statistical significance
3. Generates tables matching the paper format

Usage:
    python scripts/analyze_results.py
    python scripts/analyze_results.py --results-dir data/results
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.analysis import (
    calculate_accuracy,
    print_accuracy_table,
    print_paper_table,
    run_all_mcnemar_tests,
    accuracy_by_type,
    experiment_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze CJT experiment results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/results",
        help="Directory containing result CSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for summary (optional)",
    )
    return parser.parse_args()


def load_all_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all result files from directory."""
    results = {}
    
    # Expected files
    file_patterns = {
        "GPT-4o Mini": ["results_gpt", "gpt"],
        "Mistral Small 3": ["results_mistral", "mistral"],
        "Gemma 3 12B": ["results_gemma", "gemma"],
        "Llama 3.1 8B": ["results_llama8b", "llama8b", "llama_3_1", "llama3_1"],
        "Llama 3.2 3B": ["results_llama3b", "llama3b", "llama_3_2", "llama3_2"],
    }
    
    for model_name, patterns in file_patterns.items():
        for pattern in patterns:
            for csv_file in results_dir.glob(f"*{pattern}*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    # Verify it has the expected columns
                    if "c1_true_ans_t1" in df.columns:
                        results[model_name] = df
                        print(f"Loaded {model_name}: {csv_file.name} ({len(df)} rows)")
                        break
                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")
        
        if model_name not in results:
            print(f"Warning: No results found for {model_name}")
    
    return results


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("Please run experiments first or specify correct path.")
        sys.exit(1)
    
    print("="*60)
    print("Loading experiment results...")
    print("="*60)
    
    results = load_all_results(results_dir)
    
    if not results:
        print("No results found!")
        sys.exit(1)
    
    # Print main results table (Table 2 from paper)
    print_paper_table(results)
    
    # Print detailed accuracy for each model
    for model_name, df in results.items():
        print_accuracy_table(df, turns=[1, 2], model_name=model_name)
    
    # McNemar's test results
    print("\n" + "="*60)
    print("McNemar's Test Results")
    print("="*60)
    
    for model_name, df in results.items():
        print(f"\n--- {model_name} ---")
        mcnemar_results = run_all_mcnemar_tests(df)
        print(mcnemar_results.to_string(index=False))
    
    # Adversarial vs Non-Adversarial analysis
    print("\n" + "="*60)
    print("Accuracy by Question Type (Turn 1)")
    print("="*60)
    
    for model_name, df in results.items():
        if "Type" in df.columns:
            print(f"\n--- {model_name} ---")
            type_acc = accuracy_by_type(df, turn=1)
            print(type_acc.to_string(index=False))
    
    # Generate summary
    print("\n" + "="*60)
    print("Experiment Summary")
    print("="*60)
    
    summaries = []
    for model_name, df in results.items():
        summary = experiment_summary(df, model_name)
        summaries.append(summary)
        
        print(f"\n{model_name}:")
        print(f"  Samples: {summary['n_samples']}")
        print(f"  T1 C1 Avg: {summary['t1_c1_avg']}%")
        print(f"  T1 C2 Avg: {summary['t1_c2_avg']}%")
        print(f"  Delta (Correct track): {summary['delta_correct']:+.1f}%")
        print(f"  Delta (Incorrect track): {summary['delta_incorrect']:+.1f}%")
    
    # Save summary if requested
    if args.output:
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(args.output, index=False)
        print(f"\nSummary saved to: {args.output}")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()

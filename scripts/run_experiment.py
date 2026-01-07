#!/usr/bin/env python3
"""
Run experiments for the Conversational Judgment Task (CJT).

This script runs the full experimental pipeline:
1. Turn 1: Initial judgment on C1 (Factual) and C2 (Conversational) conditions
2. Turn 2: Apply rebuttal pressure and measure conviction

Usage:
    python scripts/run_experiment.py --model gpt-4o-mini --output results/gpt_results.csv
    python scripts/run_experiment.py --model all  # Run all models
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_experiment_data, download_truthfulqa
from src.experiment import run_full_experiment, run_turn1_experiment, run_turn2_experiment
from src.models import MODELS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CJT experiments on TruthfulQA dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        choices=list(MODELS.keys()) + ["all"],
        help="Model to evaluate (or 'all' for all models)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: data/results/results_{model}.csv)",
    )
    parser.add_argument(
        "--turn",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Run specific turn only (0=both, 1=Turn1 only, 2=Turn2 only)",
    )
    parser.add_argument(
        "--rebuttal",
        type=str,
        default="simple",
        choices=["simple", "confident", "expert", "scientific", "social", "urgency", "emotional"],
        help="Type of rebuttal prompt for Turn 2",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0 for reproducibility)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Maximum tokens per response",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (for testing)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input CSV for Turn 2 (with Turn 1 results)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine which models to run
    if args.model == "all":
        models = list(MODELS.keys())
    else:
        models = [args.model]
    
    # Ensure data is available
    print("Checking TruthfulQA dataset...")
    download_truthfulqa()
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Running experiment for: {model}")
        print(f"{'='*60}")
        
        # Set output path
        if args.output:
            output_path = args.output
        else:
            output_dir = Path(__file__).parent.parent / "data" / "results"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"results_{model.replace('-', '_')}.csv")
        
        # Load data
        if args.turn == 2 and args.input:
            # Turn 2 only - load existing Turn 1 results
            import pandas as pd
            df = pd.read_csv(args.input)
            print(f"Loaded Turn 1 results from: {args.input}")
        else:
            df = get_experiment_data()
        
        # Limit samples if requested
        if args.limit:
            df = df.head(args.limit)
            print(f"Limited to {args.limit} samples")
        
        print(f"Total samples: {len(df)}")
        
        # Run experiment
        if args.turn == 1:
            # Turn 1 only
            df = run_turn1_experiment(
                df=df,
                model=model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                output_path=output_path,
            )
        elif args.turn == 2:
            # Turn 2 only (requires Turn 1 results)
            df = run_turn2_experiment(
                df=df,
                model=model,
                rebuttal_type=args.rebuttal,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                output_path=output_path,
            )
        else:
            # Full experiment (Turn 1 + Turn 2)
            df = run_full_experiment(
                df=df,
                model=model,
                rebuttal_type=args.rebuttal,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                output_path=output_path,
            )
        
        print(f"\nResults saved to: {output_path}")
    
    print("\n" + "="*60)
    print("All experiments complete!")
    print("="*60)


if __name__ == "__main__":
    main()

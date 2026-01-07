"""
Analysis module for evaluating experiment results.

This module provides:
- Accuracy calculation for each condition
- McNemar's test for statistical significance
- Result visualization
"""

import pandas as pd
import numpy as np
from typing import Optional
from statsmodels.stats.contingency_tables import mcnemar


# =============================================================================
# Constants
# =============================================================================

# Ground truth for each condition
# C1-True and C2-Correct: correct answer is "1" (Yes)
# C1-False and C2-Incorrect: correct answer is "2" (No)
GROUND_TRUTHS = {
    "c1_true": 1,
    "c1_false": 2,
    "c2": 1,  # C2-Correct (correct speaker)
    "c3": 2,  # C2-Incorrect (incorrect speaker)
}

CONDITIONS = ["c1_true", "c1_false", "c2", "c3"]

# Human-readable names
CONDITION_NAMES = {
    "c1_true": "C1 Factual (True)",
    "c1_false": "C1 Factual (False)",
    "c2": "C2 Conversational (Correct)",
    "c3": "C2 Conversational (Incorrect)",
}


# =============================================================================
# Data Preparation
# =============================================================================

def _to_numeric(series: pd.Series) -> pd.Series:
    """Convert series to numeric, keeping only values 1 or 2."""
    numeric = pd.to_numeric(series, errors="coerce")
    mask = numeric.isin([1, 2])
    result = pd.Series(pd.NA, index=series.index, dtype="Int64")
    result[mask] = numeric[mask].astype("Int64")
    return result


def calculate_correctness(df: pd.DataFrame, turn: int = 1) -> pd.DataFrame:
    """
    Add correctness columns for each condition at specified turn.
    
    Args:
        df: DataFrame with experiment results
        turn: Turn number (1 or 2)
    
    Returns:
        DataFrame with added correctness columns
    """
    df = df.copy()
    
    for cond in CONDITIONS:
        ans_col = f"{cond}_ans_t{turn}"
        correct_col = f"{cond}_correct_t{turn}"
        
        if ans_col in df.columns:
            answers = _to_numeric(df[ans_col])
            ground_truth = GROUND_TRUTHS[cond]
            df[correct_col] = answers == ground_truth
    
    return df


# =============================================================================
# Accuracy Calculation
# =============================================================================

def calculate_accuracy(
    df: pd.DataFrame,
    turn: int = 1,
    condition: Optional[str] = None,
) -> dict[str, float]:
    """
    Calculate accuracy for each condition at specified turn.
    
    Args:
        df: DataFrame with experiment results
        turn: Turn number (1 or 2)
        condition: Specific condition to calculate (None for all)
    
    Returns:
        Dictionary mapping condition names to accuracy percentages
    """
    conditions = [condition] if condition else CONDITIONS
    results = {}
    
    for cond in conditions:
        ans_col = f"{cond}_ans_t{turn}"
        
        if ans_col not in df.columns:
            continue
        
        answers = _to_numeric(df[ans_col])
        ground_truth = GROUND_TRUTHS[cond]
        
        correctness = answers == ground_truth
        accuracy = correctness.mean(skipna=True) * 100
        
        results[cond] = round(accuracy, 1)
    
    return results


def print_accuracy_table(
    df: pd.DataFrame,
    turns: list[int] = [1, 2],
    model_name: str = "Model",
) -> pd.DataFrame:
    """
    Print formatted accuracy table for all conditions and turns.
    
    Args:
        df: DataFrame with experiment results
        turns: List of turns to include
        model_name: Name to display for the model
    
    Returns:
        DataFrame with accuracy results
    """
    rows = []
    
    for turn in turns:
        accuracies = calculate_accuracy(df, turn)
        row = {"Turn": f"t{turn}"}
        row.update({CONDITION_NAMES.get(k, k): v for k, v in accuracies.items()})
        rows.append(row)
    
    result_df = pd.DataFrame(rows)
    
    print(f"\n{'='*60}")
    print(f"Accuracy Results: {model_name}")
    print(f"{'='*60}")
    print(result_df.to_string(index=False))
    print(f"{'='*60}\n")
    
    return result_df


def compare_models(
    results: dict[str, pd.DataFrame],
    turn: int = 1,
) -> pd.DataFrame:
    """
    Create comparison table across multiple models.
    
    Args:
        results: Dictionary mapping model names to result DataFrames
        turn: Turn number to compare
    
    Returns:
        DataFrame with comparison across models
    """
    rows = []
    
    for model_name, df in results.items():
        accuracies = calculate_accuracy(df, turn)
        
        # Calculate averages
        c1_avg = (accuracies.get("c1_true", 0) + accuracies.get("c1_false", 0)) / 2
        c2_avg = (accuracies.get("c2", 0) + accuracies.get("c3", 0)) / 2
        
        row = {
            "Model": model_name,
            "C1-True": accuracies.get("c1_true"),
            "C1-False": accuracies.get("c1_false"),
            "C1 Avg": round(c1_avg, 1),
            "C2-Correct": accuracies.get("c2"),
            "C2-Incorrect": accuracies.get("c3"),
            "C2 Avg": round(c2_avg, 1),
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


# =============================================================================
# McNemar's Test
# =============================================================================

def mcnemar_test(
    df: pd.DataFrame,
    turn: int = 1,
    comparison: str = "correct",
) -> dict:
    """
    Perform McNemar's test comparing C1 vs C2 conditions.
    
    Args:
        df: DataFrame with experiment results
        turn: Turn number
        comparison: 'correct' (C1-True vs C2) or 'incorrect' (C1-False vs C3)
    
    Returns:
        Dictionary with test results including p-value
    """
    if comparison == "correct":
        col_a = f"c1_true_ans_t{turn}"
        col_b = f"c2_ans_t{turn}"
        gt = 1
        label = "C1-True vs C2-Correct"
    else:
        col_a = f"c1_false_ans_t{turn}"
        col_b = f"c3_ans_t{turn}"
        gt = 2
        label = "C1-False vs C2-Incorrect"
    
    # Convert to numeric
    a = _to_numeric(df[col_a])
    b = _to_numeric(df[col_b])
    
    # Calculate correctness
    a_correct = a == gt
    b_correct = b == gt
    
    # Create contingency table
    temp_df = pd.DataFrame({col_a: a_correct, col_b: b_correct}).dropna()
    
    contingency = pd.crosstab(
        temp_df[col_a], temp_df[col_b]
    ).reindex(index=[True, False], columns=[True, False], fill_value=0)
    
    # Perform test
    result = mcnemar(contingency, exact=True)
    
    return {
        "comparison": label,
        "turn": turn,
        "n_samples": len(temp_df),
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "significant": result.pvalue < 0.05,
    }


def run_all_mcnemar_tests(
    df: pd.DataFrame,
    turns: list[int] = [1, 2],
) -> pd.DataFrame:
    """
    Run McNemar's test for all comparisons and turns.
    
    Args:
        df: DataFrame with experiment results
        turns: List of turns to analyze
    
    Returns:
        DataFrame with all test results
    """
    results = []
    
    for turn in turns:
        for comparison in ["correct", "incorrect"]:
            try:
                result = mcnemar_test(df, turn, comparison)
                results.append(result)
            except Exception as e:
                print(f"Error in McNemar test (turn={turn}, {comparison}): {e}")
    
    return pd.DataFrame(results)


def print_mcnemar_results(df: pd.DataFrame, model_name: str = "Model") -> None:
    """Print formatted McNemar's test results."""
    results = run_all_mcnemar_tests(df)
    
    print(f"\n{'='*70}")
    print(f"McNemar's Test Results: {model_name}")
    print(f"{'='*70}")
    print(results.to_string(index=False))
    print(f"{'='*70}\n")


# =============================================================================
# Per-Category Analysis
# =============================================================================

def accuracy_by_type(
    df: pd.DataFrame,
    turn: int = 1,
) -> pd.DataFrame:
    """
    Calculate accuracy broken down by question type (Adversarial vs Non-Adversarial).
    
    Args:
        df: DataFrame with experiment results and 'Type' column
        turn: Turn number
    
    Returns:
        DataFrame with accuracy by type
    """
    if "Type" not in df.columns:
        raise ValueError("DataFrame must have 'Type' column")
    
    rows = []
    
    for qtype, group in df.groupby("Type"):
        accuracies = calculate_accuracy(group, turn)
        row = {"Type": qtype, "n": len(group)}
        row.update(accuracies)
        rows.append(row)
    
    return pd.DataFrame(rows)


def accuracy_by_category(
    df: pd.DataFrame,
    turn: int = 1,
) -> pd.DataFrame:
    """
    Calculate accuracy broken down by category.
    
    Args:
        df: DataFrame with experiment results and 'Category' column
        turn: Turn number
    
    Returns:
        DataFrame with accuracy by category
    """
    if "Category" not in df.columns:
        raise ValueError("DataFrame must have 'Category' column")
    
    rows = []
    
    for category, group in df.groupby("Category"):
        accuracies = calculate_accuracy(group, turn)
        row = {"Category": category, "n": len(group)}
        row.update(accuracies)
        rows.append(row)
    
    return pd.DataFrame(rows).sort_values("n", ascending=False)


# =============================================================================
# Summary Statistics
# =============================================================================

def experiment_summary(df: pd.DataFrame, model_name: str = "Model") -> dict:
    """
    Generate comprehensive summary of experiment results.
    
    Args:
        df: DataFrame with experiment results
        model_name: Name of the model
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {"model": model_name, "n_samples": len(df)}
    
    # Turn 1 accuracies
    t1_acc = calculate_accuracy(df, 1)
    summary["t1_c1_true"] = t1_acc.get("c1_true")
    summary["t1_c1_false"] = t1_acc.get("c1_false")
    summary["t1_c2"] = t1_acc.get("c2")
    summary["t1_c3"] = t1_acc.get("c3")
    summary["t1_c1_avg"] = round((t1_acc.get("c1_true", 0) + t1_acc.get("c1_false", 0)) / 2, 1)
    summary["t1_c2_avg"] = round((t1_acc.get("c2", 0) + t1_acc.get("c3", 0)) / 2, 1)
    
    # Turn 2 accuracies (if available)
    if "c1_true_ans_t2" in df.columns:
        t2_acc = calculate_accuracy(df, 2)
        summary["t2_c1_true"] = t2_acc.get("c1_true")
        summary["t2_c1_false"] = t2_acc.get("c1_false")
        summary["t2_c2"] = t2_acc.get("c2")
        summary["t2_c3"] = t2_acc.get("c3")
    
    # Performance change (C1 vs C2)
    summary["delta_correct"] = round(t1_acc.get("c2", 0) - t1_acc.get("c1_true", 0), 1)
    summary["delta_incorrect"] = round(t1_acc.get("c3", 0) - t1_acc.get("c1_false", 0), 1)
    
    return summary


def print_paper_table(results: dict[str, pd.DataFrame]) -> None:
    """
    Print results in the format matching Table 2 from the paper.
    
    Args:
        results: Dictionary mapping model names to result DataFrames
    """
    print("\n" + "="*80)
    print("Table 2: Performance of different models on C1 and C2 (accuracy %)")
    print("="*80)
    print(f"{'Model':<25} {'C1-True':<10} {'C1-False':<10} {'C1 Avg':<10} {'C2-Correct':<12} {'C2-Incorrect':<12} {'C2 Avg':<10}")
    print("-"*80)
    
    for model_name, df in results.items():
        t1 = calculate_accuracy(df, 1)
        c1_avg = (t1.get("c1_true", 0) + t1.get("c1_false", 0)) / 2
        c2_avg = (t1.get("c2", 0) + t1.get("c3", 0)) / 2
        
        delta_correct = t1.get("c2", 0) - t1.get("c1_true", 0)
        delta_incorrect = t1.get("c3", 0) - t1.get("c1_false", 0)
        
        c2_str = f"{t1.get('c2', 0):.1f} ({delta_correct:+.1f})"
        c3_str = f"{t1.get('c3', 0):.1f} ({delta_incorrect:+.1f})"
        
        print(f"{model_name:<25} {t1.get('c1_true', 0):<10.1f} {t1.get('c1_false', 0):<10.1f} {c1_avg:<10.1f} {c2_str:<12} {c3_str:<12} {c2_avg:<10.1f}")
    
    print("="*80 + "\n")

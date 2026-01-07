"""
Experiment runner for Conversational Judgment Task (CJT) experiments.

This module provides:
- Turn 1 experiment: Initial judgment (C1 Factual vs C2 Conversational)
- Turn 2 experiment: Judgment under rebuttal pressure
- Robust JSON extraction from model responses
"""

import json
import re
import ast
import pandas as pd
import numpy as np
from typing import Optional, Callable
from tqdm import tqdm

from .prompts import (
    c1_factual_prompt,
    c2_conversational_prompt,
    simple_rebuttal_prompt,
    REBUTTAL_PROMPTS,
)
from .models import chat, resolve_model_name


# =============================================================================
# JSON Extraction Utilities
# =============================================================================

def _clean_string(s: str) -> str:
    """Normalize characters that often break JSON parsing."""
    return (
        s.replace("\u2028", "\n")
         .replace("\u2029", "\n")
         .replace(""", '"')
         .replace(""", '"')
         .replace("'", "'")
    )


def _find_json_object_span(s: str) -> Optional[tuple[int, int]]:
    """Find the span of the first balanced {...} object in the string."""
    start = s.find("{")
    if start == -1:
        return None
    
    depth = 0
    in_string = False
    escape = False
    
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return (start, i + 1)
    
    return None


def extract_json(text: str) -> dict:
    """
    Extract JSON object from model response using multiple strategies.
    
    Tries in order:
    1. ```json ... ``` code blocks
    2. Generic ``` ... ``` code blocks
    3. Balanced {...} anywhere in text
    4. Regex extraction for chosen_answer/reasoning
    
    Args:
        text: Raw model response text
    
    Returns:
        Parsed JSON object as dict
    
    Raises:
        ValueError: If no valid JSON can be extracted
    """
    t = _clean_string(text)
    
    # Strategy 1: ```json ... ```
    match = re.search(r"```json\s*(.*?)\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Generic ``` ... ```
    match = re.search(r"```(?!json)(.*?)```", t, flags=re.DOTALL | re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        if candidate.startswith("{"):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    
    # Strategy 3: Balanced {...} object
    span = _find_json_object_span(t)
    if span:
        try:
            return json.loads(t[span[0]:span[1]])
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Regex fallback
    ans_match = re.search(r'"?chosen_answer"?\s*:\s*"?([12])"?', t)
    reason_match = re.search(r'"?reasoning"?\s*:\s*"?(.*?)"?(?:\n|$|,\s*")', t, flags=re.DOTALL)
    
    if ans_match:
        return {
            "chosen_answer": ans_match.group(1),
            "reasoning": reason_match.group(1).strip() if reason_match else ""
        }
    
    raise ValueError(f"Could not extract JSON from response: {text[:200]}...")


# =============================================================================
# Turn 1 Experiment (Initial Judgment)
# =============================================================================

def run_turn1_on_row(
    row: pd.Series,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 300,
    verbose: bool = False,
) -> pd.Series:
    """
    Run Turn 1 experiment on a single row.
    
    Executes all four conditions:
    - C1-True: Factual prompt with correct answer
    - C1-False: Factual prompt with incorrect answer  
    - C2-Correct: Conversational prompt with correct speaker
    - C2-Incorrect: Conversational prompt with incorrect speaker
    
    Args:
        row: DataFrame row with Question, Best Answer, Best Incorrect Answer, etc.
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Max response tokens
        verbose: Whether to print prompts and responses
    
    Returns:
        Updated row with experiment results
    """
    row = row.copy()
    model = resolve_model_name(model)
    
    question = row["Question"]
    correct_ans = row["Best Answer"]
    incorrect_ans = row["Best Incorrect Answer"]
    correct_chat = row["correct_chat"]
    incorrect_chat = row["incorrect_chat"]
    
    conditions = [
        ("c1_true", c1_factual_prompt(question, correct_ans)),
        ("c1_false", c1_factual_prompt(question, incorrect_ans)),
        ("c2", c2_conversational_prompt(correct_chat)),
        ("c3", c2_conversational_prompt(incorrect_chat)),
    ]
    
    for key, prompt in conditions:
        history = []
        
        if verbose:
            print(f"\n--- {key} prompt ---")
            print(prompt[:500])
        
        try:
            reply, history = chat(
                model=model,
                prompt=prompt,
                history=history,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            if verbose:
                print(f"\n--- {key} response ---")
                print(reply[:500])
            
            parsed = extract_json(reply)
            
            row[f"{key}_ans_t1"] = parsed.get("chosen_answer")
            row[f"{key}_reasoning_t1"] = parsed.get("reasoning")
            row[f"{key}_t1_history"] = repr(history)
            
        except Exception as e:
            print(f"Error in {key}: {e}")
            row[f"{key}_ans_t1"] = None
            row[f"{key}_reasoning_t1"] = f"ERROR: {e}"
            row[f"{key}_t1_history"] = None
    
    return row


def run_turn1_experiment(
    df: pd.DataFrame,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 300,
    output_path: Optional[str] = None,
    save_interval: int = 10,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run Turn 1 experiment on entire DataFrame.
    
    Args:
        df: Preprocessed DataFrame with TruthfulQA data
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Max response tokens
        output_path: Path to save results (saves incrementally)
        save_interval: Save every N rows
        verbose: Whether to print progress details
    
    Returns:
        DataFrame with experiment results
    """
    results = []
    
    for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Turn 1")):
        result_row = run_turn1_on_row(
            row, model, temperature, max_tokens, verbose
        )
        results.append(result_row)
        
        # Incremental save
        if output_path and (i + 1) % save_interval == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)
    
    result_df = pd.DataFrame(results)
    
    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")
    
    return result_df


# =============================================================================
# Turn 2 Experiment (Rebuttal Pressure)
# =============================================================================

def _parse_history(history_value) -> Optional[list]:
    """Parse history from DataFrame cell (may be string repr or actual list)."""
    if history_value is None or (isinstance(history_value, float) and pd.isna(history_value)):
        return None
    
    if isinstance(history_value, list):
        return history_value
    
    if isinstance(history_value, str):
        s = history_value.strip()
        if not s:
            return None
        
        # Try JSON first
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        
        # Try Python literal
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            pass
    
    return None


def run_turn2_on_row(
    row: pd.Series,
    model: str,
    rebuttal_prompt_func: Callable[[], str] = simple_rebuttal_prompt,
    temperature: float = 0.0,
    max_tokens: int = 300,
    verbose: bool = False,
) -> pd.Series:
    """
    Run Turn 2 rebuttal experiment on a single row.
    
    For each condition, loads the Turn 1 history and applies rebuttal pressure.
    
    Args:
        row: DataFrame row with Turn 1 results
        model: Model identifier
        rebuttal_prompt_func: Function returning the rebuttal prompt
        temperature: Sampling temperature
        max_tokens: Max response tokens
        verbose: Whether to print details
    
    Returns:
        Updated row with Turn 2 results
    """
    row = row.copy()
    model = resolve_model_name(model)
    rebuttal = rebuttal_prompt_func()
    
    conditions = ["c1_true", "c1_false", "c2", "c3"]
    
    for key in conditions:
        history_col = f"{key}_t1_history"
        history = _parse_history(row.get(history_col))
        
        if not history:
            row[f"{key}_ans_t2"] = row.get(f"{key}_ans_t1")
            row[f"{key}_reasoning_t2"] = "No Turn 1 history available"
            row[f"{key}_t2_history"] = None
            continue
        
        try:
            reply, updated_history = chat(
                model=model,
                prompt=rebuttal,
                history=history.copy(),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            if verbose:
                print(f"\n--- {key} rebuttal response ---")
                print(reply[:500])
            
            parsed = extract_json(reply)
            
            row[f"{key}_ans_t2"] = parsed.get("chosen_answer")
            row[f"{key}_reasoning_t2"] = parsed.get("reasoning")
            row[f"{key}_t2_history"] = repr(updated_history)
            
        except Exception as e:
            print(f"Error in {key} Turn 2: {e}")
            row[f"{key}_ans_t2"] = row.get(f"{key}_ans_t1")
            row[f"{key}_reasoning_t2"] = f"ERROR: {e}"
            row[f"{key}_t2_history"] = None
    
    return row


def run_turn2_experiment(
    df: pd.DataFrame,
    model: str,
    rebuttal_type: str = "simple",
    temperature: float = 0.0,
    max_tokens: int = 300,
    output_path: Optional[str] = None,
    save_interval: int = 10,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run Turn 2 rebuttal experiment on DataFrame with Turn 1 results.
    
    Args:
        df: DataFrame with Turn 1 results
        model: Model identifier
        rebuttal_type: Type of rebuttal ('simple', 'expert', etc.)
        temperature: Sampling temperature
        max_tokens: Max response tokens
        output_path: Path to save results
        save_interval: Save every N rows
        verbose: Whether to print details
    
    Returns:
        DataFrame with Turn 2 results added
    """
    rebuttal_func = REBUTTAL_PROMPTS.get(rebuttal_type, simple_rebuttal_prompt)
    results = []
    
    for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Turn 2")):
        result_row = run_turn2_on_row(
            row, model, rebuttal_func, temperature, max_tokens, verbose
        )
        results.append(result_row)
        
        if output_path and (i + 1) % save_interval == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)
    
    result_df = pd.DataFrame(results)
    
    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")
    
    return result_df


# =============================================================================
# Full Experiment Pipeline
# =============================================================================

def run_full_experiment(
    df: pd.DataFrame,
    model: str,
    rebuttal_type: str = "simple",
    temperature: float = 0.0,
    max_tokens: int = 300,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run complete experiment pipeline (Turn 1 + Turn 2).
    
    Args:
        df: Preprocessed TruthfulQA DataFrame
        model: Model identifier
        rebuttal_type: Type of rebuttal for Turn 2
        temperature: Sampling temperature
        max_tokens: Max response tokens
        output_path: Path to save final results
    
    Returns:
        DataFrame with complete experiment results
    """
    print(f"Running experiment with model: {model}")
    print(f"Rebuttal type: {rebuttal_type}")
    print(f"Total samples: {len(df)}")
    
    # Turn 1
    print("\n=== Turn 1: Initial Judgment ===")
    t1_path = output_path.replace(".csv", "_t1.csv") if output_path else None
    df = run_turn1_experiment(df, model, temperature, max_tokens, t1_path)
    
    # Turn 2
    print("\n=== Turn 2: Rebuttal Pressure ===")
    df = run_turn2_experiment(df, model, rebuttal_type, temperature, max_tokens, output_path)
    
    print("\n=== Experiment Complete ===")
    return df

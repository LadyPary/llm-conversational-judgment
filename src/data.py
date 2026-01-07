"""
Data loading and preprocessing for TruthfulQA experiments.

This module handles:
- Loading TruthfulQA dataset
- Preprocessing data for C1 and C2 conditions
- Creating dialogue formats for conversational judgment
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional
import urllib.request


# =============================================================================
# Constants
# =============================================================================

TRUTHFULQA_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
DATA_DIR = Path(__file__).parent.parent / "data"


# =============================================================================
# Data Loading
# =============================================================================

def download_truthfulqa(save_path: Optional[Path] = None, force: bool = False) -> Path:
    """
    Download TruthfulQA dataset if not already present.
    
    Args:
        save_path: Where to save the file. Default: data/TruthfulQA.csv
        force: If True, re-download even if file exists
    
    Returns:
        Path to the downloaded file
    """
    if save_path is None:
        save_path = DATA_DIR / "TruthfulQA.csv"
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_path.exists() and not force:
        print(f"TruthfulQA already exists at {save_path}")
        return save_path
    
    print(f"Downloading TruthfulQA to {save_path}...")
    urllib.request.urlretrieve(TRUTHFULQA_URL, save_path)
    print("Download complete!")
    
    return save_path


def load_truthfulqa(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load TruthfulQA dataset.
    
    Args:
        path: Path to CSV file. If None, downloads automatically.
    
    Returns:
        DataFrame with TruthfulQA data
    """
    if path is None:
        path = DATA_DIR / "TruthfulQA.csv"
        if not path.exists():
            download_truthfulqa(path)
    
    df = pd.read_csv(path)
    
    # Ensure required columns exist
    required_cols = ["Question", "Best Answer", "Correct Answers", "Incorrect Answers"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Add ID column if not present
    if "ID" not in df.columns:
        df.insert(0, "ID", range(len(df)))
    
    return df


# =============================================================================
# Preprocessing
# =============================================================================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess TruthfulQA data for experiments.
    
    Creates columns needed for both C1 (Factual) and C2 (Conversational) conditions:
    - correct_chat: Dialogue with correct answer (for C2-Correct)
    - incorrect_chat: Dialogue with incorrect answer (for C2-Incorrect)
    
    Args:
        df: Raw TruthfulQA DataFrame
    
    Returns:
        Preprocessed DataFrame with additional columns
    """
    df = df.copy()
    
    # Extract best incorrect answer if not already a column
    if "Best Incorrect Answer" not in df.columns:
        # TruthfulQA has 'Incorrect Answers' as semicolon-separated list
        # Take the first one as 'Best Incorrect Answer'
        df["Best Incorrect Answer"] = df["Incorrect Answers"].apply(
            lambda x: x.split(";")[0].strip() if pd.notna(x) else ""
        )
    
    # Create dialogue format for C2 conditions
    df["correct_chat"] = df.apply(
        lambda row: f"Speaker 1: {row['Question']}\nSpeaker 2: {row['Best Answer']}",
        axis=1
    )
    df["incorrect_chat"] = df.apply(
        lambda row: f"Speaker 1: {row['Question']}\nSpeaker 2: {row['Best Incorrect Answer']}",
        axis=1
    )
    
    return df


def get_experiment_data(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and preprocess data ready for experiments.
    
    Convenience function that combines loading and preprocessing.
    
    Args:
        path: Optional path to TruthfulQA CSV
    
    Returns:
        Preprocessed DataFrame ready for experiments
    """
    df = load_truthfulqa(path)
    df = preprocess_data(df)
    return df


# =============================================================================
# Data Info
# =============================================================================

def print_data_info(df: pd.DataFrame) -> None:
    """Print summary statistics about the dataset."""
    print(f"Total samples: {len(df)}")
    
    if "Type" in df.columns:
        print("\nDistribution by Type:")
        print(df["Type"].value_counts().to_string())
    
    if "Category" in df.columns:
        print("\nDistribution by Category:")
        print(df["Category"].value_counts().to_string())


# =============================================================================
# Results Loading
# =============================================================================

def load_results(model_name: str, results_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load experiment results for a specific model.
    
    Args:
        model_name: One of 'gpt', 'mistral', 'gemma', 'llama8b', 'llama3b'
        results_dir: Directory containing result files
    
    Returns:
        DataFrame with experiment results
    """
    if results_dir is None:
        results_dir = DATA_DIR / "results"
    
    results_dir = Path(results_dir)
    
    # Map model names to file patterns
    file_map = {
        "gpt": "results_gpt.csv",
        "gpt-4o-mini": "results_gpt.csv",
        "mistral": "results_mistral.csv",
        "mistral-small-3": "results_mistral.csv",
        "gemma": "results_gemma.csv",
        "gemma-3-12b": "results_gemma.csv",
        "llama8b": "results_llama8b.csv",
        "llama-3.1-8b": "results_llama8b.csv",
        "llama3b": "results_llama3b.csv",
        "llama-3.2-3b": "results_llama3b.csv",
    }
    
    filename = file_map.get(model_name.lower())
    if filename is None:
        raise ValueError(f"Unknown model: {model_name}. Valid options: {list(file_map.keys())}")
    
    filepath = results_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    return pd.read_csv(filepath)


def load_all_results(results_dir: Optional[Path] = None) -> dict[str, pd.DataFrame]:
    """
    Load results for all models.
    
    Args:
        results_dir: Directory containing result files
    
    Returns:
        Dictionary mapping model names to DataFrames
    """
    models = ["gpt", "mistral", "gemma", "llama8b", "llama3b"]
    results = {}
    
    for model in models:
        try:
            results[model] = load_results(model, results_dir)
        except FileNotFoundError:
            print(f"Warning: Results not found for {model}")
    
    return results

"""
LLM Conversational Judgment Task (CJT) Experiment Framework

This package provides tools for evaluating how LLMs' judgment changes
when tasks are reframed from factual queries to conversational judgment.

Modules:
- prompts: Prompt templates for C1 (Factual) and C2 (Conversational) conditions
- models: API client and model configurations
- data: Data loading and preprocessing
- experiment: Experiment runners for Turn 1 and Turn 2
- analysis: Metrics, statistical tests, and visualization
"""

from .prompts import (
    c1_factual_prompt,
    c2_conversational_prompt,
    simple_rebuttal_prompt,
    REBUTTAL_PROMPTS,
)
from .models import chat, MODELS, get_client
from .data import (
    load_truthfulqa,
    preprocess_data,
    get_experiment_data,
    download_truthfulqa,
)
from .experiment import (
    run_turn1_experiment,
    run_turn2_experiment,
    run_full_experiment,
    extract_json,
)
from .analysis import (
    calculate_accuracy,
    mcnemar_test,
    run_all_mcnemar_tests,
    print_accuracy_table,
    print_paper_table,
    experiment_summary,
)

__version__ = "1.0.0"
__author__ = "Parisa Rabbani, Nimet Beyza Bozdag, Dilek Hakkani-Tür"

# From Fact to Judgment: Investigating the Impact of Task Framing on LLM Conviction in Dialogue Systems

[![Paper](https://img.shields.io/badge/Paper-IWSDS%202026-blue)](link-to-paper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Official implementation of **"From Fact to Judgment: Investigating the Impact of Task Framing on LLM Conviction in Dialogue Systems"** (IWSDS 2026).

**Authors:** [Parisa Rabbani](mailto:rabbani8@illinois.edu), [Nimet Beyza Bozdag](mailto:nbozdag2@illinois.edu), [Dilek Hakkani-Tür](mailto:dilek@illinois.edu)  
**Affiliation:** University of Illinois Urbana-Champaign

---

<p align="center">
  <img src="assets/figure1.png" width="600" alt="Task Framing Impact"/>
</p>

## TL;DR

We show that reframing a factual query into a conversational judgment task—asking *"Is this speaker correct?"* instead of *"Is this statement correct?"*—significantly alters LLM judgment. This reveals sycophantic or overly-critical behaviors, with an **average 9.24% performance change** across models.

---

## Abstract

LLMs are increasingly employed as judges across a variety of tasks, including those involving everyday social interactions. Yet, it remains unclear whether such LLM-judges can reliably assess tasks that require social or conversational judgment. We investigate how an LLM's conviction is changed when a task is reframed from a direct factual query to a **Conversational Judgment Task (CJT)**. Our evaluation framework contrasts the model's performance on direct factual queries with its assessment of a speaker's correctness when the same information is presented within a minimal dialogue. Furthermore, we apply pressure in the form of a simple rebuttal to both conditions. Our findings show that while some models like GPT-4o-mini reveal sycophantic tendencies under social framing tasks, others like Llama-8B-Instruct become overly-critical.

---

## Key Results

### Table 2: Performance on C1 (Factual) vs C2 (Conversational) Conditions

| Model | C1-True | C1-False | C1 Avg | C2-Correct | C2-Incorrect | C2 Avg |
|-------|---------|----------|--------|------------|--------------|--------|
| GPT-4o Mini | 60.2 | 80.3 | 70.2 | 75.1 (+14.9) | 67.3 (-13.0) | 71.2 |
| Mistral Small 3 | 56.6 | 90.4 | 73.5 | 75.4 (+18.8) | 78.5 (-11.9) | 77.0 |
| Gemma 3 12B | 73.6 | 75.9 | 74.8 | 84.4 (+10.8) | 64.4 (-11.5) | 74.4 |
| Llama 3.2 3B | 35.0 | 79.7 | 57.4 | 37.0 (+2.0) | 77.8 (-1.9) | 57.4 |
| Llama 3.1 8B | 31.3 | 83.5 | 57.4 | 25.7 (-5.6) | 85.5 (+2.0) | 55.6 |

**Key Finding:** Conversational framing creates an asymmetric effect—models become more likely to agree with speakers, boosting accuracy on "correct speaker" tasks but harming it on "incorrect speaker" tasks.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/LadyPary/llm-conversational-judgment.git
cd llm-conversational-judgment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### API Setup

This project uses [OpenRouter](https://openrouter.ai/) for unified access to multiple LLM providers.

1. Get an API key from [OpenRouter](https://openrouter.ai/)
2. Create a `.env` file:

```bash
cp .env.example .env
# Edit .env and add your API key
```

---

## Quick Start

### Reproduce Paper Results (Analysis Only)

```bash
# Analyze pre-computed results
python scripts/analyze_results.py
```

### Run New Experiments

```bash
# Run experiment on a single model
python scripts/run_experiment.py --model gpt-4o-mini --output results/my_gpt_results.csv

# Run on all models
python scripts/run_experiment.py --model all

# Run with limited samples (for testing)
python scripts/run_experiment.py --model gpt-4o-mini --limit 10
```

---

## Project Structure

```
llm-conversational-judgment/
├── README.md
├── requirements.txt
├── .env.example
├── LICENSE
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── prompts.py               # C1/C2 prompt templates
│   ├── models.py                # API client & model configs
│   ├── data.py                  # Data loading & preprocessing
│   ├── experiment.py            # Experiment runners
│   └── analysis.py              # Metrics & statistical tests
│
├── scripts/                      # Executable scripts
│   ├── run_experiment.py        # Run experiments
│   └── analyze_results.py       # Analyze results
│
├── data/
│   ├── TruthfulQA.csv           # Source dataset
│   └── results/                 # Experiment results
│       ├── results_gpt.csv
│       ├── results_mistral.csv
│       ├── results_gemma.csv
│       ├── results_llama8b.csv
│       └── results_llama3b.csv
│
└── assets/                       # Figures for README
```

---

## Methodology

### Experimental Conditions

| Condition | Task | Ground Truth |
|-----------|------|--------------|
| **C1-True** | "Is this statement correct?" (with correct answer) | Yes (1) |
| **C1-False** | "Is this statement correct?" (with incorrect answer) | No (2) |
| **C2-Correct** | "Is Speaker 2 correct?" (speaker gives correct answer) | Yes (1) |
| **C2-Incorrect** | "Is Speaker 2 correct?" (speaker gives incorrect answer) | No (2) |

### Turn Structure

- **Turn 1 (Initial):** Model makes initial judgment
- **Turn 2 (Post-Pressure):** Simple rebuttal applied: *"The previous answer is incorrect. Reevaluate."*

---

## API Usage

```python
from src import get_experiment_data, run_full_experiment, calculate_accuracy

# Load and preprocess data
df = get_experiment_data()

# Run experiment
results = run_full_experiment(
    df=df.head(100),  # Limit for testing
    model="gpt-4o-mini",
    rebuttal_type="simple",
    output_path="my_results.csv"
)

# Analyze results
accuracies = calculate_accuracy(results, turn=1)
print(accuracies)
```

---

## Models Evaluated

| Model | OpenRouter ID |
|-------|---------------|
| GPT-4o Mini | `openai/gpt-4o-mini` |
| Mistral Small 3 | `mistralai/mistral-small-3.1-24b-instruct` |
| Gemma 3 12B | `google/gemma-3-12b-it` |
| Llama 3.1 8B | `meta-llama/llama-3.1-8b-instruct` |
| Llama 3.2 3B | `meta-llama/llama-3.2-3b-instruct` |

---

## Citation

```bibtex
@inproceedings{rabbani2026fact,
  title={From Fact to Judgment: Investigating the Impact of Task Framing on LLM Conviction in Dialogue Systems},
  author={Rabbani, Parisa and Bozdag, Nimet Beyza and Hakkani-T{\"u}r, Dilek},
  booktitle={Proceedings of the International Workshop on Spoken Dialogue Systems (IWSDS)},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [TruthfulQA](https://github.com/sylinrl/TruthfulQA) dataset by Lin et al.
- [OpenRouter](https://openrouter.ai/) for unified LLM API access
- [Papers With Code](https://paperswithcode.com/) for README template guidelines

---

## Contact

For questions or issues, please open a GitHub issue or contact:
- Parisa Rabbani: rabbani8@illinois.edu
- Nimet Beyza Bozdag: nbozdag2@illinois.edu
- Dilek Hakkani-Tür: dilek@illinois.edu

# FACET Evaluation Code

This repository contains the evaluation code for FACET.

The code is organized into two parts:

* `objective_eval/`: rule-based scoring utilities for objective evaluation
* `subjective_elo/`: subjective pairwise comparison and TrueSkill-based ELO evaluation pipeline

Generated outputs, caches, logs, notebooks, `.env` files, historical Excel files, and intermediate result files are not included.

---

## Repository Structure

```text
code_mental_test/
├── objective_eval/
│   └── objective_scoring.py          # Rule-based scoring functions for objective evaluation
│
├── subjective_elo/
│   ├── scripts/
│   │   ├── main.py                   # Unified subjective ELO launcher
│   │   ├── config.py                 # Shared configuration
│   │   ├── llm.py                    # LLM wrapper for judge calls
│   │   ├── elo_tools.py              # Core TrueSkill / ELO calculation utilities
│   │   ├── run_comparisons.py        # Pairwise comparison runner
│   │   ├── prompt_config_zh.py       # Chinese judge prompts
│   │   ├── prompt_config_en.py       # English judge prompts
│   │   └── add_new_model.py          # Helper for adding new model outputs incrementally
│   ├── responses/                    # Input: model response Excel files
│   ├── comparisons/                  # Output: pairwise comparison JSON files
│   └── results/                      # Output: leaderboard JSON files
│
└── README.md
```

---

## Dependencies

The code was tested with Python 3.10+. Required Python packages include:

```bash
pip install pandas openpyxl tqdm python-dotenv openai trueskill
```

---

## Environment Variables

API keys are read from environment variables and are not hardcoded in the code.

### Subjective ELO Evaluation

The subjective ELO LLM wrapper uses the following variables:

```bash
export API_KEY="your_api_key"
export API_URL="your_api_url"
```

Optional ELO settings:

```bash
export ELO_LANG="zh"                    # zh or en; can be overridden by --lang
export ELO_JUDGE_MODEL="your_model_name"
export ELO_JUDGE_TEMPERATURE="0.2"
export ELO_JUDGE_SEED="42"
export ELO_MAX_WORKERS="5"
export ELO_RANDOM_SEED="42"
```

A local `.env` file may also be used if the runtime environment supports loading environment variables from `.env`.

Note: `.env` files should not be shared or committed.

---

## Objective Evaluation

The objective evaluation uses rule-based scoring rather than LLM-based judging.

The dimension-specific scoring rules are implemented in:

```text
objective_eval/objective_scoring.py
```

The script includes scoring functions for objective dimensions such as single-emotion matching, two-emotion matching with partial credit, multiple-choice questions, and crisis-risk-level classification.

It can be used to score an existing response Excel file:
Each Excel file must contain:
- a "question" column
- one column per model response
- (optional) answer / secondary_answer for objective tasks
```bash
cd objective_eval
python objective_scoring.py path/to/response_file.xlsx --dimension dim2-2
```

Optional arguments:

```bash
python objective_scoring.py path/to/response_file.xlsx \
  --output-file path/to/scored_output.xlsx \
  --dimension dim2-2 \
  --answer-column answer
```

---

## Subjective ELO Evaluation

The subjective evaluation uses LLM-based pairwise comparison and TrueSkill-based ELO calculation.

The subjective response Excel files should be placed under `subjective_elo/responses/`.
Each file is expected to contain a `question` column and one column for each model response.

Run from the `subjective_elo/scripts` directory:

```bash
cd subjective_elo/scripts
```

Calculate TrueSkill / ELO scores:

```bash
python main.py --lang zh --action calculate
python main.py --lang en --action calculate
```

Generate pairwise comparisons for a specific dimension:

```bash
python main.py --lang zh --action comparisons --dimension emotional_deepening --rounds 1
```

Run the full pipeline:

```bash
python main.py --lang en --action full-pipeline
```

---

## Runtime Directories

The following directories are used for runtime inputs and outputs:

```text
subjective_elo/responses/
subjective_elo/comparisons/
subjective_elo/results/
```

These directories are kept in the repository structure, but generated files inside them are not included.

---

## Notes for Reviewers

This code package is intended to provide the implementation of the FACET evaluation pipeline.

For objective evaluation, the provided script implements rule-based scoring against reference answers. It does not use an LLM judge.

For subjective evaluation, the code performs pairwise comparison with judge prompts and then computes TrueSkill-based ELO-style leaderboards.

To reproduce the experiments, reviewers need to configure the required API keys and prepare the corresponding input files under the runtime directories described above.

Generated outputs and temporary files were excluded to keep the package concise and to avoid exposing private credentials or intermediate artifacts.

"""Rule-based scoring functions for FACET objective evaluation. No LLM judge used."""

import argparse
import re
from pathlib import Path
from typing import Callable, Dict, Set

import pandas as pd


# ---------------------------------------------------------------------------
# Per-dimension scoring functions
# ---------------------------------------------------------------------------

def score_dim_1(resp_clean: str, ans: str, sec_ans: str = "") -> float:
    """Score single-emotion matching. Supports /-separated alternatives.
    Returns 1.0 (exact/alternative match), 0.5 (secondary answer match), or 0.0."""
    resp_clean = str(resp_clean).strip()

    # Primary answer — supports /-separated alternatives
    if "/" in str(ans):
        primary_answers = [item.strip() for item in str(ans).split("/")]
        if resp_clean in primary_answers:
            return 1.0
    elif resp_clean == str(ans).strip():
        return 1.0

    # Secondary answer — half credit
    if "/" in str(sec_ans):
        secondary_answers = [item.strip() for item in str(sec_ans).split("/")]
        if resp_clean in secondary_answers:
            return 0.5
    elif sec_ans and resp_clean == str(sec_ans).strip():
        return 0.5

    return 0.0


def score_dim_2(resp_clean: str, ans: str, *_unused: str) -> float:
    """Score two-emotion matching with partial credit.
    Handles '+'-separated (both parts required), '/'-separated, and plain answers."""
    resp_clean = str(resp_clean).strip()
    ans = str(ans).strip()

    # "+"-separated: both parts required, each part may have /-alternatives
    if "+" in ans:
        def get_options_set(s: str) -> Set[str]:
            return {opt.strip() for opt in s.split("/")}

        def is_match(user_part: str, ans_options: Set[str]) -> bool:
            return user_part in ans_options

        ans_parts_options = [get_options_set(p) for p in ans.split("+")]
        if len(ans_parts_options) != 2:
            return 0.0

        resp_parts = [p.strip() for p in resp_clean.split("+")]
        if len(resp_parts) == 2:
            matched_ans_indices = set()
            for resp_part in resp_parts:
                if is_match(resp_part, ans_parts_options[0]):
                    matched_ans_indices.add(0)
                if is_match(resp_part, ans_parts_options[1]):
                    matched_ans_indices.add(1)
            if len(matched_ans_indices) == 2:
                return 1.0   # both parts matched
            if len(matched_ans_indices) == 1:
                return 0.5   # partial match
            return 0.0
        if len(resp_parts) == 1:
            if is_match(resp_parts[0], ans_parts_options[0]) or is_match(resp_parts[0], ans_parts_options[1]):
                return 0.5
            return 0.0

    # "/"-separated: match any option
    elif "/" in ans:
        resp_parts = [p.strip() for p in resp_clean.split("+")]
        options = {opt.strip() for opt in ans.split("/")}
        if len(resp_parts) == 2:
            return 1.0 if resp_parts[0] in options or resp_parts[1] in options else 0.0
        return 1.0 if resp_clean in options else 0.0

    # Plain string: exact or either-part match
    else:
        resp_parts = [p.strip() for p in resp_clean.split("+")]
        if len(resp_parts) == 2:
            return 1.0 if resp_parts[0] == ans or resp_parts[1] == ans else 0.0
        return 1.0 if resp_clean == ans else 0.0

    return 0.0


def score_dim7(resp_clean: str, ans: str, *_unused: str) -> float:
    """Exact-match scoring for multiple-choice questions (dim7)."""
    return 1.0 if str(resp_clean).strip() == str(ans).strip() else 0.0


def score_dim9(resp_clean: str, ans: str, *_unused: str) -> float:
    """Exact-match scoring for crisis-risk-level classification (dim9)."""
    return 1.0 if str(resp_clean).strip() == str(ans).strip() else 0.0


# Dimension name → scoring function registry
scoring_funcs: Dict[str, Callable[..., float]] = {
    "dim1-1": score_dim_1,
    "dim1-2": score_dim_2,
    "dim2-1": score_dim_1,
    "dim2-2": score_dim_2,
    "dim7": score_dim7,
    "dim8-1": score_dim_2,
    "dim8-2": score_dim_2,
    "dim9": score_dim9,
}

# Backward-compatible alias
score_dim2_1 = score_dim_2


# ---------------------------------------------------------------------------
# Batch scoring over Excel files
# ---------------------------------------------------------------------------

def score_response_file(
    input_file: str,
    output_file: str | None = None,
    dimension: str = "dim2-2",
    answer_column: str = "answer",
) -> Path:
    """Score every model column in a response Excel file.

    Cleans <answer> tags from responses, then applies the dimension-specific
    scoring function row-wise.

    Args:
        input_file: Path to the response .xlsx file.
        output_file: Output path; defaults to ``{input_stem}_scored.xlsx``.
        dimension: Scoring dimension key (must exist in ``scoring_funcs``).
        answer_column: Column name holding reference answers.

    Returns:
        Path of the scored output file.
    """
    if dimension not in scoring_funcs:
        raise ValueError(f"Unknown scoring dimension: {dimension}")

    path = Path(input_file)
    out_path = Path(output_file) if output_file else path.with_name(f"{path.stem}_scored.xlsx")
    df = pd.read_excel(path)
    score_func = scoring_funcs[dimension]

    # Identify model columns (exclude metadata, pre-computed score/clean columns)
    model_cols = [
        col for col in df.columns
        if col not in {"dimension", "Dimension", "Question", "question",
                       "Answer", "answer", "Secondary_answer", "secondary_answer"}
        and not col.endswith("_score")
        and not col.endswith("_clean")
        and not col.startswith("detail_")
    ]

    for model in model_cols:
        clean_col = f"{model}_clean"
        score_col = f"{model}_score"

        # Strip <answer> XML tags from model responses
        df[clean_col] = (
            df[model]
            .astype(str)
            .str.replace("<answer>", "", regex=False)
            .str.replace("</answer>", "", regex=False)
            .str.strip()
        )

        # Detect secondary answer column (case-insensitive)
        secondary_col = (
            "Secondary_answer" if "Secondary_answer" in df.columns
            else "secondary_answer" if "secondary_answer" in df.columns
            else None
        )

        # Apply dimension scoring function row-wise
        df[score_col] = df.apply(
            lambda row: score_func(
                str(row[clean_col]),
                str(row[answer_column]),
                str(row[secondary_col]) if secondary_col else "",
            ),
            axis=1,
        )

    df.to_excel(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def score_cli(argv: list[str] | None = None) -> None:
    """Parse CLI args and run the scoring pipeline."""
    parser = argparse.ArgumentParser(description="Score objective response Excel files.")
    parser.add_argument("input_file")
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--dimension", default="dim1-1", choices=sorted(scoring_funcs))
    parser.add_argument("--answer-column", default="answer")
    args = parser.parse_args(argv)

    out_path = score_response_file(
        input_file=args.input_file,
        output_file=args.output_file,
        dimension=args.dimension,
        answer_column=args.answer_column,
    )
    print(f"Scored file saved to {out_path}")


if __name__ == "__main__":
    score_cli()

"""Unified launcher for the FACET subjective ELO evaluation pipeline.

Actions:
  comparisons   — generate pairwise comparison data (seed + smart-matching)
  calculate     — compute leaderboards from existing comparison JSON files
  full-pipeline — comparisons + calculate
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
from config import DIMENSIONS, COMPARISONS_DIR, RESULTS_DIR, DIMENSION_TO_LEADERBOARD_FILE
from run_comparisons import generate_comparisons_for_dimension, get_models_from_excel
from elo_tools import calculate_elo_ratings


def save_leaderboard(ratings: Dict, file_path: Path):
    """Save ranked leaderboard to JSON and print to console."""
    sorted_models = sorted(ratings.items(), key=lambda item: item[1]['mu'], reverse=True)

    print(f"\n--- Leaderboard: {file_path.stem} ---")
    print(f"{'Rank':<5} {'Model':<30} {'Score (mu)':<15} {'Uncertainty (sigma)':<20}")
    print("-" * 75)

    leaderboard_data = {}
    for i, (model, scores) in enumerate(sorted_models):
        rank = i + 1
        mu, sigma = scores['mu'], scores['sigma']
        print(f"{rank:<5} {model:<30} {mu:<15.2f} {sigma:<20.2f}")
        leaderboard_data[model] = scores

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(leaderboard_data, f, indent=2)
    print(f"\nLeaderboard saved to: {file_path}")


def load_all_comparisons(dimension: str = None) -> List[Dict]:
    """Load aggregated comparison results for one dimension or all dimensions."""
    comparisons = []
    base_dir = COMPARISONS_DIR
    search_dirs = [base_dir / dimension] if dimension else [d for d in base_dir.iterdir() if d.is_dir()]

    for dim_dir in search_dirs:
        for file_path in dim_dir.glob("*/*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'aggregated_result' in data:
                        agg_res = data['aggregated_result']
                        agg_res['pair'] = data['pair']
                        comparisons.append(agg_res)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: failed to read {file_path}: {e}")
                continue
    return comparisons


def get_all_models_from_comparisons() -> List[str]:
    """Scan all comparison JSON files and return the unique set of model names."""
    model_set = set()
    for dim_dir in COMPARISONS_DIR.iterdir():
        if dim_dir.is_dir():
            for file_path in dim_dir.glob("*/*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        model_set.update(data['pair'])
                except (json.JSONDecodeError, KeyError):
                    continue
    return list(model_set)


def run_comparison_pipeline(target_dimension: str = None, rounds: int = None):
    """Run the comparison data-generation stage (seed + smart-matching)."""
    dims_to_run = [target_dimension] if target_dimension else DIMENSIONS
    print(f"Generating comparisons for: {dims_to_run} "
          f"(rounds limit: {rounds if rounds else 'all'})")

    for dimension in dims_to_run:
        generate_comparisons_for_dimension(dimension, rounds=rounds)

    print("\n  Comparison data generation complete!")


def run_calculation_pipeline(target_dimension: str = None):
    """Run the leaderboard calculation stage from existing comparison data."""
    print("\n--- Calculating final leaderboards ---")

    if target_dimension:
        all_models, _ = get_models_from_excel(target_dimension)
    else:
        all_models = get_all_models_from_comparisons()

    if not all_models:
        print("No models found, cannot compute rankings.")
        return

    dims_to_calculate = [target_dimension] if target_dimension else DIMENSIONS

    for dimension in dims_to_calculate:
        print(f"\n--- Calculating dimension: {dimension} ---")
        dim_comparisons = load_all_comparisons(dimension=dimension)
        if dim_comparisons:
            dim_ratings = calculate_elo_ratings(all_models, dim_comparisons)
            save_leaderboard(dim_ratings, DIMENSION_TO_LEADERBOARD_FILE[dimension])
        else:
            print(f"No comparison data for dimension '{dimension}', skipping.")

    # Per-dimension done — now compute Overall leaderboard
    if not target_dimension:
        print("\n--- Calculating Overall leaderboard ---")
        overall_comparisons = load_all_comparisons()
        if overall_comparisons:
            overall_ratings = calculate_elo_ratings(all_models, overall_comparisons)
            save_leaderboard(overall_ratings, DIMENSION_TO_LEADERBOARD_FILE["Overall"])
        else:
            print("No comparison data found, cannot generate Overall leaderboard.")


def main():
    parser = argparse.ArgumentParser(
        description="FACET modular ELO evaluation pipeline controller."
    )
    parser.add_argument(
        '--lang', type=str, default=None, choices=['zh', 'en'],
        help="Prompt language: zh (Chinese) or en (English). "
             "Selects prompt_config_zh.py or prompt_config_en.py."
    )
    parser.add_argument(
        '--action', type=str, required=True,
        choices=['comparisons', 'calculate', 'full-pipeline'],
        help="comparisons (generate data only), calculate (rankings only), "
             "full-pipeline (both)."
    )
    parser.add_argument(
        '--dimension', type=str, default=None, choices=DIMENSIONS,
        help="Optionally restrict to a single dimension."
    )
    parser.add_argument(
        '--rounds', type=int, default=None,
        help="Limit processing to the first N data files. "
             "If omitted, all matching files are processed."
    )
    args = parser.parse_args()

    if args.action == 'comparisons':
        run_comparison_pipeline(args.dimension, rounds=args.rounds)
    elif args.action == 'calculate':
        run_calculation_pipeline(args.dimension)
    elif args.action == 'full-pipeline':
        run_comparison_pipeline(args.dimension, rounds=args.rounds)
        run_calculation_pipeline(args.dimension)

    print("\nAll requested operations completed.")


if __name__ == "__main__":
    main()

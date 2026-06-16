"""Incremental ELO evaluation helper for adding new models.

When new model outputs become available, this script:
  1. Loads the existing baseline leaderboard for each dimension.
  2. Seeds the new model(s) at the median mu of existing models.
  3. Runs a seed round (random opponents) followed by smart-matching
     rounds that focus on the new models until their ranks converge.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple
import pandas as pd
from tqdm import tqdm
from itertools import combinations
import random
import concurrent.futures
from collections import Counter

from config import (
    RESPONSES_DIR, COMPARISONS_DIR,
    DIMENSION_TO_FILENAME, DIMENSION_TO_PROMPTS,
    DIMENSION_TO_LEADERBOARD_FILE, INITIAL_SIGMA, INITIAL_MU,
    NUM_MATCHES_PER_SMART_ROUND, MAX_SMART_ROUNDS,
    SAMPLE_COUNT_HIGH_PRIORITY, SAMPLE_COUNT_MID_PRIORITY,
    SAMPLE_COUNT_LOW_PRIORITY,
    MAX_WORKERS, RANDOM_SEED,
)
from elo_tools import calculate_elo_ratings
from run_comparisons import (
    process_single_matchup,
    load_all_comparisons_for_elo,
    get_already_played_prompts,
    get_models_from_excel,
)

random.seed(RANDOM_SEED)

# Dimensions to process (edit as needed)
DIMENSIONS = [
    "emotional_matching",
]


def find_priority_matches_for_new_models(
    new_models: List[str],
    ratings: Dict,
    already_played_pairs: Set[Tuple[str, str]],
    num_matches: int,
) -> List[Tuple[str, str]]:
    """Find priority matchups for newly added models.

    Scores each candidate pair by combined uncertainty × skill proximity,
    so the algorithm first resolves the most uncertain and closest-ranked pairs.
    """
    candidate_matches = []
    old_models = [m for m in ratings if m not in new_models]
    epsilon = 1e-6

    # New vs old model pairings
    for new_m in new_models:
        for old_m in old_models:
            match_key = tuple(sorted((new_m, old_m)))
            if match_key in already_played_pairs:
                continue
            if new_m not in ratings or old_m not in ratings:
                continue
            mu_new, sigma_new = ratings[new_m]['mu'], ratings[new_m]['sigma']
            mu_old, sigma_old = ratings[old_m]['mu'], ratings[old_m]['sigma']
            priority_score = (sigma_new + sigma_old) * (1.0 / (abs(mu_new - mu_old) + epsilon))
            candidate_matches.append({'pair': match_key, 'score': priority_score})

    # New vs new model pairings
    for model_A, model_B in combinations(new_models, 2):
        match_key = tuple(sorted((model_A, model_B)))
        if match_key in already_played_pairs:
            continue
        if model_A not in ratings or model_B not in ratings:
            continue
        mu_A, sigma_A = ratings[model_A]['mu'], ratings[model_A]['sigma']
        mu_B, sigma_B = ratings[model_B]['mu'], ratings[model_B]['sigma']
        priority_score = (sigma_A + sigma_B) * (1.0 / (abs(mu_A - mu_B) + epsilon))
        candidate_matches.append({'pair': match_key, 'score': priority_score})

    candidate_matches.sort(key=lambda x: x['score'], reverse=True)
    return [match['pair'] for match in candidate_matches[:num_matches]]


def main():
    parser = argparse.ArgumentParser(
        description="Incrementally generate ELO evaluation data for new model(s) "
                    "and iterate until their rankings converge."
    )
    parser.add_argument('new_models', type=str, nargs='+',
                        help="One or more new model names.")
    parser.add_argument('--rounds', type=int, default=None,
                        help="Limit the number of test rounds (data files) to process.")
    args = parser.parse_args()

    print(f"🌟  Starting incremental evaluation for new model(s): {args.new_models} "
          f"(convergence mode, rounds limit: {args.rounds})...")

    for dimension in DIMENSIONS:
        print(f"\n{'='*25} Processing dimension: {dimension} {'='*25}")

        # Load baseline leaderboard
        leaderboard_file = DIMENSION_TO_LEADERBOARD_FILE[dimension]
        if not leaderboard_file.exists():
            print(f"  - Warning: baseline leaderboard not found for '{dimension}'. Skipping.")
            continue
        with open(leaderboard_file, 'r', encoding='utf-8') as f:
            base_ratings = json.load(f)

        # Filter to models that are genuinely new (not in existing leaderboard)
        truly_new_models = [m for m in args.new_models if m not in base_ratings]
        existing_models = sorted(list(base_ratings.keys()))
        if not truly_new_models:
            print(f"  - All specified models already exist in '{dimension}' leaderboard. Skipping.")
            continue
        print(f"  - Will rank new model(s): {truly_new_models}")

        # Seed new models at the median mu of existing models
        all_mu_scores = [scores['mu'] for scores in base_ratings.values()]
        median_mu = sorted(all_mu_scores)[len(all_mu_scores) // 2] if all_mu_scores else INITIAL_MU
        current_ratings = base_ratings.copy()
        for new_model in truly_new_models:
            current_ratings[new_model] = {'mu': median_mu, 'sigma': INITIAL_SIGMA}
        all_models_in_play = list(current_ratings.keys())

        # Load response data
        _, df = get_models_from_excel(dimension, rounds=args.rounds)
        if df is None or df.empty:
            print(f"  - Error: cannot load data for dimension '{dimension}'.")
            continue

        prompt_template = DIMENSION_TO_PROMPTS[dimension]
        dimension_comparison_dir = COMPARISONS_DIR / dimension

        # ==================================================================
        # Phase 1: Seed round (cold-start for new models)
        # ==================================================================
        print(f"\n--- [Phase 1] Seed Round (cold-start) ---")
        num_seed_opponents = max(1, len(existing_models) // 2)
        print(f"Each prompt: {num_seed_opponents} random existing opponent(s) per new model")

        for idx, row in df.iterrows():
            tasks = []
            prompt_text = row['question']
            original_idx = row['original_index']
            version = row['source_version']
            prompt_id_str = f"prompt_{original_idx}_v{version}"

            if not isinstance(prompt_text, str) or not prompt_text.strip():
                continue

            current_seed = RANDOM_SEED + original_idx + (version * 10000)
            random.seed(current_seed)

            # Build per-prompt seed matchups
            prompt_seed_matchups = []
            for new_m in truly_new_models:
                if len(existing_models) <= num_seed_opponents:
                    selected = existing_models
                else:
                    selected = random.sample(existing_models, num_seed_opponents)
                for opp in selected:
                    prompt_seed_matchups.append(tuple(sorted((new_m, opp))))

            for model_A, model_B in prompt_seed_matchups:
                comp_file = dimension_comparison_dir / prompt_id_str / f"{model_A}_vs_{model_B}.json"
                if not comp_file.exists():
                    tasks.append({
                        "model_A": model_A, "model_B": model_B,
                        "prompt_text": prompt_text,
                        "prompt_template": prompt_template,
                        "comparison_file": comp_file,
                        "row_data": row,
                    })

            if not tasks:
                print(f"✅ [Skip] {prompt_id_str} — already complete")
            else:
                print(f"🚀 [Run] {prompt_id_str}: {len(tasks)} new matchups...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    list(tqdm(
                        executor.map(lambda p: process_single_matchup(**p), tasks),
                        total=len(tasks),
                        desc=f"Running {prompt_id_str}",
                        leave=False,
                    ))

        print("\n✅ Seed round complete for all prompts.")

        # ==================================================================
        # Phase 2: Smart-matching rounds until convergence
        # ==================================================================
        all_dim_comps_for_elo = load_all_comparisons_for_elo(
            dimension_comparison_dir, all_models_in_play
        )
        previous_ranks = {}

        for i in range(MAX_SMART_ROUNDS):
            print(f"\n--- Incremental Round {i+1}/{MAX_SMART_ROUNDS} for {dimension} ---")

            current_ratings = calculate_elo_ratings(all_models_in_play, all_dim_comps_for_elo)
            pair_counts = Counter(tuple(sorted(c['pair'])) for c in all_dim_comps_for_elo)
            total_prompts_count = len(df)

            # Check convergence: have new model ranks stabilised?
            current_ladder = sorted(current_ratings.items(),
                                    key=lambda item: item[1]['mu'], reverse=True)
            current_ranks = {model: rank + 1 for rank, (model, _) in enumerate(current_ladder)
                             if model in truly_new_models}
            print("  - Current new-model ranks:", current_ranks)
            if i > 0 and current_ranks == previous_ranks:
                print(f"\n=== New model ranks converged for '{dimension}'! ===")
                break
            previous_ranks = current_ranks

            # Pairs that have been compared on every prompt are considered exhausted
            already_played_pairs = {pair for pair, count in pair_counts.items()
                                    if count >= total_prompts_count}

            matchups_to_play = find_priority_matches_for_new_models(
                truly_new_models, current_ratings, already_played_pairs,
                NUM_MATCHES_PER_SMART_ROUND,
            )

            if not matchups_to_play:
                print("\n=== No more new-model-related matches available! ===")
                break

            # Build tasks with priority-based sampling
            tasks = []
            num_high_priority = int(len(matchups_to_play) * 0.1)
            num_mid_priority = int(len(matchups_to_play) * 0.3)

            for idx, pair in enumerate(matchups_to_play):
                model_A, model_B = pair
                if idx <= num_high_priority:
                    sample_count = SAMPLE_COUNT_HIGH_PRIORITY
                elif idx <= num_mid_priority:
                    sample_count = SAMPLE_COUNT_MID_PRIORITY
                else:
                    sample_count = SAMPLE_COUNT_LOW_PRIORITY

                already_played_prompts = get_already_played_prompts(
                    pair, df, dimension_comparison_dir
                )
                all_prompt_indices = [
                    f"prompt_{row['original_index']}_v{row['source_version']}"
                    for _, row in df.iterrows()
                ]
                available_prompts = [pid for pid in all_prompt_indices
                                     if pid not in already_played_prompts]

                random.shuffle(available_prompts)
                prompts_to_play = available_prompts[:sample_count]

                for prompt_id_str in prompts_to_play:
                    try:
                        parts = prompt_id_str.split('_v')
                        if len(parts) != 2:
                            continue
                        version = int(parts[1])
                        original_idx = int(parts[0].split('_')[-1])

                        row_data = df[
                            (df['original_index'] == original_idx) &
                            (df['source_version'] == version)
                        ].iloc[0]

                        prompt_text = row_data['question']
                        comp_file = dimension_comparison_dir / prompt_id_str / f"{model_A}_vs_{model_B}.json"
                        tasks.append({
                            "model_A": model_A, "model_B": model_B,
                            "prompt_text": prompt_text,
                            "prompt_template": prompt_template,
                            "comparison_file": comp_file,
                            "row_data": row_data,
                        })
                    except (IndexError, ValueError, pd.errors.EmptyDataError):
                        continue

            if not tasks:
                print("  - All focus pairs complete on all prompts this round.")
                continue

            print(f"  - Running {len(tasks)} adaptive-sampling matchups this round...")

            newly_completed_comps_this_round = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_task = {executor.submit(process_single_matchup, **task): task
                                  for task in tasks}
                for future in tqdm(concurrent.futures.as_completed(future_to_task),
                                   total=len(tasks),
                                   desc=f"Incremental matches for {dimension}"):
                    result = future.result()
                    if result:
                        task = future_to_task[future]
                        result['pair'] = [task['model_A'], task['model_B']]
                        newly_completed_comps_this_round.append(result)

            all_dim_comps_for_elo.extend(newly_completed_comps_this_round)
        else:
            print(f"\n=== Max rounds ({MAX_SMART_ROUNDS}) reached for '{dimension}'. ===")

    print("\nAll new-model incremental evaluation data generated!")
    print("Next step: run 'python main.py --action calculate' to update leaderboards.")


if __name__ == "__main__":
    main()

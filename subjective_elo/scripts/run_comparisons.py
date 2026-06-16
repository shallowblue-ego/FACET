"""Pairwise comparison runner for the FACET subjective ELO pipeline.

Key workflow:
  Seed round (cold-start) → Smart-matching rounds (iterative, convergence-based)
Each matchup runs two judge calls (position-swapped) to reduce position bias,
then aggregates them into a single result with winner, loser, and margin.
"""

import json
import random
from pathlib import Path
from itertools import combinations
from typing import List, Dict, Set, Tuple
import pandas as pd
from tqdm import tqdm
import re
import concurrent.futures
import shutil

from config import (
    RESPONSES_DIR, COMPARISONS_DIR, JUDGE_MODEL, JUDGE_TEMPERATURE, JUDGE_SEED,
    DIMENSION_TO_FILENAME, DIMENSION_TO_PROMPTS,
    NUM_MATCHES_PER_SMART_ROUND, MAX_SMART_ROUNDS,
    DIMENSION_TO_LEADERBOARD_FILE, MAX_WORKERS, RANDOM_SEED,
    SAMPLE_COUNT_HIGH_PRIORITY, SAMPLE_COUNT_MID_PRIORITY, SAMPLE_COUNT_LOW_PRIORITY,
)

from llm import llm_service
from elo_tools import calculate_elo_ratings

random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_answer(text):
    """Extract content inside <answer>...</answer> tags via regex.
    Falls back to the raw stripped text if no tags found."""
    if not isinstance(text, str):
        return None
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def aggregate_judgments(res_A_vs_B: Dict, res_B_vs_A: Dict,
                        model_A: str, model_B: str) -> Dict:
    """Aggregate two position-swapped judge results into one verdict.

    If both calls agree on the winner, return that winner with an averaged
    margin. If they disagree, the result is marked as DRAW / conflicted."""
    try:
        # Judgment 1: A in position A, B in position B
        winner_code_1 = ''.join(filter(str.isalpha, res_A_vs_B.get("overall_winner", "")))
        margin_1 = res_A_vs_B.get("overall_winner", "").count('+')
        winner_1 = model_A if winner_code_1 == 'A' else model_B

        # Judgment 2: positions swapped — B in position A, A in position B
        winner_code_2 = ''.join(filter(str.isalpha, res_B_vs_A.get("overall_winner", "")))
        margin_2 = res_B_vs_A.get("overall_winner", "").count('+')
        winner_2 = model_B if winner_code_2 == 'A' else model_A   # swap back

        if winner_1 == winner_2:
            return {
                "winner": winner_1,
                "loser": model_A if winner_1 == model_B else model_B,
                "margin": round((margin_1 + margin_2) / 2.0),
                "is_conflicted": False,
            }
        else:
            return {
                "winner": "DRAW", "loser": "DRAW",
                "margin": 0, "is_conflicted": True,
            }
    except Exception:
        return {"winner": "DRAW", "loser": "DRAW", "margin": 0, "is_conflicted": True}


# ---------------------------------------------------------------------------
# Single matchup processing
# ---------------------------------------------------------------------------

def process_single_matchup(model_A, model_B, prompt_text, prompt_template,
                           comparison_file, row_data):
    """Run one matchup: two judge calls (position-swapped), aggregate, save JSON.

    Returns the aggregated result dict, or None if skipped/failed."""
    if comparison_file.exists():
        return None   # already computed

    response_A_text = extract_answer(row_data.get(model_A))
    response_B_text = extract_answer(row_data.get(model_B))
    len_A = len(response_A_text) if response_A_text else 0
    len_B = len(response_B_text) if response_B_text else 0
    if not response_A_text or not response_B_text:
        return None

    # Call 1: A in position A, B in position B
    prompt1 = prompt_template.format(
        prompt_text=prompt_text,
        response_A_text=response_A_text,
        response_B_text=response_B_text,
        len_A=len_A, len_B=len_B,
    )
    res1 = llm_service(prompt=prompt1, model_name=JUDGE_MODEL,
                       temperature=JUDGE_TEMPERATURE, seed=JUDGE_SEED, is_json=True)

    # Call 2: positions swapped to reduce position bias
    prompt2 = prompt_template.format(
        prompt_text=prompt_text,
        response_A_text=response_B_text,
        response_B_text=response_A_text,
        len_A=len_B, len_B=len_A,
    )
    res2 = llm_service(prompt=prompt2, model_name=JUDGE_MODEL,
                       temperature=JUDGE_TEMPERATURE, seed=JUDGE_SEED, is_json=True)

    if res1 and res2:
        aggregated_result = aggregate_judgments(res1, res2, model_A, model_B)
        output_data = {
            'pair': [model_A, model_B],
            'aggregated_result': aggregated_result,
            'raw_judgments': {'A_vs_B': res1, 'B_vs_A': res2},
        }
        comparison_file.parent.mkdir(parents=True, exist_ok=True)
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        return aggregated_result
    return None


# ---------------------------------------------------------------------------
# Matchup generation (seed round + smart-matching)
# ---------------------------------------------------------------------------

def generate_seed_round_matchups(all_models: List[str],
                                 num_matches: int) -> List[Tuple[str, str]]:
    """Generate balanced random matchups for the cold-start seed round.
    Each model gets up to num_matches distinct opponents."""
    matchups_to_play = set()
    for model_A in all_models:
        potential_opponents = [m for m in all_models if m != model_A]
        num_to_select = min(num_matches, len(potential_opponents))
        random.shuffle(potential_opponents)
        selected_opponents = potential_opponents[:num_to_select]
        for model_B in selected_opponents:
            match_key = tuple(sorted((model_A, model_B)))
            matchups_to_play.add(match_key)
    return list(matchups_to_play)


def get_next_matchups(ratings: Dict, already_played: Set[Tuple[str, str]],
                      num_matches: int) -> List[Tuple[str, str]]:
    """Select the next round of matchups via smart-matching.

    Prioritises pairs with high combined uncertainty (sigma_A + sigma_B)
    and close skill proximity (1 / |mu_A - mu_B|)."""
    candidate_matches = []
    all_models = list(ratings.keys())
    epsilon = 1e-6
    for model_A, model_B in combinations(all_models, 2):
        match_key = tuple(sorted((model_A, model_B)))
        if model_A not in ratings or model_B not in ratings:
            continue
        mu_A, sigma_A = ratings[model_A]['mu'], ratings[model_A]['sigma']
        mu_B, sigma_B = ratings[model_B]['mu'], ratings[model_B]['sigma']
        uncertainty_score = sigma_A + sigma_B
        proximity_score = 1.0 / (abs(mu_A - mu_B) + epsilon)
        priority_score = uncertainty_score * proximity_score
        candidate_matches.append({'pair': match_key, 'score': priority_score})
    candidate_matches.sort(key=lambda x: x['score'], reverse=True)
    return [match['pair'] for match in candidate_matches[:num_matches]]


# ---------------------------------------------------------------------------
# Comparison loading & prompt tracking
# ---------------------------------------------------------------------------

def load_all_comparisons_for_elo(dimension_comparison_dir: Path,
                                 all_models: List[str]) -> List[Dict]:
    """Load all aggregate comparison results for one dimension.

    Only includes results where both models are in ``all_models``."""
    all_dim_comps_for_elo = []
    for file_path in dimension_comparison_dir.glob("*/*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            agg_res = data.get('aggregated_result', {})
            agg_res['pair'] = data.get('pair', [])
            if all(m in all_models for m in agg_res['pair']):
                all_dim_comps_for_elo.append(agg_res)
        except (json.JSONDecodeError, KeyError):
            continue
    return all_dim_comps_for_elo


def get_already_played_prompts(pair: Tuple[str, str], df_prompts: pd.DataFrame,
                               dimension_comparison_dir: Path) -> Set[str]:
    """Return the set of versioned prompt IDs already compared for a given pair."""
    played_prompts = set()
    model_A, model_B = pair
    for _, row in df_prompts.iterrows():
        prompt_id_str = f"prompt_{row['original_index']}_v{row['source_version']}"
        comp_file = dimension_comparison_dir / prompt_id_str / f"{model_A}_vs_{model_B}.json"
        if comp_file.exists():
            played_prompts.add(prompt_id_str)
    return played_prompts


# ---------------------------------------------------------------------------
# Excel data loading
# ---------------------------------------------------------------------------

def get_models_from_excel(dimension: str, rounds: int = None) -> Tuple[List[str], pd.DataFrame]:
    """Load and merge response Excel files for a dimension.

    Files are matched by DIMENSION_TO_FILENAME prefix, sorted by name, and
    concatenated. Each file gets a ``source_version`` and ``original_index``
    column added for prompt-level tracking.

    Args:
        dimension: Dimension key (e.g. "emotional_deepening").
        rounds: If set, only process the first N files.

    Returns:
        (list of model names, concatenated DataFrame)."""
    all_models_set = set()
    filename_key = DIMENSION_TO_FILENAME.get(dimension)
    if not filename_key:
        print(f"Dimension {dimension} has no configured filename key.")
        return [], None

    files = list(RESPONSES_DIR.glob(f"{filename_key}*.xlsx"))
    files.sort(key=lambda f: f.name)

    if not files:
        print(f"No XLSX files found for dimension {dimension}.")
        return [], None

    if rounds is not None:
        if len(files) < rounds:
            raise FileNotFoundError(
                f"Requested {rounds} rounds but only {len(files)} files found "
                f"for dimension '{dimension}': {[f.name for f in files]}. "
                f"Check the file path or reduce --rounds."
            )
        print(f"Processing only the first {rounds} files (--rounds limit).")
        files = files[:rounds]

    print(f"Found {len(files)} data file(s): {[f.name for f in files]}, merging...")

    combined_df_list = []
    for i, file_path in enumerate(files):
        try:
            df_temp = pd.read_excel(file_path)
            # Identify model columns (exclude metadata)
            non_model_columns = ['dimension', 'question', 'answer']
            models_in_file = [col for col in df_temp.columns if col not in non_model_columns]
            all_models_set.update(models_in_file)

            df_temp['source_version'] = i              # which file this row came from
            df_temp['original_index'] = df_temp.index  # original row index within that file
            combined_df_list.append(df_temp)
        except Exception as e:
            print(f"Failed to read {file_path.name}: {e}")
            continue

    if not combined_df_list:
        return [], None

    final_df = pd.concat(combined_df_list, ignore_index=True)
    return sorted(list(all_models_set)), final_df


# ---------------------------------------------------------------------------
# Main dimension pipeline
# ---------------------------------------------------------------------------

def generate_comparisons_for_dimension(dimension: str, rounds: int = None):
    """Run the full comparison pipeline for one dimension.

    Phases:
    1. Seed round: cold-start with balanced random matchups.
       Supports Copy-on-Reuse from a ``comparisons_exist/`` backup directory.
    2. Smart-matching rounds: iterative, convergence-driven matchmaking
       using current TrueSkill ratings to prioritise informative pairs.
    """
    print(f"\n{'='*25} Generating comparisons for dimension: {dimension} {'='*25}")

    all_models, df = get_models_from_excel(dimension, rounds=rounds)
    if df is None or df.empty:
        print("❌ [Error] Cannot load prompt data, skipping.")
        return

    dimension_comparison_dir = COMPARISONS_DIR / dimension
    prompt_template = DIMENSION_TO_PROMPTS[dimension]

    # Check for backup directory for Copy-on-Reuse
    exist_dir_name = "comparisons_exist"
    source_dim_dir = COMPARISONS_DIR.parent / exist_dir_name / dimension
    has_backup = source_dim_dir.exists()
    if has_backup:
        print(f"   - ✅ Found backup [{exist_dir_name}], Copy-on-Reuse enabled.")
    else:
        print(f"   - ⚠️  No backup [{exist_dir_name}] found, local-only mode.")

    total_new = 0
    total_copied = 0

    # ==================================================================
    # Phase 1: Seed round (cold-start)
    # ==================================================================
    print("\n--- Seed Round (cold-start, per-prompt processing) ---")
    num_seed_matches = len(all_models) // 2
    print(f"Each model will participate in ~{num_seed_matches} seed matches.")

    for idx, row in df.iterrows():
        tasks = []
        original_idx = row['original_index']
        version = row['source_version']
        prompt_id_str = f"prompt_{original_idx}_v{version}"
        prompt_text = row['question']

        if not isinstance(prompt_text, str) or not prompt_text.strip():
            continue

        # Deterministic per-prompt seed
        current_seed = RANDOM_SEED + original_idx + (version * 10000)
        random.seed(current_seed)

        seed_matchups = generate_seed_round_matchups(all_models, num_seed_matches)
        copied_backup = 0

        for model_A, model_B in seed_matchups:
            filename = f"{model_A}_vs_{model_B}.json"
            target_file = dimension_comparison_dir / prompt_id_str / filename
            source_file = source_dim_dir / prompt_id_str / filename

            if target_file.exists():
                continue   # already computed

            # Try to copy from backup first
            if has_backup and source_file.exists():
                try:
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, target_file)
                    copied_backup += 1
                    continue
                except Exception as e:
                    print(f"⚠️  [Copy failed] {filename}: {e}")

            tasks.append({
                "model_A": model_A, "model_B": model_B,
                "prompt_text": prompt_text, "prompt_template": prompt_template,
                "comparison_file": target_file, "row_data": row,
            })

        if not tasks:
            total_copied += copied_backup
            print(f"✅ [Skip] {prompt_id_str} — already complete")
        else:
            msg = f"🚀 [Run] {prompt_id_str}: {len(tasks)} new matchups..."
            if copied_backup > 0:
                msg += f" | 📥 {copied_backup} copied from backup"
            print(msg)
            total_new += len(tasks)
            total_copied += copied_backup

            # Parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                list(tqdm(
                    executor.map(lambda p: process_single_matchup(**p), tasks),
                    total=len(tasks),
                    desc=f"Running {prompt_id_str}",
                    leave=False,
                ))

    print("\n✅ Seed round complete for all prompts.")
    print(f"\n✅ [{dimension}] Done: {total_new} new, {total_copied} copied from backup.\n")

    # ==================================================================
    # Phase 2: Smart-matching rounds
    # ==================================================================
    print("\n--- Smart-Matching Mode ---")

    all_dim_comps_for_elo = load_all_comparisons_for_elo(dimension_comparison_dir, all_models)
    previous_ladder = None

    for i in range(MAX_SMART_ROUNDS):
        print(f"\n--- Smart Round {i+1}/{MAX_SMART_ROUNDS} for {dimension} ---")

        if not all_dim_comps_for_elo:
            print("No comparison data to start smart-matching.")
            break

        ratings = calculate_elo_ratings(all_models, all_dim_comps_for_elo)

        # Check convergence: has the leaderboard stabilised?
        current_ladder = sorted(ratings.items(), key=lambda item: item[1]['mu'], reverse=True)
        current_ladder_names = [model[0] for model in current_ladder]

        if i > 0 and current_ladder_names == previous_ladder:
            print(f"\n=== Leaderboard converged for dimension '{dimension}'! ===")
            break
        previous_ladder = current_ladder_names

        already_played_pairs = {tuple(sorted(c['pair'])) for c in all_dim_comps_for_elo}
        matchups_to_play = get_next_matchups(ratings, already_played_pairs,
                                             NUM_MATCHES_PER_SMART_ROUND)

        if not matchups_to_play:
            print("\n=== All possible pairs already covered, smart-matching complete! ===")
            break

        tasks = []
        num_high_priority = int(len(matchups_to_play) * 0.1)
        num_mid_priority = int(len(matchups_to_play) * 0.3)

        print(f"Evaluating {len(matchups_to_play)} candidate pairs this round...")

        for idx, pair in enumerate(matchups_to_play):
            model_A, model_B = pair

            # Assign sample count by priority tier
            if idx <= num_high_priority:
                sample_count = SAMPLE_COUNT_HIGH_PRIORITY
            elif idx <= num_mid_priority:
                sample_count = SAMPLE_COUNT_MID_PRIORITY
            else:
                sample_count = SAMPLE_COUNT_LOW_PRIORITY

            # Find prompts not yet compared for this pair
            already_played_prompts = get_already_played_prompts(pair, df, dimension_comparison_dir)
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
                        "prompt_text": prompt_text, "prompt_template": prompt_template,
                        "comparison_file": comp_file, "row_data": row_data,
                    })
                except (IndexError, ValueError, pd.errors.EmptyDataError):
                    continue

        if not tasks:
            print("  - All focus pairs already complete on all prompts, smart-matching done.")
            break

        print(f"  - Priority sampling produced {len(tasks)} matchups this round.")

        # Run in parallel
        newly_completed_comps_this_round = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_task = {executor.submit(process_single_matchup, **task): task
                              for task in tasks}
            for future in tqdm(concurrent.futures.as_completed(future_to_task),
                               total=len(tasks), desc=f"Smart Round for {dimension}"):
                result = future.result()
                if result:
                    task = future_to_task[future]
                    result['pair'] = [task['model_A'], task['model_B']]
                    newly_completed_comps_this_round.append(result)

        all_dim_comps_for_elo.extend(newly_completed_comps_this_round)
    else:
        print(f"\n=== Max rounds ({MAX_SMART_ROUNDS}) reached for dimension '{dimension}'. ===")

    print(f"--- Comparison generation complete for dimension '{dimension}' ---")

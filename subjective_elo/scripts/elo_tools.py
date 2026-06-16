"""TrueSkill-based ELO rating calculation using the pseudo-win method."""

from typing import List, Dict, Tuple
import trueskill
import math

from config import (
    INITIAL_MU, INITIAL_SIGMA, BETA, TAU,
    MAX_WIN_MARGIN, BIN_SIZE,
)


def get_trueskill_env() -> trueskill.TrueSkill:
    """Create a configured TrueSkill environment (no draws allowed)."""
    return trueskill.TrueSkill(
        mu=INITIAL_MU, sigma=INITIAL_SIGMA,
        beta=BETA, tau=TAU, draw_probability=0.0,
    )


def bin_fraction(frac: float, bin_size: int) -> Tuple[int, int]:
    """Convert a fraction in [0, 1] to asymmetric pseudo-win counts.

    Used by the pseudo-win method: a fractional margin (e.g. 0.7) is mapped
    to repeated 1v1 games. Returns (wins_stronger, wins_weaker)."""
    frac = max(0.0, min(1.0, frac))
    eps = 1e-9

    if bin_size <= 0:
        return (1, 0) if frac > 0.5 else (0, 1) if frac < 0.5 else (1, 1)

    step = 0.5 / bin_size

    if abs(frac - 0.5) < eps:
        return 1, 1   # perfect tie

    if frac > 0.5:
        margin = frac - 0.5
        value_to_ceil = round(margin / step, 6)
        wins_test = max(1, min(bin_size, math.ceil(value_to_ceil)))
        return wins_test, 0
    else:
        margin = 0.5 - frac
        value_to_ceil = round(margin / step, 6)
        wins_other = max(1, min(bin_size, math.ceil(value_to_ceil)))
        return 0, wins_other


def update_ratings_with_margin(
    env: trueskill.TrueSkill, ratings: Dict, comparison: Dict
) -> None:
    """Update two models' ratings in-place from one comparison result.

    Handles DRAW, clear win (with margin → pseudo-wins), and conflicted/invalid cases."""
    winner = comparison.get('winner')

    # Draw
    if winner == 'DRAW':
        try:
            model_A, model_B = comparison['pair']
        except (KeyError, ValueError):
            return
        if model_A not in ratings or model_B not in ratings:
            return
        rating_A, rating_B = ratings[model_A], ratings[model_B]
        new_rating_A, new_rating_B = env.rate_1vs1(rating_A, rating_B, drawn=True)
        ratings[model_A], ratings[model_B] = new_rating_A, new_rating_B
        return

    # Win/loss
    loser = comparison.get('loser')
    if not winner or not loser or winner not in ratings or loser not in ratings:
        return

    winner_rating = ratings[winner]
    loser_rating = ratings[loser]

    # Clamp margin and convert to fraction
    margin = comparison.get('margin', 1)
    clamped_margin = min(margin, MAX_WIN_MARGIN)
    fraction_for_winner = 0.5 + 0.5 * (clamped_margin / MAX_WIN_MARGIN)

    # Convert to pseudo-win counts and apply
    wins_winner, wins_loser = bin_fraction(fraction_for_winner, BIN_SIZE)

    if wins_winner == 1 and wins_loser == 1:
        winner_rating, loser_rating = env.rate_1vs1(winner_rating, loser_rating, drawn=True)
    else:
        for _ in range(wins_winner):
            winner_rating, loser_rating = env.rate_1vs1(winner_rating, loser_rating)
        for _ in range(wins_loser):
            loser_rating, winner_rating = env.rate_1vs1(loser_rating, winner_rating)

    ratings[winner] = winner_rating
    ratings[loser] = loser_rating


def calculate_elo_ratings(
    all_models: List[str], all_comparisons: List[Dict]
) -> Dict:
    """Recompute TrueSkill ratings from all comparison results.

    Returns a dict of model_name → {"mu": float, "sigma": float}."""
    print(f"\n--- Computing ELO for {len(all_models)} models "
          f"from {len(all_comparisons)} comparisons... ---")

    env = get_trueskill_env()
    ratings = {model: env.Rating() for model in all_models}

    for comp in all_comparisons:
        if 'pair' in comp:
            update_ratings_with_margin(env, ratings, comp)

    final_ratings = {
        model: {"mu": rating.mu, "sigma": rating.sigma}
        for model, rating in ratings.items()
    }

    print("ELO ranking calculation complete.")
    return final_ratings

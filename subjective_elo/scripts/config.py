"""Shared configuration for the FACET subjective ELO pipeline."""

import importlib
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Language detection (--lang flag → ELO_LANG env var → default "zh")
# ---------------------------------------------------------------------------

def _detect_lang() -> str:
    if "--lang" in sys.argv:
        idx = sys.argv.index("--lang")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1].lower()
    for arg in sys.argv:
        if arg.startswith("--lang="):
            return arg.split("=", 1)[1].lower()
    return os.getenv("ELO_LANG", "zh").lower()


LANG = _detect_lang()
if LANG not in {"zh", "en"}:
    raise ValueError("ELO language must be 'zh' or 'en'.")

# Dynamically import language-specific prompt templates
_prompt_config = importlib.import_module(f"prompt_config_{LANG}")

elo_prompt_template_for_emotional_deepening = _prompt_config.elo_prompt_template_for_emotional_deepening
elo_prompt_template_for_emotional_matching = _prompt_config.elo_prompt_template_for_emotional_matching
elo_prompt_template_for_empathetic_understanding = _prompt_config.elo_prompt_template_for_empathetic_understanding
elo_prompt_template_for_emotion_regulation = _prompt_config.elo_prompt_template_for_emotion_regulation
elo_prompt_template_for_expression_naturalness = _prompt_config.elo_prompt_template_for_expression_naturalness

# ---------------------------------------------------------------------------
# Directory paths (relative to subjective_elo/)
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent   # subjective_elo/
PROMPTS_DIR = ROOT_DIR / "prompts"
RESPONSES_DIR = ROOT_DIR / "responses"               # input: model response .xlsx files
COMPARISONS_DIR = ROOT_DIR / "comparisons"           # output: pairwise comparison .json files
RESULTS_DIR = ROOT_DIR / "results"                   # output: leaderboard .json files

# ---------------------------------------------------------------------------
# Evaluation dimensions
# ---------------------------------------------------------------------------

DIMENSIONS = [
    "emotional_deepening",
    "emotional_matching",
    "empathetic_understanding",
    "emotional_regulation",
    "expression_naturalness",
]

DIMENSION_TO_PROMPTS = {
    "emotional_deepening": elo_prompt_template_for_emotional_deepening,
    "emotional_matching": elo_prompt_template_for_emotional_matching,
    "empathetic_understanding": elo_prompt_template_for_empathetic_understanding,
    "emotional_regulation": elo_prompt_template_for_emotion_regulation,
    "expression_naturalness": elo_prompt_template_for_expression_naturalness,
}

# Dimension → response Excel filename prefix (e.g. "emotional_deepening" → dim5*.xlsx)
DIMENSION_TO_FILENAME = {
    "emotional_deepening": "dim5",
    "emotional_matching": "dim6",
    "empathetic_understanding": "dim6",
    "emotional_regulation": "dim6",
    "expression_naturalness": "dim10",
}

# Dimension → leaderboard output file path (plus "Overall")
DIMENSION_TO_LEADERBOARD_FILE = {
    dim: RESULTS_DIR / f"leaderboard_{LANG}_{dim}.json" for dim in DIMENSIONS
}
DIMENSION_TO_LEADERBOARD_FILE["Overall"] = RESULTS_DIR / f"leaderboard_{LANG}_overall.json"

# ---------------------------------------------------------------------------
# LLM Judge defaults (overridable via environment variables)
# ---------------------------------------------------------------------------

JUDGE_MODEL = os.getenv("ELO_JUDGE_MODEL", "gemini-2.5-pro")
JUDGE_TEMPERATURE = float(os.getenv("ELO_JUDGE_TEMPERATURE", "0.2"))
JUDGE_SEED = int(os.getenv("ELO_JUDGE_SEED", "42"))

# ---------------------------------------------------------------------------
# TrueSkill hyper-parameters
# ---------------------------------------------------------------------------

INITIAL_MU = 1000.0               # default skill estimate for new models
INITIAL_SIGMA = INITIAL_MU / 3    # default uncertainty
BETA = INITIAL_SIGMA / 2          # skill gap where stronger player wins ~80%
TAU = 0.0                         # per-game skill volatility

# ---------------------------------------------------------------------------
# Concurrency & reproducibility
# ---------------------------------------------------------------------------

MAX_WORKERS = int(os.getenv("ELO_MAX_WORKERS", "5"))
RANDOM_SEED = int(os.getenv("ELO_RANDOM_SEED", "42"))

# ---------------------------------------------------------------------------
# Win-margin binning (for pseudo-win conversion in elo_tools)
# ---------------------------------------------------------------------------

MAX_WIN_MARGIN = 5                # max "+" count before clamping
BIN_SIZE = 5                      # number of discrete bins

# ---------------------------------------------------------------------------
# Smart-matching round control
# ---------------------------------------------------------------------------

NUM_MATCHES_PER_SMART_ROUND = 20  # unique pairs per smart round
MAX_SMART_ROUNDS = 10             # max smart-round iterations
NUM_MATCHES_FOR_NEW_MODEL = 20    # matches for a newly added model
SAMPLE_COUNT_HIGH_PRIORITY = 8    # prompts sampled for high-priority pairs
SAMPLE_COUNT_MID_PRIORITY = 4     # prompts sampled for mid-priority pairs
SAMPLE_COUNT_LOW_PRIORITY = 2     # prompts sampled for low-priority pairs

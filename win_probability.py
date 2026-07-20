"""
win_probability.py – core utilities for NBA home-team win probability modeling.

Contains:
  - build_features(df_raw)  : causal pre-game feature construction
  - FEATURE_SETS            : S0–S3 regimes
  - Log5Model + model factories
  - MODEL_SETS              : which models run on which regimes
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _safe_pct(wins: float, games: float) -> float:
    return wins / games if games > 0 else np.nan


def _safe_mean(total: float, games: float) -> float:
    return total / games if games > 0 else np.nan


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a leakage-free feature matrix from the raw game log.

    For every game we extract features from the *current* running state of
    both teams, then update the state with the game's box-score.  This
    guarantees that no post-tip information is ever used as a predictor.
    """
    state = defaultdict(lambda: {
        # overall
        "games": 0, "wins": 0,
        "pts": 0.0, "pts_allowed": 0.0,
        "reb": 0.0, "reb_allowed": 0.0,
        "tov": 0.0, "takeaways": 0.0,
        "fouls": 0.0, "fouls_drawn": 0.0,
        # home
        "games_home": 0, "wins_home": 0,
        "pts_home": 0.0, "pts_allowed_home": 0.0,
        "reb_home": 0.0, "reb_allowed_home": 0.0,
        "tov_home": 0.0, "takeaways_home": 0.0,
        "fouls_home": 0.0, "fouls_drawn_home": 0.0,
        # road
        "games_road": 0, "wins_road": 0,
        "pts_road": 0.0, "pts_allowed_road": 0.0,
        "reb_road": 0.0, "reb_allowed_road": 0.0,
        "tov_road": 0.0, "takeaways_road": 0.0,
        "fouls_road": 0.0, "fouls_drawn_road": 0.0,
        # calendar
        "last_date": None,
    })

    rows = []

    for _, g in df_raw.iterrows():
        home = g["home"]
        away = g["away"]
        date = g["game_date"]

        hs = state[home]
        as_ = state[away]

        # ----- extract pre-game features (state BEFORE this game) -----
        feat = {
            "game_id": g["game_id"],
            "game_date": date,
            "home": home,
            "away": away,
            # target
            "home_win": int(g["home_points"] > g["away_points"]),
            # strength
            "home_win_pct_overall": _safe_pct(hs["wins"], hs["games"]),
            "away_win_pct_overall": _safe_pct(as_["wins"], as_["games"]),
            "home_win_pct_at_home": _safe_pct(hs["wins_home"], hs["games_home"]),
            "away_win_pct_on_road": _safe_pct(as_["wins_road"], as_["games_road"]),
            # points
            "home_points_overall": _safe_mean(hs["pts"], hs["games"]),
            "away_points_overall": _safe_mean(as_["pts"], as_["games"]),
            "home_points_at_home": _safe_mean(hs["pts_home"], hs["games_home"]),
            "away_points_on_road": _safe_mean(as_["pts_road"], as_["games_road"]),
            "home_points_allowed_overall": _safe_mean(hs["pts_allowed"], hs["games"]),
            "away_points_allowed_overall": _safe_mean(as_["pts_allowed"], as_["games"]),
            "home_points_allowed_at_home": _safe_mean(hs["pts_allowed_home"], hs["games_home"]),
            "away_points_allowed_on_road": _safe_mean(as_["pts_allowed_road"], as_["games_road"]),
            # rebounds + nets
            "home_rebounds_overall": _safe_mean(hs["reb"], hs["games"]),
            "away_rebounds_overall": _safe_mean(as_["reb"], as_["games"]),
            "home_rebounds_allowed_overall": _safe_mean(hs["reb_allowed"], hs["games"]),
            "away_rebounds_allowed_overall": _safe_mean(as_["reb_allowed"], as_["games"]),
            "home_rebounds_net_overall": _safe_mean(hs["reb"] - hs["reb_allowed"], hs["games"]),
            "away_rebounds_net_overall": _safe_mean(as_["reb"] - as_["reb_allowed"], as_["games"]),
            "home_rebounds_net_at_home": _safe_mean(hs["reb_home"] - hs["reb_allowed_home"], hs["games_home"]),
            "away_rebounds_net_on_road": _safe_mean(as_["reb_road"] - as_["reb_allowed_road"], as_["games_road"]),
            # turnovers + nets
            "home_turnovers_overall": _safe_mean(hs["tov"], hs["games"]),
            "away_turnovers_overall": _safe_mean(as_["tov"], as_["games"]),
            "home_takeaways_overall": _safe_mean(hs["takeaways"], hs["games"]),
            "away_takeaways_overall": _safe_mean(as_["takeaways"], as_["games"]),
            "home_turnovers_net_overall": _safe_mean(hs["tov"] - hs["takeaways"], hs["games"]),
            "away_turnovers_net_overall": _safe_mean(as_["tov"] - as_["takeaways"], as_["games"]),
            "home_turnovers_net_at_home": _safe_mean(hs["tov_home"] - hs["takeaways_home"], hs["games_home"]),
            "away_turnovers_net_on_road": _safe_mean(as_["tov_road"] - as_["takeaways_road"], as_["games_road"]),
            # fouls + nets
            "home_fouls_overall": _safe_mean(hs["fouls"], hs["games"]),
            "away_fouls_overall": _safe_mean(as_["fouls"], as_["games"]),
            "home_fouls_drawn_overall": _safe_mean(hs["fouls_drawn"], hs["games"]),
            "away_fouls_drawn_overall": _safe_mean(as_["fouls_drawn"], as_["games"]),
            "home_fouls_net_overall": _safe_mean(hs["fouls"] - hs["fouls_drawn"], hs["games"]),
            "away_fouls_net_overall": _safe_mean(as_["fouls"] - as_["fouls_drawn"], as_["games"]),
            "home_fouls_net_at_home": _safe_mean(hs["fouls_home"] - hs["fouls_drawn_home"], hs["games_home"]),
            "away_fouls_net_on_road": _safe_mean(as_["fouls_road"] - as_["fouls_drawn_road"], as_["games_road"]),
            # context
            "home_games_played": hs["games"],
            "away_games_played": as_["games"],
        }

        # rest days
        if hs["last_date"] is not None:
            feat["home_rest_days"] = (date - hs["last_date"]).days - 1
        else:
            feat["home_rest_days"] = np.nan
        if as_["last_date"] is not None:
            feat["away_rest_days"] = (date - as_["last_date"]).days - 1
        else:
            feat["away_rest_days"] = np.nan

        rows.append(feat)

        # ----- update state with THIS game (after features extracted) -----
        # home team (played at home)
        hs["games"] += 1
        hs["wins"] += int(g["home_points"] > g["away_points"])
        hs["pts"] += g["home_points"]
        hs["pts_allowed"] += g["away_points"]
        hs["reb"] += g["home_rebounds"]
        hs["reb_allowed"] += g["away_rebounds"]
        hs["tov"] += g["home_turnovers"]
        hs["takeaways"] += g["away_turnovers"]
        hs["fouls"] += g["home_fouls"]
        hs["fouls_drawn"] += g["away_fouls"]

        hs["games_home"] += 1
        hs["wins_home"] += int(g["home_points"] > g["away_points"])
        hs["pts_home"] += g["home_points"]
        hs["pts_allowed_home"] += g["away_points"]
        hs["reb_home"] += g["home_rebounds"]
        hs["reb_allowed_home"] += g["away_rebounds"]
        hs["tov_home"] += g["home_turnovers"]
        hs["takeaways_home"] += g["away_turnovers"]
        hs["fouls_home"] += g["home_fouls"]
        hs["fouls_drawn_home"] += g["away_fouls"]

        hs["last_date"] = date

        # away team (played on road)
        as_["games"] += 1
        as_["wins"] += int(g["away_points"] > g["home_points"])
        as_["pts"] += g["away_points"]
        as_["pts_allowed"] += g["home_points"]
        as_["reb"] += g["away_rebounds"]
        as_["reb_allowed"] += g["home_rebounds"]
        as_["tov"] += g["away_turnovers"]
        as_["takeaways"] += g["home_turnovers"]
        as_["fouls"] += g["away_fouls"]
        as_["fouls_drawn"] += g["home_fouls"]

        as_["games_road"] += 1
        as_["wins_road"] += int(g["away_points"] > g["home_points"])
        as_["pts_road"] += g["away_points"]
        as_["pts_allowed_road"] += g["home_points"]
        as_["reb_road"] += g["away_rebounds"]
        as_["reb_allowed_road"] += g["home_rebounds"]
        as_["tov_road"] += g["away_turnovers"]
        as_["takeaways_road"] += g["home_turnovers"]
        as_["fouls_road"] += g["away_fouls"]
        as_["fouls_drawn_road"] += g["home_fouls"]

        as_["last_date"] = date

    df = pd.DataFrame(rows)

    # derived differentials / sums (still pre-game)
    df["points_overall_diff"] = df["home_points_overall"] - df["away_points_overall"]
    df["points_allowed_overall_diff"] = df["home_points_allowed_overall"] - df["away_points_allowed_overall"]
    df["rebounds_net_overall_diff"] = df["home_rebounds_net_overall"] - df["away_rebounds_net_overall"]
    df["turnovers_net_overall_diff"] = df["home_turnovers_net_overall"] - df["away_turnovers_net_overall"]
    df["fouls_net_overall_diff"] = df["home_fouls_net_overall"] - df["away_fouls_net_overall"]
    df["rest_days_diff"] = df["home_rest_days"] - df["away_rest_days"]
    df["games_played_diff"] = df["home_games_played"] - df["away_games_played"]
    df["games_played_sum"] = df["home_games_played"] + df["away_games_played"]

    return df


# ---------------------------------------------------------------------------
# Feature regimes
# ---------------------------------------------------------------------------

S0 = [
    "home_win_pct_overall",
    "away_win_pct_overall",
]

S1 = [
    "home_win_pct_overall",
    "away_win_pct_overall",
    "points_overall_diff",
    "points_allowed_overall_diff",
    "rest_days_diff",
    "rebounds_net_overall_diff",
    "turnovers_net_overall_diff",
    "fouls_net_overall_diff",
    "games_played_diff",
    "games_played_sum",
]

S2 = S1 + [
    "home_win_pct_at_home",
    "away_win_pct_on_road",
    "home_points_at_home",
    "away_points_on_road",
    "home_points_allowed_at_home",
    "away_points_allowed_on_road",
    "home_rebounds_net_at_home",
    "away_rebounds_net_on_road",
    "home_turnovers_net_at_home",
    "away_turnovers_net_on_road",
    "home_fouls_net_at_home",
    "away_fouls_net_on_road",
]

S3 = [
    # strength
    "home_win_pct_overall", "away_win_pct_overall",
    "home_win_pct_at_home", "away_win_pct_on_road",
    # offense
    "home_points_overall", "away_points_overall",
    "home_points_at_home", "away_points_on_road",
    # defense
    "home_points_allowed_overall", "away_points_allowed_overall",
    "home_points_allowed_at_home", "away_points_allowed_on_road",
    # rebounding nets
    "home_rebounds_net_overall", "away_rebounds_net_overall",
    "home_rebounds_net_at_home", "away_rebounds_net_on_road",
    # turnover nets
    "home_turnovers_net_overall", "away_turnovers_net_overall",
    "home_turnovers_net_at_home", "away_turnovers_net_on_road",
    # foul nets
    "home_fouls_net_overall", "away_fouls_net_overall",
    "home_fouls_net_at_home", "away_fouls_net_on_road",
    # context
    "rest_days_diff", "games_played_diff", "games_played_sum",
]

FEATURE_SETS: Dict[str, List[str]] = {"S0": S0, "S1": S1, "S2": S2, "S3": S3}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def log5_proba(home_wp: np.ndarray, away_wp: np.ndarray) -> np.ndarray:
    H = np.clip(home_wp, 1e-6, 1 - 1e-6)
    A = np.clip(away_wp, 1e-6, 1 - 1e-6)
    numer = H - H * A
    denom = H + A - 2 * H * A
    p = np.where(np.abs(denom) < 1e-9, 0.5, numer / denom)
    return np.clip(p, 1e-6, 1 - 1e-6)


class Log5Model:
    """Closed-form Log5 win-probability model (no training required)."""

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        p = log5_proba(
            X["home_win_pct_overall"].values,
            X["away_win_pct_overall"].values,
        )
        return np.column_stack([1 - p, p])


def make_logistic():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)),
    ])


def make_rf():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def make_xgb():
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )


MODELS = {
    "log5": Log5Model,
    "logistic": make_logistic,
    "rf": make_rf,
    "xgboost": make_xgb,
}

# Log5 only on S0; others on all regimes
MODEL_SETS = {
    "log5": ["S0"],
    "logistic": ["S0", "S1", "S2", "S3"],
    "rf": ["S0", "S1", "S2", "S3"],
    "xgboost": ["S0", "S1", "S2", "S3"],
}

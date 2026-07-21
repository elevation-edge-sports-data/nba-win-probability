# NBA Home Win Probability

Predict home-team win probability using only information available before tip-off.

## Data
- Single CSV of 1,230 NBA games from the 2025-26 season (`nba-win-probability-data.csv`).
- Columns: game identifiers, date, team abbreviations, and the box-score totals (points, turnovers, fouls, rebounds) for each side.

## Approach
- Features are built with a strict causal loop: for every game the current running team statistics are read first, then the box score is used to update those statistics. No post-tip information enters the feature matrix.
- Four nested feature sets (S0–S3) of increasing size:
  - S0: overall win percentages only (2 features)
  - S1: S0 + point differentials, rest, rebound/turnover/foul nets, games played (10 features)
  - S2: S1 + home/road splits (22 features)
  - S3: fullest set of overall + location-specific strength, offense, and defense (27 features)
- Models: Log5 (closed-form), logistic regression, random forest, XGBoost.
- Split by calendar date only:
  - Train: games before 1 March 2026 (895 games)
  - Validation: March 2026 (239 games) — used for model/feature-set selection and Platt scaling
  - Hold-out: April 2026 (96 games) — final evaluation only

Primary metrics are Brier score and log-loss. Accuracy and AUC are reported for reference.

## Key hold-out results (April 2026)
| Model          | Features | Brier  | Log-loss | Accuracy |
|----------------|----------|--------|----------|----------|
| Log5           | S0       | 0.1697 | 0.5175   | 0.750    |
| Logistic       | S0       | 0.1890 | 0.5668   | 0.823    |
| Random Forest  | S1       | 0.1661 | 0.5126   | 0.792    |
| XGBoost        | S3       | 0.1796 | 0.5363   | 0.708    |

Random Forest on the 10-feature set posts the lowest Brier and log-loss. The two-feature Log5 baseline remains competitive. Expanding beyond S1 does not improve probability quality on the hold-out set.

## Files
- `win_probability.py` — feature construction, feature-set definitions, and model factories
- `nba_home_win_probability.ipynb` — full analysis, diagnostics, and results
- `nba-win-probability-data.csv` — source data
- `requirements.txt` — Python dependencies

## Reproduce
```bash
pip install -r requirements.txt
jupyter notebook nba_home_win_probability.ipynb
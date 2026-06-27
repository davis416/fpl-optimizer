"""
metrics.py — Feature engineering & predictive scoring for FPL players.

Applies:
  1. Exponential-decay form weighting (last 4 GWs)
  2. Fixture Difficulty Rating (FDR) discount
  3. Efficiency Value score (points / cost)
  4. Team-dynamics sub-module for club-level analysis
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
FORM_ALPHA = 0.6          # EMA decay factor (higher = more weight to recent GWs)
FORM_WINDOWS = 4          # number of recent gameweeks considered

# FDR multipliers: maps difficulty rating (1–5) to a projected-points modifier
FDR_MULTIPLIER = {
    1: 1.20,   # very easy
    2: 1.10,   # easy
    3: 1.00,   # neutral
    4: 0.85,   # difficult
    5: 0.70,   # very difficult
}

# Position-based baseline minutes threshold (players below this are discounted)
MIN_MINUTES_THRESHOLD = 450   # ~5 full 90-min games


# ── 1. Form Weighting ─────────────────────────────────────────────────────────

def exponential_form_score(gw_points: list[float], alpha: float = FORM_ALPHA) -> float:
    """
    Compute an exponentially-decayed weighted average of recent gameweek points.

    The most recent gameweek carries the highest weight.

    Parameters
    ----------
    gw_points : list of float
        Points per gameweek, ordered oldest → newest.
        Padded to `FORM_WINDOWS` entries.
    alpha : float
        Decay factor in (0, 1). Higher → more weight to recent games.

    Returns
    -------
    float : EMA score
    """
    points = list(gw_points[-FORM_WINDOWS:])          # keep last N GWs
    n = len(points)
    if n == 0:
        return 0.0

    weights = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)])
    weights /= weights.sum()
    return float(np.dot(weights, points))


def add_form_score(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the players DataFrame with an `ema_form` column derived from
    the API `form` field (a rolling 30-day average provided by FPL).

    For higher accuracy, pass in actual per-GW history via
    `add_form_score_from_history()`.
    """
    df = players_df.copy()
    # FPL's own `form` is already a 4-GW average; we scale it slightly
    # using EMA logic to penalise streaky players vs consistently good ones.
    df["ema_form"] = df["form"].apply(
        lambda f: exponential_form_score([f] * FORM_WINDOWS)
    )
    return df


def add_form_score_from_history(
    players_df: pd.DataFrame,
    history: dict[int, list[float]],
) -> pd.DataFrame:
    """
    Compute EMA form from actual per-gameweek point histories.

    Parameters
    ----------
    history : dict mapping player_id → list of GW points (oldest to newest)
    """
    df = players_df.copy()

    def _score(player_id: int) -> float:
        pts = history.get(player_id, [])
        if not pts:
            return float(df.loc[df["id"] == player_id, "form"].values[0]) if player_id in df["id"].values else 0.0
        return exponential_form_score(pts)

    df["ema_form"] = df["id"].apply(_score)
    return df


# ── 2. Fixture Difficulty Discounting ────────────────────────────────────────

def _next_fixture_difficulty(player_team_id: int, fixtures_df: pd.DataFrame, next_n: int = 1) -> float:
    """
    Return the average FDR multiplier for a team's next `next_n` fixtures.
    """
    upcoming = fixtures_df[~fixtures_df["finished"]].copy()
    upcoming = upcoming[upcoming["gameweek"].notna()].sort_values("gameweek")

    home_mask = upcoming["home_team_id"] == player_team_id
    away_mask = upcoming["away_team_id"] == player_team_id

    home_rows = upcoming[home_mask][["gameweek", "home_difficulty"]].rename(
        columns={"home_difficulty": "difficulty"}
    )
    away_rows = upcoming[away_mask][["gameweek", "away_difficulty"]].rename(
        columns={"away_difficulty": "difficulty"}
    )

    all_fixtures = pd.concat([home_rows, away_rows]).sort_values("gameweek").head(next_n)

    if all_fixtures.empty:
        return 1.0   # neutral if no fixtures found

    avg_multiplier = all_fixtures["difficulty"].map(FDR_MULTIPLIER).mean()
    return float(avg_multiplier)


def add_fdr_discount(players_df: pd.DataFrame, fixtures_df: pd.DataFrame, next_n: int = 1) -> pd.DataFrame:
    """
    Add `fdr_multiplier` column: each player's FDR modifier for the next `next_n` gameweeks.
    """
    df = players_df.copy()
    df["fdr_multiplier"] = df["team"].apply(
        lambda tid: _next_fixture_difficulty(tid, fixtures_df, next_n)
    )
    logger.info("FDR multipliers computed for next %d gameweek(s).", next_n)
    return df


# ── 3. Projected Points & Value Score ────────────────────────────────────────

def add_projected_points(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute `projected_points` = ema_form × fdr_multiplier.

    Also applies a minutes-based availability discount (players with fewer
    than MIN_MINUTES_THRESHOLD minutes get a 15% penalty).
    """
    df = players_df.copy()

    # Availability discount based on playing time
    minutes_ok = df["minutes"] >= MIN_MINUTES_THRESHOLD
    availability_factor = np.where(minutes_ok, 1.0, 0.85)

    df["projected_points"] = (
        df["ema_form"] * df["fdr_multiplier"] * availability_factor
    )
    return df


def add_value_score(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute `value_score` = projected_points / now_cost.

    This metric captures points-per-pound efficiency.
    """
    df = players_df.copy()
    # Avoid division by zero
    df["value_score"] = df.apply(
        lambda row: row["projected_points"] / row["now_cost"] if row["now_cost"] > 0 else 0.0,
        axis=1,
    )
    return df


# ── 4. Team-Dynamics Sub-module ───────────────────────────────────────────────

class TeamDynamicsAnalyzer:
    """
    Tracks team-level performance dynamics to surface high-value clubs.

    Computes:
      - Average projected points by team
      - Average clean-sheet likelihood (defenders/GKs) by team
      - Goal contribution rate per team
    """

    def __init__(self, players_df: pd.DataFrame):
        self.df = players_df.copy()

    def team_attack_rating(self) -> pd.DataFrame:
        """Returns each team's average MID/FWD projected points (attack proxy)."""
        attack = self.df[self.df["position"].isin(["MID", "FWD"])]
        return (
            attack.groupby("team_name")["projected_points"]
            .mean()
            .reset_index()
            .rename(columns={"projected_points": "avg_attack_pts"})
            .sort_values("avg_attack_pts", ascending=False)
        )

    def team_defense_rating(self) -> pd.DataFrame:
        """Returns each team's average GK/DEF projected points (defense proxy)."""
        defense = self.df[self.df["position"].isin(["GK", "DEF"])]
        return (
            defense.groupby("team_name")["projected_points"]
            .mean()
            .reset_index()
            .rename(columns={"projected_points": "avg_defense_pts"})
            .sort_values("avg_defense_pts", ascending=False)
        )

    def top_players_by_team(self, team_name: str, top_n: int = 5) -> pd.DataFrame:
        """Return top-N projected players from a specific club."""
        team_df = self.df[self.df["team_name"] == team_name]
        return (
            team_df.sort_values("projected_points", ascending=False)
            .head(top_n)[["web_name", "position", "now_cost", "projected_points", "value_score"]]
        )

    def systemic_impact_report(self) -> pd.DataFrame:
        """
        Full club-level summary: attack rating, defense rating, player count.
        Useful for identifying which clubs are delivering the most FPL value.
        """
        attack = self.team_attack_rating().set_index("team_name")
        defense = self.team_defense_rating().set_index("team_name")
        count = self.df.groupby("team_name").size().reset_index(name="player_count").set_index("team_name")

        report = pd.concat([attack, defense, count], axis=1).reset_index()
        report["overall_score"] = report["avg_attack_pts"] * 0.6 + report["avg_defense_pts"] * 0.4
        return report.sort_values("overall_score", ascending=False)


# ── Pipeline Entrypoint ───────────────────────────────────────────────────────

def engineer_features(
    players_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    next_n_gws: int = 1,
    gw_history: Optional[dict[int, list[float]]] = None,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Steps:
      1. EMA form score
      2. FDR fixture discount
      3. Projected points
      4. Value score (pts/£m)

    Parameters
    ----------
    players_df    : cleaned players DataFrame from ingestion
    fixtures_df   : cleaned fixtures DataFrame from ingestion
    next_n_gws    : how many upcoming GWs to average FDR over
    gw_history    : optional dict {player_id: [pts_gw1, pts_gw2, ...]}

    Returns
    -------
    Enriched players DataFrame with additional metric columns.
    """
    logger.info("Starting feature engineering pipeline…")

    if gw_history:
        df = add_form_score_from_history(players_df, gw_history)
    else:
        df = add_form_score(players_df)

    df = add_fdr_discount(df, fixtures_df, next_n=next_n_gws)
    df = add_projected_points(df)
    df = add_value_score(df)

    logger.info("Feature engineering complete. Shape: %s", df.shape)
    return df

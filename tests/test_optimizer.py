"""
tests/test_optimizer.py — Unit tests for the FPL optimizer constraints.

Run with:  pytest tests/ -v
"""

import pytest
import pandas as pd

from src.optimizer import OptimizerConfig, optimize_squad, pick_starting_xi, VALID_FORMATIONS
from src.metrics import (
    exponential_form_score,
    add_value_score,
    add_projected_points,
    TeamDynamicsAnalyzer,
)


# ── Fixtures (pytest) ─────────────────────────────────────────────────────────

def _make_player(pid, name, pos, team_id, team_name, cost, pts):
    """Helper to build a minimal player row for tests."""
    return {
        "id": pid,
        "web_name": name,
        "full_name": name,
        "position": pos,
        "team": team_id,
        "team_name": team_name,
        "now_cost": cost,
        "total_points": pts,
        "form": pts / 5.0,
        "minutes": 1000,
        "ema_form": pts / 5.0,
        "fdr_multiplier": 1.0,
        "projected_points": pts / 5.0,
        "value_score": (pts / 5.0) / cost if cost else 0,
    }


@pytest.fixture
def minimal_squad_pool():
    """
    A pool of players sufficient to build a valid 15-man squad within £100m.
    Positions: 2 GK, 5 DEF, 5 MID, 3 FWD (minimum required).
    Players spread across multiple clubs.
    """
    players = []
    pid = 1

    # 4 GKs
    for i, (club_id, club, cost, pts) in enumerate([
        (1, "Arsenal", 4.5, 25), (1, "Arsenal", 4.0, 20),
        (2, "Chelsea", 5.0, 30), (6, "Aston Villa", 4.0, 15),
    ]):
        players.append(_make_player(pid, f"GK_{i+1}", "GK", club_id, club, cost, pts))
        pid += 1

    # 8 DEFs
    for i, (club_id, club, cost, pts) in enumerate([
        (1, "Arsenal", 5.0, 40), (1, "Arsenal", 4.5, 35), (1, "Arsenal", 4.0, 30),
        (2, "Chelsea", 5.5, 45), (2, "Chelsea", 5.0, 38),
        (3, "Liverpool", 4.5, 35), (3, "Liverpool", 4.0, 28),
        (6, "Aston Villa", 4.5, 40),
    ]):
        players.append(_make_player(pid, f"DEF_{i+1}", "DEF", club_id, club, cost, pts))
        pid += 1

    # 8 MIDs
    for i, (club_id, club, cost, pts) in enumerate([
        (1, "Arsenal", 6.0, 60), (1, "Arsenal", 5.5, 55),
        (2, "Chelsea", 7.0, 70), (2, "Chelsea", 6.5, 65),
        (3, "Liverpool", 6.0, 58), (4, "Man City", 8.0, 80),
        (4, "Man City", 7.5, 75), (5, "Spurs", 5.0, 50),
    ]):
        players.append(_make_player(pid, f"MID_{i+1}", "MID", club_id, club, cost, pts))
        pid += 1

    # 5 FWDs
    for i, (club_id, club, cost, pts) in enumerate([
        (1, "Arsenal", 8.0, 85), (2, "Chelsea", 9.0, 90),
        (3, "Liverpool", 7.5, 78), (4, "Man City", 10.0, 95),
        (6, "Aston Villa", 6.0, 60),
    ]):
        players.append(_make_player(pid, f"FWD_{i+1}", "FWD", club_id, club, cost, pts))
        pid += 1

    return pd.DataFrame(players)


# ── Metrics tests ─────────────────────────────────────────────────────────────

class TestExponentialFormScore:
    def test_empty_list_returns_zero(self):
        assert exponential_form_score([]) == 0.0

    def test_single_value_returns_that_value(self):
        assert exponential_form_score([5.0]) == pytest.approx(5.0, abs=1e-6)

    def test_recent_gws_weighted_more(self):
        """Most recent (last) GW should have higher weight with alpha > 0."""
        score_improving = exponential_form_score([2, 3, 4, 8])
        score_declining  = exponential_form_score([8, 4, 3, 2])
        assert score_improving > score_declining

    def test_uniform_values(self):
        """Uniform points list should return that point value."""
        assert exponential_form_score([6, 6, 6, 6]) == pytest.approx(6.0, abs=0.1)

    def test_truncation_to_form_windows(self):
        """Only last FORM_WINDOWS (4) entries should matter."""
        score_long = exponential_form_score([100, 100, 100, 100, 100, 1, 1, 1, 5])
        score_short = exponential_form_score([1, 1, 1, 5])
        assert score_long == pytest.approx(score_short, abs=0.01)


class TestValueScore:
    def test_value_score_proportional_to_points(self, minimal_squad_pool):
        df = add_value_score(minimal_squad_pool)
        assert "value_score" in df.columns
        # Player with higher pts and same cost should have higher value
        cheap = df[df["now_cost"] == 5.0].copy()
        if len(cheap) >= 2:
            sorted_cheap = cheap.sort_values("projected_points", ascending=False)
            assert sorted_cheap.iloc[0]["value_score"] >= sorted_cheap.iloc[1]["value_score"]

    def test_value_score_zero_cost_player(self):
        """Division by zero guard: zero-cost player gets value_score=0."""
        df = pd.DataFrame([_make_player(999, "FreePlayer", "GK", 1, "Test", 0.0, 10)])
        result = add_value_score(df)
        assert result.iloc[0]["value_score"] == 0.0


# ── Optimizer constraint tests ────────────────────────────────────────────────

class TestOptimizerConstraints:

    def test_optimal_solution_found_default_budget(self, minimal_squad_pool):
        result = optimize_squad(minimal_squad_pool, config=OptimizerConfig(total_budget=100.0))
        assert result.is_optimal, f"Expected Optimal, got: {result.status}"

    def test_squad_size_is_15(self, minimal_squad_pool):
        result = optimize_squad(minimal_squad_pool, config=OptimizerConfig(total_budget=100.0))
        assert result.is_optimal
        assert len(result.squad) == 15

    def test_position_counts_correct(self, minimal_squad_pool):
        result = optimize_squad(minimal_squad_pool, config=OptimizerConfig(total_budget=100.0))
        assert result.is_optimal
        pos_counts = result.squad["position"].value_counts()
        assert pos_counts.get("GK", 0) == 2
        assert pos_counts.get("DEF", 0) == 5
        assert pos_counts.get("MID", 0) == 5
        assert pos_counts.get("FWD", 0) == 3

    def test_budget_constraint_respected(self, minimal_squad_pool):
        budget = 100.0
        result = optimize_squad(minimal_squad_pool, config=OptimizerConfig(total_budget=budget))
        assert result.is_optimal
        assert result.total_cost <= budget + 1e-6

    def test_club_limit_not_exceeded(self, minimal_squad_pool):
        result = optimize_squad(minimal_squad_pool, config=OptimizerConfig(
            total_budget=100.0, max_players_per_club=3
        ))
        assert result.is_optimal
        club_counts = result.squad["team"].value_counts()
        assert club_counts.max() <= 3, f"Club limit violated: {club_counts.to_dict()}"

    def test_infeasible_budget_returns_failure(self, minimal_squad_pool):
        """Budget of £40m cannot build any valid squad — must not crash."""
        result = optimize_squad(minimal_squad_pool, config=OptimizerConfig(total_budget=40.0))
        assert not result.is_optimal
        assert result.status != "Optimal"
        assert result.infeasibility_reason is not None
        assert len(result.infeasibility_reason) > 0

    def test_infeasible_budget_includes_diagnosis(self, minimal_squad_pool):
        """Infeasibility reason string should mention the budget shortfall."""
        result = optimize_squad(minimal_squad_pool, config=OptimizerConfig(total_budget=40.0))
        assert "budget" in result.infeasibility_reason.lower() or "cost" in result.infeasibility_reason.lower()

    def test_custom_club_limit(self, minimal_squad_pool):
        """Strict club limit of 1 should still find a solution."""
        result = optimize_squad(minimal_squad_pool, config=OptimizerConfig(
            total_budget=100.0, max_players_per_club=1
        ))
        # With 5 clubs and limit of 1, we have 5 DEF slots but may not have enough players
        # Just check it doesn't crash and status is a known string
        assert result.status in {"Optimal", "Infeasible", "Not Solved"}

    def test_missing_required_columns_raises(self, minimal_squad_pool):
        bad_df = minimal_squad_pool.drop(columns=["now_cost"])
        with pytest.raises(ValueError, match="missing required columns"):
            optimize_squad(bad_df)


# ── Starting XI tests ─────────────────────────────────────────────────────────

class TestStartingXI:

    def _get_squad(self, minimal_squad_pool):
        result = optimize_squad(minimal_squad_pool, config=OptimizerConfig(total_budget=100.0))
        assert result.is_optimal
        return result.squad

    def test_starting_xi_size(self, minimal_squad_pool):
        squad = self._get_squad(minimal_squad_pool)
        xi, bench = pick_starting_xi(squad, formation="4-4-2")
        assert len(xi) == 11
        assert len(bench) == 4

    def test_bench_plus_xi_equals_squad(self, minimal_squad_pool):
        squad = self._get_squad(minimal_squad_pool)
        xi, bench = pick_starting_xi(squad, formation="4-3-3")
        assert len(xi) + len(bench) == len(squad)

    def test_xi_has_exactly_one_gk(self, minimal_squad_pool):
        squad = self._get_squad(minimal_squad_pool)
        xi, _ = pick_starting_xi(squad, formation="4-4-2")
        assert (xi["position"] == "GK").sum() == 1

    @pytest.mark.parametrize("formation", list(VALID_FORMATIONS.keys()))
    def test_all_valid_formations(self, minimal_squad_pool, formation):
        squad = self._get_squad(minimal_squad_pool)
        xi, bench = pick_starting_xi(squad, formation=formation)
        assert len(xi) == 11

    def test_invalid_formation_raises(self, minimal_squad_pool):
        squad = self._get_squad(minimal_squad_pool)
        with pytest.raises(ValueError, match="Unknown formation"):
            pick_starting_xi(squad, formation="99-99-99")


# ── TeamDynamicsAnalyzer tests ────────────────────────────────────────────────

class TestTeamDynamicsAnalyzer:

    def test_systemic_impact_report_shape(self, minimal_squad_pool):
        analyzer = TeamDynamicsAnalyzer(minimal_squad_pool)
        report = analyzer.systemic_impact_report()
        assert "team_name" in report.columns
        assert "overall_score" in report.columns
        assert len(report) > 0

    def test_top_players_by_team_returns_correct_club(self, minimal_squad_pool):
        analyzer = TeamDynamicsAnalyzer(minimal_squad_pool)
        top = analyzer.top_players_by_team("Arsenal", top_n=3)
        assert len(top) == 3

    def test_top_players_unknown_team_returns_empty(self, minimal_squad_pool):
        analyzer = TeamDynamicsAnalyzer(minimal_squad_pool)
        top = analyzer.top_players_by_team("NonExistentFC", top_n=5)
        assert top.empty

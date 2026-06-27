"""
optimizer.py — Integer Linear Programming squad optimizer for FPL.

Formulates the FPL squad selection as a Bounded Knapsack / ILP problem using PuLP.

Objective:
    Maximize  Σ (projected_points_i × x_i)

Subject to:
    Budget    : Σ (cost_i × x_i) ≤ total_budget
    GK count  : Σ x_i [position=GK]  = 2
    DEF count : Σ x_i [position=DEF] = 5
    MID count : Σ x_i [position=MID] = 5
    FWD count : Σ x_i [position=FWD] = 3
    Club limit: Σ x_i [team=t] ≤ 3  ∀ t
    Binary    : x_i ∈ {0, 1}
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import pulp
except ImportError as exc:
    raise ImportError(
        "PuLP is required for the optimizer. Install it with: pip install pulp"
    ) from exc


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class OptimizerConfig:
    """Holds all tunable parameters for the ILP solver."""
    total_budget: float = 100.0          # £m
    gk_count: int = 2
    def_count: int = 5
    mid_count: int = 5
    fwd_count: int = 3
    max_players_per_club: int = 3
    objective_weight: str = "projected_points"   # column to maximise
    solver_time_limit_secs: int = 30


SQUAD_SIZE = 15   # 2 + 5 + 5 + 3


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    """Holds the output of a solved squad optimization."""
    squad: pd.DataFrame
    total_cost: float
    total_projected_points: float
    status: str
    config: OptimizerConfig
    infeasibility_reason: Optional[str] = None

    @property
    def is_optimal(self) -> bool:
        return self.status == "Optimal"

    def summary(self) -> str:
        lines = [
            f"  Status             : {self.status}",
            f"  Total Cost         : £{self.total_cost:.1f}m  (budget £{self.config.total_budget:.1f}m)",
            f"  Projected Points   : {self.total_projected_points:.2f}",
            f"  Squad Size         : {len(self.squad)}",
        ]
        if self.infeasibility_reason:
            lines.append(f"  ⚠  Reason           : {self.infeasibility_reason}")
        return "\n".join(lines)


# ── Core ILP Solver ───────────────────────────────────────────────────────────

def _infer_infeasibility_reason(players_df: pd.DataFrame, config: OptimizerConfig) -> str:
    """Attempt to diagnose why the problem is infeasible."""
    reasons = []

    min_budget = (
        players_df[players_df["position"] == "GK"]["now_cost"].nsmallest(config.gk_count).sum()
        + players_df[players_df["position"] == "DEF"]["now_cost"].nsmallest(config.def_count).sum()
        + players_df[players_df["position"] == "MID"]["now_cost"].nsmallest(config.mid_count).sum()
        + players_df[players_df["position"] == "FWD"]["now_cost"].nsmallest(config.fwd_count).sum()
    )
    if min_budget > config.total_budget:
        reasons.append(
            f"Minimum squad cost is £{min_budget:.1f}m but budget is only £{config.total_budget:.1f}m."
        )

    for pos, required in [
        ("GK", config.gk_count), ("DEF", config.def_count),
        ("MID", config.mid_count), ("FWD", config.fwd_count),
    ]:
        available = len(players_df[players_df["position"] == pos])
        if available < required:
            reasons.append(
                f"Only {available} available {pos}s but need {required}."
            )

    return " | ".join(reasons) if reasons else "Unknown infeasibility — check constraints."


def optimize_squad(
    players_df: pd.DataFrame,
    config: Optional[OptimizerConfig] = None,
) -> OptimizationResult:
    """
    Solve the FPL squad selection ILP.

    Parameters
    ----------
    players_df : DataFrame
        Must contain columns: id, web_name, position, team, team_name,
        now_cost, projected_points (from metrics.engineer_features).
    config : OptimizerConfig, optional
        Solver configuration. Uses defaults if not provided.

    Returns
    -------
    OptimizationResult
    """
    if config is None:
        config = OptimizerConfig()

    required_cols = {"id", "web_name", "position", "team", "team_name", "now_cost", config.objective_weight}
    missing = required_cols - set(players_df.columns)
    if missing:
        raise ValueError(f"players_df is missing required columns: {missing}")

    df = players_df.copy().reset_index(drop=True)
    n = len(df)

    logger.info("Setting up ILP with %d players, budget £%.1fm…", n, config.total_budget)

    # ── Decision variables ────────────────────────────────────────────────────
    prob = pulp.LpProblem("FPL_Squad_Optimizer", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    # ── Objective ─────────────────────────────────────────────────────────────
    obj_values = df[config.objective_weight].tolist()
    prob += pulp.lpSum(obj_values[i] * x[i] for i in range(n)), "Maximize_Projected_Points"

    # ── Budget constraint ─────────────────────────────────────────────────────
    costs = df["now_cost"].tolist()
    prob += (
        pulp.lpSum(costs[i] * x[i] for i in range(n)) <= config.total_budget,
        "Budget_Constraint",
    )

    # ── Position constraints ──────────────────────────────────────────────────
    for pos, required in [
        ("GK", config.gk_count),
        ("DEF", config.def_count),
        ("MID", config.mid_count),
        ("FWD", config.fwd_count),
    ]:
        indices = df.index[df["position"] == pos].tolist()
        prob += (
            pulp.lpSum(x[i] for i in indices) == required,
            f"Position_{pos}_Count",
        )

    # ── Club limit constraint ─────────────────────────────────────────────────
    for team_id in df["team"].unique():
        indices = df.index[df["team"] == team_id].tolist()
        team_name = df.loc[df["team"] == team_id, "team_name"].iloc[0]
        prob += (
            pulp.lpSum(x[i] for i in indices) <= config.max_players_per_club,
            f"Club_Limit_{team_name.replace(' ', '_')}",
        )

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(
        msg=0,
        timeLimit=config.solver_time_limit_secs,
    )
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    logger.info("Solver status: %s", status)

    # ── Extract results ───────────────────────────────────────────────────────
    if status == "Optimal":
        selected_indices = [i for i in range(n) if pulp.value(x[i]) == 1]
        squad_df = df.iloc[selected_indices].copy()
        squad_df = squad_df.sort_values(["position", "projected_points"], ascending=[True, False])

        total_cost = squad_df["now_cost"].sum()
        total_pts = squad_df[config.objective_weight].sum()

        return OptimizationResult(
            squad=squad_df,
            total_cost=total_cost,
            total_projected_points=total_pts,
            status="Optimal",
            config=config,
        )
    else:
        reason = _infer_infeasibility_reason(df, config)
        logger.error("Optimization failed: %s — %s", status, reason)
        return OptimizationResult(
            squad=pd.DataFrame(),
            total_cost=0.0,
            total_projected_points=0.0,
            status=status,
            config=config,
            infeasibility_reason=reason,
        )


# ── Starting XI picker (bonus) ────────────────────────────────────────────────

VALID_FORMATIONS = {
    "3-4-3": (3, 4, 3),
    "3-5-2": (3, 5, 2),
    "4-3-3": (4, 3, 3),
    "4-4-2": (4, 4, 2),
    "4-5-1": (4, 5, 1),
    "5-3-2": (5, 3, 2),
    "5-4-1": (5, 4, 1),
}


def pick_starting_xi(squad_df: pd.DataFrame, formation: str = "4-4-2") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select the best-projected starting XI from the 15-man squad.

    Parameters
    ----------
    squad_df  : full 15-player squad
    formation : string like '4-4-2', '4-3-3', etc.

    Returns
    -------
    (starting_xi_df, bench_df)
    """
    if formation not in VALID_FORMATIONS:
        raise ValueError(f"Unknown formation '{formation}'. Valid: {list(VALID_FORMATIONS)}")

    def_n, mid_n, fwd_n = VALID_FORMATIONS[formation]

    # Always 1 GK in starting XI
    gk = squad_df[squad_df["position"] == "GK"].sort_values("projected_points", ascending=False).head(1)
    defs = squad_df[squad_df["position"] == "DEF"].sort_values("projected_points", ascending=False).head(def_n)
    mids = squad_df[squad_df["position"] == "MID"].sort_values("projected_points", ascending=False).head(mid_n)
    fwds = squad_df[squad_df["position"] == "FWD"].sort_values("projected_points", ascending=False).head(fwd_n)

    starting_xi = pd.concat([gk, defs, mids, fwds])
    bench = squad_df[~squad_df.index.isin(starting_xi.index)]

    return starting_xi, bench

"""
main.py — CLI entry point for the FPL Squad Optimizer.

Usage examples:
    python -m src.main
    python -m src.main --budget 95.0 --gws 2 --formation 4-3-3
    python -m src.main --team "Tottenham Hotspur" --refresh
    python -m src.main --budget 40.0   # will show infeasibility diagnosis
"""

import argparse
import logging
import sys
from pathlib import Path

# ── Colorama for styled terminal output ───────────────────────────────────────
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    class Fore:
        GREEN = CYAN = YELLOW = RED = MAGENTA = WHITE = ""
    class Style:
        BRIGHT = RESET_ALL = DIM = ""

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

from src.ingestion import load_all_data
from src.metrics import engineer_features, TeamDynamicsAnalyzer
from src.optimizer import OptimizerConfig, optimize_squad, pick_starting_xi

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fpl_optimizer")


# ── Display helpers ───────────────────────────────────────────────────────────

POSITION_COLOR = {
    "GK":  Fore.YELLOW,
    "DEF": Fore.GREEN,
    "MID": Fore.CYAN,
    "FWD": Fore.RED,
}


def _color_position(pos: str) -> str:
    return POSITION_COLOR.get(pos, "") + pos + Style.RESET_ALL if HAS_COLOR else pos


def _print_banner():
    banner = r"""
  ███████╗██████╗ ██╗      ██████╗ ██████╗ ████████╗
  ██╔════╝██╔══██╗██║     ██╔═══██╗██╔══██╗╚══██╔══╝
  █████╗  ██████╔╝██║     ██║   ██║██████╔╝   ██║
  ██╔══╝  ██╔═══╝ ██║     ██║   ██║██╔═══╝    ██║
  ██║     ██║     ███████╗╚██████╔╝██║         ██║
  ╚═╝     ╚═╝     ╚══════╝ ╚═════╝ ╚═╝         ╚═╝
        Fantasy Premier League Squad Optimizer
    """
    print(Fore.CYAN + Style.BRIGHT + banner + Style.RESET_ALL)


def _print_squad_table(squad_df, title: str = "OPTIMAL SQUAD"):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'━'*60}")
    print(f"  {title}")
    print(f"{'━'*60}{Style.RESET_ALL}")

    rows = []
    for _, row in squad_df.iterrows():
        rows.append([
            _color_position(row["position"]),
            row.get("web_name", row.get("full_name", "?")),
            row.get("team_name", ""),
            f"£{row['now_cost']:.1f}m",
            f"{row.get('projected_points', 0):.2f}",
            f"{row.get('value_score', 0):.3f}",
            f"{row.get('fdr_multiplier', 1.0):.2f}",
        ])

    headers = ["Pos", "Player", "Club", "Cost", "Proj. Pts", "Value", "FDR×"]
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        print("  " + "  ".join(f"{h:<12}" for h in headers))
        for r in rows:
            print("  " + "  ".join(f"{str(v):<12}" for v in r))


def _print_team_report(analyzer: TeamDynamicsAnalyzer, team_name: str):
    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}━━━  Team Dynamics: {team_name}  ━━━{Style.RESET_ALL}")
    top = analyzer.top_players_by_team(team_name, top_n=10)
    if top.empty:
        print(f"  {Fore.YELLOW}No players found for '{team_name}'.{Style.RESET_ALL}")
        return
    rows = [
        [_color_position(r["position"]), r["web_name"], f"£{r['now_cost']:.1f}m",
         f"{r['projected_points']:.2f}", f"{r['value_score']:.3f}"]
        for _, r in top.iterrows()
    ]
    headers = ["Pos", "Player", "Cost", "Proj. Pts", "Value"]
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        print("  " + "  ".join(f"{h:<12}" for h in headers))
        for r in rows:
            print("  " + "  ".join(f"{str(v):<12}" for v in r))

    print(f"\n{Fore.MAGENTA}  Club Systemic Impact Report:{Style.RESET_ALL}")
    report = analyzer.systemic_impact_report()
    report_rows = [
        [r["team_name"], f"{r.get('avg_attack_pts', 0):.2f}",
         f"{r.get('avg_defense_pts', 0):.2f}", f"{r.get('overall_score', 0):.2f}",
         int(r.get("player_count", 0))]
        for _, r in report.head(10).iterrows()
    ]
    rep_headers = ["Club", "Avg Atk Pts", "Avg Def Pts", "Overall Score", "Players"]
    if HAS_TABULATE:
        print(tabulate(report_rows, headers=rep_headers, tablefmt="rounded_outline"))


# ── CLI argument parsing ──────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fpl-optimizer",
        description="FPL Squad Optimizer — Integer Linear Programming squad selector.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--budget", type=float, default=100.0,
                   help="Total squad budget in £m (default: 100.0)")
    p.add_argument("--gws", type=int, default=1,
                   help="Number of upcoming GWs to average FDR over (default: 1)")
    p.add_argument("--formation", type=str, default="4-4-2",
                   help="Starting XI formation, e.g. 4-3-3 (default: 4-4-2)")
    p.add_argument("--objective", type=str, default="projected_points",
                   choices=["projected_points", "value_score", "total_points"],
                   help="Column to maximise in the solver (default: projected_points)")
    p.add_argument("--team", type=str, default=None,
                   help="Show team-dynamics report for this club name (e.g. 'Tottenham Hotspur')")
    p.add_argument("--refresh", action="store_true",
                   help="Force-refresh cached API data")
    p.add_argument("--max-per-club", type=int, default=3,
                   help="Max players allowed from one club (default: 3)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress the ASCII banner")
    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.quiet:
        _print_banner()

    # ── Phase 1: Data ingestion ───────────────────────────────────────────────
    print(f"{Fore.CYAN}▶ Fetching FPL data…{Style.RESET_ALL}")
    try:
        players_df, fixtures_df = load_all_data(force_refresh=args.refresh)
    except Exception as exc:
        print(f"{Fore.RED}✗ Failed to load data: {exc}{Style.RESET_ALL}")
        sys.exit(1)

    print(f"{Fore.GREEN}✓ Loaded {len(players_df)} players, {len(fixtures_df)} fixtures.{Style.RESET_ALL}")

    # ── Phase 2: Feature engineering ─────────────────────────────────────────
    print(f"{Fore.CYAN}▶ Engineering features…{Style.RESET_ALL}")
    enriched_df = engineer_features(players_df, fixtures_df, next_n_gws=args.gws)
    print(f"{Fore.GREEN}✓ Features computed.{Style.RESET_ALL}")

    # ── Optional: Team dynamics report ───────────────────────────────────────
    if args.team:
        analyzer = TeamDynamicsAnalyzer(enriched_df)
        _print_team_report(analyzer, args.team)

    # ── Phase 3: Optimization ─────────────────────────────────────────────────
    print(f"\n{Fore.CYAN}▶ Running ILP optimizer (budget £{args.budget:.1f}m, objective: {args.objective})…{Style.RESET_ALL}")

    config = OptimizerConfig(
        total_budget=args.budget,
        max_players_per_club=args.max_per_club,
        objective_weight=args.objective,
    )

    result = optimize_squad(enriched_df, config=config)

    # ── Phase 4: Presentation ─────────────────────────────────────────────────
    print(f"\n{Fore.WHITE}{Style.BRIGHT}  Optimization Result:{Style.RESET_ALL}")
    print(result.summary())

    if result.is_optimal:
        _print_squad_table(result.squad, title="✦  OPTIMAL 15-MAN SQUAD")

        try:
            starting_xi, bench = pick_starting_xi(result.squad, formation=args.formation)
            _print_squad_table(starting_xi, title=f"✦  STARTING XI  [{args.formation}]")
            _print_squad_table(bench, title="✦  BENCH")
        except ValueError as exc:
            print(f"{Fore.YELLOW}⚠  Formation error: {exc}{Style.RESET_ALL}")

        print(f"\n{Fore.GREEN}{Style.BRIGHT}  ✓ Squad selected successfully!{Style.RESET_ALL}\n")
    else:
        print(f"\n{Fore.RED}  ✗ Could not find a valid squad. {result.infeasibility_reason}{Style.RESET_ALL}\n")
        sys.exit(2)


if __name__ == "__main__":
    main()

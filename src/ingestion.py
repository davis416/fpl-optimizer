"""
ingestion.py — FPL API client with local JSON caching.

Fetches player, team, and fixture data from the official FPL endpoints.
Cleans and returns structured pandas DataFrames ready for feature engineering.
"""

import json
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
BASE_URL = "https://fantasy.premierleague.com/api"
BOOTSTRAP_URL = f"{BASE_URL}/bootstrap-static/"
FIXTURES_URL = f"{BASE_URL}/fixtures/"
PLAYER_HISTORY_URL = f"{BASE_URL}/element-summary/{{player_id}}/"

CACHE_DIR = Path(__file__).parent.parent / "data"
CACHE_TTL_HOURS = 6          # refresh cache after 6 hours

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Players with these status codes are excluded from the pool
UNAVAILABLE_STATUSES = {"u", "i"}   # 'u' = unavailable, 'i' = injured/suspended


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def _fetch_json(url: str) -> dict | list:
    """Fetch JSON from a URL with a browser-like User-Agent."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as exc:
        logger.error("HTTP error fetching %s: %s", url, exc)
        raise
    except requests.RequestException as exc:
        logger.error("Network error fetching %s: %s", url, exc)
        raise


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.json"


def _is_cache_fresh(path: Path, ttl_hours: int = CACHE_TTL_HOURS) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=ttl_hours)


def _load_cached(name: str) -> Optional[dict | list]:
    path = _cache_path(name)
    if _is_cache_fresh(path):
        logger.info("Loading '%s' from cache (%s).", name, path)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_cache(name: str, data: dict | list) -> None:
    path = _cache_path(name)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Cached '%s' → %s", name, path)


# ── Public fetch functions ────────────────────────────────────────────────────

def fetch_bootstrap(force_refresh: bool = False) -> dict:
    """Return raw bootstrap-static JSON (cached unless forced)."""
    name = "bootstrap_static"
    if not force_refresh:
        cached = _load_cached(name)
        if cached:
            return cached
    logger.info("Fetching bootstrap data from FPL API…")
    data = _fetch_json(BOOTSTRAP_URL)
    _save_cache(name, data)
    return data


def fetch_fixtures(force_refresh: bool = False) -> list:
    """Return raw fixtures JSON (cached unless forced)."""
    name = "fixtures"
    if not force_refresh:
        cached = _load_cached(name)
        if cached:
            return cached
    logger.info("Fetching fixtures data from FPL API…")
    data = _fetch_json(FIXTURES_URL)
    _save_cache(name, data)
    return data


def fetch_player_history(player_id: int) -> dict:
    """Return gameweek-by-gameweek history for a single player (always live)."""
    url = PLAYER_HISTORY_URL.format(player_id=player_id)
    return _fetch_json(url)


# ── Data Cleaning ─────────────────────────────────────────────────────────────

def build_players_df(bootstrap: dict) -> pd.DataFrame:
    """
    Construct a clean players DataFrame from the bootstrap payload.

    Columns of interest:
        id, web_name, full_name, team, team_name, position,
        now_cost (in £m), total_points, form, status, chance_of_playing_next_round
    """
    elements = bootstrap["elements"]
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}

    df = pd.DataFrame(elements)

    # ── Rename & select key columns ──────────────────────────────────────────
    df["full_name"] = df["first_name"] + " " + df["second_name"]
    df["position"] = df["element_type"].map(POSITION_MAP)
    df["team_name"] = df["team"].map(teams)
    df["now_cost"] = df["now_cost"] / 10.0          # convert to £m

    keep = [
        "id", "web_name", "full_name", "team", "team_name", "position",
        "now_cost", "total_points", "form", "status",
        "chance_of_playing_next_round", "points_per_game",
        "minutes", "goals_scored", "assists", "clean_sheets",
        "bonus", "bps", "influence", "creativity", "threat", "ict_index",
        "selected_by_percent",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    # ── Type coercion ────────────────────────────────────────────────────────
    for col in ["form", "points_per_game", "influence", "creativity",
                "threat", "ict_index", "selected_by_percent"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # ── Filter unavailable players ───────────────────────────────────────────
    before = len(df)
    df = df[~df["status"].isin(UNAVAILABLE_STATUSES)].copy()
    logger.info(
        "Filtered %d unavailable players → %d available.", before - len(df), len(df)
    )

    df.reset_index(drop=True, inplace=True)
    return df


def build_fixtures_df(fixtures_raw: list) -> pd.DataFrame:
    """
    Build a structured fixtures DataFrame.

    Returns only upcoming (not finished) fixtures with columns:
        fixture_id, gameweek, home_team_id, away_team_id,
        home_difficulty, away_difficulty, finished
    """
    df = pd.DataFrame(fixtures_raw)
    if df.empty:
        return df

    df = df.rename(columns={
        "id": "fixture_id",
        "event": "gameweek",
        "team_h": "home_team_id",
        "team_a": "away_team_id",
        "team_h_difficulty": "home_difficulty",
        "team_a_difficulty": "away_difficulty",
    })

    cols = [
        "fixture_id", "gameweek", "home_team_id", "away_team_id",
        "home_difficulty", "away_difficulty", "finished",
    ]
    df = df[[c for c in cols if c in df.columns]].copy()
    return df


def load_all_data(force_refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Top-level loader: returns (players_df, fixtures_df).

    Parameters
    ----------
    force_refresh : bool
        If True, bypass the local cache and pull fresh data from the API.
    """
    bootstrap = fetch_bootstrap(force_refresh=force_refresh)
    fixtures_raw = fetch_fixtures(force_refresh=force_refresh)

    players_df = build_players_df(bootstrap)
    fixtures_df = build_fixtures_df(fixtures_raw)

    logger.info(
        "Loaded %d players and %d fixtures.", len(players_df), len(fixtures_df)
    )
    return players_df, fixtures_df

# ONE WAY PICKZ — TENNIS V2 FULL CLEAN
# Full tennis prop projection app using a V1/V11-style workflow:
# - Underdog active line pull with manual/upload fallback
# - ATP/WTA historical match-stat pull
# - Player master stat log from day one
# - Underdog player/line history log
# - Elite player identification
# - Aces, Player Games, Total Games, Break Points, Breaks, Tiebreak watch, Fantasy Points
# - Serving, returning, rally/shot proxy, physical/workload, surface, H2H, match-length engine
# - Save snapshots, after-grade learning, player/bucket bias correction
#
# Run:
#   pip install streamlit pandas numpy requests
#   streamlit run one_way_pickz_tennis_v2_full_clean.py

import json
import math
import os
import re
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

warnings.filterwarnings("ignore")

APP_VERSION = "ONE WAY PICKZ — TENNIS V3 LEARNING + TRUE METRIC OVERLAY"
CURRENT_YEAR = datetime.now().year
DATA_DIR = "tennis_v2_data"
os.makedirs(DATA_DIR, exist_ok=True)

SNAPSHOT_FILE = os.path.join(DATA_DIR, "tennis_projection_snapshots.csv")
GRADE_FILE = os.path.join(DATA_DIR, "tennis_after_grades.csv")
LEARNING_FILE = os.path.join(DATA_DIR, "tennis_learning_memory.csv")
MASTER_LOG_FILE = os.path.join(DATA_DIR, "tennis_player_master_log.csv")
UD_LOG_FILE = os.path.join(DATA_DIR, "tennis_underdog_line_log.csv")
ELITE_FILE = os.path.join(DATA_DIR, "tennis_elite_player_tags.csv")
CHARTING_FILE = os.path.join(DATA_DIR, "tennis_charting_true_metrics.csv")
STATUS_FILE = os.path.join(DATA_DIR, "tennis_status_flags.csv")
DRAW_FILE = os.path.join(DATA_DIR, "tennis_draw_status.csv")

UNDERDOG_ENDPOINTS = [
    "https://api.underdogfantasy.com/beta/v5/over_under_lines",
    "https://api.underdogfantasy.com/beta/v4/over_under_lines",
    "https://api.underdogfantasy.com/beta/v3/over_under_lines",
]
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
    "Origin": "https://underdogfantasy.com",
    "Referer": "https://underdogfantasy.com/",
}

TENNIS_KEYWORDS = ["tennis", "atp", "wta", "challenger", "itf", "wimbledon", "roland", "us open", "australian open"]
SURFACE_FACTOR = {"Hard": 1.00, "Clay": 0.92, "Grass": 1.13, "Carpet": 1.06, "Unknown": 1.00}
INDOOR_FACTOR = {"Outdoor": 1.00, "Indoor": 1.055, "Unknown": 1.00}
TOURNEY_LEVEL_FACTOR = {
    "Grand Slam": 1.08,
    "Masters / WTA 1000": 1.045,
    "ATP/WTA 500": 1.015,
    "ATP/WTA 250": 1.00,
    "Challenger / Qualifier": 0.955,
    "Unknown": 1.00,
}
LEVEL_MAP = {"G": "Grand Slam", "M": "Masters / WTA 1000", "A": "ATP/WTA 250", "D": "Team", "F": "Finals", "C": "Challenger / Qualifier", "S": "Unknown"}

# ------------------------------ UI ------------------------------
st.set_page_config(page_title=APP_VERSION, page_icon="🎾", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.stApp {background:#090d11;color:#e7fff2;}
section[data-testid="stSidebar"] {background:#0d131a;}
.big-title {font-size:35px;font-weight:900;color:#00ff88;letter-spacing:.3px;margin-bottom:0}
.sub-title {font-size:14px;color:#9fb2ac;margin-bottom:15px}
.card {background:#111820;border:1px solid #23313d;border-radius:18px;padding:16px;margin:10px 0;box-shadow:0 0 18px rgba(0,255,136,.06)}
.muted{color:#8ca09a}.good{color:#00ff88;font-weight:900}.warn{color:#ffd166;font-weight:900}.bad{color:#ff4d6d;font-weight:900}
.kpi {background:#0e151b;border:1px solid #1d2b35;border-radius:14px;padding:12px;margin:4px}.kpi-value {font-size:22px;font-weight:900;color:#e7fff2}.kpi-label{font-size:12px;color:#8ca09a}
</style>
""", unsafe_allow_html=True)

# ------------------------------ helpers ------------------------------
def clean_name(x) -> str:
    if x is None:
        return ""
    x = str(x).replace("_", " ").replace("-", " ")
    x = re.sub(r"[^A-Za-zÀ-ÿ' .]", "", x)
    return re.sub(r"\s+", " ", x).strip()

def norm_name(x) -> str:
    return clean_name(x).lower()

def safe_float(x, default=np.nan):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def normal_prob_over(edge, sigma):
    if sigma <= 0 or pd.isna(edge):
        return np.nan
    return 100 * sigmoid(1.702 * (edge / sigma))

def prop_bucket(stat: str) -> str:
    s = str(stat).lower()
    if "ace" in s:
        return "ACES"
    if "double fault" in s or "fault" in s:
        return "DOUBLE_FAULTS"
    if "break point" in s:
        return "BREAK_POINTS"
    if "break" in s:
        return "BREAKS"
    if "tie" in s and "break" in s:
        return "TIEBREAK"
    if "total" in s and "game" in s:
        return "TOTAL_GAMES"
    if "game" in s:
        return "PLAYER_GAMES"
    if "fantasy" in s:
        return "FANTASY_POINTS"
    if "set" in s:
        return "SETS"
    if "match" in s or "winner" in s:
        return "MATCH_WINNER"
    return "OTHER"

def read_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def append_csv(path, df):
    if df is None or df.empty:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old = read_csv_safe(path)
    out = pd.concat([old, df], ignore_index=True) if not old.empty else df.copy()
    out.to_csv(path, index=False)

def write_dedup_csv(path, df, subset=None):
    if df is None or df.empty:
        return
    out = df.copy()
    if subset:
        out = out.drop_duplicates(subset=subset, keep="last")
    out.to_csv(path, index=False)

# ------------------------------ Underdog parser ------------------------------
@st.cache_data(ttl=120, show_spinner=False)
def fetch_underdog_raw() -> Tuple[dict, str, str]:
    last = ""
    for url in UNDERDOG_ENDPOINTS:
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            if r.status_code == 200 and r.text.strip().startswith("{"):
                return r.json(), url, ""
            last = f"{url} HTTP {r.status_code}"
        except Exception as e:
            last = f"{url} {e}"
    return {}, "", last

def _idx(items):
    return {str(x.get("id")): x for x in (items or []) if isinstance(x, dict) and x.get("id") is not None}

def _line_value(line):
    candidates = [line.get("stat_value"), line.get("line"), line.get("value")]
    ou = line.get("over_under") if isinstance(line.get("over_under"), dict) else {}
    candidates += [ou.get("stat_value"), ou.get("line")]
    for c in candidates:
        v = safe_float(c)
        if not pd.isna(v):
            return v
    return np.nan

def _stat_title(line):
    for k in ["stat", "stat_type", "stat_type_display", "display_stat", "title", "stat_title", "name"]:
        v = line.get(k)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, dict):
            for kk in ["display_stat", "stat", "name", "title"]:
                if v.get(kk):
                    return str(v.get(kk))
    ou = line.get("over_under") if isinstance(line.get("over_under"), dict) else {}
    app_stat = ou.get("appearance_stat") if isinstance(ou.get("appearance_stat"), dict) else {}
    for kk in ["display_stat", "stat", "name", "title"]:
        if app_stat.get(kk):
            return str(app_stat.get(kk))
    return "Unknown"

def parse_underdog(payload: dict) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    data = payload.get("data", payload)
    if isinstance(data, dict):
        lines = data.get("over_under_lines") or data.get("lines") or data.get("over_unders") or []
        appearances = data.get("appearances") or []
        players = data.get("players") or []
        games = data.get("games") or data.get("solo_games") or []
    elif isinstance(data, list):
        lines, appearances, players, games = data, [], [], []
    else:
        return pd.DataFrame()
    app_i, player_i, game_i = _idx(appearances), _idx(players), _idx(games)
    rows = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        ou = line.get("over_under") if isinstance(line.get("over_under"), dict) else {}
        app_stat = ou.get("appearance_stat") if isinstance(ou.get("appearance_stat"), dict) else {}
        appearance_id = app_stat.get("appearance_id") or line.get("appearance_id") or ou.get("appearance_id")
        app = app_i.get(str(appearance_id), {}) if appearance_id is not None else {}
        player_id = app.get("player_id") or line.get("player_id")
        player = player_i.get(str(player_id), {}) if player_id is not None else {}
        name = player.get("display_name") or player.get("name") or app.get("player_name") or line.get("player_name") or line.get("title") or ""
        stat = _stat_title(line)
        bucket = prop_bucket(stat)
        full_text = (json.dumps(line) + json.dumps(app) + json.dumps(player)).lower()
        is_tennis = any(k in full_text for k in TENNIS_KEYWORDS) or bucket in ["ACES", "PLAYER_GAMES", "TOTAL_GAMES", "BREAK_POINTS", "BREAKS", "TIEBREAK", "SETS", "DOUBLE_FAULTS"]
        if not is_tennis:
            continue
        match_id = app.get("match_id") or app.get("game_id") or line.get("game_id")
        game = game_i.get(str(match_id), {}) if match_id is not None else {}
        matchup = game.get("title") or game.get("match_title") or game.get("name") or app.get("matchup") or line.get("matchup") or ""
        rows.append({
            "Player": clean_name(name), "Opponent": "", "Matchup": matchup, "Stat": stat, "Bucket": bucket,
            "UD/Line": _line_value(line), "Line Source": "Underdog", "Start Time": game.get("scheduled_at") or app.get("scheduled_at") or line.get("scheduled_at") or "", "Raw ID": line.get("id", "")
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["Player"].astype(str).str.len() > 1]
    df = df.drop_duplicates(subset=["Player", "Stat", "UD/Line", "Matchup"])
    df["Pulled At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return df.reset_index(drop=True)

# ------------------------------ Tennis historical loader ------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_matches(years: List[int]) -> pd.DataFrame:
    frames = []
    for tour in ["atp", "wta"]:
        repo = "tennis_atp" if tour == "atp" else "tennis_wta"
        prefix = "atp" if tour == "atp" else "wta"
        for y in years:
            url = f"https://raw.githubusercontent.com/JeffSackmann/{repo}/master/{prefix}_matches_{y}.csv"
            try:
                d = pd.read_csv(url, low_memory=False)
                d["tour"] = tour.upper()
                frames.append(d)
            except Exception:
                pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["tourney_date"] = pd.to_numeric(df.get("tourney_date"), errors="coerce")
    return df.sort_values("tourney_date").reset_index(drop=True)

def _row_for_player(r, won: bool) -> dict:
    p = "w" if won else "l"
    o = "loser" if won else "winner"
    # opponent-side stats are crucial for ace allowed, return, pressure estimates
    op = "l" if won else "w"
    return {
        "date": safe_float(r.get("tourney_date")),
        "surface": r.get("surface", "Unknown") or "Unknown",
        "tour": r.get("tour", ""),
        "level_raw": r.get("tourney_level", ""),
        "round": r.get("round", ""),
        "won": int(won),
        "player_name": r.get(f"{ 'winner' if won else 'loser' }_name", ""),
        "opp_name": r.get(f"{o}_name", ""),
        "rank": safe_float(r.get(f"{p}_rank")),
        "rank_points": safe_float(r.get(f"{p}_rank_points")),
        "opp_rank": safe_float(r.get(f"{o}_rank")),
        "aces": safe_float(r.get(f"{p}_ace"), 0),
        "df": safe_float(r.get(f"{p}_df"), 0),
        "svpt": safe_float(r.get(f"{p}_svpt")),
        "first_in": safe_float(r.get(f"{p}_1stIn")),
        "first_won": safe_float(r.get(f"{p}_1stWon")),
        "second_won": safe_float(r.get(f"{p}_2ndWon")),
        "service_games": safe_float(r.get(f"{p}_SvGms")),
        "bp_saved": safe_float(r.get(f"{p}_bpSaved")),
        "bp_faced": safe_float(r.get(f"{p}_bpFaced")),
        "opp_aces": safe_float(r.get(f"{op}_ace"), 0),
        "opp_df": safe_float(r.get(f"{op}_df"), 0),
        "opp_svpt": safe_float(r.get(f"{op}_svpt")),
        "opp_first_in": safe_float(r.get(f"{op}_1stIn")),
        "opp_first_won": safe_float(r.get(f"{op}_1stWon")),
        "opp_second_won": safe_float(r.get(f"{op}_2ndWon")),
        "opp_service_games": safe_float(r.get(f"{op}_SvGms")),
        "opp_bp_saved": safe_float(r.get(f"{op}_bpSaved")),
        "opp_bp_faced": safe_float(r.get(f"{op}_bpFaced")),
        "score": r.get("score", ""),
        "best_of": safe_float(r.get("best_of"), 3),
        "minutes": safe_float(r.get("minutes")),
    }

def player_rows(matches: pd.DataFrame, player: str, limit=160) -> pd.DataFrame:
    if matches.empty or not player:
        return pd.DataFrame()
    p = norm_name(player)
    w = matches[matches["winner_name"].astype(str).map(norm_name).str.contains(p, regex=False, na=False)]
    l = matches[matches["loser_name"].astype(str).map(norm_name).str.contains(p, regex=False, na=False)]
    rows = [_row_for_player(r, True) for _, r in w.iterrows()] + [_row_for_player(r, False) for _, r in l.iterrows()]
    if not rows:
        last = p.split(" ")[-1] if p else ""
        if len(last) >= 4:
            w = matches[matches["winner_name"].astype(str).map(norm_name).str.contains(last, regex=False, na=False)]
            l = matches[matches["loser_name"].astype(str).map(norm_name).str.contains(last, regex=False, na=False)]
            rows = [_row_for_player(r, True) for _, r in w.iterrows()] + [_row_for_player(r, False) for _, r in l.iterrows()]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("date").tail(limit).reset_index(drop=True)

def parse_score_games(score: str) -> Tuple[int, int, int, int]:
    if not isinstance(score, str):
        return 0, 0, 0, 0
    sets = re.findall(r"(\d+)\-([0-9]+)", score)
    p_games = sum(int(a) for a, _ in sets)
    o_games = sum(int(b) for _, b in sets)
    tbs = len(re.findall(r"7\-6|6\-7", score))
    set_count = len(sets)
    return p_games, o_games, tbs, set_count

def h2h_summary(matches: pd.DataFrame, player: str, opp: str, surface="Unknown") -> dict:
    if matches.empty or not player or not opp:
        return {"H2H Matches": 0, "H2H Win %": 50.0, "H2H Surface Matches": 0}
    p, o = norm_name(player), norm_name(opp)
    mask1 = matches["winner_name"].astype(str).map(norm_name).str.contains(p, regex=False, na=False) & matches["loser_name"].astype(str).map(norm_name).str.contains(o, regex=False, na=False)
    mask2 = matches["winner_name"].astype(str).map(norm_name).str.contains(o, regex=False, na=False) & matches["loser_name"].astype(str).map(norm_name).str.contains(p, regex=False, na=False)
    h = matches[mask1 | mask2].copy()
    if h.empty:
        return {"H2H Matches": 0, "H2H Win %": 50.0, "H2H Surface Matches": 0}
    if surface != "Unknown":
        hs = h[h["surface"].astype(str).str.lower() == surface.lower()]
    else:
        hs = h
    use = hs if len(hs) else h
    wins = use["winner_name"].astype(str).map(norm_name).str.contains(p, regex=False, na=False).mean()
    return {"H2H Matches": int(len(h)), "H2H Win %": round(100 * wins, 1), "H2H Surface Matches": int(len(use))}

def infer_level(raw):
    s = str(raw)
    return LEVEL_MAP.get(s, "Unknown")

def default_summary(player):
    return {
        "player": player, "matches": 0, "surface_matches": 0, "win_pct": .50, "last10_win_pct": .50, "last25_win_pct": .50,
        "rank": np.nan, "rank_points": np.nan, "ace_per_service_game": .52, "ace_per_svpt": .062, "opponent_ace_allowed_per_service_game": .52,
        "df_per_service_game": .24, "first_in_pct": .61, "first_win_pct": .69, "second_win_pct": .50, "service_points_won_pct": .62,
        "serve_effectiveness_pct": .66, "unreturned_serve_proxy_pct": .28, "first_return_won_proxy_pct": .30, "second_return_won_proxy_pct": .49,
        "return_points_won_proxy": .37, "bp_save_pct": .60, "bp_convert_proxy_pct": .39, "bp_created_per_return_game": .55,
        "hold_pct": .78, "break_pct": .22, "return_games_won_pct": .22, "tiebreak_rate": .22, "tiebreak_win_proxy_pct": .50,
        "games_won_avg": 11.2, "games_total_avg": 22.4, "sets_avg": 2.25, "serve_strength": 62.5, "return_strength": 41.0, "overall_strength": 53.0,
        "winner_proxy": 22.0, "unforced_error_proxy": 24.0, "forced_error_proxy": 18.0, "winner_error_ratio_proxy": .92, "shot_quality_proxy": 5.0,
        "short_rally_edge_proxy": 5.0, "long_rally_edge_proxy": 5.0, "minutes_avg": np.nan, "matches_last14": 0, "workload_index": 50.0, "fatigue_tax": 0.0,
        "rest_days": np.nan, "elite_tag": "UNKNOWN_SAMPLE", "reliability": 34.0
    }

def summarize(matches: pd.DataFrame, player: str, surface="Unknown") -> dict:
    rows = player_rows(matches, player, 160)
    if rows.empty:
        return default_summary(player)
    surf = rows[rows["surface"].astype(str).str.lower() == surface.lower()] if surface != "Unknown" else rows
    use = surf if len(surf) >= 6 else rows
    last10 = rows.tail(10)
    last25 = rows.tail(25)
    sg = use["service_games"].replace(0, np.nan)
    opp_sg = use["opp_service_games"].replace(0, np.nan)
    svpt = use["svpt"].replace(0, np.nan)
    opp_svpt = use["opp_svpt"].replace(0, np.nan)
    first_in = use["first_in"].sum()
    second_pts = use["svpt"].sum() - first_in
    opp_first_in = use["opp_first_in"].sum()
    opp_second_pts = use["opp_svpt"].sum() - opp_first_in

    ace_sg = use["aces"].sum() / sg.sum() if sg.sum() and not pd.isna(sg.sum()) else .52
    ace_svpt = use["aces"].sum() / svpt.sum() if svpt.sum() and not pd.isna(svpt.sum()) else .062
    opp_ace_allowed_sg = use["opp_aces"].sum() / opp_sg.sum() if opp_sg.sum() and not pd.isna(opp_sg.sum()) else .52
    df_sg = use["df"].sum() / sg.sum() if sg.sum() and not pd.isna(sg.sum()) else .24
    first_in_pct = first_in / svpt.sum() if svpt.sum() and not pd.isna(svpt.sum()) else .61
    first_win_pct = use["first_won"].sum() / first_in if first_in and not pd.isna(first_in) else .69
    second_win_pct = use["second_won"].sum() / second_pts if second_pts and not pd.isna(second_pts) else .50
    service_points_won = (use["first_won"].sum() + use["second_won"].sum()) / svpt.sum() if svpt.sum() and not pd.isna(svpt.sum()) else .62
    bp_save = use["bp_saved"].sum() / use["bp_faced"].sum() if use["bp_faced"].sum() and not pd.isna(use["bp_faced"].sum()) else .60
    # True return split is not always present in free CSV, so derive from opponent service points in the same matches.
    return_points_won = 1 - ((use["opp_first_won"].sum() + use["opp_second_won"].sum()) / opp_svpt.sum()) if opp_svpt.sum() and not pd.isna(opp_svpt.sum()) else .37
    first_return_won = 1 - (use["opp_first_won"].sum() / opp_first_in) if opp_first_in and not pd.isna(opp_first_in) else .30
    second_return_won = 1 - (use["opp_second_won"].sum() / opp_second_pts) if opp_second_pts and not pd.isna(opp_second_pts) else .49
    bp_convert_proxy = use["opp_bp_faced"].sum() - use["opp_bp_saved"].sum()
    bp_convert_proxy = bp_convert_proxy / use["opp_bp_faced"].sum() if use["opp_bp_faced"].sum() and not pd.isna(use["opp_bp_faced"].sum()) else .39
    bp_created_per_rg = use["opp_bp_faced"].sum() / opp_sg.sum() if opp_sg.sum() and not pd.isna(opp_sg.sum()) else .55

    hold_pct = clamp(0.49 + 0.58 * service_points_won + 0.06 * bp_save - 0.045 * df_sg, 0.48, 0.94)
    break_pct = clamp(0.10 + 1.75 * (return_points_won - .34) + .08 * bp_convert_proxy, .06, .48)
    return_games_won_pct = break_pct

    win_pct = float(use["won"].mean())
    last10_win = float(last10["won"].mean()) if len(last10) else win_pct
    last25_win = float(last25["won"].mean()) if len(last25) else win_pct
    rank = use["rank"].dropna().tail(1).mean() if use["rank"].notna().any() else np.nan
    rank_points = use["rank_points"].dropna().tail(1).mean() if use["rank_points"].notna().any() else np.nan

    tiebreak_count = 0
    tb_wins = 0
    games_for, games_against, set_counts = [], [], []
    for _, rr in use.iterrows():
        gf, ga, tb, sc = parse_score_games(rr.get("score", ""))
        tiebreak_count += tb
        # coarse proxy: if won match with a tiebreak, credit slightly more TB strength
        if tb > 0 and rr.get("won", 0) == 1:
            tb_wins += 1
        if gf + ga > 0:
            if rr.get("won", 0) == 1:
                games_for.append(gf); games_against.append(ga)
            else:
                games_for.append(ga); games_against.append(gf)
            set_counts.append(sc)
    games_won_avg = float(np.mean(games_for)) if games_for else 11.2
    games_total_avg = float(np.mean(np.array(games_for) + np.array(games_against))) if games_for else 22.4
    sets_avg = float(np.mean(set_counts)) if set_counts else 2.25
    tiebreak_rate = tiebreak_count / max(len(use), 1)
    tiebreak_win_proxy = clamp(.48 + .18 * (hold_pct - .78) + .12 * (win_pct - .5) + .04 * tb_wins / max(tiebreak_count, 1), .34, .68)

    # Rally/shot metrics are rarely available free pre-match; use transparent proxies from serve/return/error profile.
    serve_effectiveness = clamp(.47 + .45 * service_points_won + .70 * ace_svpt - .05 * df_sg, .45, .82)
    unreturned_proxy = clamp(.12 + 2.15 * ace_svpt + .22 * (first_win_pct - .66), .12, .46)
    winner_proxy = clamp(12 + 55 * ace_svpt + 18 * (first_win_pct - .66) + 7 * (return_points_won - .37), 8, 42)
    unforced_error_proxy = clamp(17 + 18 * df_sg + 12 * (1 - second_win_pct) - 4 * (win_pct - .5), 12, 42)
    forced_error_proxy = clamp(14 + 24 * return_points_won + 8 * break_pct, 10, 36)
    wer = winner_proxy / max(unforced_error_proxy, 1)
    shot_quality = clamp(5.0 + 4.0 * (service_points_won - .62) + 3.0 * (return_points_won - .37) + 0.7 * (wer - .9), 1.0, 10.0)
    short_rally_edge = clamp(5.0 + 10 * (serve_effectiveness - .66) + 3 * (ace_svpt - .062), 1, 10)
    long_rally_edge = clamp(5.0 + 9 * (return_points_won - .37) + 2.5 * (second_win_pct - .50) - .08 * (unforced_error_proxy - 24), 1, 10)

    rest_days = np.nan
    matches_last14 = 0
    if rows["date"].notna().any():
        try:
            dates = pd.to_datetime(rows["date"].dropna().astype(int).astype(str), format="%Y%m%d")
            last_dt = dates.iloc[-1]
            now = pd.Timestamp(datetime.now().date())
            rest_days = int((now - last_dt).days)
            matches_last14 = int((dates >= (now - pd.Timedelta(days=14))).sum())
        except Exception:
            pass
    minutes_avg = use["minutes"].dropna().mean() if use["minutes"].notna().any() else np.nan
    workload_index = 50 + 5 * matches_last14 + (0 if pd.isna(minutes_avg) else clamp((minutes_avg - 95) / 3, -12, 18))
    fatigue_tax = 0.0
    if not pd.isna(rest_days):
        if rest_days <= 1:
            fatigue_tax -= .045
        elif rest_days <= 3:
            fatigue_tax -= .015
        elif rest_days >= 21:
            fatigue_tax -= .025
    if workload_index >= 75:
        fatigue_tax -= .025

    serve_strength = 100 * service_points_won + 9 * ace_svpt + 6 * (hold_pct - .75) + 3.0 * (last10_win - .5) - 1.8 * df_sg
    return_strength = 100 * return_points_won + 18 * break_pct + 3 * (bp_convert_proxy - .39) + 2.5 * (last10_win - .5)
    rank_adj = 0 if pd.isna(rank) else clamp((75 - rank) / 22, -2.2, 2.2)
    overall_strength = 0.54 * serve_strength + 0.46 * return_strength + rank_adj + 0.8 * (shot_quality - 5)
    reliability = clamp(34 + len(use) * 1.0 + len(rows) * .25 + (6 if len(surf) >= 8 else 0), 30, 96)

    elite_tag = "STANDARD"
    tags = []
    if service_points_won >= .66 or ace_sg >= .78 or hold_pct >= .84:
        tags.append("ELITE_SERVER")
    if return_points_won >= .405 or break_pct >= .29:
        tags.append("ELITE_RETURNER")
    if not pd.isna(rank) and rank <= 25:
        tags.append("TOP_25")
    if last10_win >= .70:
        tags.append("HOT_FORM")
    if shot_quality >= 6.4:
        tags.append("HIGH_SHOT_QUALITY")
    if workload_index >= 78:
        tags.append("FATIGUE_RISK")
    elite_tag = ",".join(tags) if tags else "STANDARD"

    return {
        "player": player, "matches": int(len(rows)), "surface_matches": int(len(use)), "win_pct": win_pct, "last10_win_pct": last10_win, "last25_win_pct": last25_win,
        "rank": rank, "rank_points": rank_points, "ace_per_service_game": float(ace_sg), "ace_per_svpt": float(ace_svpt), "opponent_ace_allowed_per_service_game": float(opp_ace_allowed_sg),
        "df_per_service_game": float(df_sg), "first_in_pct": float(first_in_pct), "first_win_pct": float(first_win_pct), "second_win_pct": float(second_win_pct),
        "service_points_won_pct": float(service_points_won), "serve_effectiveness_pct": float(serve_effectiveness), "unreturned_serve_proxy_pct": float(unreturned_proxy),
        "first_return_won_proxy_pct": float(first_return_won), "second_return_won_proxy_pct": float(second_return_won), "return_points_won_proxy": float(return_points_won),
        "bp_save_pct": float(bp_save), "bp_convert_proxy_pct": float(bp_convert_proxy), "bp_created_per_return_game": float(bp_created_per_rg),
        "hold_pct": float(hold_pct), "break_pct": float(break_pct), "return_games_won_pct": float(return_games_won_pct), "tiebreak_rate": float(tiebreak_rate), "tiebreak_win_proxy_pct": float(tiebreak_win_proxy),
        "games_won_avg": float(games_won_avg), "games_total_avg": float(games_total_avg), "sets_avg": float(sets_avg),
        "serve_strength": float(serve_strength), "return_strength": float(return_strength), "overall_strength": float(overall_strength),
        "winner_proxy": float(winner_proxy), "unforced_error_proxy": float(unforced_error_proxy), "forced_error_proxy": float(forced_error_proxy), "winner_error_ratio_proxy": float(wer), "shot_quality_proxy": float(shot_quality),
        "short_rally_edge_proxy": float(short_rally_edge), "long_rally_edge_proxy": float(long_rally_edge), "minutes_avg": minutes_avg, "matches_last14": matches_last14, "workload_index": float(workload_index), "fatigue_tax": float(fatigue_tax),
        "rest_days": rest_days, "elite_tag": elite_tag, "reliability": float(reliability)
    }

# ------------------------------ Logs / learning ------------------------------
def _mean_error(df):
    if df is None or df.empty or "Error" not in df.columns:
        return np.nan, 0
    vals = pd.to_numeric(df["Error"], errors="coerce").dropna().tail(40)
    if len(vals) < 2:
        return np.nan, len(vals)
    return float(vals.mean()), len(vals)

def learning_bias(player, bucket, surface="", tourney_level="", opponent=""):
    """Tennis learning engine. Blends player+prop, player, prop, surface and tournament-level bias.
    Error = Actual - Projection, so a positive bias raises future projections.
    This mirrors the MLB learning concept but is safer: it needs sample size and caps adjustments.
    """
    mem = read_csv_safe(LEARNING_FILE)
    if mem.empty:
        return 0.0, "NO_LEARNING"
    m = mem.copy()
    for col in ["Player", "Opponent", "Bucket", "Surface", "Tournament Level"]:
        if col not in m.columns:
            m[col] = ""
    pk, ok = norm_name(player), norm_name(opponent)
    b = str(bucket)
    layers = []
    specs = [
        ("PLY_PROP", .46, (m["Player"].map(norm_name)==pk) & (m["Bucket"].astype(str)==b)),
        ("PLY_ALL", .18, (m["Player"].map(norm_name)==pk)),
        ("PROP", .18, (m["Bucket"].astype(str)==b)),
        ("SURF_PROP", .10, (m["Surface"].astype(str)==str(surface)) & (m["Bucket"].astype(str)==b)),
        ("TOUR_PROP", .08, (m["Tournament Level"].astype(str)==str(tourney_level)) & (m["Bucket"].astype(str)==b)),
    ]
    # small optional opponent layer, useful for ace-allowed and break-prop tendencies
    if ok:
        specs.append(("OPP_PROP", .08, (m["Opponent"].map(norm_name)==ok) & (m["Bucket"].astype(str)==b)))
    total_w, adj, labels = 0.0, 0.0, []
    for name, w, mask in specs:
        err, n = _mean_error(m[mask])
        if not pd.isna(err) and n >= 3:
            shrink = min(1.0, n / 18)
            adj += w * shrink * err
            total_w += w * shrink
            labels.append(f"{name}:{err:+.2f}/{n}")
    if total_w <= 0:
        return 0.0, "LOW_SAMPLE"
    raw = adj / total_w
    cap = .95 if b in ["ACES", "PLAYER_GAMES", "BREAK_POINTS", "BREAKS", "DOUBLE_FAULTS"] else 1.45
    bias = clamp(raw, -cap, cap)
    return bias, " | ".join(labels[:3])

def read_optional_overlay(path):
    df = read_csv_safe(path)
    if df.empty:
        return df
    if "Player" in df.columns:
        df["PlayerKey"] = df["Player"].map(norm_name)
    return df

def overlay_charting_metrics(summary, player, charting_df):
    """Overlay true/free charted metrics when available.
    Expected columns are flexible: Player plus any of Winners, Unforced Errors, Forced Errors, Rally Length, Distance Covered, Workload.
    If absent, the app keeps transparent proxy values.
    """
    if charting_df is None or charting_df.empty or "PlayerKey" not in charting_df.columns:
        summary["true_metric_source"] = "PROXY_ONLY"
        return summary
    sub = charting_df[charting_df["PlayerKey"] == norm_name(player)]
    if sub.empty:
        summary["true_metric_source"] = "PROXY_ONLY"
        return summary
    row = sub.tail(20)
    def avg_any(names, default=np.nan):
        for name in names:
            if name in row.columns:
                v = pd.to_numeric(row[name], errors="coerce").dropna()
                if len(v): return float(v.mean())
        return default
    winners = avg_any(["Winners", "winner_count", "winners", "Winner Count"])
    ufe = avg_any(["Unforced Errors", "unforced_errors", "UFE", "true_unforced_errors"])
    fe = avg_any(["Forced Errors", "forced_errors", "FE", "true_forced_errors"])
    rally = avg_any(["Rally Length", "avg_rally_length", "Average Rally", "rally_length"])
    dist = avg_any(["Distance Covered", "distance_covered", "Distance", "meters_covered"])
    if not pd.isna(winners): summary["winner_proxy"] = winners
    if not pd.isna(ufe): summary["unforced_error_proxy"] = ufe
    if not pd.isna(fe): summary["forced_error_proxy"] = fe
    if not pd.isna(rally):
        summary["avg_rally_length_true"] = rally
        summary["short_rally_edge_proxy"] = clamp(8.0 - rally, 1, 10)
        summary["long_rally_edge_proxy"] = clamp(2.0 + rally, 1, 10)
    else:
        summary["avg_rally_length_true"] = np.nan
    if not pd.isna(dist):
        summary["distance_covered_true"] = dist
        # More recent true physical workload raises fatigue risk if extreme.
        summary["workload_index"] = clamp(summary.get("workload_index", 50) + max((dist - 2800)/140, 0), 1, 99)
    else:
        summary["distance_covered_true"] = np.nan
    summary["winner_error_ratio_proxy"] = summary.get("winner_proxy", 1) / max(summary.get("unforced_error_proxy", 1), 1)
    summary["true_metric_source"] = "CHARTING_OVERLAY"
    return summary

def lookup_risk_flags(player, opponent, status_df, draw_df):
    flags, tax, block = [], 0.0, False
    for df, source in [(status_df, "STATUS"), (draw_df, "DRAW")]:
        if df is None or df.empty or "PlayerKey" not in df.columns:
            continue
        for who in [player, opponent]:
            if not who: continue
            sub = df[df["PlayerKey"] == norm_name(who)]
            if sub.empty: continue
            row = sub.tail(1).iloc[0]
            txt = " ".join([str(row.get(c, "")) for c in row.index]).lower()
            if any(x in txt for x in ["retired", "withdrawn", "walkover", "out", "cancelled"]):
                block = True; flags.append(f"{source}:{who}:BLOCK")
            elif any(x in txt for x in ["injury", "questionable", "illness", "limited", "medical"]):
                tax += .10; flags.append(f"{source}:{who}:RISK")
            elif any(x in txt for x in ["confirmed", "active", "scheduled", "in draw"]):
                flags.append(f"{source}:{who}:OK")
    return "; ".join(flags) if flags else "NO_LIVE_FLAG", clamp(tax, 0, .25), block

def save_master_logs(engine_df, player_summaries):
    if engine_df is None or engine_df.empty:
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ud_cols = ["Player", "Opponent", "Matchup", "Stat", "Bucket", "UD/Line", "Line Source", "Start Time", "Raw ID", "TENNIS PROJ", "Decision", "Confidence %", "Grade"]
    ud_log = engine_df[[c for c in ud_cols if c in engine_df.columns]].copy()
    ud_log["Logged At"] = now
    append_csv(UD_LOG_FILE, ud_log)

    prof_rows = []
    for name, s in player_summaries.items():
        row = {"Player": name, "Logged At": now}
        row.update({k: v for k, v in s.items() if k != "player"})
        prof_rows.append(row)
    prof = pd.DataFrame(prof_rows)
    if not prof.empty:
        append_csv(MASTER_LOG_FILE, prof)
        elite = prof[["Player", "elite_tag", "rank", "rank_points", "serve_strength", "return_strength", "overall_strength", "reliability", "Logged At"]].copy()
        write_dedup_csv(ELITE_FILE, elite, subset=["Player"])

def grade_from_result_file(graded: pd.DataFrame, current: pd.DataFrame):
    if graded is None or graded.empty or current is None or current.empty:
        return pd.DataFrame()
    need = {"Player", "Stat", "Actual"}
    if not need.issubset(set(graded.columns)):
        return pd.DataFrame()
    g = graded.copy()
    g["PlayerKey"] = g["Player"].map(norm_name)
    current = current.copy()
    current["PlayerKey"] = current["Player"].map(norm_name)
    merged = current.merge(g[["PlayerKey", "Stat", "Actual"]], on=["PlayerKey", "Stat"], how="left")
    merged["Actual"] = pd.to_numeric(merged["Actual"], errors="coerce")
    merged["Error"] = merged["Actual"] - pd.to_numeric(merged["TENNIS PROJ"], errors="coerce")
    def res(row):
        if pd.isna(row["Actual"]): return ""
        line = row.get("UD/Line", np.nan)
        dec = row.get("Decision", "")
        if pd.isna(line): return ""
        if row["Actual"] == line: return "PUSH"
        if dec == "OVER": return "WIN" if row["Actual"] > line else "LOSS"
        if dec == "UNDER": return "WIN" if row["Actual"] < line else "LOSS"
        return ""
    merged["Result"] = merged.apply(res, axis=1)
    mem_cols = ["Player", "Opponent", "Stat", "Bucket", "UD/Line", "TENNIS PROJ", "Decision", "Actual", "Error", "Result", "Surface", "Tournament Level", "Indoor/Outdoor", "Best Of", "Confidence %", "Lean Gap", "Grade", "Official Filter", "Logged At"]
    merged["Logged At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    append_csv(GRADE_FILE, merged[[c for c in mem_cols if c in merged.columns]])
    append_csv(LEARNING_FILE, merged[[c for c in mem_cols if c in merged.columns]])
    return merged

# ------------------------------ Projection engine ------------------------------
def infer_opponent(row):
    if str(row.get("Opponent", "")).strip():
        return clean_name(row.get("Opponent"))
    player = norm_name(row.get("Player", ""))
    matchup = str(row.get("Matchup", ""))
    parts = re.split(r"\s+v(?:s\.?|ersus)?\s+| @ | - |/", matchup, flags=re.I)
    parts = [clean_name(x) for x in parts if clean_name(x)]
    for p in parts:
        n = norm_name(p)
        if n and n != player and player not in n and n not in player:
            return p
    return ""

def expected_sets(best_of, p, opp, h2h_win=50):
    gap = p["overall_strength"] - opp["overall_strength"]
    h2h_adj = (h2h_win - 50) / 100
    close = 1 - min(abs(gap + h2h_adj * 5) / 18, .45)
    if best_of == 5:
        return clamp(3.18 + 1.12 * close, 3.02, 4.75)
    return clamp(2.01 + .58 * close, 2.00, 2.95)

def straight_set_probability(best_of, p, opp):
    gap = abs(p["overall_strength"] - opp["overall_strength"])
    if best_of == 5:
        return clamp(.18 + gap / 55, .12, .54)
    return clamp(.39 + gap / 42, .36, .76)

def three_set_probability(best_of, p, opp):
    if best_of != 3:
        return 0.0
    return round(100 * (1 - straight_set_probability(best_of, p, opp)), 1)

def expected_match_games(best_of, p, opp, surface, h2h_win=50):
    sets = expected_sets(best_of, p, opp, h2h_win)
    hold_combo = (p["hold_pct"] + opp["hold_pct"]) / 2
    close = 1 - min(abs(p["overall_strength"] - opp["overall_strength"]) / 18, .45)
    surface_game = .65 if surface == "Grass" else (-.38 if surface == "Clay" else 0)
    return clamp(9.30 * sets + 2.35 * close + 3.85 * (hold_combo - .76) + surface_game, 17 if best_of == 3 else 28, 35 if best_of == 3 else 60)

def official_gate(bucket, edge, conf, rel, vol, line, p, opp):
    if bucket == "TIEBREAK":
        # allow only very strict tiebreak setups as WATCH, not official lock
        hold_combo = (p["hold_pct"] + opp["hold_pct"]) / 2
        if hold_combo >= .84 and abs(edge) >= .22 and conf >= 60:
            return False, "WATCH ONLY — elite hold profile but tiebreak still volatile"
        return False, "TIEBREAK FADE/WATCH — high variance"
    if bucket == "OTHER":
        return False, "Unsupported prop"
    if pd.isna(edge) or pd.isna(conf):
        return False, "Missing line/projection"
    absedge = abs(edge)
    if bucket == "ACES" and absedge >= .55 and conf >= 56 and rel >= 45:
        return True, "PASS — Aces main market with volume support"
    if bucket == "PLAYER_GAMES" and absedge >= .65 and conf >= 57 and rel >= 47:
        return True, "PASS — Player games main market"
    if bucket == "TOTAL_GAMES" and absedge >= .90 and conf >= 59 and rel >= 52 and vol <= 1.42:
        return True, "PASS — Total games qualified"
    if bucket in ["BREAK_POINTS", "BREAKS"] and absedge >= .80 and conf >= 60 and rel >= 55 and vol <= 1.58:
        return True, "PASS — Break market qualified"
    if bucket == "DOUBLE_FAULTS" and absedge >= .45 and conf >= 58 and rel >= 50:
        return True, "PASS — Double faults qualified"
    if bucket == "FANTASY_POINTS" and absedge >= 1.10 and conf >= 59 and rel >= 52:
        return True, "PASS — Fantasy points qualified"
    return False, "NO PLAY — edge/conf/reliability short"

def grade(conf, official, volatility, reliability):
    if not official:
        return "C / WATCH"
    if conf >= 68 and volatility <= 1.20 and reliability >= 62:
        return "S 🔒"
    if conf >= 63 and reliability >= 55:
        return "A"
    if conf >= 58:
        return "B"
    return "C"

def project(row, p, opp, h2h, surface, indoor, tourney_level, best_of):
    bucket = row.get("Bucket", "OTHER")
    line = safe_float(row.get("UD/Line"))
    surf = SURFACE_FACTOR.get(surface, 1.0)
    indoor_f = INDOOR_FACTOR.get(indoor, 1.0)
    level_f = TOURNEY_LEVEL_FACTOR.get(tourney_level, 1.0)
    h2h_win = h2h.get("H2H Win %", 50.0)
    sets = expected_sets(best_of, p, opp, h2h_win)
    match_games = expected_match_games(best_of, p, opp, surface, h2h_win)
    player_service_games = match_games / 2 + clamp((p["overall_strength"] - opp["overall_strength"]) / 18, -1.15, 1.15)
    player_return_games = match_games / 2
    strength_gap = p["overall_strength"] - opp["overall_strength"]
    serve_gap = p["serve_strength"] - opp["return_strength"]
    return_gap = p["return_strength"] - opp["serve_strength"]
    h2h_edge = clamp((h2h_win - 50) / 20, -1.0, 1.0) if h2h.get("H2H Matches", 0) >= 2 else 0.0
    learning_adj, learning_label = learning_bias(row.get("Player", ""), bucket, surface, tourney_level, row.get("Opponent", ""))
    risk_label = row.get("Risk Flags", "NO_LIVE_FLAG")
    risk_tax = safe_float(row.get("Risk Tax", 0.0), 0.0)
    risk_block = bool(row.get("Risk Block", False))

    fatigue_factor = 1 + p.get("fatigue_tax", 0.0)
    opp_ace_allow_factor = clamp(1 + (opp["opponent_ace_allowed_per_service_game"] - .52) * .42, .88, 1.18)

    if bucket == "ACES":
        proj = p["ace_per_service_game"] * player_service_games * surf * indoor_f * opp_ace_allow_factor * (1 + .035 * serve_gap / 10) * fatigue_factor
        sigma = clamp(1.05 + proj * .34, 1.25, 5.2)
    elif bucket == "DOUBLE_FAULTS":
        proj = p["df_per_service_game"] * player_service_games * (1 + .04 * max(opp["return_strength"] - 41, 0) / 10) * (1 - p.get("fatigue_tax", 0.0))
        sigma = clamp(.75 + proj * .45, .85, 3.8)
    elif bucket == "PLAYER_GAMES":
        proj = (match_games / 2) + clamp(strength_gap / 12, -2.25, 2.25) + .25 * h2h_edge
        sigma = 2.10 if best_of == 3 else 3.60
    elif bucket == "TOTAL_GAMES":
        proj = match_games
        sigma = 3.30 if best_of == 3 else 5.45
    elif bucket == "BREAK_POINTS":
        opp_bp_faced_rate = clamp(opp["bp_created_per_return_game"] + (1 - opp["hold_pct"]) * .33, .32, .82)
        proj = player_return_games * opp_bp_faced_rate * (1 + .055 * return_gap / 10) * level_f
        sigma = clamp(1.20 + proj * .50, 1.35, 5.2)
    elif bucket == "BREAKS":
        break_chance = clamp(p["break_pct"] + .015 * return_gap / 10 + .01 * h2h_edge, .06, .50)
        proj = player_return_games * break_chance * level_f
        sigma = clamp(.95 + proj * .58, 1.05, 4.5)
    elif bucket == "TIEBREAK":
        hold_combo = (p["hold_pct"] + opp["hold_pct"]) / 2
        tb_rate = (p["tiebreak_rate"] + opp["tiebreak_rate"]) / 2
        proj = clamp(.10 + .62 * tb_rate + .72 * max(hold_combo - .78, 0) + (.08 if surface == "Grass" else 0), .04, 1.20)
        sigma = .78
    elif bucket == "FANTASY_POINTS":
        ace_component = p["ace_per_service_game"] * player_service_games * surf * indoor_f * opp_ace_allow_factor
        games_component = (match_games / 2) + clamp(strength_gap / 12, -2.25, 2.25)
        breaks_component = player_return_games * clamp(p["break_pct"] + .015 * return_gap / 10, .06, .50)
        proj = games_component + .85 * ace_component + 1.15 * breaks_component + 2.0 * sigmoid(strength_gap / 8)
        sigma = clamp(2.4 + proj * .13, 2.8, 6.8)
    else:
        proj, sigma = np.nan, 2.0

    if not pd.isna(proj):
        proj += learning_adj
    edge = proj - line if not pd.isna(proj) and not pd.isna(line) else np.nan
    over = normal_prob_over(edge, sigma)
    under = 100 - over if not pd.isna(over) else np.nan
    conf = max(over, under) if not pd.isna(over) else np.nan
    decision = "OVER" if not pd.isna(edge) and edge > 0 else ("UNDER" if not pd.isna(edge) else "NO PLAY")
    vol = round(clamp(sigma / max(abs(proj), 1), .45, 2.55) + (.18 if bucket in ["TIEBREAK", "BREAK_POINTS", "BREAKS"] else 0), 2)
    rel = min(p["reliability"], opp["reliability"] if opp["matches"] else p["reliability"] - 8)
    official, reason = official_gate(bucket, edge, conf, rel, vol, line, p, opp)
    if risk_block:
        official, reason = False, "NO PLAY — live status/draw block: " + str(risk_label)
    elif risk_tax > 0:
        conf = max(0, conf - risk_tax * 100) if not pd.isna(conf) else conf
        rel = max(0, rel - risk_tax * 75)
        if official and risk_tax >= .10:
            official, reason = False, "WATCH — injury/status risk tax: " + str(risk_label)

    return {
        "TENNIS PROJ": round(proj, 2) if not pd.isna(proj) else np.nan,
        "Floor": round(proj - sigma, 2) if not pd.isna(proj) else np.nan,
        "Median": round(proj, 2) if not pd.isna(proj) else np.nan,
        "Ceiling": round(proj + sigma, 2) if not pd.isna(proj) else np.nan,
        "Volatility": vol, "Over Sim %": round(over, 1) if not pd.isna(over) else np.nan, "Under Sim %": round(under, 1) if not pd.isna(under) else np.nan,
        "Decision": decision, "Model Lean": decision, "Lean Gap": round(edge, 2) if not pd.isna(edge) else np.nan,
        "Confidence %": round(conf, 1) if not pd.isna(conf) else np.nan, "Reliability": round(rel, 1),
        "Official Filter": "PASS" if official else "NO PLAY", "Official Reason": reason, "Grade": grade(conf if not pd.isna(conf) else 0, official, vol, rel),
        "Learning Adj": round(learning_adj, 2), "Learning Label": learning_label,
        "Expected Sets": round(sets, 2), "3 Set Prob %": three_set_probability(best_of, p, opp), "Straight Set Prob %": round(100 * straight_set_probability(best_of, p, opp), 1),
        "Expected Match Games": round(match_games, 2), "Expected Service Games": round(player_service_games, 2), "Expected Return Games": round(player_return_games, 2),
        "Surface": surface, "Indoor/Outdoor": indoor, "Tournament Level": tourney_level, "Best Of": best_of,
        "Rank": p["rank"], "Rank Points": p["rank_points"], "Elite Tag": p["elite_tag"], "Player Win %": round(100*p["win_pct"], 1), "Last10 Win %": round(100*p["last10_win_pct"], 1), "Last25 Win %": round(100*p["last25_win_pct"], 1),
        "Aces/Service Game": round(p["ace_per_service_game"], 3), "Opponent Ace Allowed/SG": round(opp["opponent_ace_allowed_per_service_game"], 3), "DF/Service Game": round(p["df_per_service_game"], 3),
        "1st Serve In %": round(100*p["first_in_pct"], 1), "1st Serve Won %": round(100*p["first_win_pct"], 1), "2nd Serve Won %": round(100*p["second_win_pct"], 1), "Service Points Won %": round(100*p["service_points_won_pct"], 1),
        "Serve Effectiveness %": round(100*p["serve_effectiveness_pct"], 1), "Unreturned Serve Proxy %": round(100*p["unreturned_serve_proxy_pct"], 1),
        "1st Return Won Proxy %": round(100*p["first_return_won_proxy_pct"], 1), "2nd Return Won Proxy %": round(100*p["second_return_won_proxy_pct"], 1), "Return Points Won %": round(100*p["return_points_won_proxy"], 1),
        "BP Save %": round(100*p["bp_save_pct"], 1), "BP Convert Proxy %": round(100*p["bp_convert_proxy_pct"], 1), "BP Created/Return Game": round(p["bp_created_per_return_game"], 3),
        "Hold %": round(100*p["hold_pct"], 1), "Break %": round(100*p["break_pct"], 1), "Return Games Won %": round(100*p["return_games_won_pct"], 1), "Tiebreak Rate": round(100*p["tiebreak_rate"], 1), "Tiebreak Win Proxy %": round(100*p["tiebreak_win_proxy_pct"], 1),
        "Winner Proxy": round(p["winner_proxy"], 1), "Unforced Error Proxy": round(p["unforced_error_proxy"], 1), "Forced Error Proxy": round(p["forced_error_proxy"], 1), "Winner/Error Ratio Proxy": round(p["winner_error_ratio_proxy"], 2), "Shot Quality Proxy": round(p["shot_quality_proxy"], 1),
        "Short Rally Edge Proxy": round(p["short_rally_edge_proxy"], 1), "Long Rally Edge Proxy": round(p["long_rally_edge_proxy"], 1), "Avg Rally Length True": p.get("avg_rally_length_true", np.nan), "Distance Covered True": p.get("distance_covered_true", np.nan), "True Metric Source": p.get("true_metric_source", "PROXY_ONLY"), "Risk Flags": risk_label, "Risk Tax": round(risk_tax, 3), "Workload Index": round(p["workload_index"], 1), "Matches Last 14": p["matches_last14"], "Rest Days": p["rest_days"], "Fatigue Tax": round(p["fatigue_tax"], 3),
        "Serve Strength": round(p["serve_strength"], 1), "Return Strength": round(p["return_strength"], 1), "Overall Strength": round(p["overall_strength"], 1), "Opponent Strength": round(opp["overall_strength"], 1),
        "H2H Matches": h2h.get("H2H Matches", 0), "H2H Win %": h2h.get("H2H Win %", 50.0), "H2H Surface Matches": h2h.get("H2H Surface Matches", 0),
        "Player Matches": p["matches"], "Surface Matches": p["surface_matches"], "Opp Matches": opp["matches"],
    }

def build_engine(lines, matches, surface, indoor, tourney_level, best_of, charting_df=None, status_df=None, draw_df=None):
    rows, summaries = [], {}
    for _, r in lines.iterrows():
        base = r.to_dict()
        base["Player"] = clean_name(base.get("Player"))
        base["Opponent"] = infer_opponent(base)
        base["Bucket"] = base.get("Bucket") or prop_bucket(base.get("Stat", ""))
        p = summarize(matches, base["Player"], surface)
        opp = summarize(matches, base["Opponent"], surface) if base["Opponent"] else default_summary("Unknown")
        p = overlay_charting_metrics(p, base["Player"], charting_df)
        if base["Opponent"]:
            opp = overlay_charting_metrics(opp, base["Opponent"], charting_df)
        flags, tax, block = lookup_risk_flags(base["Player"], base["Opponent"], status_df, draw_df)
        base["Risk Flags"], base["Risk Tax"], base["Risk Block"] = flags, tax, block
        h2h = h2h_summary(matches, base["Player"], base["Opponent"], surface) if base["Opponent"] else {"H2H Matches":0,"H2H Win %":50.0,"H2H Surface Matches":0}
        summaries[base["Player"]] = p
        if base["Opponent"]:
            summaries[base["Opponent"]] = opp
        base.update(project(base, p, opp, h2h, surface, indoor, tourney_level, best_of))
        rows.append(base)
    df = pd.DataFrame(rows)
    if df.empty:
        return df, summaries
    df["Abs Edge"] = pd.to_numeric(df["Lean Gap"], errors="coerce").abs()
    return df.sort_values(["Official Filter", "Confidence %", "Abs Edge"], ascending=[True, False, False]).reset_index(drop=True), summaries

# ------------------------------ Render helpers ------------------------------
def render_card(r: pd.Series):
    cls = "good" if r.get("Official Filter") == "PASS" else "warn"
    st.markdown(f"""
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div><div style="font-size:23px;font-weight:900;">{r.get('Player','')} — {r.get('Stat','')}</div><div class="muted">{r.get('Matchup','')} {('vs ' + str(r.get('Opponent',''))) if r.get('Opponent','') else ''}</div></div>
        <div style="text-align:right;"><div class="{cls}" style="font-size:24px;">{r.get('Decision','')}</div><div class="muted">{r.get('Grade','')} · {r.get('Official Filter','')}</div></div>
      </div>
      <hr style="border-color:#23313d;">
      <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:10px;">
        <div><div class="muted">Proj</div><div style="font-size:22px;font-weight:900;">{r.get('TENNIS PROJ','')}</div></div>
        <div><div class="muted">UD Line</div><div style="font-size:22px;font-weight:900;">{r.get('UD/Line','')}</div></div>
        <div><div class="muted">Edge</div><div style="font-size:22px;font-weight:900;">{r.get('Lean Gap','')}</div></div>
        <div><div class="muted">Conf</div><div style="font-size:22px;font-weight:900;">{r.get('Confidence %','')}%</div></div>
        <div><div class="muted">Volume</div><div style="font-size:22px;font-weight:900;">{r.get('Expected Service Games','')}</div></div>
        <div><div class="muted">Elite</div><div style="font-size:13px;font-weight:900;">{r.get('Elite Tag','')}</div></div>
      </div>
      <div class="muted" style="margin-top:9px;">{r.get('Official Reason','')} · Learning: {r.get('Learning Label','')}</div>
    </div>
    """, unsafe_allow_html=True)

def show_table(df):
    preferred = [
        "Player", "Opponent", "Matchup", "Stat", "Bucket", "TENNIS PROJ", "Floor", "Median", "Ceiling", "Volatility", "Over Sim %", "Under Sim %", "UD/Line", "Decision", "Lean Gap", "Confidence %", "Grade", "Official Filter", "Official Reason",
        "Expected Sets", "3 Set Prob %", "Straight Set Prob %", "Expected Match Games", "Expected Service Games", "Surface", "Indoor/Outdoor", "Tournament Level", "Best Of", "Elite Tag",
        "Rank", "Rank Points", "Player Win %", "Last10 Win %", "Last25 Win %", "Aces/Service Game", "Opponent Ace Allowed/SG", "DF/Service Game", "1st Serve In %", "1st Serve Won %", "2nd Serve Won %", "Service Points Won %", "Serve Effectiveness %", "Unreturned Serve Proxy %",
        "1st Return Won Proxy %", "2nd Return Won Proxy %", "Return Points Won %", "BP Save %", "BP Convert Proxy %", "BP Created/Return Game", "Hold %", "Break %", "Return Games Won %", "Tiebreak Rate", "Tiebreak Win Proxy %",
        "Winner Proxy", "Unforced Error Proxy", "Forced Error Proxy", "Winner/Error Ratio Proxy", "Shot Quality Proxy", "Short Rally Edge Proxy", "Long Rally Edge Proxy", "Avg Rally Length True", "Distance Covered True", "True Metric Source", "Risk Flags", "Risk Tax", "Workload Index", "Matches Last 14", "Rest Days", "Fatigue Tax",
        "Serve Strength", "Return Strength", "Overall Strength", "Opponent Strength", "H2H Matches", "H2H Win %", "Player Matches", "Surface Matches", "Opp Matches", "Learning Adj"
    ]
    use = [c for c in preferred if c in df.columns]
    st.dataframe(df[use], use_container_width=True, height=520)

# ------------------------------ Sidebar / app ------------------------------
st.sidebar.markdown("## 🎾 Tennis Engine Controls")
surface = st.sidebar.selectbox("Surface", ["Hard", "Clay", "Grass", "Carpet", "Unknown"], index=0)
indoor = st.sidebar.selectbox("Indoor / Outdoor", ["Outdoor", "Indoor", "Unknown"], index=0)
tourney_level = st.sidebar.selectbox("Tournament Level", list(TOURNEY_LEVEL_FACTOR.keys()), index=5)
best_of = st.sidebar.selectbox("Best of", [3, 5], index=0)
years_back = st.sidebar.slider("Historical years", 1, 8, 4)
min_conf = st.sidebar.slider("Minimum confidence", 50, 78, 54)
official_only = st.sidebar.toggle("Official PASS only", value=False)
prop_filter = st.sidebar.multiselect("Prop Types", ["ACES", "PLAYER_GAMES", "TOTAL_GAMES", "BREAK_POINTS", "BREAKS", "DOUBLE_FAULTS", "TIEBREAK", "FANTASY_POINTS", "OTHER"], default=["ACES", "PLAYER_GAMES", "TOTAL_GAMES", "BREAK_POINTS", "BREAKS", "DOUBLE_FAULTS"])
page = st.sidebar.radio("Page", ["Dashboard", "Aces", "Games", "Breaks", "Elite Players", "Search", "Upload / Manual", "True Metrics / Status", "Learning Engine", "Logs", "After Grading"])

st.markdown(f'<div class="big-title">{APP_VERSION}</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Underdog lines + ATP/WTA history + charting/status overlays + learning engine + elite tags + official filters.</div>', unsafe_allow_html=True)

with st.spinner("Loading ATP/WTA historical match database..."):
    years = list(range(CURRENT_YEAR - years_back + 1, CURRENT_YEAR + 1))
    matches = load_matches(years)

raw, ud_url, ud_err = fetch_underdog_raw()
ud_lines = parse_underdog(raw)
charting_overlay = read_optional_overlay(CHARTING_FILE)
status_flags = read_optional_overlay(STATUS_FILE)
draw_status = read_optional_overlay(DRAW_FILE)

if "manual_lines" not in st.session_state:
    st.session_state["manual_lines"] = pd.DataFrame()

lines = ud_lines if not ud_lines.empty else st.session_state["manual_lines"]
engine, summaries = build_engine(lines, matches, surface, indoor, tourney_level, best_of, charting_overlay, status_flags, draw_status) if not lines.empty else (pd.DataFrame(), {})

if not engine.empty:
    engine = engine[engine["Bucket"].isin(prop_filter)]
    engine = engine[pd.to_numeric(engine["Confidence %"], errors="coerce").fillna(0) >= min_conf]
    if official_only:
        engine = engine[engine["Official Filter"] == "PASS"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Underdog Lines", len(ud_lines))
c2.metric("Historical Matches", len(matches))
c3.metric("Official PASS", int((engine["Official Filter"] == "PASS").sum()) if not engine.empty else 0)
c4.metric("Elite Profiles", len([s for s in summaries.values() if s.get("elite_tag") not in ["STANDARD", "UNKNOWN_SAMPLE"]]))

if not ud_lines.empty:
    st.success(f"Underdog connected: {ud_url}")
    if st.button("Save Underdog + Player Master Logs"):
        save_master_logs(engine, summaries)
        st.success("Saved master player stats, elite tags, and Underdog line history.")
else:
    st.warning(f"No Underdog tennis board found right now. Use Upload / Manual. Last error: {ud_err}")

if page == "Dashboard":
    st.markdown("### 🟢 Best Board")
    if engine.empty:
        st.info("No current board loaded. Use Upload / Manual if Underdog has no active tennis lines.")
    else:
        for _, r in engine.head(8).iterrows():
            render_card(r)
        st.markdown("### Full Projection Board")
        show_table(engine)
        if st.button("Save Projection Snapshot"):
            snap = engine.copy(); snap["Saved At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            append_csv(SNAPSHOT_FILE, snap)
            save_master_logs(engine, summaries)
            st.success("Snapshot + logs saved.")

elif page == "Aces":
    st.markdown("### 🎯 Aces Engine")
    show_table(engine[engine["Bucket"] == "ACES"] if not engine.empty else pd.DataFrame())

elif page == "Games":
    st.markdown("### 🎾 Games Engine")
    show_table(engine[engine["Bucket"].isin(["PLAYER_GAMES", "TOTAL_GAMES"])] if not engine.empty else pd.DataFrame())

elif page == "Breaks":
    st.markdown("### 🔁 Breaks / Break Points Engine")
    show_table(engine[engine["Bucket"].isin(["BREAK_POINTS", "BREAKS"])] if not engine.empty else pd.DataFrame())

elif page == "Elite Players":
    st.markdown("### ⭐ Elite Player Tags")
    prof = pd.DataFrame([{**{"Player": k}, **v} for k, v in summaries.items()]) if summaries else read_csv_safe(ELITE_FILE)
    if prof.empty:
        st.info("No elite profile log yet. Run a board and save logs.")
    else:
        cols = ["Player", "elite_tag", "rank", "rank_points", "serve_strength", "return_strength", "overall_strength", "service_points_won_pct", "return_points_won_proxy", "hold_pct", "break_pct", "shot_quality_proxy", "workload_index", "reliability"]
        cols = [c for c in cols if c in prof.columns]
        st.dataframe(prof[cols].sort_values(cols[-1] if cols else "Player", ascending=False), use_container_width=True, height=520)

elif page == "Search":
    st.markdown("### 🔎 Search")
    q = st.text_input("Search player, matchup, prop")
    if q and not engine.empty:
        df = engine[engine.astype(str).apply(lambda col: col.str.contains(q, case=False, na=False)).any(axis=1)]
        for _, r in df.head(10).iterrows():
            render_card(r)
        show_table(df)

elif page == "Upload / Manual":
    st.markdown("### Upload / Manual Board")
    st.caption("Required CSV columns: Player, Stat, UD/Line. Optional: Opponent, Matchup.")
    f = st.file_uploader("Upload Tennis Lines CSV", type=["csv"])
    if f is not None:
        df = pd.read_csv(f)
        if {"Player", "Stat", "UD/Line"}.issubset(df.columns):
            if "Opponent" not in df.columns: df["Opponent"] = ""
            if "Matchup" not in df.columns: df["Matchup"] = ""
            df["Bucket"] = df["Stat"].map(prop_bucket)
            df["Line Source"] = "Manual/Upload"
            df["Pulled At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state["manual_lines"] = df
            st.success("Manual board loaded.")
            st.dataframe(df, use_container_width=True)
        else:
            st.error("CSV needs Player, Stat, UD/Line.")
    with st.form("manual_add"):
        p = st.text_input("Player")
        o = st.text_input("Opponent")
        stat = st.selectbox("Prop", ["Aces", "Games Won", "Total Games", "Break Points Won", "Breaks", "Double Faults", "Tiebreaks", "Fantasy Points"])
        line = st.number_input("Underdog Line", min_value=0.0, value=4.5, step=.5)
        matchup = st.text_input("Matchup")
        ok = st.form_submit_button("Add")
        if ok:
            new = pd.DataFrame([{"Player": clean_name(p), "Opponent": clean_name(o), "Matchup": matchup, "Stat": stat, "Bucket": prop_bucket(stat), "UD/Line": line, "Line Source": "Manual", "Pulled At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}])
            st.session_state["manual_lines"] = pd.concat([st.session_state["manual_lines"], new], ignore_index=True)
            st.success("Added manual line.")
    if not st.session_state["manual_lines"].empty:
        st.dataframe(st.session_state["manual_lines"], use_container_width=True)
        if st.button("Clear Manual Lines"):
            st.session_state["manual_lines"] = pd.DataFrame(); st.rerun()

elif page == "True Metrics / Status":
    st.markdown("### 🎾 True Metrics / Live Status Overlay")
    st.caption("Free/realistic layer. Upload Tennis Abstract charting-style data or your own CSV. These true values override proxy winners/errors/rally/distance when available.")
    tab_a, tab_b, tab_c = st.tabs(["Charting True Metrics", "Injury / Retirement Flags", "Draw / Match Status"])
    with tab_a:
        st.write("Accepted columns: Player, Winners, Unforced Errors, Forced Errors, Rally Length, Distance Covered. Extra columns are kept.")
        f = st.file_uploader("Upload charting true metrics CSV", type=["csv"], key="charting_upload")
        if f is not None:
            df = pd.read_csv(f)
            if "Player" in df.columns:
                write_dedup_csv(CHARTING_FILE, df, subset=["Player"] if "Date" not in df.columns else ["Player", "Date"])
                st.success("Charting/true metric overlay saved.")
            else:
                st.error("CSV needs a Player column.")
        st.dataframe(read_csv_safe(CHARTING_FILE).tail(300), use_container_width=True, height=360)
    with tab_b:
        st.write("Use this free fallback for live injuries/retirements when no paid API is connected. Status examples: Active, Questionable, Injury, Retired, Withdrawn.")
        f = st.file_uploader("Upload status flags CSV", type=["csv"], key="status_upload")
        if f is not None:
            df = pd.read_csv(f)
            if "Player" in df.columns:
                write_dedup_csv(STATUS_FILE, df, subset=["Player"])
                st.success("Status flags saved.")
            else:
                st.error("CSV needs a Player column.")
        st.dataframe(read_csv_safe(STATUS_FILE).tail(300), use_container_width=True, height=360)
    with tab_c:
        st.write("Use this for official draw/match status. Status examples: Scheduled, Confirmed, In Draw, Withdrawn, Walkover, Cancelled.")
        f = st.file_uploader("Upload draw status CSV", type=["csv"], key="draw_upload")
        if f is not None:
            df = pd.read_csv(f)
            if "Player" in df.columns:
                write_dedup_csv(DRAW_FILE, df, subset=["Player"] if "Tournament" not in df.columns else ["Player", "Tournament"])
                st.success("Draw status saved.")
            else:
                st.error("CSV needs a Player column.")
        st.dataframe(read_csv_safe(DRAW_FILE).tail(300), use_container_width=True, height=360)

elif page == "Learning Engine":
    st.markdown("### 🧠 Tennis Learning Engine")
    mem = read_csv_safe(LEARNING_FILE)
    if mem.empty:
        st.info("No learning memory yet. Save projections, then upload graded results with Player, Stat, Actual.")
    else:
        mem["Error"] = pd.to_numeric(mem.get("Error"), errors="coerce")
        tabs = st.tabs(["Player Bias", "Prop Bias", "Surface Bias", "Tournament Bias", "Raw Memory"])
        with tabs[0]:
            grp = mem.groupby(["Player", "Bucket"], dropna=False).agg(Sample=("Error", "count"), Avg_Error=("Error", "mean"), Hit_Rate=("Result", lambda x: (x.astype(str)=="WIN").mean()*100)).reset_index()
            st.dataframe(grp[grp["Sample"]>=2].sort_values(["Sample", "Avg_Error"], ascending=[False, False]), use_container_width=True, height=420)
        with tabs[1]:
            grp = mem.groupby(["Bucket"], dropna=False).agg(Sample=("Error", "count"), Avg_Error=("Error", "mean"), Hit_Rate=("Result", lambda x: (x.astype(str)=="WIN").mean()*100)).reset_index()
            st.dataframe(grp.sort_values("Sample", ascending=False), use_container_width=True, height=420)
        with tabs[2]:
            grp = mem.groupby(["Surface", "Bucket"], dropna=False).agg(Sample=("Error", "count"), Avg_Error=("Error", "mean"), Hit_Rate=("Result", lambda x: (x.astype(str)=="WIN").mean()*100)).reset_index()
            st.dataframe(grp.sort_values("Sample", ascending=False), use_container_width=True, height=420)
        with tabs[3]:
            grp = mem.groupby(["Tournament Level", "Bucket"], dropna=False).agg(Sample=("Error", "count"), Avg_Error=("Error", "mean"), Hit_Rate=("Result", lambda x: (x.astype(str)=="WIN").mean()*100)).reset_index()
            st.dataframe(grp.sort_values("Sample", ascending=False), use_container_width=True, height=420)
        with tabs[4]:
            st.dataframe(mem.tail(700), use_container_width=True, height=420)

elif page == "Logs":
    st.markdown("### 📁 Logs")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Player Master Log", "Underdog Line Log", "Elite Tags", "Snapshots", "Charting Overlay", "Status Flags", "Draw Status"])
    with tab1: st.dataframe(read_csv_safe(MASTER_LOG_FILE).tail(500), use_container_width=True, height=480)
    with tab2: st.dataframe(read_csv_safe(UD_LOG_FILE).tail(500), use_container_width=True, height=480)
    with tab3: st.dataframe(read_csv_safe(ELITE_FILE), use_container_width=True, height=480)
    with tab4: st.dataframe(read_csv_safe(SNAPSHOT_FILE).tail(500), use_container_width=True, height=480)
    with tab5: st.dataframe(read_csv_safe(CHARTING_FILE).tail(500), use_container_width=True, height=480)
    with tab6: st.dataframe(read_csv_safe(STATUS_FILE).tail(500), use_container_width=True, height=480)
    with tab7: st.dataframe(read_csv_safe(DRAW_FILE).tail(500), use_container_width=True, height=480)

elif page == "After Grading":
    st.markdown("### ✅ After Grading / Learning")
    st.caption("Upload CSV with Player, Stat, Actual. It will calculate WIN/LOSS and projection error, then feed the learning file.")
    gf = st.file_uploader("Upload graded results CSV", type=["csv"], key="grade")
    if gf is not None:
        gd = pd.read_csv(gf)
        res = grade_from_result_file(gd, engine)
        if res.empty:
            st.error("Need columns Player, Stat, Actual and current board loaded.")
        else:
            st.success("Grades saved to learning memory.")
            st.dataframe(res, use_container_width=True, height=520)
            hit = res["Result"].astype(str).eq("WIN").mean()
            st.metric("Current Upload Hit Rate", f"{hit*100:.1f}%")
    st.markdown("#### Learning Memory")
    st.dataframe(read_csv_safe(LEARNING_FILE).tail(500), use_container_width=True, height=420)

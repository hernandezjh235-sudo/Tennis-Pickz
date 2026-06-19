
# ONE WAY PICKZ — TENNIS V6 CLEAN BOARD
# Clean Streamlit tennis app:
# - Underdog pull attempt
# - Easy manual board / screenshot-style paste fallback
# - Auto surface / best-of / tournament inference
# - Seeded starter profiles if public history is unavailable
# - Tabs by prop type like MLB
# - Learning / grading logs

import io, os, re, json, math, warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

warnings.filterwarnings("ignore")

APP_VERSION = "ONE WAY PICKZ — TENNIS V7 UNDERDOG CLEAN MATCHER"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
SAMPLE_DIR = BASE_DIR / "samples"
for d in [DATA_DIR, LOG_DIR, SAMPLE_DIR]:
    d.mkdir(exist_ok=True)

SNAPSHOT_FILE = LOG_DIR / "projection_snapshots.csv"
GRADE_FILE = LOG_DIR / "graded_results.csv"
LEARNING_FILE = LOG_DIR / "learning_memory.csv"
UD_LOG_FILE = LOG_DIR / "underdog_line_log.csv"
HISTORY_CACHE = DATA_DIR / "tennis_history_cache.csv"
STARTER_PROFILES = DATA_DIR / "starter_player_profiles.csv"

UNDERDOG_ENDPOINTS = [
    "https://api.underdogfantasy.com/beta/v5/over_under_lines",
    "https://api.underdogfantasy.com/beta/v4/over_under_lines",
    "https://api.underdogfantasy.com/beta/v3/over_under_lines",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Origin": "https://underdogfantasy.com",
    "Referer": "https://underdogfantasy.com/",
}

SURFACE_FACTOR = {"Hard": 1.00, "Clay": 0.92, "Grass": 1.13, "Carpet": 1.05, "Unknown": 1.00}
INDOOR_FACTOR = {"Indoor": 1.055, "Outdoor": 1.00, "Unknown": 1.00}
LEVEL_FACTOR = {
    "Grand Slam": 1.08,
    "Masters/WTA 1000": 1.045,
    "ATP/WTA 500": 1.015,
    "ATP/WTA 250": 1.00,
    "Challenger/Qualifier": 0.955,
    "Unknown": 1.00,
}

# ------------------------- UI -------------------------

st.set_page_config(page_title=APP_VERSION, page_icon="🎾", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.stApp{background:#080d12;color:#eafff3}
.big-title{font-size:38px;font-weight:950;color:#00ff88;line-height:1.05}
.sub-title{font-size:14px;color:#a9bbb5;margin-bottom:12px}
.card{background:#101821;border:1px solid #243546;border-radius:18px;padding:16px;margin:10px 0;box-shadow:0 0 18px rgba(0,255,136,.05)}
.kpi{background:#0e151d;border:1px solid #263744;border-radius:16px;padding:16px}
.kpi-v{font-size:28px;font-weight:950;color:#fff}
.kpi-l{font-size:12px;color:#91a49e}
.good{color:#00ff88;font-weight:950}.warn{color:#ffd166;font-weight:950}.bad{color:#ff4d6d;font-weight:950}.muted{color:#91a49e}
.stTabs [data-baseweb="tab"]{font-weight:800}
</style>
""", unsafe_allow_html=True)

# ------------------------- helpers -------------------------

def clean_name(x):
    if x is None:
        return ""
    x = str(x).replace("_", " ").replace("-", " ")
    x = re.sub(r"[^A-Za-zÀ-ÿ' .]", "", x)
    return re.sub(r"\s+", " ", x).strip()

def norm(x):
    return clean_name(x).lower()

def sf(x, default=np.nan):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def prob_over(edge, sigma):
    return 100 * sigmoid(1.702 * (edge / max(sigma, 0.01)))

def append_csv(path, df):
    if df is None or df.empty:
        return
    try:
        old = pd.read_csv(path)
        out = pd.concat([old, df], ignore_index=True)
    except Exception:
        out = df.copy()
    out.to_csv(path, index=False)

def read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def prop_bucket(stat):
    s = str(stat).lower().strip()
    s = s.replace("breakpoints", "break points")
    if "ace" in s:
        return "ACES"
    if "double" in s and "fault" in s:
        return "DOUBLE_FAULTS"
    if "1st" in s and "set" in s and "games played" in s:
        return "FIRST_SET_TOTAL_GAMES"
    if "1st" in s and "set" in s and "games won" in s:
        return "FIRST_SET_PLAYER_GAMES"
    if "first" in s and "set" in s and "games played" in s:
        return "FIRST_SET_TOTAL_GAMES"
    if "first" in s and "set" in s and "games won" in s:
        return "FIRST_SET_PLAYER_GAMES"
    if "sets played" in s:
        return "SETS_PLAYED"
    if "sets won" in s:
        return "SETS_WON"
    if "set" in s and "won" in s:
        return "SETS_WON"
    if "set" in s and "played" in s:
        return "SETS_PLAYED"
    if "break point" in s:
        return "BREAK_POINTS"
    if "break" in s and "tie" not in s:
        return "BREAKS"
    if "tie" in s and "break" in s:
        return "TIEBREAK"
    if "games played" in s or ("total" in s and "game" in s):
        return "TOTAL_GAMES"
    if "games won" in s:
        return "PLAYER_GAMES"
    if "game" in s:
        return "PLAYER_GAMES"
    if "winner" in s or "moneyline" in s or "match result" in s:
        return "MATCH_WINNER"
    return "OTHER"

def infer_match_context(matchup="", tournament=""):
    text = f"{matchup} {tournament}".lower()
    # Surface inference by current common event names.
    grass_words = ["halle", "london", "queen", "queens", "berlin", "wimbledon", "stuttgart", "nottingham", "eastbourne", "mallorca"]
    clay_words = ["roland", "french", "monte carlo", "madrid", "rome", "barcelona", "hamburg", "gstaad", "bastad", "umag", "clay"]
    hard_words = ["australian", "us open", "miami", "indian wells", "cincinnati", "canada", "shanghai", "beijing", "tokyo", "hard"]
    if any(w in text for w in grass_words):
        surface = "Grass"
    elif any(w in text for w in clay_words):
        surface = "Clay"
    elif any(w in text for w in hard_words):
        surface = "Hard"
    else:
        surface = "Unknown"

    if any(w in text for w in ["wimbledon", "roland", "french open", "australian open", "us open"]):
        level = "Grand Slam"
    elif any(w in text for w in ["masters", "1000", "indian wells", "miami", "madrid", "rome", "monte carlo", "cincinnati", "shanghai"]):
        level = "Masters/WTA 1000"
    elif any(w in text for w in ["halle", "berlin", "queen", "queens", "london"]):
        level = "ATP/WTA 500"
    else:
        level = "Unknown"

    best_of = 5 if level == "Grand Slam" and "wta" not in text else 3
    indoor = "Indoor" if "indoor" in text else "Outdoor"
    return surface, best_of, indoor, level

# ------------------------- starter profiles -------------------------

def ensure_starter_profiles():
    if STARTER_PROFILES.exists():
        return
    rows = [
        # starter profiles are fallback baselines so the app never starts at zero when public feeds fail.
        # They can be replaced by history or grading logs as soon as real boards are graded.
        ["Taylor Fritz", 12, 3400, 0.59, 0.62, 0.74, 0.67, 0.52, 0.65, 0.37, 0.82, 0.22, 0.90, 0.22, "ELITE_SERVER,TOP_25"],
        ["Ben Shelton", 14, 3000, 0.58, 0.61, 0.79, 0.68, 0.51, 0.64, 0.36, 0.83, 0.21, 1.02, 0.27, "ELITE_SERVER,TOP_25"],
        ["Alexander Zverev", 3, 6500, 0.68, 0.70, 0.72, 0.69, 0.53, 0.66, 0.38, 0.84, 0.24, 0.72, 0.20, "ELITE_SERVER,TOP_25"],
        ["Raphael Collignon", 90, 700, 0.51, 0.50, 0.63, 0.64, 0.49, 0.60, 0.35, 0.76, 0.20, 0.38, 0.24, "STANDARD"],
        ["Daniel Altmaier", 65, 880, 0.50, 0.49, 0.62, 0.64, 0.50, 0.61, 0.37, 0.77, 0.23, 0.42, 0.25, "STANDARD"],
        ["Carlos Alcaraz", 2, 8500, 0.76, 0.73, 0.67, 0.70, 0.57, 0.68, 0.42, 0.86, 0.31, 0.55, 0.20, "ELITE_RETURNER,TOP_25"],
        ["Jannik Sinner", 1, 10000, 0.79, 0.74, 0.70, 0.72, 0.56, 0.69, 0.41, 0.87, 0.30, 0.66, 0.18, "ELITE_SERVER,ELITE_RETURNER,TOP_25"],
        ["Novak Djokovic", 5, 5200, 0.72, 0.69, 0.66, 0.71, 0.55, 0.68, 0.43, 0.86, 0.31, 0.54, 0.18, "ELITE_RETURNER,TOP_25"],
        ["Daniil Medvedev", 10, 3900, 0.63, 0.61, 0.65, 0.67, 0.52, 0.65, 0.40, 0.82, 0.27, 0.50, 0.22, "TOP_25"],
        ["Hubert Hurkacz", 18, 2400, 0.57, 0.58, 0.82, 0.69, 0.52, 0.66, 0.35, 0.86, 0.19, 1.12, 0.19, "ELITE_SERVER,TOP_25"],
        ["Aryna Sabalenka", 1, 9000, 0.75, 0.71, 0.72, 0.68, 0.51, 0.65, 0.39, 0.83, 0.27, 0.62, 0.32, "ELITE_SERVER,TOP_25"],
        ["Iga Swiatek", 2, 8700, 0.78, 0.74, 0.66, 0.70, 0.54, 0.67, 0.45, 0.85, 0.34, 0.38, 0.17, "ELITE_RETURNER,TOP_25"],
        ["Coco Gauff", 3, 6500, 0.70, 0.68, 0.64, 0.66, 0.52, 0.64, 0.42, 0.81, 0.31, 0.42, 0.28, "ELITE_RETURNER,TOP_25"],
    ]
    cols = ["Player","Rank","Rank Points","Win %","Last10 Win %","First Serve %","First Won %","Second Won %","Serve Pts Won %","Return Pts Won %","Hold %","Break %","Ace/SG","DF/SG","Elite Tag"]
    pd.DataFrame(rows, columns=cols).to_csv(STARTER_PROFILES, index=False)

ensure_starter_profiles()

def default_profile(player):
    return {
        "player": player, "source": "DEFAULT", "rank": np.nan, "rank_points": np.nan,
        "win_pct": .50, "last10_win_pct": .50, "first_in_pct": .63,
        "first_win_pct": .66, "second_win_pct": .50, "service_points_won_pct": .61,
        "return_points_won_pct": .36, "hold_pct": .78, "break_pct": .22,
        "ace_per_service_game": .55, "df_per_service_game": .24,
        "serve_strength": 62, "return_strength": 36, "overall_strength": 50,
        "elite_tag": "NO_HISTORY", "reliability": 35,
    }

def starter_profile(player):
    df = pd.read_csv(STARTER_PROFILES)
    p = norm(player)
    if p:
        sub = df[df["Player"].astype(str).map(norm).str.contains(p, regex=False, na=False)]
        if sub.empty:
            last = p.split(" ")[-1]
            sub = df[df["Player"].astype(str).map(norm).str.contains(last, regex=False, na=False)]
        if not sub.empty:
            r = sub.iloc[0]
            spw = sf(r["Serve Pts Won %"], .61)
            rpw = sf(r["Return Pts Won %"], .36)
            return {
                "player": r["Player"], "source": "STARTER_PROFILE", "rank": sf(r["Rank"]), "rank_points": sf(r["Rank Points"]),
                "win_pct": sf(r["Win %"], .50), "last10_win_pct": sf(r["Last10 Win %"], .50),
                "first_in_pct": sf(r["First Serve %"], .63), "first_win_pct": sf(r["First Won %"], .66),
                "second_win_pct": sf(r["Second Won %"], .50), "service_points_won_pct": spw,
                "return_points_won_pct": rpw, "hold_pct": sf(r["Hold %"], .78), "break_pct": sf(r["Break %"], .22),
                "ace_per_service_game": sf(r["Ace/SG"], .55), "df_per_service_game": sf(r["DF/SG"], .24),
                "serve_strength": 100*spw + 8*sf(r["Ace/SG"], .55) - 2*sf(r["DF/SG"], .24),
                "return_strength": 100*rpw + 14*sf(r["Break %"], .22),
                "overall_strength": 52*spw + 48*rpw + (0 if pd.isna(sf(r["Rank"])) else clamp((75-sf(r["Rank"]))/18, -3, 3)),
                "elite_tag": str(r["Elite Tag"]), "reliability": 55,
            }
    return default_profile(player)

# ------------------------- history -------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_history(years_back=4):
    if HISTORY_CACHE.exists():
        try:
            c = pd.read_csv(HISTORY_CACHE, low_memory=False)
            if len(c) > 1000:
                return c, "LOCAL_CACHE", ""
        except Exception:
            pass

    current = datetime.now().year
    years = list(range(current - years_back, current + 1))
    frames, errors = [], []
    # Try both raw.githubusercontent and jsdelivr. Some hosts block one but not the other.
    for tour in ["atp", "wta"]:
        repo = "tennis_atp" if tour == "atp" else "tennis_wta"
        pref = "atp" if tour == "atp" else "wta"
        for y in years:
            urls = [
                f"https://raw.githubusercontent.com/JeffSackmann/{repo}/master/{pref}_matches_{y}.csv",
                f"https://cdn.jsdelivr.net/gh/JeffSackmann/{repo}@master/{pref}_matches_{y}.csv",
            ]
            ok = False
            for url in urls:
                try:
                    r = requests.get(url, headers={"User-Agent": HEADERS["User-Agent"]}, timeout=16)
                    if r.status_code == 200 and len(r.text) > 1000 and "winner_name" in r.text[:2000]:
                        d = pd.read_csv(io.StringIO(r.text), low_memory=False)
                        d["tour"] = tour.upper()
                        frames.append(d)
                        ok = True
                        break
                    else:
                        errors.append(f"{pref} {y}: HTTP {r.status_code}")
                except Exception as e:
                    errors.append(f"{pref} {y}: {str(e)[:60]}")
            if not ok:
                continue

    if frames:
        df = pd.concat(frames, ignore_index=True)
        df["tourney_date"] = pd.to_numeric(df.get("tourney_date"), errors="coerce")
        df = df.sort_values("tourney_date").reset_index(drop=True)
        try:
            df.to_csv(HISTORY_CACHE, index=False)
        except Exception:
            pass
        return df, "JEFF_SACKMANN", " | ".join(errors[-6:])
    return pd.DataFrame(), "STARTER_PROFILES_ONLY", "Public history unavailable. Using starter player profiles."

def row_for(r, won):
    p = "w" if won else "l"
    op = "l" if won else "w"
    name = "winner" if won else "loser"
    opp = "loser" if won else "winner"
    return {
        "date": sf(r.get("tourney_date")),
        "surface": r.get("surface", "Unknown") or "Unknown",
        "won": int(won),
        "player_name": r.get(f"{name}_name", ""),
        "opp_name": r.get(f"{opp}_name", ""),
        "rank": sf(r.get(f"{p}_rank")),
        "rank_points": sf(r.get(f"{p}_rank_points")),
        "aces": sf(r.get(f"{p}_ace"), 0),
        "df": sf(r.get(f"{p}_df"), 0),
        "svpt": sf(r.get(f"{p}_svpt")),
        "first_in": sf(r.get(f"{p}_1stIn")),
        "first_won": sf(r.get(f"{p}_1stWon")),
        "second_won": sf(r.get(f"{p}_2ndWon")),
        "service_games": sf(r.get(f"{p}_SvGms")),
        "bp_saved": sf(r.get(f"{p}_bpSaved")),
        "bp_faced": sf(r.get(f"{p}_bpFaced")),
        "opp_svpt": sf(r.get(f"{op}_svpt")),
        "opp_first_in": sf(r.get(f"{op}_1stIn")),
        "opp_first_won": sf(r.get(f"{op}_1stWon")),
        "opp_second_won": sf(r.get(f"{op}_2ndWon")),
        "opp_bp_saved": sf(r.get(f"{op}_bpSaved")),
        "opp_bp_faced": sf(r.get(f"{op}_bpFaced")),
        "score": r.get("score", ""),
        "best_of": sf(r.get("best_of"), 3),
    }

def player_rows(hist, player, limit=180):
    if hist.empty or not player or "winner_name" not in hist:
        return pd.DataFrame()
    p = norm(player)
    w = hist[hist["winner_name"].astype(str).map(norm).str.contains(p, regex=False, na=False)]
    l = hist[hist["loser_name"].astype(str).map(norm).str.contains(p, regex=False, na=False)]
    rows = [row_for(r, True) for _, r in w.iterrows()] + [row_for(r, False) for _, r in l.iterrows()]
    if not rows:
        last = p.split(" ")[-1] if p else ""
        if len(last) >= 4:
            w = hist[hist["winner_name"].astype(str).map(norm).str.contains(last, regex=False, na=False)]
            l = hist[hist["loser_name"].astype(str).map(norm).str.contains(last, regex=False, na=False)]
            rows = [row_for(r, True) for _, r in w.iterrows()] + [row_for(r, False) for _, r in l.iterrows()]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("date").tail(limit).reset_index(drop=True)

def summarize(hist, player, surface="Unknown"):
    rows = player_rows(hist, player)
    if rows.empty:
        return starter_profile(player)

    surf = rows[rows["surface"].astype(str).str.lower() == surface.lower()] if surface != "Unknown" else rows
    use = surf if len(surf) >= 6 else rows
    last10 = rows.tail(10)
    sg = use["service_games"].replace(0, np.nan)
    svpt = use["svpt"].replace(0, np.nan)
    sv_sum = svpt.sum()
    first_in = use["first_in"].sum()
    second_pts = sv_sum - first_in

    ace_sg = use["aces"].sum() / sg.sum() if sg.sum() and not pd.isna(sg.sum()) else starter_profile(player)["ace_per_service_game"]
    df_sg = use["df"].sum() / sg.sum() if sg.sum() and not pd.isna(sg.sum()) else starter_profile(player)["df_per_service_game"]
    first_in_pct = first_in / sv_sum if sv_sum and not pd.isna(sv_sum) else .63
    first_win = use["first_won"].sum() / first_in if first_in and not pd.isna(first_in) else .66
    second_win = use["second_won"].sum() / second_pts if second_pts and not pd.isna(second_pts) else .50
    spw = (use["first_won"].sum() + use["second_won"].sum()) / sv_sum if sv_sum and not pd.isna(sv_sum) else .61

    opp_sv = use["opp_svpt"].replace(0, np.nan).sum()
    opp_first = use["opp_first_in"].sum()
    opp_second = opp_sv - opp_first
    rpw = 1 - ((use["opp_first_won"].sum() + use["opp_second_won"].sum()) / opp_sv) if opp_sv and not pd.isna(opp_sv) else .36

    bp_save = use["bp_saved"].sum() / use["bp_faced"].sum() if use["bp_faced"].sum() and not pd.isna(use["bp_faced"].sum()) else .60
    bp_convert = (use["opp_bp_faced"].sum() - use["opp_bp_saved"].sum()) / use["opp_bp_faced"].sum() if use["opp_bp_faced"].sum() and not pd.isna(use["opp_bp_faced"].sum()) else .39

    hold = clamp(.49 + .58 * spw + .06 * bp_save - .04 * df_sg, .48, .94)
    brk = clamp(.10 + 1.75 * (rpw - .34) + .08 * bp_convert, .06, .48)

    win = float(use["won"].mean())
    last10w = float(last10["won"].mean()) if len(last10) else win
    rank = use["rank"].dropna().tail(1).mean() if use["rank"].notna().any() else np.nan
    rank_points = use["rank_points"].dropna().tail(1).mean() if use["rank_points"].notna().any() else np.nan

    serve = 100 * spw + 8 * ace_sg - 2 * df_sg
    ret = 100 * rpw + 14 * brk
    rank_adj = 0 if pd.isna(rank) else clamp((75 - rank) / 18, -3, 3)
    overall = 52 * spw + 48 * rpw + rank_adj

    tags = []
    if spw >= .65 or ace_sg >= .78 or hold >= .84:
        tags.append("ELITE_SERVER")
    if rpw >= .40 or brk >= .29:
        tags.append("ELITE_RETURNER")
    if not pd.isna(rank) and rank <= 25:
        tags.append("TOP_25")
    if last10w >= .70:
        tags.append("HOT_FORM")

    return {
        "player": player, "source": "MATCH_HISTORY", "rank": rank, "rank_points": rank_points,
        "win_pct": win, "last10_win_pct": last10w, "first_in_pct": first_in_pct,
        "first_win_pct": first_win, "second_win_pct": second_win, "service_points_won_pct": spw,
        "return_points_won_pct": rpw, "hold_pct": hold, "break_pct": brk,
        "ace_per_service_game": ace_sg, "df_per_service_game": df_sg,
        "serve_strength": serve, "return_strength": ret, "overall_strength": overall,
        "elite_tag": ",".join(tags) if tags else "STANDARD",
        "reliability": clamp(42 + len(use) * .8 + len(rows) * .18, 35, 96),
    }

# ------------------------- Underdog parser -------------------------

def stat_title(line):
    for k in ["stat", "stat_type", "stat_type_display", "display_stat", "title", "stat_title", "name"]:
        v = line.get(k) if isinstance(line, dict) else None
        if isinstance(v, str) and v:
            return v
        if isinstance(v, dict):
            for kk in ["display_stat", "stat", "name", "title"]:
                if v.get(kk):
                    return str(v.get(kk))
    ou = line.get("over_under") if isinstance(line.get("over_under"), dict) else {}
    app = ou.get("appearance_stat") if isinstance(ou.get("appearance_stat"), dict) else {}
    for kk in ["display_stat", "stat", "name", "title"]:
        if app.get(kk):
            return str(app.get(kk))
    return "Unknown"

def line_value(line):
    vals = [line.get("stat_value"), line.get("line"), line.get("value")]
    ou = line.get("over_under") if isinstance(line.get("over_under"), dict) else {}
    vals += [ou.get("stat_value"), ou.get("line")]
    for x in vals:
        v = sf(x)
        if not pd.isna(v):
            return v
    return np.nan

def idx(items):
    return {str(x.get("id")): x for x in (items or []) if isinstance(x, dict) and x.get("id") is not None}

@st.cache_data(ttl=120, show_spinner=False)
def fetch_underdog():
    last = ""
    for url in UNDERDOG_ENDPOINTS:
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            if r.status_code == 200 and r.text.strip().startswith("{"):
                return r.json(), url, ""
            last = f"{url} HTTP {r.status_code}"
        except Exception as e:
            last = f"{url}: {str(e)[:90]}"
    return {}, "", last

def parse_underdog(payload):
    if not payload:
        return pd.DataFrame()
    data = payload.get("data", payload)
    if isinstance(data, dict):
        lines = data.get("over_under_lines") or data.get("lines") or data.get("over_unders") or []
        apps = data.get("appearances") or []
        players = data.get("players") or []
        games = data.get("games") or data.get("solo_games") or []
    elif isinstance(data, list):
        lines, apps, players, games = data, [], [], []
    else:
        return pd.DataFrame()

    ai, pi, gi = idx(apps), idx(players), idx(games)
    rows = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        ou = line.get("over_under") if isinstance(line.get("over_under"), dict) else {}
        ast = ou.get("appearance_stat") if isinstance(ou.get("appearance_stat"), dict) else {}
        aid = ast.get("appearance_id") or line.get("appearance_id") or ou.get("appearance_id")
        app = ai.get(str(aid), {}) if aid is not None else {}
        pid = app.get("player_id") or line.get("player_id") or line.get("participant_id")
        pl = pi.get(str(pid), {}) if pid is not None else {}

        name = pl.get("display_name") or pl.get("name") or app.get("player_name") or line.get("player_name") or line.get("title") or ""
        stat = stat_title(line)
        bucket = prop_bucket(stat)

        gid = app.get("match_id") or app.get("game_id") or line.get("game_id")
        g = gi.get(str(gid), {}) if gid is not None else {}
        matchup = g.get("title") or g.get("match_title") or g.get("name") or app.get("matchup") or line.get("matchup") or ""
        tournament = g.get("sport_title") or g.get("league") or g.get("tournament") or line.get("tournament") or ""

        full = (json.dumps(line) + json.dumps(app) + json.dumps(pl) + json.dumps(g)).lower()
        is_tennis = (
            "tennis" in full or "atp" in full or "wta" in full
            or bucket in ["ACES","PLAYER_GAMES","TOTAL_GAMES","FIRST_SET_TOTAL_GAMES","FIRST_SET_PLAYER_GAMES","BREAK_POINTS","BREAKS","SETS_WON","SETS_PLAYED","DOUBLE_FAULTS","MATCH_WINNER"]
        )
        if not is_tennis:
            continue
        v = line_value(line)
        if clean_name(name) and not pd.isna(v):
            rows.append({
                "Player": clean_name(name),
                "Opponent": "",
                "Matchup": matchup,
                "Tournament": tournament,
                "Stat": stat,
                "Bucket": bucket,
                "UD/Line": v,
                "Line Source": "Underdog",
                "Start Time": g.get("scheduled_at") or app.get("scheduled_at") or line.get("scheduled_at") or "",
                "Raw ID": line.get("id", ""),
                "Pulled At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["Player"].astype(str).str.len() > 1]
    return df.drop_duplicates(subset=["Player","Stat","UD/Line","Matchup"]).reset_index(drop=True)



def flatten_json_records(obj):
    """Find likely prop rows inside unknown provider JSON."""
    found = []
    def walk(x):
        if isinstance(x, dict):
            keys = set(k.lower() for k in x.keys())
            if any(k in keys for k in ["player", "player_name", "name", "participant_name"]) and any(k in keys for k in ["line", "stat_value", "projection", "value"]):
                found.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(obj)
    return found

def parse_external_lines_payload(payload, source_name="External API/JSON"):
    """
    Flexible normalizer for Apify/SharpAPI/CSV-like JSON.
    Expected fields can be any of:
    player/player_name/name, stat/market/stat_type, line/projection/value,
    opponent/matchup/event/tournament.
    """
    if not payload:
        return pd.DataFrame()
    if isinstance(payload, list):
        records = payload
    else:
        records = flatten_json_records(payload)

    rows = []
    for r in records:
        if not isinstance(r, dict):
            continue
        player = clean_name(
            r.get("player") or r.get("player_name") or r.get("participant_name") or
            r.get("name") or r.get("athlete") or r.get("display_name") or ""
        )
        stat = (
            r.get("stat") or r.get("stat_type") or r.get("market") or r.get("prop") or
            r.get("selection") or r.get("display_stat") or r.get("bet_type") or ""
        )
        val = sf(r.get("line") or r.get("stat_value") or r.get("projection") or r.get("value") or r.get("handicap"))
        opponent = clean_name(r.get("opponent") or r.get("opponent_name") or "")
        matchup = r.get("matchup") or r.get("event") or r.get("game") or r.get("fixture") or ""
        tournament = r.get("tournament") or r.get("league") or r.get("competition") or ""
        if not player or not stat or pd.isna(val):
            continue
        # Keep tennis only if source provides league/sport OR if prop name is tennis-style.
        sport_blob = " ".join(str(r.get(k, "")) for k in ["sport","league","competition","tournament","event","matchup"]).lower()
        b = prop_bucket(stat)
        is_tennis = ("tennis" in sport_blob or "atp" in sport_blob or "wta" in sport_blob or b != "OTHER")
        if not is_tennis:
            continue
        rows.append({
            "Player": player,
            "Opponent": opponent,
            "Matchup": matchup,
            "Tournament": tournament,
            "Stat": stat,
            "Bucket": b,
            "UD/Line": val,
            "Line Source": source_name,
            "Start Time": r.get("start_time") or r.get("startTime") or r.get("commence_time") or "",
            "Raw ID": r.get("id", ""),
            "Pulled At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame()

@st.cache_data(ttl=120, show_spinner=False)
def fetch_custom_json_url(url):
    if not url:
        return pd.DataFrame(), ""
    try:
        r = requests.get(url, headers=HEADERS, timeout=18)
        if r.status_code != 200:
            return pd.DataFrame(), f"Custom URL HTTP {r.status_code}"
        payload = r.json()
        return parse_external_lines_payload(payload, "Custom JSON/API"), ""
    except Exception as e:
        return pd.DataFrame(), str(e)[:180]

# ------------------------- manual board -------------------------

def parse_paste(text):
    """
    Supported:
    Taylor Fritz, Aces, 13.5, Ben Shelton, ATP Halle
    Taylor Fritz - Aces - 13.5 - Ben Shelton - ATP Halle
    Ben Shelton vs Taylor Fritz | Taylor Fritz | Aces | 13.5
    """
    rows = []
    for line in str(text).splitlines():
        s = line.strip()
        if not s:
            continue
        parts = [p.strip() for p in re.split(r"\||,|\s+—\s+|\s+-\s+", s) if p.strip()]
        if len(parts) < 3:
            continue

        if " vs " in parts[0].lower() and len(parts) >= 4:
            matchup = parts[0]
            player = clean_name(parts[1])
            stat = parts[2]
            val = sf(parts[3])
            other_names = re.split(r"\s+vs\s+", matchup, flags=re.I)
            opp = ""
            for n in other_names:
                if norm(n) != norm(player):
                    opp = clean_name(n)
            tournament = parts[4] if len(parts) >= 5 else ""
        else:
            player = clean_name(parts[0])
            stat = parts[1]
            val = sf(parts[2])
            opp = clean_name(parts[3]) if len(parts) >= 4 else ""
            tournament = parts[4] if len(parts) >= 5 else ""
            matchup = f"{player} vs {opp}" if opp else "Manual"

        if player and not pd.isna(val):
            rows.append({
                "Player": player,
                "Opponent": opp,
                "Matchup": matchup,
                "Tournament": tournament,
                "Stat": stat,
                "Bucket": prop_bucket(stat),
                "UD/Line": val,
                "Line Source": "Manual",
                "Start Time": "",
                "Raw ID": "",
                "Pulled At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
    return pd.DataFrame(rows)

def infer_opp(row):
    if str(row.get("Opponent", "")).strip():
        return clean_name(row.get("Opponent", ""))
    m = str(row.get("Matchup", ""))
    p = norm(row.get("Player", ""))
    chunks = [clean_name(c) for c in re.split(r"\s+v(?:s\.?|ersus)?\s+| @ | / |,", m, flags=re.I) if clean_name(c)]
    for c in chunks:
        if norm(c) != p and p not in norm(c):
            return c
    return ""

# ------------------------- learning + projection -------------------------

def learning_bias(player, bucket, surface):
    mem = read_csv(LEARNING_FILE)
    if mem.empty or "Error" not in mem.columns:
        return 0, "NO_LEARNING"
    mem["Error"] = pd.to_numeric(mem["Error"], errors="coerce")
    vals = []
    masks = [
        ("PLAYER_PROP", (mem.get("Player", "").astype(str).map(norm) == norm(player)) & (mem.get("Bucket", "").astype(str) == bucket), 3, .60),
        ("PROP", mem.get("Bucket", "").astype(str) == bucket, 8, .35),
        ("SURFACE", mem.get("Surface", "").astype(str) == surface, 8, .25),
    ]
    for label, mask, minn, cap in masks:
        sub = mem[mask].dropna(subset=["Error"])
        if len(sub) >= minn:
            vals.append((label, clamp(float(sub["Error"].tail(40).mean()) * .35, -cap, cap)))
    if not vals:
        return 0, "NO_LEARNING"
    return round(sum(v for _, v in vals), 3), "+".join(k for k, _ in vals)

def project(row, hist):
    player = row.get("Player", "")
    opp_name = infer_opp(row)
    surface, best_of, indoor, level = infer_match_context(row.get("Matchup", ""), row.get("Tournament", ""))
    p = summarize(hist, player, surface)
    o = summarize(hist, opp_name, surface) if opp_name else default_profile("Unknown")

    bucket = row.get("Bucket", prop_bucket(row.get("Stat", "")))
    line = sf(row.get("UD/Line"))
    strength_gap = p["overall_strength"] - o["overall_strength"]
    close = 1 - min(abs(strength_gap) / 28, .38)
    sets = 2.10 + .42 * close if best_of == 3 else 3.35 + .80 * close
    sets = clamp(sets, 2.0, 3.0 if best_of == 3 else 5.0)
    service_games = (5.0 * sets) + (.55 * close) + (.25 if strength_gap > 4 else 0)
    sfac = SURFACE_FACTOR.get(surface, 1) * INDOOR_FACTOR.get(indoor, 1) * LEVEL_FACTOR.get(level, 1)

    win_prob = clamp(.50 + strength_gap / 42, .18, .82)
    three_set_prob = clamp(.18 + .42 * close, .16, .62) if best_of == 3 else clamp(.34 + .46 * close, .25, .78)
    first_set_total = clamp(9.35 + 2.1 * close + (0.25 if surface == "Grass" else 0) - (0.15 if surface == "Clay" else 0), 8.4, 12.9)
    first_set_player_games = clamp(first_set_total * win_prob + 0.15 * np.sign(strength_gap), 2.6, 7.2)

    if bucket == "ACES":
        # Aces = ace rate per service game * projected service games * surface/indoor/event context.
        # Grass/indoor raises ace volume; elite returners reduce it.
        return_tax = 1 - clamp((o["return_points_won_pct"] - .37) * .25, -.08, .10)
        proj = p["ace_per_service_game"] * service_games * sfac * return_tax
        sigma = clamp(1.15 + proj * .32, 1.35, 4.8)
    elif bucket == "DOUBLE_FAULTS":
        pressure = 1.06 if o["return_points_won_pct"] > .39 else 1.0
        proj = p["df_per_service_game"] * service_games * pressure
        sigma = clamp(.85 + proj * .40, 1.0, 3.2)
    elif bucket == "PLAYER_GAMES":
        proj = 4.85 * sets + clamp(strength_gap / 17, -1.45, 1.45)
        sigma = 2.15 if best_of == 3 else 3.3
    elif bucket == "TOTAL_GAMES":
        proj = 9.65 * sets + 1.85 * close
        sigma = 3.25 if best_of == 3 else 5.1
    elif bucket == "FIRST_SET_TOTAL_GAMES":
        proj = first_set_total
        sigma = 1.45
    elif bucket == "FIRST_SET_PLAYER_GAMES":
        proj = first_set_player_games
        sigma = 1.25
    elif bucket == "SETS_WON":
        # Expected sets won. For 0.5 lines, this acts like chance to win at least one set.
        straight_loss_risk = clamp((.50 - win_prob) * 1.35 + (1-close) * .22, .04, .70)
        proj = clamp((1 - straight_loss_risk) * (1 + 0.65 * three_set_prob), .12, 2.65 if best_of == 3 else 4.65)
        sigma = .72
    elif bucket == "SETS_PLAYED":
        proj = 2 + three_set_prob if best_of == 3 else 3.1 + 1.4 * three_set_prob
        sigma = .62 if best_of == 3 else .95
    elif bucket in ["BREAKS", "BREAK_POINTS"]:
        opp_weak = clamp((.80 - o["hold_pct"]) * 2.0 + (p["return_points_won_pct"] - .36) * 2.2, -.55, .95)
        base = (.95 * sets) if bucket == "BREAKS" else (2.15 * sets)
        proj = base * (1 + .26 * opp_weak)
        sigma = clamp(1.15 + proj * .52, 1.3, 4.1)
    elif bucket == "MATCH_WINNER":
        # For ML rows, Projection is win probability, line is informational if no true decimal/American edge is available.
        proj = clamp(50 + strength_gap * 1.65, 15, 85)
        sigma = 14
    else:
        proj = np.nan
        sigma = 2

    bias, bias_src = learning_bias(player, bucket, surface)
    if not pd.isna(proj) and bucket != "MATCH_WINNER":
        proj += bias
    edge = proj - line if not pd.isna(proj) and not pd.isna(line) else np.nan
    over = prob_over(edge, sigma) if not pd.isna(edge) else np.nan
    under = 100 - over if not pd.isna(over) else np.nan
    decision = "OVER" if not pd.isna(edge) and edge > 0 else "UNDER" if not pd.isna(edge) else "NO PLAY"
    conf = max(over, under) if not pd.isna(over) else np.nan

    reasons = []
    official = True
    if p["reliability"] < 42:
        official = False; reasons.append("LOW_SAMPLE")
    if bucket == "TIEBREAK":
        official = False; reasons.append("TIEBREAK_VOLATILE")
    if bucket == "OTHER":
        official = False; reasons.append("UNSUPPORTED")
    if bucket == "ACES" and (abs(edge) < .60 or conf < 56):
        official = False; reasons.append("ACE_EDGE_LOW")
    elif bucket in ["PLAYER_GAMES","TOTAL_GAMES","FIRST_SET_TOTAL_GAMES","FIRST_SET_PLAYER_GAMES","SETS_WON","SETS_PLAYED"] and (abs(edge) < .55 or conf < 56):
        official = False; reasons.append("GAME_EDGE_LOW")
    elif bucket in ["BREAKS","BREAK_POINTS","DOUBLE_FAULTS"] and (abs(edge) < .65 or conf < 57):
        official = False; reasons.append("VOL_EDGE_LOW")
    elif bucket == "SETS_WON" and (abs(edge) < .16 or conf < 56):
        official = False; reasons.append("SET_EDGE_LOW")
    elif bucket == "MATCH_WINNER" and conf < 56:
        official = False; reasons.append("ML_EDGE_LOW")

    if official and conf >= 66:
        grade = "S 🔒"
    elif official and conf >= 61:
        grade = "A"
    elif official:
        grade = "B"
    else:
        grade = "C / WATCH"

    return {
        **row.to_dict(),
        "Opponent": opp_name,
        "Surface": surface,
        "Best Of": best_of,
        "Indoor": indoor,
        "Tournament Level": level,
        "Projection": round(proj, 2) if not pd.isna(proj) else np.nan,
        "Floor": round(proj - sigma, 2) if not pd.isna(proj) else np.nan,
        "Ceiling": round(proj + sigma, 2) if not pd.isna(proj) else np.nan,
        "Over Sim %": round(over, 1) if not pd.isna(over) else np.nan,
        "Under Sim %": round(under, 1) if not pd.isna(under) else np.nan,
        "Decision": decision,
        "Lean Gap": round(edge, 2) if not pd.isna(edge) else np.nan,
        "Confidence %": round(conf, 1) if not pd.isna(conf) else np.nan,
        "Official Filter": "PASS" if official else "NO PLAY",
        "Official Reason": "PASS" if official else ",".join(reasons[:4]),
        "Grade": grade,
        "Reliability": round(p["reliability"], 1),
        "Learning Adj": bias,
        "Learning Source": bias_src,
        "Profile Source": p["source"],
        "Rank": p["rank"],
        "Elite Tag": p["elite_tag"],
        "Ace/SG": round(p["ace_per_service_game"], 3),
        "DF/SG": round(p["df_per_service_game"], 3),
        "1st Serve %": round(100 * p["first_in_pct"], 1),
        "1st Won %": round(100 * p["first_win_pct"], 1),
        "2nd Won %": round(100 * p["second_win_pct"], 1),
        "Serve Pts Won %": round(100 * p["service_points_won_pct"], 1),
        "Return Pts Won %": round(100 * p["return_points_won_pct"], 1),
        "Hold %": round(100 * p["hold_pct"], 1),
        "Break %": round(100 * p["break_pct"], 1),
    }

def run_engine(board, hist, min_conf=50, official_only=False):
    if board.empty:
        return pd.DataFrame()
    rows = [project(r, hist) for _, r in board.iterrows()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Abs Edge"] = pd.to_numeric(df["Lean Gap"], errors="coerce").abs()
    df = df[pd.to_numeric(df["Confidence %"], errors="coerce").fillna(0) >= min_conf]
    if official_only:
        df = df[df["Official Filter"] == "PASS"]
    return df.sort_values(["Official Filter","Confidence %","Abs Edge"], ascending=[True, False, False]).reset_index(drop=True)

def render_card(r):
    cls = "good" if r.get("Official Filter") == "PASS" else "warn"
    st.markdown(f"""
    <div class='card'>
      <div style='display:flex;justify-content:space-between;gap:10px'>
        <div>
          <b style='font-size:22px'>{r.get('Player','')} — {r.get('Stat','')}</b>
          <div class='muted'>{r.get('Matchup','')} {('vs '+r.get('Opponent','')) if r.get('Opponent','') else ''}</div>
        </div>
        <div style='text-align:right'>
          <div class='{cls}' style='font-size:24px'>{r.get('Decision','')}</div>
          <div>{r.get('Grade','')}</div>
        </div>
      </div>
      <hr>
      <div style='display:grid;grid-template-columns:repeat(6,1fr);gap:8px'>
        <div><span class='muted'>Proj</span><br><b>{r.get('Projection','')}</b></div>
        <div><span class='muted'>Line</span><br><b>{r.get('UD/Line','')}</b></div>
        <div><span class='muted'>Edge</span><br><b>{r.get('Lean Gap','')}</b></div>
        <div><span class='muted'>Conf</span><br><b>{r.get('Confidence %','')}%</b></div>
        <div><span class='muted'>Official</span><br><b>{r.get('Official Filter','')}</b></div>
        <div><span class='muted'>Source</span><br><b>{r.get('Profile Source','')}</b></div>
      </div>
      <div class='muted' style='margin-top:8px'>Reason: {r.get('Official Reason','')} | Surface: {r.get('Surface','')} | {r.get('Elite Tag','')}</div>
    </div>
    """, unsafe_allow_html=True)

def show_table(df):
    cols = ["Player","Opponent","Matchup","Tournament","Stat","Bucket","Projection","UD/Line","Decision","Lean Gap","Confidence %","Official Filter","Grade","Profile Source","Surface","Rank","Elite Tag","Ace/SG","DF/SG","1st Serve %","Serve Pts Won %","Return Pts Won %","Hold %","Break %","Learning Adj"]
    st.dataframe(df[[c for c in cols if c in df.columns]], use_container_width=True, height=520)

# ------------------------- page state -------------------------

if "manual_board" not in st.session_state:
    st.session_state.manual_board = pd.DataFrame()

with st.sidebar:
    st.header("⚙️ Simple Controls")
    min_conf = st.slider("Minimum confidence", 50, 75, 54)
    official_only = st.toggle("Official PASS only", False)
    custom_url = st.text_input("Optional Underdog/API JSON URL", value="", help="Use this only if you have a stable provider/API URL. The app will normalize the JSON into tennis lines.")
    if st.button("Clear manual board"):
        st.session_state.manual_board = pd.DataFrame()
        st.rerun()
    st.caption("V7 auto-detects surface, best-of, indoor/outdoor, and tournament level from the match/tournament text.")

st.markdown(f"<div class='big-title'>{APP_VERSION}</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Clean board: Underdog attempt + easy manual fallback + automatic tennis context + tabs by prop.</div>", unsafe_allow_html=True)

hist, hist_source, hist_err = load_history(4)
payload, ud_url, ud_err = fetch_underdog()
ud_board = parse_underdog(payload)
custom_board, custom_err = fetch_custom_json_url(custom_url) if 'custom_url' in globals() and custom_url else (pd.DataFrame(), "")
if not custom_board.empty:
    append_csv(UD_LOG_FILE, custom_board)
if not ud_board.empty:
    append_csv(UD_LOG_FILE, ud_board)

# Priority: Custom JSON/API URL > direct Underdog attempt > manual board.
board = custom_board if not custom_board.empty else (ud_board if not ud_board.empty else st.session_state.manual_board)
engine = run_engine(board, hist, min_conf=min_conf, official_only=official_only) if not board.empty else pd.DataFrame()

k1,k2,k3,k4,k5 = st.columns(5)
metrics = [
    ("Auto Lines", len(ud_board)),
    ("History Rows", len(hist)),
    ("Board Lines", len(board)),
    ("Official PASS", int((engine.get("Official Filter", pd.Series(dtype=str)) == "PASS").sum()) if not engine.empty else 0),
    ("Elite Profiles", int(engine["Elite Tag"].astype(str).str.contains("ELITE|TOP_25", na=False).sum()) if not engine.empty else 0),
]
for col, (label, val) in zip([k1,k2,k3,k4,k5], metrics):
    col.markdown(f"<div class='kpi'><div class='kpi-v'>{val}</div><div class='kpi-l'>{label}</div></div>", unsafe_allow_html=True)

if hist.empty:
    st.info("Public ATP/WTA history is not loaded, so V6 is using starter player profiles + grading memory. The app will still project known players and learn from your results.")
elif hist_source != "JEFF_SACKMANN":
    st.warning(f"History source: {hist_source}. {hist_err}")

if ud_board.empty:
    st.warning("Auto line pull returned 0 here. Use Board Builder paste, upload CSV, or add a stable API/JSON URL in the sidebar.")

tabs = st.tabs(["🏠 Dashboard", "🎯 Aces", "🎾 Games/Sets", "💥 Breaks/DF", "⬆️ Board Builder", "✅ Grade + Learning", "🩺 Data Health", "📚 Logs"])

with tabs[0]:
    st.subheader("🟢 Best Board")
    if engine.empty:
        st.info("No board loaded. Go to Board Builder and paste your Underdog props.")
    else:
        for _, r in engine.head(10).iterrows():
            render_card(r)
        show_table(engine)
        if st.button("Save projection snapshot"):
            snap = engine.copy()
            snap["Saved At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            append_csv(SNAPSHOT_FILE, snap)
            st.success("Saved.")

with tabs[1]:
    st.subheader("🎯 Aces")
    df = engine[engine["Bucket"] == "ACES"] if not engine.empty else pd.DataFrame()
    show_table(df) if not df.empty else st.info("No ace props loaded.")

with tabs[2]:
    st.subheader("🎾 Games / Sets / Moneyline")
    df = engine[engine["Bucket"].isin(["PLAYER_GAMES","TOTAL_GAMES","FIRST_SET_TOTAL_GAMES","FIRST_SET_PLAYER_GAMES","SETS_WON","SETS_PLAYED","MATCH_WINNER"])] if not engine.empty else pd.DataFrame()
    show_table(df) if not df.empty else st.info("No games/sets props loaded.")

with tabs[3]:
    st.subheader("💥 Breaks / Double Faults")
    df = engine[engine["Bucket"].isin(["BREAKS","BREAK_POINTS","DOUBLE_FAULTS","TIEBREAK"])] if not engine.empty else pd.DataFrame()
    show_table(df) if not df.empty else st.info("No breaks/DF props loaded.")

with tabs[4]:
    st.subheader("⬆️ Board Builder")
    st.write("Paste Underdog lines exactly in this simple format:")
    st.code("Taylor Fritz, Aces, 13.5, Ben Shelton, ATP Halle\nBen Shelton, Sets Won, 0.5, Taylor Fritz, ATP Halle\nTaylor Fritz, Games Played, 26.5, Ben Shelton, ATP Halle")
    paste = st.text_area("Paste lines", height=170)
    c1,c2,c3 = st.columns(3)
    if c1.button("Load pasted board"):
        df = parse_paste(paste)
        if not df.empty:
            st.session_state.manual_board = df
            st.success(f"Loaded {len(df)} lines.")
            st.rerun()
        else:
            st.error("No valid lines found.")
    if c2.button("Load Shelton/Fritz screenshot sample"):
        st.session_state.manual_board = pd.DataFrame([
            {"Player":"Taylor Fritz","Opponent":"Ben Shelton","Matchup":"Ben Shelton vs Taylor Fritz","Tournament":"ATP Halle","Stat":"Aces","Bucket":"ACES","UD/Line":13.5,"Line Source":"Sample","Pulled At":datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"Player":"Taylor Fritz","Opponent":"Ben Shelton","Matchup":"Ben Shelton vs Taylor Fritz","Tournament":"ATP Halle","Stat":"Sets Won","Bucket":"SETS_WON","UD/Line":0.5,"Line Source":"Sample","Pulled At":datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"Player":"Ben Shelton","Opponent":"Taylor Fritz","Matchup":"Ben Shelton vs Taylor Fritz","Tournament":"ATP Halle","Stat":"Sets Won","Bucket":"SETS_WON","UD/Line":0.5,"Line Source":"Sample","Pulled At":datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"Player":"Taylor Fritz","Opponent":"Ben Shelton","Matchup":"Ben Shelton vs Taylor Fritz","Tournament":"ATP Halle","Stat":"Games Played","Bucket":"TOTAL_GAMES","UD/Line":26.5,"Line Source":"Sample","Pulled At":datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"Player":"Alexander Zverev","Opponent":"Raphael Collignon","Matchup":"Alexander Zverev vs Raphael Collignon","Tournament":"ATP Halle","Stat":"Match Winner","Bucket":"MATCH_WINNER","UD/Line":50,"Line Source":"Sample","Pulled At":datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        ])
        st.success("Loaded sample board.")
        st.rerun()
    if c3.button("Use Underdog pull if available"):
        if not ud_board.empty:
            st.session_state.manual_board = pd.DataFrame()
            st.success("Using Underdog board.")
            st.rerun()
        else:
            st.error("Underdog endpoint returned 0 here. Paste the board instead.")

    f = st.file_uploader("Upload board CSV", type=["csv"])
    if f is not None:
        df = pd.read_csv(f)
        if {"Player","Stat","UD/Line"}.issubset(df.columns):
            if "Opponent" not in df.columns: df["Opponent"] = ""
            if "Matchup" not in df.columns: df["Matchup"] = ""
            if "Tournament" not in df.columns: df["Tournament"] = ""
            df["Bucket"] = df["Stat"].map(prop_bucket)
            df["Line Source"] = "Upload"
            df["Pulled At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.manual_board = df
            st.success(f"Loaded {len(df)} rows.")
            st.rerun()
        else:
            st.error("CSV needs Player, Stat, UD/Line.")

    st.markdown("### Current Board")
    st.dataframe(board, use_container_width=True)

with tabs[5]:
    st.subheader("✅ Grade + Learning")
    st.caption("Upload results after matches. Required: Player, Stat, Actual. Optional: Result.")
    gf = st.file_uploader("Upload graded results CSV", type=["csv"], key="grade_file")
    if gf is not None:
        g = pd.read_csv(gf)
        st.dataframe(g, use_container_width=True)
        if {"Player","Stat","Actual"}.issubset(g.columns):
            snaps = read_csv(SNAPSHOT_FILE)
            if not snaps.empty:
                snaps["Key"] = snaps["Player"].astype(str).map(norm) + "|" + snaps["Stat"].astype(str).map(norm)
                g["Key"] = g["Player"].astype(str).map(norm) + "|" + g["Stat"].astype(str).map(norm)
                m = snaps.merge(g[["Key","Actual"]], on="Key", how="inner")
                if not m.empty:
                    m["Actual"] = pd.to_numeric(m["Actual"], errors="coerce")
                    m["Projection"] = pd.to_numeric(m["Projection"], errors="coerce")
                    m["Error"] = m["Actual"] - m["Projection"]
                    mem = m[["Player","Stat","Bucket","Surface","Projection","Actual","Error","Decision","UD/Line","Confidence %","Official Filter","Saved At"]].dropna(subset=["Error"])
                    append_csv(LEARNING_FILE, mem)
                    append_csv(GRADE_FILE, g)
                    st.success(f"Learning updated with {len(mem)} graded rows.")
                else:
                    st.warning("No matching saved snapshots. Save projections before grading.")
            else:
                st.warning("No snapshots yet. Save projections first.")
        else:
            st.error("Grading CSV needs Player, Stat, Actual.")

    mem = read_csv(LEARNING_FILE)
    st.markdown("### Learning Memory")
    st.dataframe(mem.tail(250), use_container_width=True)

with tabs[6]:
    st.subheader("🩺 Data Health")
    st.write({"History Source": hist_source, "History Error": hist_err, "Underdog URL": ud_url, "Underdog Error": ud_err})
    st.markdown("### Starter Profiles")
    st.dataframe(pd.read_csv(STARTER_PROFILES), use_container_width=True)
    if not hist.empty:
        st.markdown("### History sample")
        st.dataframe(hist.tail(50), use_container_width=True)

with tabs[7]:
    st.subheader("📚 Logs")
    a,b,c,d = st.tabs(["Snapshots","Underdog Pulls","Learning","Grades"])
    with a: st.dataframe(read_csv(SNAPSHOT_FILE).tail(250), use_container_width=True)
    with b: st.dataframe(read_csv(UD_LOG_FILE).tail(250), use_container_width=True)
    with c: st.dataframe(read_csv(LEARNING_FILE).tail(250), use_container_width=True)
    with d: st.dataframe(read_csv(GRADE_FILE).tail(250), use_container_width=True)

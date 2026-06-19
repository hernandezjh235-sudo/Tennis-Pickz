# ONE WAY PICKZ — TENNIS V5 EASY RUN
# Streamlit Cloud ready. Real data first, manual/upload fallback always available.

import io, os, re, json, math, time, warnings
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

warnings.filterwarnings('ignore')

APP_VERSION = "ONE WAY PICKZ — TENNIS V5 EASY RUN"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
SAMPLE_DIR = os.path.join(BASE_DIR, 'samples')
for d in [DATA_DIR, LOG_DIR, SAMPLE_DIR]: os.makedirs(d, exist_ok=True)

SNAPSHOT_FILE = os.path.join(LOG_DIR, 'projection_snapshots.csv')
GRADE_FILE = os.path.join(LOG_DIR, 'graded_results.csv')
LEARNING_FILE = os.path.join(LOG_DIR, 'learning_memory.csv')
UD_LOG_FILE = os.path.join(LOG_DIR, 'underdog_line_log.csv')
MASTER_FILE = os.path.join(LOG_DIR, 'player_master_stats.csv')
HISTORY_CACHE = os.path.join(DATA_DIR, 'tennis_history_cache.csv')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36',
    'Accept': 'application/json,text/plain,*/*',
    'Origin': 'https://underdogfantasy.com',
    'Referer': 'https://underdogfantasy.com/'
}
UNDERDOG_ENDPOINTS = [
    'https://api.underdogfantasy.com/beta/v5/over_under_lines',
    'https://api.underdogfantasy.com/beta/v4/over_under_lines',
    'https://api.underdogfantasy.com/beta/v3/over_under_lines',
]
SURFACE_FACTOR = {'Hard':1.00,'Clay':0.92,'Grass':1.13,'Carpet':1.06,'Unknown':1.00}
INDOOR_FACTOR = {'Outdoor':1.00,'Indoor':1.055,'Unknown':1.00}
LEVEL_FACTOR = {'Grand Slam':1.08,'Masters / WTA 1000':1.045,'ATP/WTA 500':1.015,'ATP/WTA 250':1.00,'Challenger / Qualifier':0.955,'Unknown':1.00}

st.set_page_config(page_title=APP_VERSION, page_icon='🎾', layout='wide', initial_sidebar_state='expanded')
st.markdown('''
<style>
.stApp{background:#090d11;color:#e7fff2}.big-title{font-size:34px;font-weight:900;color:#00ff88}.sub-title{color:#a9bbb5;font-size:14px}.card{background:#111820;border:1px solid #24323f;border-radius:18px;padding:16px;margin:10px 0}.good{color:#00ff88;font-weight:900}.warn{color:#ffd166;font-weight:900}.bad{color:#ff4d6d;font-weight:900}.muted{color:#91a49e}.kpi{background:#0e151b;border:1px solid #1d2b35;border-radius:14px;padding:12px}.kpi-v{font-size:25px;font-weight:900}.kpi-l{font-size:12px;color:#91a49e}
</style>
''', unsafe_allow_html=True)

def clean_name(x):
    if x is None: return ''
    x=str(x).replace('_',' ').replace('-',' ')
    x=re.sub(r"[^A-Za-zÀ-ÿ' .]",'',x)
    return re.sub(r'\s+',' ',x).strip()

def norm(x): return clean_name(x).lower()
def sf(x, default=np.nan):
    try:
        if x is None or x=='': return default
        return float(x)
    except Exception: return default

def clamp(x,a,b): return max(a,min(b,x))
def sigmoid(x): return 1/(1+math.exp(-x))
def prob_over(edge,sigma): return 100*sigmoid(1.702*(edge/max(sigma,.01)))

def prop_bucket(stat):
    s=str(stat).lower()
    if 'ace' in s: return 'ACES'
    if 'double' in s and 'fault' in s: return 'DOUBLE_FAULTS'
    if 'break point' in s: return 'BREAK_POINTS'
    if 'break' in s and 'tie' not in s: return 'BREAKS'
    if 'tie' in s and 'break' in s: return 'TIEBREAK'
    if 'total' in s and 'game' in s: return 'TOTAL_GAMES'
    if 'games played' in s: return 'TOTAL_GAMES'
    if 'game' in s: return 'PLAYER_GAMES'
    if 'set' in s: return 'SETS'
    if 'winner' in s or 'moneyline' in s: return 'MATCH_WINNER'
    return 'OTHER'

def read_csv(path):
    try: return pd.read_csv(path)
    except Exception: return pd.DataFrame()

def append_csv(path, df):
    if df is None or df.empty: return
    old=read_csv(path)
    out=pd.concat([old,df],ignore_index=True) if not old.empty else df.copy()
    out.to_csv(path,index=False)

@st.cache_data(ttl=3600, show_spinner=False)
def load_history(years_back:int=4, force_refresh:bool=False):
    if os.path.exists(HISTORY_CACHE) and not force_refresh:
        try:
            c=pd.read_csv(HISTORY_CACHE, low_memory=False)
            if len(c)>1000: return c, 'LOCAL_CACHE', ''
        except Exception: pass
    current=datetime.now().year
    years=list(range(current-years_back, current+1))
    frames=[]; errors=[]
    for tour in ['atp','wta']:
        repo='tennis_atp' if tour=='atp' else 'tennis_wta'
        pref='atp' if tour=='atp' else 'wta'
        for y in years:
            url=f'https://raw.githubusercontent.com/JeffSackmann/{repo}/master/{pref}_matches_{y}.csv'
            try:
                r=requests.get(url,headers={'User-Agent':HEADERS['User-Agent']},timeout=18)
                if r.status_code==200 and len(r.text)>1000:
                    d=pd.read_csv(io.StringIO(r.text), low_memory=False)
                    d['tour']=tour.upper(); frames.append(d)
                else:
                    errors.append(f'{pref} {y}: HTTP {r.status_code}')
            except Exception as e:
                errors.append(f'{pref} {y}: {str(e)[:70]}')
    if frames:
        df=pd.concat(frames, ignore_index=True)
        df['tourney_date']=pd.to_numeric(df.get('tourney_date'), errors='coerce')
        df=df.sort_values('tourney_date').reset_index(drop=True)
        try: df.to_csv(HISTORY_CACHE,index=False)
        except Exception: pass
        return df, 'JEFF_SACKMANN_GITHUB', ' | '.join(errors[-4:])
    seed=os.path.join(SAMPLE_DIR,'seed_history.csv')
    if os.path.exists(seed):
        try: return pd.read_csv(seed), 'SEED_SAMPLE_FALLBACK', 'Real history failed: '+' | '.join(errors[-4:])
        except Exception: pass
    return pd.DataFrame(), 'EMPTY', ' | '.join(errors[-8:])

def row_for(r, won):
    p='w' if won else 'l'; op='l' if won else 'w'; opp='loser' if won else 'winner'
    name='winner' if won else 'loser'
    return {'date':sf(r.get('tourney_date')),'surface':r.get('surface','Unknown') or 'Unknown','tour':r.get('tour',''),'level':r.get('tourney_level',''),'round':r.get('round',''), 'won':int(won),'player_name':r.get(f'{name}_name',''),'opp_name':r.get(f'{opp}_name',''), 'rank':sf(r.get(f'{p}_rank')), 'rank_points':sf(r.get(f'{p}_rank_points')), 'opp_rank':sf(r.get(f'{opp}_rank')), 'aces':sf(r.get(f'{p}_ace'),0),'df':sf(r.get(f'{p}_df'),0),'svpt':sf(r.get(f'{p}_svpt')),'first_in':sf(r.get(f'{p}_1stIn')),'first_won':sf(r.get(f'{p}_1stWon')),'second_won':sf(r.get(f'{p}_2ndWon')),'service_games':sf(r.get(f'{p}_SvGms')),'bp_saved':sf(r.get(f'{p}_bpSaved')),'bp_faced':sf(r.get(f'{p}_bpFaced')),'opp_aces':sf(r.get(f'{op}_ace'),0),'opp_svpt':sf(r.get(f'{op}_svpt')),'opp_first_in':sf(r.get(f'{op}_1stIn')),'opp_first_won':sf(r.get(f'{op}_1stWon')),'opp_second_won':sf(r.get(f'{op}_2ndWon')),'opp_service_games':sf(r.get(f'{op}_SvGms')),'opp_bp_saved':sf(r.get(f'{op}_bpSaved')),'opp_bp_faced':sf(r.get(f'{op}_bpFaced')),'score':r.get('score',''),'best_of':sf(r.get('best_of'),3),'minutes':sf(r.get('minutes'))}

def player_rows(hist, player, limit=180):
    if hist.empty or not player: return pd.DataFrame()
    p=norm(player)
    w=hist[hist['winner_name'].astype(str).map(norm).str.contains(p,regex=False,na=False)] if 'winner_name' in hist else pd.DataFrame()
    l=hist[hist['loser_name'].astype(str).map(norm).str.contains(p,regex=False,na=False)] if 'loser_name' in hist else pd.DataFrame()
    rows=[row_for(r,True) for _,r in w.iterrows()]+[row_for(r,False) for _,r in l.iterrows()]
    if not rows:
        last=p.split(' ')[-1] if p else ''
        if len(last)>=4:
            w=hist[hist['winner_name'].astype(str).map(norm).str.contains(last,regex=False,na=False)]
            l=hist[hist['loser_name'].astype(str).map(norm).str.contains(last,regex=False,na=False)]
            rows=[row_for(r,True) for _,r in w.iterrows()]+[row_for(r,False) for _,r in l.iterrows()]
    out=pd.DataFrame(rows)
    if out.empty: return out
    return out.sort_values('date').tail(limit).reset_index(drop=True)

def parse_score(score):
    if not isinstance(score,str): return 0,0,0,0
    sets=re.findall(r'(\d+)\-([0-9]+)', score)
    gf=sum(int(a) for a,b in sets); ga=sum(int(b) for a,b in sets)
    tb=len(re.findall(r'7\-6|6\-7', score)); return gf,ga,tb,len(sets)

def default_summary(player):
    return {'player':player,'matches':0,'surface_matches':0,'win_pct':.50,'last10_win_pct':.50,'last25_win_pct':.50,'rank':np.nan,'rank_points':np.nan,'ace_per_service_game':.55,'ace_per_svpt':.065,'opponent_ace_allowed_per_service_game':.55,'df_per_service_game':.25,'first_in_pct':.61,'first_win_pct':.69,'second_win_pct':.50,'service_points_won_pct':.62,'return_points_won_pct':.37,'first_return_won_pct':.30,'second_return_won_pct':.49,'bp_save_pct':.60,'bp_convert_pct':.39,'bp_created_per_return_game':.55,'hold_pct':.78,'break_pct':.22,'tiebreak_rate':.18,'tiebreak_win_pct':.50,'games_won_avg':11.2,'games_total_avg':22.4,'sets_avg':2.25,'serve_strength':62,'return_strength':37,'overall_strength':50,'winner_proxy':18,'unforced_error_proxy':24,'forced_error_proxy':18,'shot_quality_proxy':5,'short_rally_edge_proxy':5,'long_rally_edge_proxy':5,'minutes_avg':np.nan,'matches_last14':0,'workload_index':50,'rest_days':np.nan,'fatigue_tax':0,'elite_tag':'NO_HISTORY','reliability':32}

def summarize(hist, player, surface='Unknown'):
    rows=player_rows(hist, player)
    if rows.empty: return default_summary(player)
    surf=rows[rows['surface'].astype(str).str.lower()==surface.lower()] if surface!='Unknown' else rows
    use=surf if len(surf)>=6 else rows
    last10=rows.tail(10); last25=rows.tail(25)
    sg=use['service_games'].replace(0,np.nan); opp_sg=use['opp_service_games'].replace(0,np.nan)
    svpt=use['svpt'].replace(0,np.nan); opp_svpt=use['opp_svpt'].replace(0,np.nan)
    sv_sum=svpt.sum(); opp_sv_sum=opp_svpt.sum(); first_in=use['first_in'].sum(); opp_first_in=use['opp_first_in'].sum()
    second_pts=sv_sum-first_in; opp_second_pts=opp_sv_sum-opp_first_in
    ace_sg=use['aces'].sum()/sg.sum() if sg.sum() and not pd.isna(sg.sum()) else .55
    ace_svpt=use['aces'].sum()/sv_sum if sv_sum and not pd.isna(sv_sum) else .065
    opp_ace_allowed=use['opp_aces'].sum()/opp_sg.sum() if opp_sg.sum() and not pd.isna(opp_sg.sum()) else .55
    df_sg=use['df'].sum()/sg.sum() if sg.sum() and not pd.isna(sg.sum()) else .25
    first_in_pct=first_in/sv_sum if sv_sum and not pd.isna(sv_sum) else .61
    first_win=use['first_won'].sum()/first_in if first_in and not pd.isna(first_in) else .69
    second_win=use['second_won'].sum()/second_pts if second_pts and not pd.isna(second_pts) else .50
    spw=(use['first_won'].sum()+use['second_won'].sum())/sv_sum if sv_sum and not pd.isna(sv_sum) else .62
    rpw=1-((use['opp_first_won'].sum()+use['opp_second_won'].sum())/opp_sv_sum) if opp_sv_sum and not pd.isna(opp_sv_sum) else .37
    frw=1-(use['opp_first_won'].sum()/opp_first_in) if opp_first_in and not pd.isna(opp_first_in) else .30
    srw=1-(use['opp_second_won'].sum()/opp_second_pts) if opp_second_pts and not pd.isna(opp_second_pts) else .49
    bp_save=use['bp_saved'].sum()/use['bp_faced'].sum() if use['bp_faced'].sum() and not pd.isna(use['bp_faced'].sum()) else .60
    bp_convert=(use['opp_bp_faced'].sum()-use['opp_bp_saved'].sum())/use['opp_bp_faced'].sum() if use['opp_bp_faced'].sum() and not pd.isna(use['opp_bp_faced'].sum()) else .39
    bp_created=use['opp_bp_faced'].sum()/opp_sg.sum() if opp_sg.sum() and not pd.isna(opp_sg.sum()) else .55
    hold=clamp(.49+.58*spw+.06*bp_save-.045*df_sg,.48,.94)
    brk=clamp(.10+1.75*(rpw-.34)+.08*bp_convert,.06,.48)
    win=float(use['won'].mean()); last10w=float(last10['won'].mean()) if len(last10) else win; last25w=float(last25['won'].mean()) if len(last25) else win
    rank=use['rank'].dropna().tail(1).mean() if use['rank'].notna().any() else np.nan
    rp=use['rank_points'].dropna().tail(1).mean() if use['rank_points'].notna().any() else np.nan
    gfs=[]; gas=[]; sets=[]; tbs=0
    for _,rr in use.iterrows():
        gf,ga,tb,sc=parse_score(rr.get('score','')); tbs+=tb
        if gf+ga>0:
            if rr.get('won',0)==1: gfs.append(gf); gas.append(ga)
            else: gfs.append(ga); gas.append(gf)
            sets.append(sc)
    games_avg=float(np.mean(gfs)) if gfs else 11.2; total_avg=float(np.mean(np.array(gfs)+np.array(gas))) if gfs else 22.4; sets_avg=float(np.mean(sets)) if sets else 2.25
    tb_rate=tbs/max(len(use),1); tb_win=clamp(.48+.18*(hold-.78)+.12*(win-.5),.34,.68)
    winner=clamp(12+55*ace_svpt+18*(first_win-.66)+7*(rpw-.37),8,42)
    ue=clamp(17+18*df_sg+12*(1-second_win)-4*(win-.5),12,42)
    fe=clamp(14+24*rpw+8*brk,10,36)
    shot=clamp(5+4*(spw-.62)+3*(rpw-.37)+.7*(winner/max(ue,1)-.9),1,10)
    short=clamp(5+10*((.47+.45*spw+0.7*ace_svpt)-.66)+3*(ace_svpt-.062),1,10)
    long=clamp(5+9*(rpw-.37)+2.5*(second_win-.50)-.08*(ue-24),1,10)
    rest=np.nan; last14=0
    if rows['date'].notna().any():
        try:
            dates=pd.to_datetime(rows['date'].dropna().astype(int).astype(str),format='%Y%m%d')
            now=pd.Timestamp(datetime.now().date()); rest=int((now-dates.iloc[-1]).days); last14=int((dates>=now-pd.Timedelta(days=14)).sum())
        except Exception: pass
    mins=use['minutes'].dropna().mean() if use['minutes'].notna().any() else np.nan
    workload=50+5*last14+(0 if pd.isna(mins) else clamp((mins-95)/3,-12,18))
    fatigue=0
    if not pd.isna(rest):
        if rest<=1: fatigue-=.045
        elif rest<=3: fatigue-=.015
        elif rest>=21: fatigue-=.025
    if workload>=75: fatigue-=.025
    serve=100*spw+9*ace_svpt+6*(hold-.75)+3*(last10w-.5)-1.8*df_sg
    ret=100*rpw+18*brk+3*(bp_convert-.39)+2.5*(last10w-.5)
    rank_adj=0 if pd.isna(rank) else clamp((75-rank)/22,-2.2,2.2)
    overall=.54*serve+.46*ret+rank_adj+.8*(shot-5)
    tags=[]
    if spw>=.66 or ace_sg>=.78 or hold>=.84: tags.append('ELITE_SERVER')
    if rpw>=.405 or brk>=.29: tags.append('ELITE_RETURNER')
    if not pd.isna(rank) and rank<=25: tags.append('TOP_25')
    if last10w>=.70: tags.append('HOT_FORM')
    if workload>=78: tags.append('FATIGUE_RISK')
    rel=clamp(34+len(use)*1.0+len(rows)*.25+(6 if len(surf)>=8 else 0),30,96)
    return {'player':player,'matches':int(len(rows)),'surface_matches':int(len(use)),'win_pct':win,'last10_win_pct':last10w,'last25_win_pct':last25w,'rank':rank,'rank_points':rp,'ace_per_service_game':ace_sg,'ace_per_svpt':ace_svpt,'opponent_ace_allowed_per_service_game':opp_ace_allowed,'df_per_service_game':df_sg,'first_in_pct':first_in_pct,'first_win_pct':first_win,'second_win_pct':second_win,'service_points_won_pct':spw,'return_points_won_pct':rpw,'first_return_won_pct':frw,'second_return_won_pct':srw,'bp_save_pct':bp_save,'bp_convert_pct':bp_convert,'bp_created_per_return_game':bp_created,'hold_pct':hold,'break_pct':brk,'tiebreak_rate':tb_rate,'tiebreak_win_pct':tb_win,'games_won_avg':games_avg,'games_total_avg':total_avg,'sets_avg':sets_avg,'serve_strength':serve,'return_strength':ret,'overall_strength':overall,'winner_proxy':winner,'unforced_error_proxy':ue,'forced_error_proxy':fe,'shot_quality_proxy':shot,'short_rally_edge_proxy':short,'long_rally_edge_proxy':long,'minutes_avg':mins,'matches_last14':last14,'workload_index':workload,'rest_days':rest,'fatigue_tax':fatigue,'elite_tag':','.join(tags) if tags else 'STANDARD','reliability':rel}

def stat_title(line):
    for k in ['stat','stat_type','stat_type_display','display_stat','title','stat_title','name']:
        v=line.get(k) if isinstance(line,dict) else None
        if isinstance(v,str) and v: return v
        if isinstance(v,dict):
            for kk in ['display_stat','stat','name','title']:
                if v.get(kk): return str(v.get(kk))
    ou=line.get('over_under') if isinstance(line.get('over_under'),dict) else {}
    app=ou.get('appearance_stat') if isinstance(ou.get('appearance_stat'),dict) else {}
    for kk in ['display_stat','stat','name','title']:
        if app.get(kk): return str(app.get(kk))
    return 'Unknown'

def line_value(line):
    c=[line.get('stat_value'),line.get('line'),line.get('value')]
    ou=line.get('over_under') if isinstance(line.get('over_under'),dict) else {}
    c += [ou.get('stat_value'), ou.get('line')]
    for x in c:
        v=sf(x)
        if not pd.isna(v): return v
    return np.nan

@st.cache_data(ttl=120, show_spinner=False)
def fetch_underdog():
    last=''
    for url in UNDERDOG_ENDPOINTS:
        try:
            r=requests.get(url, headers=HEADERS, timeout=12)
            if r.status_code==200 and r.text.strip().startswith('{'):
                return r.json(), url, ''
            last=f'{url} HTTP {r.status_code}'
        except Exception as e: last=f'{url} {str(e)[:80]}'
    return {}, '', last

def idx(items): return {str(x.get('id')):x for x in (items or []) if isinstance(x,dict) and x.get('id') is not None}

def parse_underdog(payload):
    if not payload: return pd.DataFrame()
    data=payload.get('data',payload)
    if isinstance(data,dict):
        lines=data.get('over_under_lines') or data.get('lines') or data.get('over_unders') or []
        apps=data.get('appearances') or [] ; players=data.get('players') or [] ; games=data.get('games') or data.get('solo_games') or []
    elif isinstance(data,list): lines,apps,players,games=data,[],[],[]
    else: return pd.DataFrame()
    ai,pi,gi=idx(apps),idx(players),idx(games)
    rows=[]
    for line in lines:
        if not isinstance(line,dict): continue
        ou=line.get('over_under') if isinstance(line.get('over_under'),dict) else {}
        ast=ou.get('appearance_stat') if isinstance(ou.get('appearance_stat'),dict) else {}
        aid=ast.get('appearance_id') or line.get('appearance_id') or ou.get('appearance_id')
        app=ai.get(str(aid),{}) if aid is not None else {}
        pid=app.get('player_id') or line.get('player_id')
        pl=pi.get(str(pid),{}) if pid is not None else {}
        name=pl.get('display_name') or pl.get('name') or app.get('player_name') or line.get('player_name') or line.get('title') or ''
        stat=stat_title(line); bucket=prop_bucket(stat)
        full=(json.dumps(line)+json.dumps(app)+json.dumps(pl)).lower()
        is_tennis=('tennis' in full or 'atp' in full or 'wta' in full or bucket in ['ACES','PLAYER_GAMES','TOTAL_GAMES','BREAK_POINTS','BREAKS','SETS','DOUBLE_FAULTS'])
        if not is_tennis: continue
        gid=app.get('match_id') or app.get('game_id') or line.get('game_id')
        g=gi.get(str(gid),{}) if gid is not None else {}
        matchup=g.get('title') or g.get('match_title') or g.get('name') or app.get('matchup') or line.get('matchup') or ''
        rows.append({'Player':clean_name(name),'Opponent':'','Matchup':matchup,'Stat':stat,'Bucket':bucket,'UD/Line':line_value(line),'Line Source':'Underdog','Start Time':g.get('scheduled_at') or app.get('scheduled_at') or line.get('scheduled_at') or '', 'Raw ID':line.get('id',''), 'Pulled At':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    df=pd.DataFrame(rows)
    if df.empty: return df
    df=df[df['Player'].astype(str).str.len()>1].drop_duplicates(subset=['Player','Stat','UD/Line','Matchup'])
    return df.reset_index(drop=True)

def parse_paste(text):
    rows=[]
    for line in str(text).splitlines():
        s=line.strip()
        if not s: continue
        # formats: Taylor Fritz, Aces, 13.5, Opponent OR Taylor Fritz - Aces - 13.5 - Ben Shelton
        parts=[p.strip() for p in re.split(r',|\s+—\s+|\s+-\s+', s) if p.strip()]
        if len(parts)>=3:
            player=clean_name(parts[0]); stat=parts[1]; val=sf(parts[2]); opp=clean_name(parts[3]) if len(parts)>=4 else ''
            if player and not pd.isna(val): rows.append({'Player':player,'Opponent':opp,'Matchup':f'{player} vs {opp}' if opp else 'Manual','Stat':stat,'Bucket':prop_bucket(stat),'UD/Line':val,'Line Source':'Paste','Start Time':'','Raw ID':'','Pulled At':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    return pd.DataFrame(rows)

def infer_opp(row):
    if str(row.get('Opponent','')).strip(): return clean_name(row.get('Opponent',''))
    m=str(row.get('Matchup',''))
    p=norm(row.get('Player',''))
    chunks=[clean_name(c) for c in re.split(r'\s+v(?:s\.?|ersus)?\s+| @ | / |,', m, flags=re.I) if clean_name(c)]
    for c in chunks:
        if norm(c)!=p and p not in norm(c): return c
    return ''

def h2h(hist, player, opp, surface='Unknown'):
    if hist.empty or not player or not opp or 'winner_name' not in hist: return {'H2H Matches':0,'H2H Win %':50.0}
    p=norm(player); o=norm(opp)
    w=hist['winner_name'].astype(str).map(norm); l=hist['loser_name'].astype(str).map(norm)
    h=hist[(w.str.contains(p,regex=False,na=False)&l.str.contains(o,regex=False,na=False)) | (w.str.contains(o,regex=False,na=False)&l.str.contains(p,regex=False,na=False))]
    if surface!='Unknown' and 'surface' in h: 
        hs=h[h['surface'].astype(str).str.lower()==surface.lower()]
        if len(hs): h=hs
    if h.empty: return {'H2H Matches':0,'H2H Win %':50.0}
    return {'H2H Matches':int(len(h)),'H2H Win %':round(100*h['winner_name'].astype(str).map(norm).str.contains(p,regex=False,na=False).mean(),1)}

def learning_bias(player,bucket,surface,level):
    mem=read_csv(LEARNING_FILE)
    if mem.empty or 'Error' not in mem.columns: return 0, 'NO_LEARNING'
    mem['Error']=pd.to_numeric(mem['Error'],errors='coerce')
    cuts=[]
    for label,mask,minn,cap in [
        ('PLAYER_PROP', (mem.get('Player','').astype(str).map(norm)==norm(player)) & (mem.get('Bucket','').astype(str)==bucket), 3, .55),
        ('PROP', mem.get('Bucket','').astype(str)==bucket, 8, .30),
        ('SURFACE', mem.get('Surface','').astype(str)==surface, 8, .20),
        ('LEVEL', mem.get('Tournament Level','').astype(str)==level, 8, .18),
    ]:
        sub=mem[mask].dropna(subset=['Error']) if hasattr(mask,'__len__') else pd.DataFrame()
        if len(sub)>=minn:
            cuts.append((label, clamp(float(sub['Error'].tail(40).mean())*.35, -cap, cap)))
    if not cuts: return 0, 'NO_LEARNING'
    val=sum(x[1] for x in cuts)
    return round(val,3), '+'.join(x[0] for x in cuts)

def official_filter(bucket, edge, conf, reliability, p, opp, h2hdat, risk):
    reasons=[]; ok=True
    if risk: ok=False; reasons.append('RISK_FLAG')
    if bucket=='TIEBREAK': ok=False; reasons.append('TIEBREAK_VOLATILE')
    if bucket=='OTHER': ok=False; reasons.append('UNSUPPORTED')
    if reliability<45: ok=False; reasons.append('LOW_SAMPLE')
    if bucket=='ACES':
        if abs(edge)<0.65 or conf<56: ok=False; reasons.append('ACE_EDGE_LOW')
        if p['surface_matches']<4 and p['matches']<10: ok=False; reasons.append('ACE_SAMPLE_LOW')
    elif bucket in ['PLAYER_GAMES','TOTAL_GAMES']:
        if abs(edge)<0.75 or conf<57: ok=False; reasons.append('GAME_EDGE_LOW')
    elif bucket in ['BREAKS','BREAK_POINTS','DOUBLE_FAULTS']:
        if abs(edge)<0.70 or conf<58: ok=False; reasons.append('VOL_EDGE_LOW')
    elif bucket=='SETS':
        if abs(edge)<0.18 or conf<57: ok=False; reasons.append('SETS_EDGE_LOW')
    return ('PASS' if ok else 'NO PLAY'), ('PASS' if ok else ','.join(reasons[:4]))

def project(row,hist,surface,best_of,indoor,level):
    player=row.get('Player',''); opp_name=infer_opp(row)
    p=summarize(hist,player,surface); o=summarize(hist,opp_name,surface) if opp_name else default_summary('Unknown')
    h=h2h(hist,player,opp_name,surface)
    bucket=row.get('Bucket',prop_bucket(row.get('Stat',''))); line=sf(row.get('UD/Line'))
    strength_gap=p['overall_strength']-o['overall_strength']
    close=1-min(abs(strength_gap)/28,.38)
    sets=2.10+.42*close if best_of==3 else 3.35+.80*close
    if bucket=='SETS': sets=clamp(sets,2,3 if best_of==3 else 5)
    svc_games=(5.0*sets)+(0.55*close)+(0.25 if strength_gap>4 else 0)
    ret_games=svc_games
    sfac=SURFACE_FACTOR.get(surface,1)*INDOOR_FACTOR.get(indoor,1)*LEVEL_FACTOR.get(level,1)
    if bucket=='ACES':
        opp_allow=(o['opponent_ace_allowed_per_service_game']+.55)/1.10
        proj=p['ace_per_service_game']*svc_games*sfac*(0.94+0.12*opp_allow)*(1+p['fatigue_tax'])
        sigma=clamp(1.15+proj*.32,1.35,4.6)
    elif bucket=='DOUBLE_FAULTS':
        proj=p['df_per_service_game']*svc_games*(1+max(0,o['return_points_won_pct']-.38)*.7)*(1-p['fatigue_tax'])
        sigma=clamp(.75+proj*.45,1,3.4)
    elif bucket=='PLAYER_GAMES':
        proj=4.80*sets+clamp(strength_gap/18,-1.45,1.45)+.30*(p['hold_pct']-.78)*10
        sigma=2.15 if best_of==3 else 3.35
    elif bucket=='TOTAL_GAMES':
        proj=9.65*sets+1.65*close-.45*abs(strength_gap)/12
        sigma=3.3 if best_of==3 else 5.1
    elif bucket=='BREAKS':
        proj=(.85*sets)*(1+1.2*(p['break_pct']-.22)+.55*(1-o['hold_pct']-.22))
        sigma=clamp(.9+proj*.55,1.1,3.6)
    elif bucket=='BREAK_POINTS':
        proj=p['bp_created_per_return_game']*ret_games*(1+.30*(p['return_points_won_pct']-.37))
        sigma=clamp(1.2+proj*.45,1.4,4.2)
    elif bucket=='SETS':
        win_prob=sigmoid(strength_gap/8.0)
        proj=1.0+win_prob*(1.0 if best_of==3 else 2.0)+.12*close
        sigma=.55 if best_of==3 else .85
    elif bucket=='MATCH_WINNER':
        proj=sigmoid(strength_gap/8.0)*100
        sigma=8; line=50 if pd.isna(line) else line
    else:
        proj=np.nan; sigma=2
    lbias,lsrc=learning_bias(player,bucket,surface,level)
    if bucket!='MATCH_WINNER' and not pd.isna(proj): proj+=lbias
    edge=proj-line if not pd.isna(proj) and not pd.isna(line) else np.nan
    over=prob_over(edge,sigma) if not pd.isna(edge) else np.nan; under=100-over if not pd.isna(over) else np.nan
    dec='OVER' if edge>0 else 'UNDER' if not pd.isna(edge) else 'NO LINE'
    conf=max(over,under) if not pd.isna(over) else np.nan
    risk=False
    status=read_csv(os.path.join(DATA_DIR,'status_flags.csv'))
    if not status.empty and 'Player' in status:
        sub=status[status['Player'].astype(str).map(norm)==norm(player)]
        if not sub.empty and str(sub.tail(1).get('Risk Flag',pd.Series([''])).iloc[0]).upper() in ['YES','TRUE','1','RISK','INJURY','RETIREMENT']:
            risk=True
    reliability=min(p['reliability'], o['reliability'] if o['matches'] else p['reliability']-6)
    official,reason=official_filter(bucket, edge if not pd.isna(edge) else 0, conf if not pd.isna(conf) else 0, reliability, p, o, h, risk)
    grade='S 🔒' if official=='PASS' and conf>=65 and reliability>=65 else 'A' if official=='PASS' and conf>=61 else 'B' if official=='PASS' else 'C / WATCH'
    return {**row.to_dict(),'Opponent':opp_name,'Projection':round(proj,2) if not pd.isna(proj) else np.nan,'Floor':round(proj-sigma,2) if not pd.isna(proj) else np.nan,'Median':round(proj,2) if not pd.isna(proj) else np.nan,'Ceiling':round(proj+sigma,2) if not pd.isna(proj) else np.nan,'Over Sim %':round(over,1) if not pd.isna(over) else np.nan,'Under Sim %':round(under,1) if not pd.isna(under) else np.nan,'Decision':dec,'Lean Gap':round(edge,2) if not pd.isna(edge) else np.nan,'Confidence %':round(conf,1) if not pd.isna(conf) else np.nan,'Official Filter':official,'Official Reason':reason,'Grade':grade,'Reliability':round(reliability,1),'Learning Bias':lbias,'Learning Source':lsrc,'Surface':surface,'Best Of':best_of,'Indoor':indoor,'Tournament Level':level,'Expected Sets':round(sets,2),'Expected Service Games':round(svc_games,2),'Player Matches':p['matches'],'Surface Matches':p['surface_matches'],'Rank':p['rank'],'Rank Points':p['rank_points'],'Elite Tag':p['elite_tag'],'Ace/SG':round(p['ace_per_service_game'],3),'Opp Ace Allowed/SG':round(o['opponent_ace_allowed_per_service_game'],3),'1st Serve %':round(100*p['first_in_pct'],1),'1st Won %':round(100*p['first_win_pct'],1),'2nd Won %':round(100*p['second_win_pct'],1),'Serve Pts Won %':round(100*p['service_points_won_pct'],1),'Return Pts Won %':round(100*p['return_points_won_pct'],1),'Hold %':round(100*p['hold_pct'],1),'Break %':round(100*p['break_pct'],1),'BP Save %':round(100*p['bp_save_pct'],1),'BP Convert %':round(100*p['bp_convert_pct'],1),'Winner Proxy':round(p['winner_proxy'],1),'UE Proxy':round(p['unforced_error_proxy'],1),'Forced Error Proxy':round(p['forced_error_proxy'],1),'Shot Quality':round(p['shot_quality_proxy'],1),'Workload':round(p['workload_index'],1),'Rest Days':p['rest_days'],'H2H Matches':h['H2H Matches'],'H2H Win %':h['H2H Win %']}

def run_engine(board,hist,surface,best_of,indoor,level):
    if board.empty: return pd.DataFrame()
    rows=[project(r,hist,surface,best_of,indoor,level) for _,r in board.iterrows()]
    df=pd.DataFrame(rows)
    if df.empty: return df
    df['Abs Edge']=pd.to_numeric(df['Lean Gap'],errors='coerce').abs()
    df=df.sort_values(['Official Filter','Confidence %','Abs Edge'],ascending=[True,False,False])
    return df.reset_index(drop=True)

def render_card(r):
    cls='good' if r.get('Official Filter')=='PASS' else 'warn'
    st.markdown(f"""<div class='card'><div style='display:flex;justify-content:space-between'><div><b style='font-size:22px'>{r.get('Player','')} — {r.get('Stat','')}</b><div class='muted'>{r.get('Matchup','')} {('vs '+r.get('Opponent','')) if r.get('Opponent','') else ''}</div></div><div style='text-align:right'><div class='{cls}' style='font-size:24px'>{r.get('Decision','')}</div><div>{r.get('Grade','')}</div></div></div><hr><div style='display:grid;grid-template-columns:repeat(6,1fr);gap:8px'><div><span class='muted'>Proj</span><br><b>{r.get('Projection','')}</b></div><div><span class='muted'>Line</span><br><b>{r.get('UD/Line','')}</b></div><div><span class='muted'>Edge</span><br><b>{r.get('Lean Gap','')}</b></div><div><span class='muted'>Conf</span><br><b>{r.get('Confidence %','')}%</b></div><div><span class='muted'>Official</span><br><b>{r.get('Official Filter','')}</b></div><div><span class='muted'>Reliability</span><br><b>{r.get('Reliability','')}</b></div></div><div class='muted' style='margin-top:8px'>Reason: {r.get('Official Reason','')} | {r.get('Elite Tag','')}</div></div>""", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.header('🎾 Controls')
years_back=st.sidebar.slider('Historical years',1,8,4)
surface=st.sidebar.selectbox('Surface',['Grass','Hard','Clay','Carpet','Unknown'],0)
best_of=st.sidebar.selectbox('Best of',[3,5],0)
indoor=st.sidebar.selectbox('Indoor/Outdoor',['Outdoor','Indoor','Unknown'],0)
level=st.sidebar.selectbox('Tournament Level',['ATP/WTA 500','Grand Slam','Masters / WTA 1000','ATP/WTA 250','Challenger / Qualifier','Unknown'],0)
min_conf=st.sidebar.slider('Minimum confidence',50,75,54)
official_only=st.sidebar.toggle('Official PASS only',False)
prop_types=st.sidebar.multiselect('Prop Types',['ACES','PLAYER_GAMES','TOTAL_GAMES','SETS','BREAKS','BREAK_POINTS','DOUBLE_FAULTS','TIEBREAK','MATCH_WINNER','OTHER'],default=['ACES','PLAYER_GAMES','TOTAL_GAMES','SETS','BREAKS','DOUBLE_FAULTS'])
force_refresh=st.sidebar.button('Refresh historical data')

st.markdown(f"<div class='big-title'>{APP_VERSION}</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Real ATP/WTA history + Underdog line pull + manual board fallback + tabs like MLB.</div>", unsafe_allow_html=True)

hist, hist_source, hist_err = load_history(years_back, force_refresh)
payload, ud_url, ud_err = fetch_underdog()
ud_board=parse_underdog(payload)
if not ud_board.empty: append_csv(UD_LOG_FILE, ud_board)

if 'manual_board' not in st.session_state: st.session_state.manual_board=pd.DataFrame()
board = ud_board if not ud_board.empty else st.session_state.manual_board
engine = run_engine(board,hist,surface,best_of,indoor,level) if not board.empty else pd.DataFrame()
if not engine.empty:
    engine=engine[engine['Bucket'].isin(prop_types)]
    engine=engine[pd.to_numeric(engine['Confidence %'],errors='coerce').fillna(0)>=min_conf]
    if official_only: engine=engine[engine['Official Filter']=='PASS']
    master_cols=['Player','Rank','Rank Points','Elite Tag','Player Matches','Surface Matches','Ace/SG','1st Serve %','Serve Pts Won %','Return Pts Won %','Hold %','Break %','Workload','Rest Days']
    append_csv(MASTER_FILE, engine[[c for c in master_cols if c in engine.columns]].drop_duplicates('Player'))

k1,k2,k3,k4,k5=st.columns(5)
for col,label,val in [(k1,'Underdog Lines',len(ud_board)),(k2,'Historical Matches',len(hist)),(k3,'Board Lines',len(board)),(k4,'Official PASS',int((engine.get('Official Filter',pd.Series([]))=='PASS').sum()) if not engine.empty else 0),(k5,'Elite Profiles',engine['Elite Tag'].astype(str).str.contains('ELITE|TOP_25',na=False).sum() if not engine.empty else 0)]:
    col.markdown(f"<div class='kpi'><div class='kpi-v'>{val}</div><div class='kpi-l'>{label}</div></div>", unsafe_allow_html=True)

if hist.empty: st.error('Historical matches are 0. Open Data Health tab and use Manual Board until data source is reachable.')
elif hist_source!='JEFF_SACKMANN_GITHUB': st.warning(f'History source: {hist_source}. {hist_err}')
if ud_board.empty: st.warning(f'No Underdog tennis board found from public endpoint. Use Upload / Manual tab. Last error: {ud_err}')

tabs=st.tabs(['🏠 Dashboard','🎯 Aces','🎾 Games/Sets','💥 Breaks/DF','⬆️ Upload / Manual','🩺 Data Health','✅ Grade + Learning','📚 Logs'])
with tabs[0]:
    st.subheader('🟢 Best Board')
    if engine.empty: st.info('No current board loaded. Go to Upload / Manual and paste Underdog lines or upload CSV.')
    else:
        for _,r in engine.head(10).iterrows(): render_card(r)
        st.dataframe(engine, use_container_width=True, height=420)
        if st.button('Save Projection Snapshot'):
            snap=engine.copy(); snap['Saved At']=datetime.now().strftime('%Y-%m-%d %H:%M:%S'); append_csv(SNAPSHOT_FILE,snap); st.success('Saved.')
with tabs[1]:
    df=engine[engine['Bucket']=='ACES'] if not engine.empty else pd.DataFrame(); st.dataframe(df,use_container_width=True,height=600)
with tabs[2]:
    df=engine[engine['Bucket'].isin(['PLAYER_GAMES','TOTAL_GAMES','SETS','MATCH_WINNER'])] if not engine.empty else pd.DataFrame(); st.dataframe(df,use_container_width=True,height=600)
with tabs[3]:
    df=engine[engine['Bucket'].isin(['BREAKS','BREAK_POINTS','DOUBLE_FAULTS','TIEBREAK'])] if not engine.empty else pd.DataFrame(); st.dataframe(df,use_container_width=True,height=600)
with tabs[4]:
    st.subheader('Manual / Upload Board')
    st.caption('CSV columns: Player, Stat, UD/Line. Optional: Opponent, Matchup. Paste format: Taylor Fritz, Aces, 13.5, Ben Shelton')
    paste=st.text_area('Paste Underdog lines', height=150, placeholder='Taylor Fritz, Aces, 13.5, Ben Shelton\nBen Shelton, Sets Won, 0.5, Taylor Fritz\nTaylor Fritz, Games Played, 26.5, Ben Shelton')
    c1,c2,c3=st.columns(3)
    if c1.button('Load pasted board'):
        df=parse_paste(paste)
        if not df.empty: st.session_state.manual_board=df; st.success(f'Loaded {len(df)} lines.'); st.rerun()
        else: st.error('No valid lines found.')
    if c2.button('Load sample from screenshot'):
        st.session_state.manual_board=pd.DataFrame([
            {'Player':'Taylor Fritz','Opponent':'Ben Shelton','Matchup':'Ben Shelton vs Taylor Fritz','Stat':'Aces','Bucket':'ACES','UD/Line':13.5,'Line Source':'Sample','Start Time':'','Raw ID':'','Pulled At':datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
            {'Player':'Taylor Fritz','Opponent':'Ben Shelton','Matchup':'Ben Shelton vs Taylor Fritz','Stat':'Sets Won','Bucket':'SETS','UD/Line':0.5,'Line Source':'Sample','Start Time':'','Raw ID':'','Pulled At':datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
            {'Player':'Ben Shelton','Opponent':'Taylor Fritz','Matchup':'Ben Shelton vs Taylor Fritz','Stat':'Sets Won','Bucket':'SETS','UD/Line':0.5,'Line Source':'Sample','Start Time':'','Raw ID':'','Pulled At':datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
            {'Player':'Taylor Fritz','Opponent':'Ben Shelton','Matchup':'Ben Shelton vs Taylor Fritz','Stat':'Games Played','Bucket':'TOTAL_GAMES','UD/Line':26.5,'Line Source':'Sample','Start Time':'','Raw ID':'','Pulled At':datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
            {'Player':'Alexander Zverev','Opponent':'Raphael Collignon','Matchup':'Alexander Zverev vs Raphael Collignon','Stat':'Match Winner','Bucket':'MATCH_WINNER','UD/Line':50,'Line Source':'Sample','Start Time':'','Raw ID':'','Pulled At':datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
        ]); st.success('Loaded screenshot sample.'); st.rerun()
    if c3.button('Clear manual board'):
        st.session_state.manual_board=pd.DataFrame(); st.rerun()
    f=st.file_uploader('Upload board CSV', type=['csv'])
    if f is not None:
        df=pd.read_csv(f); 
        if {'Player','Stat','UD/Line'}.issubset(df.columns):
            if 'Opponent' not in df: df['Opponent']=''
            if 'Matchup' not in df: df['Matchup']=''
            df['Bucket']=df['Stat'].map(prop_bucket); df['Line Source']='Upload'; df['Pulled At']=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            st.session_state.manual_board=df; st.success(f'Loaded {len(df)} uploaded lines.'); st.rerun()
        else: st.error('CSV needs Player, Stat, UD/Line')
    st.dataframe(st.session_state.manual_board, use_container_width=True)
with tabs[5]:
    st.subheader('Data Health')
    st.write({'History Source':hist_source,'History Error':hist_err,'Underdog URL':ud_url,'Underdog Error':ud_err})
    st.write('Historical columns:', list(hist.columns)[:40] if not hist.empty else [])
    st.dataframe(hist.tail(20), use_container_width=True)
    st.caption('If Underdog is 0, use Upload/Manual. If history is 0, click refresh or redeploy. The app now shows the real error here.')
with tabs[6]:
    st.subheader('After Grade / Learning Engine')
    st.caption('Upload results with: Player, Stat or Bucket, Projection, Actual. The app learns projection error by player/prop/surface/tournament.')
    gf=st.file_uploader('Upload graded results CSV', type=['csv'], key='grade')
    if gf is not None:
        gd=pd.read_csv(gf)
        if 'Bucket' not in gd and 'Stat' in gd: gd['Bucket']=gd['Stat'].map(prop_bucket)
        if 'Error' not in gd and {'Actual','Projection'}.issubset(gd.columns): gd['Error']=pd.to_numeric(gd['Actual'],errors='coerce')-pd.to_numeric(gd['Projection'],errors='coerce')
        for col,val in [('Surface',surface),('Tournament Level',level),('Graded At',datetime.now().strftime('%Y-%m-%d %H:%M:%S'))]:
            if col not in gd: gd[col]=val
        append_csv(GRADE_FILE,gd); append_csv(LEARNING_FILE,gd); st.success('Grades saved to learning memory.')
    st.dataframe(read_csv(LEARNING_FILE).tail(500), use_container_width=True, height=420)
with tabs[7]:
    st.subheader('Logs')
    a,b,c,d=st.tabs(['Snapshots','Underdog Lines','Player Master','Grades'])
    with a: st.dataframe(read_csv(SNAPSHOT_FILE).tail(500),use_container_width=True,height=430)
    with b: st.dataframe(read_csv(UD_LOG_FILE).tail(500),use_container_width=True,height=430)
    with c: st.dataframe(read_csv(MASTER_FILE).tail(500),use_container_width=True,height=430)
    with d: st.dataframe(read_csv(GRADE_FILE).tail(500),use_container_width=True,height=430)

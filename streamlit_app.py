"""
Paid Ads Intelligence Dashboard
Self-explaining · Storytelling · Hover Insights · Issue→Analysis→Benefit
Google · Meta · Microsoft | Claude + GPT-4o
"""
import sqlite3, warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import anthropic
from openai import OpenAI

warnings.filterwarnings("ignore")

OPENAI_API_KEY = "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"
CLAUDE_API_KEY = "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"
DB_PATH = "ads_dashboard.db"

# ── Palette ──────────────────────────────────────────────────
BG   = "#13161F"; SURF = "#1C2030"; SURF2 = "#252A3D"; SURF3 = "#2E3450"
BDR  = "#363D5C"; BDR2 = "#4E567A"
TP   = "#E8EDFF"; TS   = "#9BA8CC"; TM   = "#5E6A8F"
GRN  = "#06D6A0"; RED  = "#FF5370"; AMB  = "#FFD166"
BLU  = "#5B9EFF"; PUR  = "#B57AFF"; TEA  = "#22D3EE"
G_BG = "rgba(6,214,160,0.12)";   R_BG = "rgba(255,83,112,0.12)"
A_BG = "rgba(255,209,102,0.12)"; B_BG = "rgba(91,158,255,0.12)"
P_BG = "rgba(181,122,255,0.12)"; T_BG = "rgba(34,211,238,0.12)"
GCOL = "#5C9EFF"; MCOL = "#4A90F5"; MSCOL = "#00BFFF"
PC   = {"google": GCOL, "meta": MCOL, "microsoft": MSCOL}

def hr(h, a):
    x = h.lstrip("#")
    return f"rgba({int(x[0:2],16)},{int(x[2:4],16)},{int(x[4:6],16)},{a})"

PL = dict(
    template="plotly_dark", plot_bgcolor=SURF, paper_bgcolor=SURF,
    font=dict(family="Inter,system-ui,sans-serif", color=TS, size=12),
    xaxis=dict(gridcolor="#2A304A", linecolor=BDR, zeroline=False,
               tickfont=dict(size=11, color=TS), title_font=dict(color=TS)),
    yaxis=dict(gridcolor="#2A304A", linecolor=BDR, zeroline=False,
               tickfont=dict(size=11, color=TS), title_font=dict(color=TS)),
    margin=dict(l=10, r=10, t=44, b=10),
    legend=dict(bgcolor="rgba(28,32,48,0.9)", bordercolor=BDR, borderwidth=1,
                font=dict(size=11, color=TS)),
    hoverlabel=dict(bgcolor=SURF2, bordercolor=BDR2, font=dict(size=12, color=TP)),
)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Ads Intelligence Dashboard",
                   page_icon="📡", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ──────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp{{font-family:'Inter',system-ui,sans-serif!important;
  background:{BG}!important;color:{TP}!important;}}
.block-container{{padding:1rem 1.6rem 2rem!important;max-width:100%!important;}}
#MainMenu,footer,header{{visibility:hidden;}}
section[data-testid="stSidebar"]{{background:{SURF}!important;border-right:1px solid {BDR}!important;}}
section[data-testid="stSidebar"] *{{color:{TS}!important;}}
[data-testid="stSidebarNav"]{{display:none;}}
/* KPI card */
.kc{{background:{SURF};border:1px solid {BDR};border-radius:10px;
  padding:14px 16px 12px;margin-bottom:8px;position:relative;overflow:hidden;}}
.kc:hover{{border-color:{BDR2};box-shadow:0 0 0 1px {BDR2};}}
.kc-bar{{position:absolute;top:0;left:0;width:4px;height:100%;}}
.kc-ico{{font-size:16px;display:block;margin-bottom:4px;}}
.kc-lbl{{font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:{TM};}}
.kc-val{{font-size:28px;font-weight:800;color:{TP};line-height:1.1;margin:4px 0 3px;letter-spacing:-0.8px;}}
.kc-up{{font-size:11px;font-weight:700;color:{GRN};}}
.kc-dn{{font-size:11px;font-weight:700;color:{RED};}}
.kc-fl{{font-size:11px;font-weight:500;color:{TM};}}
.kc-sub{{font-size:10px;color:{TM};margin-top:3px;}}
/* Tooltip */
.tw{{position:relative;display:inline-block;cursor:help;}}
.ti{{display:inline-flex;align-items:center;justify-content:center;width:15px;height:15px;
  background:{BDR};border-radius:50%;font-size:9px;color:{TS};font-weight:700;
  margin-left:4px;vertical-align:middle;}}
.tb{{visibility:hidden;opacity:0;transition:opacity .18s;position:absolute;z-index:9999;
  bottom:130%;left:50%;transform:translateX(-50%);background:{SURF3};
  border:1px solid {BDR2};border-radius:10px;padding:12px 14px;
  min-width:260px;max-width:340px;box-shadow:0 8px 32px rgba(0,0,0,.55);}}
.tw:hover .tb{{visibility:visible;opacity:1;}}
.tb-title{{font-size:11px;font-weight:700;color:{BLU};letter-spacing:.6px;
  text-transform:uppercase;margin-bottom:6px;}}
.tb-body{{font-size:12px;color:{TS};line-height:1.65;}}
.tb-issue{{font-size:11px;color:{RED};font-weight:600;margin-top:7px;}}
.tb-fix{{font-size:11px;color:{GRN};font-weight:600;margin-top:4px;}}
.tb-ben{{font-size:11px;color:{AMB};font-weight:600;margin-top:4px;}}
/* Story block */
.sb{{background:{SURF};border:1px solid {BDR};border-left:4px solid {BLU};
  border-radius:10px;padding:14px 16px;margin:10px 0;}}
.sb-head{{font-size:13px;font-weight:700;color:{TP};margin-bottom:6px;}}
.sb-body{{font-size:12px;color:{TS};line-height:1.7;}}
.tag-issue{{display:inline-block;background:{R_BG};color:{RED};
  font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px;margin:0 4px 4px 0;}}
.tag-act{{display:inline-block;background:{B_BG};color:{BLU};
  font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px;margin:0 4px 4px 0;}}
.tag-ben{{display:inline-block;background:{G_BG};color:{GRN};
  font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px;margin:0 4px 4px 0;}}
/* Ribbon */
.rib{{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 12px;}}
.ri{{background:{SURF2};border:1px solid {BDR};border-radius:8px;
  padding:8px 12px;flex:1;min-width:140px;}}
.ri-lbl{{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:{TM};}}
.ri-val{{font-size:15px;font-weight:700;color:{TP};margin:2px 0;}}
.ri-note{{font-size:10px;color:{TS};}}
/* Section header */
.sh{{display:flex;align-items:center;gap:8px;font-size:11px;font-weight:700;
  color:{TS};text-transform:uppercase;letter-spacing:1.2px;
  margin:16px 0 8px;padding-bottom:7px;border-bottom:1px solid {BDR};}}
.sh-dot{{width:7px;height:7px;border-radius:50%;flex-shrink:0;}}
/* Banner */
.bn{{background:{SURF};border:1px solid {BDR};border-left:4px solid {BLU};
  border-radius:10px;padding:12px 18px;margin-bottom:14px;
  display:flex;align-items:center;gap:12px;}}
.bn-ico{{font-size:22px;}}
.bn-title{{font-size:16px;font-weight:700;color:{TP};}}
.bn-sub{{font-size:11px;color:{TM};margin-top:2px;}}
/* Alerts */
.alert{{border-radius:8px;padding:12px 16px;margin:8px 0;
  font-size:13px;line-height:1.7;color:{TP};white-space:pre-wrap;}}
.al-crit{{background:{R_BG};border-left:4px solid {RED};}}
.al-warn{{background:{A_BG};border-left:4px solid {AMB};}}
.al-ok{{background:{G_BG};border-left:4px solid {GRN};}}
.al-info{{background:{B_BG};border-left:4px solid {BLU};}}
.al-ai{{background:{P_BG};border-left:4px solid {PUR};}}
.al-lbl{{font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;opacity:.8;}}
/* Chat */
.chu{{background:{SURF2};border:1px solid {BDR2};border-radius:12px 12px 2px 12px;
  padding:10px 14px;margin:8px 0 4px;color:{TP};font-size:13px;line-height:1.6;}}
.cha{{background:{SURF};border:1px solid {BDR};border-left:4px solid {PUR};
  border-radius:2px 12px 12px 12px;padding:12px 14px;margin:4px 0 8px;
  color:{TS};font-size:13px;line-height:1.75;}}
.chl{{font-size:10px;font-weight:700;letter-spacing:.6px;text-transform:uppercase;margin-bottom:5px;}}
/* Overrides */
div[data-testid="stMetricValue"]{{color:{TP}!important;font-size:22px!important;font-weight:700!important;}}
div[data-testid="stMetricLabel"]{{color:{TM}!important;font-size:11px!important;
  text-transform:uppercase;letter-spacing:.8px;}}
.stTabs [data-baseweb="tab-list"]{{background:{SURF}!important;border-bottom:1px solid {BDR}!important;}}
.stTabs [data-baseweb="tab"]{{background:transparent!important;color:{TM}!important;
  font-size:12px!important;font-weight:500!important;
  padding:9px 18px!important;border-bottom:2px solid transparent!important;}}
.stTabs [aria-selected="true"]{{color:{BLU}!important;border-bottom-color:{BLU}!important;}}
div[data-testid="stDataFrame"]{{background:{SURF}!important;
  border:1px solid {BDR}!important;border-radius:8px!important;}}
.stButton>button{{background:{SURF2}!important;border:1px solid {BDR2}!important;
  color:{TP}!important;border-radius:7px!important;
  font-size:12px!important;font-weight:600!important;}}
.stButton>button:hover{{background:{B_BG}!important;
  border-color:{BLU}!important;color:{BLU}!important;}}
div[data-baseweb="select"]>div{{background:{SURF2}!important;
  border-color:{BDR}!important;color:{TP}!important;}}
div[data-baseweb="tag"]{{background:{SURF3}!important;color:{TP}!important;}}
p,label,span{{color:inherit!important;}}
</style>""", unsafe_allow_html=True)

# ── Data ─────────────────────────────────────────────────────
@st.cache_resource
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

@st.cache_data(ttl=120)
def load_pm(days=30):
    co = (datetime.utcnow()-timedelta(days=days)).strftime("%Y-%m-%d")
    df = pd.read_sql("SELECT * FROM platform_metrics WHERE date>=? ORDER BY date,hour,platform",
                     get_conn(), params=(co,))
    df["date"] = pd.to_datetime(df["date"]); return df

@st.cache_data(ttl=120)
def load_cc(days=30):
    co = (datetime.utcnow()-timedelta(days=days)).strftime("%Y-%m-%d")
    df = pd.read_sql("SELECT * FROM call_center WHERE date>=? ORDER BY date,hour",
                     get_conn(), params=(co,))
    df["date"] = pd.to_datetime(df["date"]); return df

@st.cache_data(ttl=120)
def load_cq(days=30):
    co = (datetime.utcnow()-timedelta(days=days)).strftime("%Y-%m-%d")
    df = pd.read_sql("SELECT * FROM call_quality WHERE date>=? ORDER BY date,hour",
                     get_conn(), params=(co,))
    df["date"] = pd.to_datetime(df["date"]); return df

@st.cache_data(ttl=120)
def load_attr(days=30):
    co = (datetime.utcnow()-timedelta(days=days)).strftime("%Y-%m-%d")
    df = pd.read_sql("SELECT * FROM attribution_rates WHERE date>=? ORDER BY date,hour",
                     get_conn(), params=(co,))
    df["date"] = pd.to_datetime(df["date"]); return df

@st.cache_data(ttl=300)
def load_insights(n=20):
    return pd.read_sql("SELECT * FROM ai_insights ORDER BY generated_at DESC LIMIT ?",
                       get_conn(), params=(n,))

def daily_agg(pm, cc):
    a = pm.groupby("date").agg(total_cost=("cost","sum"), total_clicks=("clicks","sum"),
        total_impressions=("impressions","sum"), total_conversions=("conversions","sum")).reset_index()
    b = cc.groupby("date").agg(total_revenue=("total_revenue","sum"),
        margin_dollar=("margin_dollar","sum"), inbound_calls=("inbound_calls","sum"),
        data_submit_forms=("data_submit_forms","sum")).reset_index()
    m = a.merge(b, on="date", how="left")
    z = lambda x: x.replace(0, np.nan)
    m["CPL"]        = (m.total_cost    / z(m.total_conversions)).round(2)
    m["RPL"]        = (m.total_revenue / z(m.total_conversions)).round(2)
    m["CPC"]        = (m.total_cost    / z(m.total_clicks)).round(2)
    m["CVR"]        = (m.total_conversions / z(m.total_clicks)).round(4)
    m["ROAS"]       = (m.total_revenue / z(m.total_cost)).round(2)
    m["margin_pct"] = (m.margin_dollar / z(m.total_revenue)).round(4)
    return m

# ── AI ───────────────────────────────────────────────────────
@st.cache_resource
def get_claude(): return anthropic.Anthropic(api_key=CLAUDE_API_KEY)
@st.cache_resource
def get_oai():    return OpenAI(api_key=OPENAI_API_KEY)

def ask_claude(p, n=1000):
    try:
        r = get_claude().messages.create(model="claude-opus-4-5", max_tokens=n,
                messages=[{"role":"user","content":p}])
        return r.content[0].text
    except Exception as e: return f"⚠️ Claude: {e}"

def ask_oai(p):
    try:
        r = get_oai().chat.completions.create(model="gpt-4o", max_tokens=800,
            messages=[{"role":"system","content":"Senior paid media analytics expert. Concise, data-driven."},
                      {"role":"user","content":p}])
        return r.choices[0].message.content
    except Exception as e: return f"⚠️ OpenAI: {e}"

def ai(p, eng, n=1000): return ask_claude(p, n) if eng=="Claude" else ask_oai(p)

# ── ML ───────────────────────────────────────────────────────
def iso_forest(df, feats):
    df = df.copy().dropna(subset=feats)
    if len(df) < 10: df["anomaly"]=1; df["score"]=0.0; return df
    X = StandardScaler().fit_transform(df[feats])
    m = IsolationForest(contamination=0.08, random_state=42, n_estimators=150)
    df["anomaly"] = m.fit_predict(X); df["score"] = m.score_samples(X); return df

def prophet_fc(df, col, periods=14):
    try:
        from prophet import Prophet
        ts = df[["date",col]].rename(columns={"date":"ds",col:"y"}).dropna()
        ts["ds"] = pd.to_datetime(ts["ds"])
        m = Prophet(daily_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.15)
        m.fit(ts); fc = m.predict(m.make_future_dataframe(periods=periods))
        return fc[["ds","yhat","yhat_lower","yhat_upper"]]
    except: return pd.DataFrame()

def linear_fc(df, col, periods=14):
    ts = df[["date",col]].dropna().copy(); ts["t"] = np.arange(len(ts))
    lr = LinearRegression().fit(ts[["t"]], ts[col])
    fd = [ts["date"].iloc[-1]+timedelta(days=i+1) for i in range(periods)]
    pv = lr.predict(np.arange(len(ts), len(ts)+periods).reshape(-1,1))
    return pd.concat([ts.rename(columns={"date":"ds",col:"yhat"})[["ds","yhat"]],
                      pd.DataFrame({"ds":fd,"yhat":pv})], ignore_index=True)

# ── UI helpers ───────────────────────────────────────────────
def sv(df, col, d=0.0):
    try:
        v = df[col].iloc[0]; return float(v) if not pd.isna(v) else d
    except: return d

def pdelta(td, pd_, col):
    t = sv(td, col); p = sv(pd_, col); return (t-p)/abs(p) if p else None

def sh(title, icon="", color=BLU):
    st.markdown(f'<div class="sh"><span class="sh-dot" style="background:{color}"></span>'
                f'{icon+" " if icon else ""}{title}</div>', unsafe_allow_html=True)

def banner(icon, title, sub):
    st.markdown(f'<div class="bn"><span class="bn-ico">{icon}</span><div>'
                f'<div class="bn-title">{title}</div>'
                f'<div class="bn-sub">{sub}</div></div></div>', unsafe_allow_html=True)

def alrt(text, kind="info", label=""):
    cls = {"info":"al-info","ok":"al-ok","warn":"al-warn","crit":"al-crit","ai":"al-ai"}.get(kind,"al-info")
    lh = f'<div class="al-lbl">{label}</div>' if label else ""
    st.markdown(f'<div class="alert {cls}">{lh}{text}</div>', unsafe_allow_html=True)

def story(icon, head, body, issue="", action="", benefit=""):
    tags = ""
    if issue:   tags += f'<span class="tag-issue">⚠ {issue}</span>'
    if action:  tags += f'<span class="tag-act">→ {action}</span>'
    if benefit: tags += f'<span class="tag-ben">★ {benefit}</span>'
    st.markdown(f'<div class="sb"><div class="sb-head">{icon} {head}</div>'
                f'<div class="sb-body">{tags+"<br>" if tags else ""}{body}</div></div>',
                unsafe_allow_html=True)

def tooltip_html(label, title, body, issue="", fix="", benefit=""):
    ih = f'<div class="tb-issue">⚠ ISSUE: {issue}</div>' if issue else ""
    fh = f'<div class="tb-fix">✓ FIX: {fix}</div>' if fix else ""
    bh = f'<div class="tb-ben">★ BENEFIT: {benefit}</div>' if benefit else ""
    return (f'<span class="tw">{label}<span class="ti">?</span>'
            f'<div class="tb"><div class="tb-title">{title}</div>'
            f'<div class="tb-body">{body}</div>{ih}{fh}{bh}</div></span>')

def ribbon(items):
    html = '<div class="rib">'
    for lbl, val, note in items:
        html += (f'<div class="ri"><div class="ri-lbl">{lbl}</div>'
                 f'<div class="ri-val">{val}</div>'
                 f'<div class="ri-note">{note}</div></div>')
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def kc(label, value, delta=None, pre="", suf="", fmt=".0f",
       icon="", color=BLU, sub=""):
    try: vs = f"{pre}{value:{fmt}}{suf}"
    except: vs = str(value)
    if delta is not None:
        dp = delta*100
        dh = (f'<div class="kc-up">▲ {dp:.1f}% vs yesterday</div>' if delta > 0.005
              else f'<div class="kc-dn">▼ {abs(dp):.1f}% vs yesterday</div>' if delta < -0.005
              else f'<div class="kc-fl">— flat</div>')
    else: dh = ""
    sh_ = f'<div class="kc-sub">{sub}</div>' if sub else ""
    ih  = f'<span class="kc-ico">{icon}</span>' if icon else ""
    return (f'<div class="kc"><div class="kc-bar" style="background:{color}"></div>'
            f'{ih}<div class="kc-lbl">{label}</div>'
            f'<div class="kc-val">{vs}</div>{dh}{sh_}</div>')

def pcfg(fig, h=320, title="", lh=False):
    u = {**PL, "height": h}
    if title: u["title"] = dict(text=title, font=dict(size=13, color=TS), x=0.01, xanchor="left")
    if lh:    u["legend"] = {**PL["legend"], "orientation":"h", "y":1.08, "x":0}
    fig.update_layout(**u); return fig

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:10px 4px 6px">
      <div style="font-size:17px;font-weight:800;color:{TP};letter-spacing:-.4px">
        📡 Ads Intelligence</div>
      <div style="font-size:9px;color:{TM};letter-spacing:1px;
        text-transform:uppercase;margin-top:2px">Performance Command Center</div>
    </div>
    <hr style="border:none;border-top:1px solid {BDR};margin:6px 0 10px">
    """, unsafe_allow_html=True)

    page = st.radio("nav", [
        "🏠  Executive Overview",
        "📈  Performance Analysis",
        "🤖  ML & Forecasting",
        "📞  Call & Attribution",
        "🧠  AI Strategy Engine",
        "⚡  Real-Time Monitor",
    ], label_visibility="collapsed")

    st.markdown(f'<hr style="border:none;border-top:1px solid {BDR};margin:10px 0">',
                unsafe_allow_html=True)
    days_back = st.slider("Lookback days", 7, 30, 30)
    pf  = st.multiselect("Platforms", ["google","meta","microsoft"],
                         default=["google","meta","microsoft"])
    eng = st.radio("AI Engine", ["Claude","OpenAI GPT-4o"], index=0)

    st.markdown(f'<hr style="border:none;border-top:1px solid {BDR};margin:10px 0">',
                unsafe_allow_html=True)
    if st.button("⟳  Refresh Data", use_container_width=True):
        st.cache_data.clear(); st.rerun()

    st.markdown(f"""
    <div style="background:{SURF2};border:1px solid {BDR};border-radius:8px;
      padding:10px 12px;margin-top:10px">
      <div style="font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;
        color:{BLU};margin-bottom:6px">What This Dashboard Does</div>
      <div style="font-size:11px;color:{TS};line-height:1.75">
        📊 <b style="color:{TP}">Tracks</b> spend, revenue &amp; margin across 3 platforms<br>
        🤖 <b style="color:{TP}">Detects</b> anomalies with Isolation Forest ML<br>
        📉 <b style="color:{TP}">Forecasts</b> KPIs 14 days ahead via Prophet<br>
        🧠 <b style="color:{TP}">Explains</b> every issue with AI root-cause analysis<br>
        💡 <b style="color:{TP}">Recommends</b> budget shifts and creative fixes
      </div>
    </div>
    <div style="font-size:10px;color:{TM};margin-top:8px">
      Updated: {datetime.now().strftime("%b %d · %H:%M")}</div>
    """, unsafe_allow_html=True)

# ── Load data ────────────────────────────────────────────────
pm_raw  = load_pm(days_back)
cc_raw  = load_cc(days_back)
cq_raw  = load_cq(days_back)
at_raw  = load_attr(days_back)
pm      = pm_raw[pm_raw["platform"].isin(pf)] if pf else pm_raw
daily   = daily_agg(pm, cc_raw)
td      = daily["date"].max() if not daily.empty else pd.Timestamp.utcnow()
td_str  = pd.Timestamp(td).strftime("%Y-%m-%d")
tday    = daily[daily["date"] == td]
yday    = daily[daily["date"] == td - timedelta(days=1)]
mstart  = pd.Timestamp(td).replace(day=1)
mtd     = daily[daily["date"] >= mstart]

# ═══════════════════════════════════════════════════════════
#  PAGE 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════
if page == "🏠  Executive Overview":
    banner("🏠", "Executive Overview",
           f"Paid ads profitability at a glance · {td_str} · {', '.join(pf)}")

    story("📖", "What You're Looking At",
          "This page answers: <b>Are we spending money profitably?</b> "
          "KPIs show today vs yesterday. The chart shows whether revenue outpaces spend — "
          "if those lines cross, you have a margin problem. Everything here flows from "
          "raw spend → clicks → conversions → revenue → margin.",
          issue="No single cross-platform profitability view",
          action="Unified KPIs across Google · Meta · Microsoft",
          benefit="Catch margin erosion in hours, not at month-end review")

    # ── KPIs ─────────────────────────────────────────────────
    spend  = sv(tday, "total_cost");    rev   = sv(tday, "total_revenue")
    margin = sv(tday, "margin_dollar"); mgpct = sv(tday, "margin_pct")*100
    cpl    = sv(tday, "CPL");          roas  = sv(tday, "ROAS")
    cvr    = sv(tday, "CVR")*100;      cpc   = sv(tday, "CPC")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    d_cpl = pdelta(tday, yday, "CPL")
    with c1: st.markdown(kc("Total Spend",   spend,  pdelta(tday,yday,"total_cost"),    "$","",",.0f","💸",RED,  "Budget consumed today"),    unsafe_allow_html=True)
    with c2: st.markdown(kc("Total Revenue", rev,    pdelta(tday,yday,"total_revenue"), "$","",",.0f","💰",GRN,  "Revenue generated"),         unsafe_allow_html=True)
    with c3: st.markdown(kc("Gross Margin",  margin, pdelta(tday,yday,"margin_dollar"), "$","",",.0f","📊",BLU,  f"{mgpct:.1f}% margin rate"), unsafe_allow_html=True)
    with c4: st.markdown(kc("Cost per Lead", cpl,    -d_cpl if d_cpl else None,         "$","",".2f", "🎯",AMB,  "Avg acquisition cost"),      unsafe_allow_html=True)
    with c5: st.markdown(kc("ROAS",          roas,   pdelta(tday,yday,"ROAS"),           "","x",".2f", "📈",PUR,  "Return per $1 spent"),        unsafe_allow_html=True)
    with c6: st.markdown(kc("Conv. Rate",    cvr,    pdelta(tday,yday,"CVR"),            "","%" ,".1f","✅",TEA,  "Click → conversion"),         unsafe_allow_html=True)

    # ── Hover explanation strip ───────────────────────────────
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 14px;font-size:12px;color:{TM}">'
        + tooltip_html("CPL", "Cost per Lead — Core Efficiency",
                       "CPL = Spend ÷ Conversions. The price you pay per acquired lead.",
                       "Rising CPL means spend up or conversions down",
                       "Pause weak ad groups, tighten targeting",
                       "Every $1 CPL reduction directly improves margin")
        + "&ensp;"
        + tooltip_html("ROAS", "Return on Ad Spend",
                       "ROAS = Revenue ÷ Spend. Below 1.0x = losing money on ads.",
                       "Low ROAS means campaigns overspend vs return",
                       "Shift budget to highest-ROAS platforms first",
                       "+0.2x ROAS improvement = thousands more in monthly margin")
        + "&ensp;"
        + tooltip_html("CVR", "Conversion Rate — Quality Signal",
                       "CVR = Conversions ÷ Clicks. Reflects landing page + audience quality.",
                       "Dropping CVR = traffic quality or page quality issue",
                       "A/B test landing pages, refine audience segments",
                       "1% CVR improvement drops CPL by 15–20% automatically")
        + "&ensp;"
        + tooltip_html("Margin", "Gross Margin — Bottom Line",
                       "Margin = Revenue − Spend. What you actually keep after ad costs.",
                       "Low margin % means spend is eating into revenue",
                       "Cut high-CPL campaigns, scale high-ROAS ones",
                       "Protecting margin % matters more than raw revenue growth")
        + '</div>', unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────
    cl, cr = st.columns([3, 1])
    with cl:
        sh("Revenue vs Spend — Daily Trend", "📉", BLU)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["total_revenue"], name="Revenue",
            fill="tozeroy", fillcolor=hr(GRN, 0.12), line=dict(color=GRN, width=2.5),
            hovertemplate="<b>%{x|%b %d}</b><br>Revenue: $%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["total_cost"], name="Ad Spend",
            fill="tozeroy", fillcolor=hr(RED, 0.1), line=dict(color=RED, width=2),
            hovertemplate="<b>%{x|%b %d}</b><br>Spend: $%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["margin_dollar"], name="Margin",
            line=dict(color=AMB, width=2, dash="dot"),
            hovertemplate="<b>%{x|%b %d}</b><br>Margin: $%{y:,.0f}<extra></extra>"))
        pcfg(fig, 300, lh=True)
        st.plotly_chart(fig, use_container_width=True)
        # Dynamic narrative
        gap = rev - spend
        if gap > 0:
            story("✅", "Revenue Outpacing Spend",
                  f"Revenue (${rev:,.0f}) exceeds spend (${spend:,.0f}) by <b>${gap:,.0f}</b> — "
                  f"campaigns are profitable. Margin rate: {mgpct:.1f}%.",
                  benefit=f"${gap:,.0f} positive gap — healthy profitability")
        else:
            story("🚨", "Spend Exceeding Revenue",
                  f"Spend (${spend:,.0f}) exceeds revenue (${rev:,.0f}) by <b>${abs(gap):,.0f}</b>. "
                  f"Campaigns are currently loss-making.",
                  issue=f"${abs(gap):,.0f} deficit — campaigns losing money",
                  action="Pause lowest-ROAS campaigns, review bid strategies",
                  benefit="Stopping losses early prevents month-end budget overruns")

    with cr:
        sh("Today's Spend Split", "🥧", PUR)
        spd = pm[pm["date"]==pm["date"].max()].groupby("platform")["cost"].sum().reset_index()
        fig2 = go.Figure(go.Pie(
            labels=[p.title() for p in spd["platform"]], values=spd["cost"], hole=0.6,
            marker=dict(colors=[PC.get(p, BLU) for p in spd["platform"]],
                        line=dict(color=BG, width=2)),
            textfont=dict(size=11, color=TP),
            hovertemplate="<b>%{label}</b><br>$%{value:,.0f} (%{percent})<extra></extra>"))
        fig2.update_layout(**{**PL, "height":260,
                              "margin":dict(l=0,r=0,t=10,b=30),
                              "legend":dict(orientation="h", y=-0.1, font=dict(size=11))})
        st.plotly_chart(fig2, use_container_width=True)
        for _, row in spd.iterrows():
            p = row["platform"]; pct = row["cost"]/spd["cost"].sum()*100 if spd["cost"].sum() else 0
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:4px 0;border-bottom:1px solid {BDR};font-size:11px">'
                f'<span style="color:{PC.get(p,BLU)};font-weight:700">{p.title()}</span>'
                f'<span style="color:{TP}">${row["cost"]:,.0f}</span>'
                f'<span style="color:{TM}">{pct:.1f}%</span></div>', unsafe_allow_html=True)

    # ── MTD ───────────────────────────────────────────────────
    sh("Month-to-Date Performance", "📅", TEA)
    ribbon([
        ("MTD Spend",   f"${mtd['total_cost'].sum():,.0f}",    "Total budget consumed"),
        ("MTD Revenue", f"${mtd['total_revenue'].sum():,.0f}", "Revenue from all campaigns"),
        ("MTD Margin",  f"${mtd['margin_dollar'].sum():,.0f}", "Net value after spend"),
        ("MTD Leads",   f"{mtd['total_conversions'].sum():,.0f}", "Total conversions"),
        ("Avg ROAS",    f"{mtd['ROAS'].mean():.2f}x",          "Return on spend MTD"),
        ("Avg CPL",     f"${mtd['CPL'].mean():.2f}",           "Cost per lead MTD"),
    ])

    # ── WoW ───────────────────────────────────────────────────
    sh("Week-over-Week", "📊", AMB)
    if len(daily) >= 14:
        tw = daily.tail(7); lw = daily.iloc[-14:-7]
        wc = ["total_cost","total_revenue","margin_dollar","CPL","ROAS","CVR"]
        wl = ["Spend","Revenue","Margin","CPL","ROAS","CVR"]
        twv = [tw[c].mean() for c in wc]; lwv = [lw[c].mean() for c in wc]
        wp  = [((t-l)/abs(l)*100 * (-1 if c=="CPL" else 1)) if l else 0
               for t,l,c in zip(twv,lwv,wc)]
        clrs = [GRN if v>=0 else RED for v in wp]

        w1, w2 = st.columns([2,1])
        with w1:
            fig_w = go.Figure(go.Bar(x=wl, y=wp,
                marker=dict(color=clrs, line=dict(width=0)),
                text=[f"{v:+.1f}%" for v in wp], textposition="outside",
                textfont=dict(size=12, color=TP),
                hovertemplate="<b>%{x}</b><br>WoW: %{y:+.1f}%<extra></extra>"))
            fig_w.update_layout(**{**PL, "height":230, "showlegend":False,
                "yaxis_title":"WoW Change (%)",
                "shapes":[dict(type="line",x0=-.5,x1=5.5,y0=0,y1=0,
                               line=dict(color=BDR2,width=1,dash="dot"))]})
            st.plotly_chart(fig_w, use_container_width=True)
        with w2:
            sh("WoW Scorecard", "", BLU)
            interp = {"CPL":"✓ cheaper leads|⚠ leads cost more",
                      "ROAS":"✓ better returns|⚠ worse returns",
                      "CVR":"✓ better converting|⚠ traffic quality drop"}
            for lbl, pct in zip(wl, wp):
                cls   = "kc-up" if pct>=0 else "kc-dn"; arrow = "▲" if pct>=0 else "▼"
                parts = interp.get(lbl,"").split("|")
                note  = (parts[0] if pct>=0 else parts[1]) if parts[0] else ""
                st.markdown(
                    f'<div style="padding:5px 0;border-bottom:1px solid {BDR};font-size:11px">'
                    f'<div style="display:flex;justify-content:space-between">'
                    f'<span style="color:{TS}">{lbl}</span>'
                    f'<span class="{cls}">{arrow} {abs(pct):.1f}%</span></div>'
                    f'<div style="font-size:10px;color:{TM}">{note}</div></div>',
                    unsafe_allow_html=True)

        best = wl[wp.index(max(wp))]; worst = wl[wp.index(min(wp))]
        story("📖", "Week-over-Week Story",
              f"Biggest gain: <b>{best}</b> ({max(wp):+.1f}%). "
              f"Biggest concern: <b>{worst}</b> ({min(wp):+.1f}%). "
              f"WoW tracking detects trend shifts before they compound into monthly losses.",
              issue=f"{worst} declining WoW" if min(wp) < -5 else "All metrics within range",
              action=f"Investigate {worst}, protect {best} momentum",
              benefit="Early WoW signals save 3–5 days of wasted spend")

    # ── Platform scorecard ────────────────────────────────────
    sh("Platform Scorecard — Today", "🏆", GRN)
    ps = pm[pm["date"]==pm["date"].max()].groupby("platform").agg(
        Spend=("cost","sum"), Clicks=("clicks","sum"),
        Impressions=("impressions","sum"), Conversions=("conversions","sum")).reset_index()
    ps["CTR"]  = (ps["Clicks"]/ps["Impressions"].replace(0,np.nan)*100).round(2)
    ps["CPC"]  = (ps["Spend"]/ps["Clicks"].replace(0,np.nan)).round(2)
    ps["CPL"]  = (ps["Spend"]/ps["Conversions"].replace(0,np.nan)).round(2)
    ps["CVR%"] = (ps["Conversions"]/ps["Clicks"].replace(0,np.nan)*100).round(2)
    st.dataframe(ps.style
        .format({"Spend":"${:,.0f}","Clicks":"{:,.0f}","Impressions":"{:,.0f}",
                 "Conversions":"{:,.0f}","CTR":"{:.2f}%","CPC":"${:.2f}",
                 "CPL":"${:.2f}","CVR%":"{:.2f}%"})
        .background_gradient(subset=["CPL"], cmap="RdYlGn_r")
        .background_gradient(subset=["CVR%","CTR"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True)
    st.caption("💡 Green CPL = cheaper leads (inverted). Green CVR/CTR = better performance.")

# ═══════════════════════════════════════════════════════════
#  PAGE 2 — PERFORMANCE ANALYSIS
# ═══════════════════════════════════════════════════════════
elif page == "📈  Performance Analysis":
    banner("📈", "Performance Analysis", "Spend · CTR · CPC · CVR · Funnel — from click to conversion")

    story("📖", "Why This Page Matters",
          "Spend numbers tell you <i>how much</i>. This page tells you <i>how well</i>. "
          "CTR shows if ads attract attention. CPC shows what you pay for it. "
          "CVR shows if that attention converts. Together they pinpoint where money leaks.",
          issue="Spend can rise while quality falls — invisible without these metrics",
          action="Monitor CTR · CPC · CVR daily per platform",
          benefit="Finding one leaky funnel stage recovers 15–30% of wasted spend")

    pm_d = pm.groupby(["date","platform"]).agg(cost=("cost","sum"), clicks=("clicks","sum"),
        impressions=("impressions","sum"), conversions=("conversions","sum")).reset_index()
    pm_d["CPC"] = (pm_d["cost"]/pm_d["clicks"].replace(0,np.nan)).round(2)
    pm_d["CTR"] = (pm_d["clicks"]/pm_d["impressions"].replace(0,np.nan)*100).round(2)
    pm_d["CVR"] = (pm_d["conversions"]/pm_d["clicks"].replace(0,np.nan)*100).round(2)
    pm_d["CPL"] = (pm_d["cost"]/pm_d["conversions"].replace(0,np.nan)).round(2)

    t1,t2,t3,t4 = st.tabs(["📊 Spend & Volume","🖱️ CTR · CPC · CVR","🕐 Intraday","📋 KPI Table"])

    with t1:
        c1, c2 = st.columns(2)
        with c1:
            sh("Daily Spend by Platform","💸",RED)
            fig = go.Figure()
            for p in pm_d["platform"].unique():
                d = pm_d[pm_d["platform"]==p]
                fig.add_trace(go.Scatter(x=d["date"], y=d["cost"], name=p.title(),
                    fill="tozeroy", fillcolor=hr(PC[p], 0.1), line=dict(color=PC[p], width=2),
                    hovertemplate=f"<b>{p.title()}</b> %{{x|%b %d}}: $%{{y:,.0f}}<extra></extra>"))
            pcfg(fig, 290, lh=True); st.plotly_chart(fig, use_container_width=True)

        with c2:
            sh("Daily Conversions by Platform","🎯",GRN)
            fig2 = px.bar(pm_d, x="date", y="conversions", color="platform",
                color_discrete_map=PC, barmode="stack")
            fig2.update_traces(marker_line_width=0)
            pcfg(fig2, 290, lh=True); st.plotly_chart(fig2, use_container_width=True)

        sh("Conversion Funnel — Period Total","🔽",TEA)
        ti = pm_d["impressions"].sum(); tc = pm_d["clicks"].sum(); tv = pm_d["conversions"].sum()
        fig_f = go.Figure(go.Funnel(
            y=["Impressions","Clicks","Conversions"], x=[ti,tc,tv],
            textposition="inside", texttemplate="%{value:,.0f}  (%{percentInitial:.2%})",
            marker=dict(color=[BLU,AMB,GRN], line=dict(color=BG,width=2)),
            connector=dict(line=dict(color=BDR2,width=1))))
        pcfg(fig_f, 220); fig_f.update_layout(margin=dict(l=140,r=20,t=20,b=10))
        st.plotly_chart(fig_f, use_container_width=True)
        ctr_a = tc/ti*100 if ti else 0; cvr_a = tv/tc*100 if tc else 0
        ribbon([("Imp→Click",f"{ctr_a:.2f}%","Industry avg 3–6%"),
                ("Click→Conv",f"{cvr_a:.2f}%","Industry avg 3–8%"),
                ("Impressions",f"{ti:,.0f}","Reach"),
                ("Clicks",f"{tc:,.0f}","Engaged traffic"),
                ("Conversions",f"{tv:,.0f}","Qualified leads")])

    with t2:
        st.markdown(
            f'<div style="font-size:12px;color:{TM};margin-bottom:10px">'
            + tooltip_html("Why CTR·CPC·CVR together?",
                           "The Efficiency Triangle",
                           "CTR = ad relevance. CPC = auction cost. CVR = landing page quality. "
                           "A problem in any one corrupts CPL.",
                           "High CPC + Low CVR = paying premium for unconverting traffic",
                           "Fix the lowest-CVR platform first — biggest CPL leverage",
                           "Improving CVR by 1% drops CPL by 15–20%")
            + '</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_ctr = go.Figure()
            for p in pm_d["platform"].unique():
                d = pm_d[pm_d["platform"]==p]
                fig_ctr.add_trace(go.Scatter(x=d["date"], y=d["CTR"], name=p.title(),
                    mode="lines+markers", line=dict(color=PC[p],width=2), marker=dict(size=5),
                    hovertemplate=f"<b>{p.title()}</b> CTR: %{{y:.2f}}%<extra></extra>"))
            pcfg(fig_ctr, 280, lh=True)
            fig_ctr.update_layout(yaxis_title="CTR %")
            st.plotly_chart(fig_ctr, use_container_width=True)
        with c2:
            fig_cpc = go.Figure()
            for p in pm_d["platform"].unique():
                d = pm_d[pm_d["platform"]==p]
                fig_cpc.add_trace(go.Scatter(x=d["date"], y=d["CPC"], name=p.title(),
                    mode="lines+markers", line=dict(color=PC[p],width=2), marker=dict(size=5),
                    hovertemplate=f"<b>{p.title()}</b> CPC: $%{{y:.2f}}<extra></extra>"))
            pcfg(fig_cpc, 280, lh=True)
            fig_cpc.update_layout(yaxis_title="CPC $")
            st.plotly_chart(fig_cpc, use_container_width=True)

        sh("Efficiency Matrix — CPC vs CVR","🎯",PUR)
        fig_sc = go.Figure()
        for p in pm_d["platform"].unique():
            d = pm_d[pm_d["platform"]==p]
            fig_sc.add_trace(go.Scatter(x=d["CPC"], y=d["CVR"], name=p.title(),
                mode="markers",
                marker=dict(color=PC[p], size=d["cost"]/d["cost"].max()*20+6 if d["cost"].max()>0 else 8,
                            line=dict(color=BG,width=1), opacity=0.85),
                text=d["date"].dt.strftime("%b %d"),
                hovertemplate=f"<b>{p.title()}</b><br>%{{text}}<br>CPC: $%{{x:.2f}}<br>CVR: %{{y:.2f}}%<extra></extra>"))
        cpc_m = pm_d["CPC"].mean(); cvr_m = pm_d["CVR"].mean()
        fig_sc.add_hline(y=cvr_m, line_dash="dot", line_color=BDR2, line_width=1)
        fig_sc.add_vline(x=cpc_m, line_dash="dot", line_color=BDR2, line_width=1)
        fig_sc.add_annotation(x=cpc_m*0.6, y=cvr_m*1.3, text="✓ BEST\nLow CPC+High CVR",
            font=dict(color=GRN,size=10), showarrow=False, bgcolor=SURF2, borderpad=4)
        fig_sc.add_annotation(x=cpc_m*1.4, y=cvr_m*0.6, text="⚠ WORST\nHigh CPC+Low CVR",
            font=dict(color=RED,size=10), showarrow=False, bgcolor=SURF2, borderpad=4)
        pcfg(fig_sc, 360, "Bubble = spend · Upper-left = best efficiency", lh=True)
        fig_sc.update_layout(xaxis_title="CPC ($)", yaxis_title="CVR (%)")
        st.plotly_chart(fig_sc, use_container_width=True)

    with t3:
        sh("Today — Hourly Spend & Conversions","🕐",BLU)
        th = pm[pm["date"]==pm["date"].max()].groupby(["hour","platform"]).agg(
            cost=("cost","sum"), clicks=("clicks","sum"), conversions=("conversions","sum")).reset_index()
        fig_h = make_subplots(rows=2, cols=1,
            subplot_titles=["Hourly Spend ($)","Hourly Conversions"],
            shared_xaxes=True, vertical_spacing=0.12)
        for p in th["platform"].unique():
            d = th[th["platform"]==p]
            fig_h.add_trace(go.Bar(x=d["hour"], y=d["cost"], name=p.title(),
                marker=dict(color=PC[p],line=dict(width=0))), row=1, col=1)
            fig_h.add_trace(go.Scatter(x=d["hour"], y=d["conversions"],
                line=dict(color=PC[p],width=2), mode="lines+markers", showlegend=False), row=2, col=1)
        fig_h.update_layout(**{**PL,"height":500,"barmode":"group",
                                "legend":dict(orientation="h",y=1.05,x=0)})
        fig_h.update_xaxes(title_text="Hour (UTC)", row=2)
        st.plotly_chart(fig_h, use_container_width=True)
        story("💡","Reading the Hourly Pattern",
              "Peak spend hours should align with peak conversion hours. "
              "If spend peaks earlier than conversions, dayparting is misaligned — "
              "you're paying for clicks before your audience is ready to convert.",
              issue="Spend and conversion peaks misaligned by hour",
              action="Adjust bid modifiers or ad scheduling",
              benefit="Aligning spend to conversion hours improves CVR by 10–25%")

    with t4:
        sh("Full KPI Table","📋",BLU)
        disp = daily[["date","total_cost","total_revenue","margin_dollar","margin_pct",
                       "CPL","RPL","CPC","CVR","ROAS","total_conversions","total_clicks"]].copy()
        disp["margin_pct"] = (disp["margin_pct"]*100).round(1)
        disp["CVR"] = (disp["CVR"]*100).round(2)
        disp = disp.sort_values("date", ascending=False)
        disp.columns = ["Date","Spend","Revenue","Margin $","Margin %",
                         "CPL","RPL","CPC","CVR %","ROAS","Conversions","Clicks"]
        st.dataframe(disp.style
            .format({"Spend":"${:,.0f}","Revenue":"${:,.0f}","Margin $":"${:,.0f}",
                     "Margin %":"{:.1f}%","CPL":"${:.2f}","RPL":"${:.2f}",
                     "CPC":"${:.2f}","CVR %":"{:.2f}%","ROAS":"{:.2f}x",
                     "Conversions":"{:,.0f}","Clicks":"{:,.0f}"})
            .background_gradient(subset=["Margin $"], cmap="RdYlGn")
            .background_gradient(subset=["ROAS"],     cmap="RdYlGn")
            .background_gradient(subset=["CPL"],      cmap="RdYlGn_r"),
            use_container_width=True, height=520, hide_index=True)
        st.caption("💡 Margin $ and ROAS: green=good · CPL: green=cheap leads (color inverted)")

# ═══════════════════════════════════════════════════════════
#  PAGE 3 — ML & FORECASTING
# ═══════════════════════════════════════════════════════════
elif page == "🤖  ML & Forecasting":
    banner("🤖","ML & Forecasting",
           "Isolation Forest · Prophet Forecasting · K-Means Clustering")

    story("📖","What ML Adds That Humans Miss",
          "Manual review catches obvious problems. ML catches subtle compound patterns — "
          "a CPL 18% above normal across 3 days, or a revenue forecast showing a cliff in 9 days. "
          "These signals exist in data but are invisible without algorithms.",
          issue="Manual monitoring misses multi-metric compound anomalies",
          action="Isolation Forest scans 6 KPIs simultaneously",
          benefit="ML catches issues 2–4 days earlier than manual review")

    ta,tf,tc = st.tabs(["🔴 Anomaly Detection","📉 Forecasting","🗂️ Clusters"])

    with ta:
        feats = ["total_cost","total_revenue","margin_dollar","CPL","CVR","ROAS"]
        df_ml = iso_forest(daily.dropna(subset=feats).copy(), feats)
        anoms = df_ml[df_ml["anomaly"]==-1]; norm = df_ml[df_ml["anomaly"]==1]

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(kc("Days Analyzed",len(df_ml),None,"","",".0f","📅",BLU), unsafe_allow_html=True)
        with c2: st.markdown(kc("Anomalies Found",len(anoms),None,"","",".0f","🚨",RED), unsafe_allow_html=True)
        with c3: st.markdown(kc("Anomaly Rate",len(anoms)/max(len(df_ml),1)*100,None,"","%",".1f","📊",AMB), unsafe_allow_html=True)
        with c4: st.markdown(kc("Clean Days",len(norm),None,"","",".0f","✅",GRN), unsafe_allow_html=True)

        st.markdown(
            f'<div style="font-size:12px;color:{TM};margin:6px 0 10px">'
            + tooltip_html("How does Isolation Forest work?",
                           "Isolation Forest — Unsupervised ML",
                           "Randomly partitions data. Points isolated quickly (fewer splits) are anomalies.",
                           "Normal days cluster together — anomalous days stand apart statistically",
                           "Cross-reference flagged dates with your campaign change log",
                           "Catching one anomalous spend day/month saves 3–8% of budget")
            + '</div>', unsafe_allow_html=True)

        ax, ay = st.columns(2)
        mx = ax.selectbox("X-Axis", feats, index=0, key="ax")
        my = ay.selectbox("Y-Axis", feats, index=3, key="ay")

        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(x=norm[mx], y=norm[my], mode="markers", name="Normal",
            text=norm["date"].dt.strftime("%b %d"),
            marker=dict(color=GRN,size=9,opacity=0.75,line=dict(color=BG,width=1)),
            hovertemplate="<b>%{text}</b><br>"+mx+": %{x:.2f}<br>"+my+": %{y:.2f}<extra>Normal</extra>"))
        if not anoms.empty:
            fig_a.add_trace(go.Scatter(x=anoms[mx], y=anoms[my], mode="markers", name="⚠️ Anomaly",
                text=anoms["date"].dt.strftime("%b %d"),
                marker=dict(color=RED,size=14,symbol="x",line=dict(color=RED,width=2.5)),
                hovertemplate="<b>%{text} — ANOMALY</b><br>"+mx+": %{x:.2f}<br>"+my+": %{y:.2f}<extra></extra>"))
        pcfg(fig_a, 360, f"Anomaly Scatter: {mx} vs {my}", lh=True)
        fig_a.update_layout(xaxis_title=mx, yaxis_title=my)
        st.plotly_chart(fig_a, use_container_width=True)

        if not anoms.empty:
            sh("Flagged Anomalous Days","🚨",RED)
            ad = anoms[["date","total_cost","total_revenue","margin_dollar","CPL","ROAS","score"]].copy()
            ad["date"] = ad["date"].dt.strftime("%Y-%m-%d")
            ad["score"] = ad["score"].round(4)
            st.dataframe(ad.sort_values("score").style
                .format({"total_cost":"${:,.0f}","total_revenue":"${:,.0f}",
                         "margin_dollar":"${:,.0f}","CPL":"${:.2f}","ROAS":"{:.2f}x"})
                .background_gradient(subset=["score"], cmap="Reds_r"),
                use_container_width=True, hide_index=True)
            st.caption("Lower anomaly score = more unusual. Cross-reference with campaign changes.")
            if st.button("🧠 AI Root Cause Analysis"):
                worst = ad.head(3).to_string(index=False)
                prompt = (f"ML flagged these anomalous days:\n{worst}\n\n"
                          "For EACH date:\n🔴 ISSUE: what went wrong\n"
                          "🔍 ROOT CAUSE: why\n✅ ACTION: specific fix\n★ BENEFIT: expected gain")
                with st.spinner("Analyzing..."):
                    alrt(ai(prompt,eng,900), "ai", f"🤖 {eng} · Anomaly Analysis")

    with tf:
        story("📖","Why Forecast?",
              "Looking back shows what happened. Forecasting shows what's coming. "
              "A 14-day forecast lets you adjust budgets before problems materialize.",
              issue="Budget decisions made without forward visibility",
              action="Prophet models daily/weekly seasonality automatically",
              benefit="Proactive adjustments prevent overspend by 10–20%")

        fm = st.selectbox("Metric to Forecast",
            ["total_revenue","total_cost","margin_dollar","CPL","ROAS","CVR"])
        fd = st.slider("Forecast horizon (days)", 7, 30, 14, key="fd")

        with st.spinner("Fitting model..."):
            fc = prophet_fc(daily, fm, fd); method = "Prophet"
            if fc.empty: fc = linear_fc(daily, fm, fd); method = "Linear Trend"

        if not fc.empty:
            he = daily["date"].max()
            hist = daily[["date",fm]].dropna()
            fut  = fc[fc["ds"] > he]
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=hist["date"], y=hist[fm], name="Actual",
                line=dict(color=TP,width=2.5),
                hovertemplate="%{x|%b %d}: %{y:,.2f}<extra>Actual</extra>"))
            if "yhat_lower" in fc.columns:
                fig_fc.add_trace(go.Scatter(
                    x=pd.concat([fut["ds"],fut["ds"][::-1]]),
                    y=pd.concat([fut["yhat_upper"],fut["yhat_lower"][::-1]]),
                    fill="toself", fillcolor=hr(BLU,0.1),
                    line=dict(color="rgba(0,0,0,0)"), name="Confidence Band"))
            fig_fc.add_trace(go.Scatter(x=fut["ds"], y=fut["yhat"],
                name=f"Forecast ({method})",
                line=dict(color=BLU,width=2.5,dash="dash"),
                hovertemplate="%{x|%b %d}: %{y:,.2f}<extra>Forecast</extra>"))
            fig_fc.add_vline(x=str(he), line_dash="dot", line_color=AMB, line_width=1.5)
            fig_fc.add_annotation(x=str(he), y=1, yref="paper",
                text="← Past | Future →", font=dict(color=AMB,size=10),
                showarrow=False, bgcolor=SURF2, yshift=14)
            pcfg(fig_fc, 400, f"{fm} — {fd}-Day Forecast ({method})", lh=True)
            st.plotly_chart(fig_fc, use_container_width=True)
            f1,f2,f3,f4 = st.columns(4)
            trend = (fut["yhat"].iloc[-1]-fut["yhat"].iloc[0])/abs(fut["yhat"].iloc[0])*100 if len(fut)>1 else 0
            with f1: st.markdown(kc("7-Day Avg",fut.head(7)["yhat"].mean(),None,"","",",.2f","📈",BLU), unsafe_allow_html=True)
            with f2: st.markdown(kc("Forecast High",fut["yhat"].max(),None,"","",",.2f","⬆️",GRN), unsafe_allow_html=True)
            with f3: st.markdown(kc("Forecast Low",fut["yhat"].min(),None,"","",",.2f","⬇️",AMB), unsafe_allow_html=True)
            with f4: st.markdown(kc("Trend",abs(trend),None,"","%",".1f","📊",GRN if trend>0 else RED,
                                    "improving" if trend>0 else "declining"), unsafe_allow_html=True)

    with tc:
        story("📖","What Clustering Reveals",
              "K-Means groups days into performance archetypes — great/average/bad. "
              "Once clustered, ask: what made the great days great? "
              "That's your growth playbook.",
              issue="Can't identify patterns in noisy daily data",
              action="K-Means clusters days by 5-metric performance profile",
              benefit="Replicating 'great day' conditions lifts average by 15–30%")

        cl_f = ["total_cost","total_revenue","CPL","ROAS","CVR"]
        df_cl = daily.dropna(subset=cl_f).copy()
        if len(df_cl) >= 6:
            nk = st.slider("Clusters (K)", 2, 5, 3, key="kn")
            Xs = StandardScaler().fit_transform(df_cl[cl_f])
            km = KMeans(n_clusters=nk, random_state=42, n_init=10)
            df_cl["Cluster"] = ["Cluster "+str(i+1) for i in km.fit_predict(Xs)]
            cl_clrs = {f"Cluster {i+1}": c for i,c in enumerate([BLU,GRN,AMB,RED,PUR])}

            cc1, cc2 = st.columns([3,2])
            with cc1:
                fig_cl = go.Figure()
                for cn, color in cl_clrs.items():
                    d = df_cl[df_cl["Cluster"]==cn]
                    if not d.empty:
                        fig_cl.add_trace(go.Scatter(x=d["total_cost"], y=d["total_revenue"],
                            name=cn, mode="markers",
                            marker=dict(color=color, size=d["ROAS"]*4+6,
                                        line=dict(color=BG,width=1), opacity=0.85),
                            text=d["date"].dt.strftime("%b %d"),
                            hovertemplate=f"<b>{cn}</b><br>%{{text}}<br>Spend:$%{{x:,.0f}}<br>Rev:$%{{y:,.0f}}<extra></extra>"))
                pcfg(fig_cl, 380, "Clusters — Spend vs Revenue (bubble=ROAS)", lh=True)
                fig_cl.update_layout(xaxis_title="Spend ($)", yaxis_title="Revenue ($)")
                st.plotly_chart(fig_cl, use_container_width=True)
            with cc2:
                sh("Cluster Profiles","📊",PUR)
                centers = df_cl.groupby("Cluster")[cl_f].mean().round(2)
                med_roas = centers["ROAS"].median()
                for cn, row in centers.iterrows():
                    color = cl_clrs.get(cn, BLU)
                    tag = "⭐ Top Performer" if row["ROAS"]>med_roas else "⚠ Below Average"
                    st.markdown(
                        f'<div style="background:{SURF2};border:1px solid {color}50;'
                        f'border-left:3px solid {color};border-radius:8px;'
                        f'padding:10px 14px;margin:6px 0">'
                        f'<div style="font-size:11px;font-weight:700;color:{color};margin-bottom:5px">'
                        f'{cn} · {tag}</div>'
                        f'<div style="font-size:11px;color:{TS};line-height:1.8">'
                        f'Spend: ${row["total_cost"]:,.0f} · Rev: ${row["total_revenue"]:,.0f}<br>'
                        f'CPL: ${row["CPL"]:.2f} · ROAS: {row["ROAS"]:.2f}x · CVR: {row["CVR"]*100:.2f}%'
                        f'</div></div>', unsafe_allow_html=True)
        else:
            alrt("Need at least 6 days of data for clustering.", "warn")


# ═══════════════════════════════════════════════════════════
#  PAGE 4 — CALL & ATTRIBUTION
# ═══════════════════════════════════════════════════════════
elif page == "📞  Call & Attribution":
    banner("📞","Call & Attribution",
           "Revenue by type · Call quality ratios · Attribution rates · Issue→Impact→Fix")

    story("📖","Why Call & Attribution Data Is Critical",
          "Platform conversion counts include low-quality leads that never generate revenue. "
          "This page tracks what happens <b>after the click</b>: call duration, income verification, "
          "and carrier status — the real signals of lead quality.",
          issue="Platform conversions inflate numbers with unqualified leads",
          action="Track inbound call quality, duration, and verified income rates",
          benefit="Optimizing toward quality conversions improves revenue-per-lead by 30–50%")

    t1,t2,t3 = st.tabs(["💰 Call Center Revenue","📞 Call Quality","🔗 Attribution Rates"])

    with t1:
        cc_d = cc_raw.groupby("date").agg(
            calls_revenue=("calls_revenue","sum"), data_revenue=("data_revenue","sum"),
            total_revenue=("total_revenue","sum"), inbound_calls=("inbound_calls","sum"),
            data_submit_forms=("data_submit_forms","sum"),
            margin_dollar=("margin_dollar","sum")).reset_index()

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(kc("Total Revenue",cc_d["total_revenue"].sum(),None,"$","",",.0f","💰",GRN), unsafe_allow_html=True)
        with c2: st.markdown(kc("Calls Revenue",cc_d["calls_revenue"].sum(),None,"$","",",.0f","📞",BLU), unsafe_allow_html=True)
        with c3: st.markdown(kc("Data Revenue", cc_d["data_revenue"].sum(), None,"$","",",.0f","📋",TEA), unsafe_allow_html=True)
        with c4: st.markdown(kc("Inbound Calls",cc_d["inbound_calls"].sum(),None,"","",",.0f","📱",PUR), unsafe_allow_html=True)

        sh("Revenue by Type — Calls vs Data","📈",GRN)
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=cc_d["date"],y=cc_d["calls_revenue"],name="Call Revenue",
            fill="tozeroy",fillcolor=hr(BLU,0.12),line=dict(color=BLU,width=2.5),
            hovertemplate="%{x|%b %d}<br>Calls: $%{y:,.0f}<extra></extra>"))
        fig_r.add_trace(go.Scatter(x=cc_d["date"],y=cc_d["data_revenue"],name="Data Revenue",
            fill="tonexty",fillcolor=hr(TEA,0.1),line=dict(color=TEA,width=2),
            hovertemplate="%{x|%b %d}<br>Data: $%{y:,.0f}<extra></extra>"))
        pcfg(fig_r,280,lh=True)
        st.plotly_chart(fig_r,use_container_width=True)

        calls_r = cc_d["calls_revenue"].sum(); data_r = cc_d["data_revenue"].sum()
        story("💡","Understanding Revenue Mix",
              f"Calls generate ${calls_r:,.0f} vs data submissions ${data_r:,.0f}. "
              "If call revenue is declining, check: Are ads reaching the right income segments? "
              "Are call center agents converting properly?",
              issue="Revenue mix shifting toward low-value data submissions",
              action="Review call scripts and ad targeting for phone-intent keywords",
              benefit="Shifting 10% of data leads to calls can increase RPL by $30–60")

        sh("Daily Margin $","📊",AMB)
        fig_m = go.Figure(go.Bar(
            x=cc_d["date"],y=cc_d["margin_dollar"],
            marker=dict(color=[GRN if v>=0 else RED for v in cc_d["margin_dollar"]],
                        line=dict(width=0)),
            hovertemplate="%{x|%b %d}<br>Margin: $%{y:,.0f}<extra></extra>"))
        pcfg(fig_m,200); fig_m.update_layout(showlegend=False)
        st.plotly_chart(fig_m,use_container_width=True)

        sh("Today — Hourly Revenue Breakdown","🕐",TEA)
        tc = cc_raw[cc_raw["date"]==cc_raw["date"].max()]
        if not tc.empty:
            fig_h = go.Figure()
            fig_h.add_trace(go.Bar(x=tc["hour"],y=tc["calls_revenue"],name="Calls",
                marker=dict(color=BLU,line=dict(width=0))))
            fig_h.add_trace(go.Bar(x=tc["hour"],y=tc["data_revenue"],name="Data",
                marker=dict(color=TEA,line=dict(width=0))))
            pcfg(fig_h,240,lh=True)
            fig_h.update_layout(barmode="stack",xaxis_title="Hour (UTC)")
            st.plotly_chart(fig_h,use_container_width=True)

    with t2:
        cq_d = cq_raw.groupby("date").agg(total_calls=("total_calls","sum"),
            r30=("medium_income_call_ratio_30s","mean"),
            r60=("medium_income_call_ratio_60s","mean"),
            r90=("medium_income_call_ratio_90s","mean")).reset_index()

        st.markdown(
            f'<div style="font-size:12px;color:{TM};margin-bottom:10px">'
            + tooltip_html("What do 30s/60s/90s ratios mean?",
                           "Call Duration Ratios — Lead Quality Proxy",
                           "These ratios show % of medium-income calls reaching each duration. "
                           "Longer calls strongly correlate with qualified leads and revenue. "
                           "Low 90s ratio = calls dropping off before qualification.",
                           "Low 90s ratio means calls ending before lead is qualified",
                           "Review call scripts and IVR flows for early drop-off",
                           "Improving 90s ratio by 10pp raises RPL by $15–25")
            + '</div>',unsafe_allow_html=True)

        sh("Medium Income Call Ratio by Duration","📊",AMB)
        fig_cq = go.Figure()
        for col,lbl,color in [("r30","30s Ratio",BLU),("r60","60s Ratio",AMB),("r90","90s Ratio",GRN)]:
            fig_cq.add_trace(go.Scatter(x=cq_d["date"],y=cq_d[col]*100,name=lbl,
                mode="lines+markers",line=dict(color=color,width=2),marker=dict(size=5),
                hovertemplate=f"<b>{lbl}</b>: %{{y:.1f}}%<br>%{{x|%b %d}}<extra></extra>"))
        pcfg(fig_cq,300,lh=True)
        fig_cq.update_layout(yaxis_title="Ratio (%)")
        st.plotly_chart(fig_cq,use_container_width=True)

        sh("Call Quality Heatmap — Hour × Day","🌡️",PUR)
        if not cq_raw.empty:
            cq_h = cq_raw.copy(); cq_h["day"] = cq_h["date"].dt.strftime("%m/%d")
            piv = cq_h.pivot_table(index="hour",columns="day",
                                    values="medium_income_call_ratio_30s",aggfunc="mean")
            fig_hm = go.Figure(go.Heatmap(
                z=piv.values*100,x=piv.columns.tolist(),
                y=[f"{h:02d}:00" for h in piv.index],
                colorscale=[[0,SURF2],[0.4,AMB],[1,GRN]],
                hovertemplate="Day: %{x}<br>Hour: %{y}<br>30s Ratio: %{z:.1f}%<extra></extra>",
                colorbar=dict(title="Ratio %",thickness=12,tickfont=dict(size=10,color=TS))))
            pcfg(fig_hm,320)
            fig_hm.update_layout(xaxis=dict(tickangle=-45,tickfont=dict(size=9)),
                                  yaxis=dict(tickfont=dict(size=9)))
            st.plotly_chart(fig_hm,use_container_width=True)
            st.caption("Green = high call quality. Use this to schedule agent shifts and dayparting.")

    with t3:
        at_d = at_raw.groupby("date").agg(
            inbound_call_rate=("inbound_call_rate","mean"),
            medium_income_verified_rate=("medium_income_verified_rate","mean"),
            any_data_rate=("any_data_rate","mean"),
            high_income_verified_carrier_rate=("high_income_verified_carrier_rate","mean")).reset_index()

        st.markdown(
            f'<div style="font-size:12px;color:{TM};margin-bottom:10px">'
            + tooltip_html("What do attribution rates measure?",
                           "Attribution Rates — Lead Quality Ladder",
                           "Inbound Call Rate: % impressions → call. "
                           "Med. Income Verified: % verified at this income tier. "
                           "Any Data Rate: % submitting any form. "
                           "High Inc. Carrier: premium leads with highest RPL.",
                           "Declining rates = ad quality or targeting degrading",
                           "Re-segment audiences, refresh creative, tighten income targeting",
                           "5pp improvement in High Inc. Carrier rate adds $40–80 RPL")
            + '</div>',unsafe_allow_html=True)

        sh("Attribution Rate Trends","📈",TEA)
        fig_at = go.Figure()
        for col,lbl,color in [
            ("inbound_call_rate","Inbound Call",BLU),
            ("medium_income_verified_rate","Med. Income Verified",GRN),
            ("any_data_rate","Any Data",AMB),
            ("high_income_verified_carrier_rate","High Inc. Carrier",TEA)]:
            fig_at.add_trace(go.Scatter(x=at_d["date"],y=at_d[col]*100,name=lbl,
                mode="lines+markers",line=dict(color=color,width=2),marker=dict(size=5),
                hovertemplate=f"<b>{lbl}</b>: %{{y:.1f}}%<br>%{{x|%b %d}}<extra></extra>"))
        pcfg(fig_at,320,lh=True)
        fig_at.update_layout(yaxis_title="Rate (%)")
        st.plotly_chart(fig_at,use_container_width=True)

        sh("Latest Rates — Gauges","🎯",BLU)
        latest = at_raw[at_raw["date"]==at_raw["date"].max()].mean(numeric_only=True)
        gi = [("Inbound Call","inbound_call_rate",BLU),
              ("Med. Verified","medium_income_verified_rate",GRN),
              ("Any Data","any_data_rate",AMB),
              ("High Inc. Carrier","high_income_verified_carrier_rate",TEA)]
        gc = st.columns(4)
        for col,(lbl,fld,color) in zip(gc,gi):
            val = latest.get(fld,0)
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",value=val*100,
                number=dict(suffix="%",font=dict(size=26,color=TP)),
                title=dict(text=lbl,font=dict(size=10,color=TS)),
                gauge=dict(
                    axis=dict(range=[0,100],tickcolor=TM,tickfont=dict(size=9,color=TM)),
                    bar=dict(color=color,thickness=0.65),bgcolor=SURF2,bordercolor=BDR,
                    steps=[dict(range=[0,30],color=hr(RED,0.1)),
                           dict(range=[30,60],color=hr(AMB,0.08)),
                           dict(range=[60,100],color=hr(GRN,0.08))],
                    threshold=dict(line=dict(color=color,width=2),value=val*100))))
            fig_g.update_layout(height=200,margin=dict(l=12,r=12,t=36,b=8),
                                 paper_bgcolor=SURF,font_color=TS)
            col.plotly_chart(fig_g,use_container_width=True)

# ═══════════════════════════════════════════════════════════
#  PAGE 5 — AI STRATEGY ENGINE
# ═══════════════════════════════════════════════════════════
elif page == "🧠  AI Strategy Engine":
    banner("🧠","AI Strategy Engine",
           f"Live-data context · Issue→Root Cause→Action→Benefit · Engine: {eng}")

    story("📖","How This AI Engine Works",
          "Every response is grounded in your <b>live dashboard data</b> — today's KPIs, "
          "MTD totals, platform breakdown — auto-injected as context. "
          "No generic advice. It reasons about your actual numbers.",
          issue="Interpreting performance data takes hours of analyst time",
          action=f"AI ({eng}) analyzes your live numbers on demand",
          benefit="Reduces time-to-insight from hours to seconds")

    live_ctx = (f"\n📊 LIVE DATA ({td_str}):"
                f"\n• Spend: ${sv(tday,'total_cost'):,.0f} | Revenue: ${sv(tday,'total_revenue'):,.0f}"
                f" | Margin: ${sv(tday,'margin_dollar'):,.0f} ({sv(tday,'margin_pct')*100:.1f}%)"
                f"\n• CPL: ${sv(tday,'CPL'):.2f} | ROAS: {sv(tday,'ROAS'):.2f}x | CVR: {sv(tday,'CVR')*100:.2f}%"
                f"\n• MTD Spend: ${mtd['total_cost'].sum():,.0f} | MTD Revenue: ${mtd['total_revenue'].sum():,.0f}"
                f"\n• MTD Margin: ${mtd['margin_dollar'].sum():,.0f} | Avg ROAS: {mtd['ROAS'].mean():.2f}x"
                f"\n• Platforms: {', '.join(pf)}\n")

    t1,t2,t3 = st.tabs(["💬 AI Chat","⚡ Auto Reports","📜 History"])

    with t1:
        st.markdown(f'<div style="font-size:11px;color:{TM};margin-bottom:10px">'
                    f'Grounded in your live data · Engine: <b style="color:{PUR}">{eng}</b> · '
                    f'Responses follow: Answer → Root Cause → Action → Benefit</div>',
                    unsafe_allow_html=True)

        if "history" not in st.session_state: st.session_state.history = []

        st.markdown(f'<div style="font-size:10px;font-weight:700;letter-spacing:1px;'
                    f'text-transform:uppercase;color:{TM};margin-bottom:6px">Quick Prompts</div>',
                    unsafe_allow_html=True)
        qps = [
            ("🔍 Why is CPL high?",
             "Analyze today's CPL. Give root cause and 3 specific fixes with expected CPL impact."),
            ("💰 Where to move budget?",
             "Based on today's platform ROAS and CPL, recommend specific $ reallocation and expected outcome."),
            ("⚠️ Red flags?",
             "Review today's KPIs. Flag anomalies, rate severity, give immediate action for each."),
            ("📋 Daily brief",
             f"Daily performance brief for {td_str}: health status, 2 wins, 2 risks, 1 action. Under 200 words."),
            ("🎨 Creative fatigue?",
             "Signs of creative or search fatigue? Which platform needs refresh first and what to test?"),
            ("📈 7-day outlook",
             "Based on current trends, what to expect next 7 days? What could improve or derail performance?"),
        ]
        qc = st.columns(3)
        for i,(lbl,prm) in enumerate(qps):
            if qc[i%3].button(lbl, use_container_width=True, key=f"qp{i}"):
                st.session_state["pq"] = prm

        st.markdown(f'<hr style="border:none;border-top:1px solid {BDR};margin:10px 0">',
                    unsafe_allow_html=True)

        for msg in st.session_state.history:
            if msg["role"]=="user":
                st.markdown(f'<div class="chu"><div class="chl" style="color:{BLU}">You</div>'
                             f'{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="cha"><div class="chl" style="color:{PUR}">🤖 {eng}</div>'
                             f'{msg["content"]}</div>', unsafe_allow_html=True)

        user_input = st.chat_input(f"Ask {eng} about your campaigns...")
        if "pq" in st.session_state: user_input = st.session_state.pop("pq")

        if user_input:
            st.session_state.history.append({"role":"user","content":user_input})
            full = (f"You are a senior paid media analytics expert.\n{live_ctx}\n"
                    f"Question: {user_input}\n\n"
                    f"Structure: (1) Direct Answer, (2) Root Cause, "
                    f"(3) Specific Action with numbers, (4) Expected Benefit.")
            with st.spinner("Thinking..."):
                resp = ai(full, eng, 700)
            st.session_state.history.append({"role":"assistant","content":resp})
            st.rerun()

        if st.session_state.history:
            if st.button("🗑️ Clear Chat", key="clr"):
                st.session_state.history = []; st.rerun()

    with t2:
        st.markdown(f'<div style="font-size:11px;color:{TM};margin-bottom:12px">'
                    f'Each report uses your live data. Structured as: '
                    f'Problem → Root Cause → Action → Benefit.</div>', unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        with r1:
            sh("Health Check & Anomalies","🔍",RED)
            if st.button("▶ Run Health Check", use_container_width=True, key="ra"):
                pm_p = pm[pm["date"]==pm["date"].max()].groupby("platform").agg(
                    cost=("cost","sum"),conversions=("conversions","sum"),
                    clicks=("clicks","sum")).reset_index()
                pm_p["CPL"] = (pm_p["cost"]/pm_p["conversions"].replace(0,np.nan)).round(2)
                prompt = (f"Paid media health check — {td_str}.\n{live_ctx}\n"
                          f"Platform data:\n{pm_p.to_string(index=False)}\n\n"
                          "For each issue:\n🔴 ISSUE\n🔍 ROOT CAUSE\n✅ ACTION (with $)\n★ BENEFIT\n"
                          "End with health score 1–10 and top priority.")
                with st.spinner("Running..."):
                    alrt(ai(prompt,eng,900),"ai",f"🤖 Health Check · {td_str}")

            sh("Creative Fatigue","🎨",PUR)
            if st.button("▶ Check Fatigue", use_container_width=True, key="rf"):
                prompt = (f"Analyze creative/search fatigue.\n{live_ctx}\n"
                          "Per platform (Google, Meta, Microsoft):\n"
                          "1. Fatigue signal (CTR/CVR/CPL trend)\n"
                          "2. Creative refresh recommendation\n"
                          "3. Query expansion opportunity\n"
                          "4. Top 2 A/B test ideas")
                with st.spinner("Analyzing..."):
                    alrt(ai(prompt,eng,700),"ai",f"🎨 Creative Fatigue Analysis")

        with r2:
            sh("Budget Reallocation","💸",AMB)
            if st.button("▶ Reallocation Plan", use_container_width=True, key="rb"):
                pm_p = pm[pm["date"]==pm["date"].max()].groupby("platform").agg(
                    cost=("cost","sum"),conversions=("conversions","sum"),
                    clicks=("clicks","sum")).reset_index()
                pm_p["CPL"] = (pm_p["cost"]/pm_p["conversions"].replace(0,np.nan)).round(2)
                prompt = (f"Budget reallocation plan.\n{live_ctx}\n"
                          f"Platform data:\n{pm_p.to_string(index=False)}\n\n"
                          "Provide:\n🔴 PROBLEM: underperforming platform + why\n"
                          "💸 ACTION: exact $ shift (e.g. Move $500/day from X to Y)\n"
                          "📊 RATIONALE: CPL/ROAS logic\n"
                          "★ EXPECTED RESULT: projected improvement\n"
                          "⚠️ RISK: what to watch after reallocation")
                with st.spinner("Building plan..."):
                    alrt(ai(prompt,eng,700),"warn","💸 Budget Reallocation Plan")

            sh("Daily Performance Brief","📋",GRN)
            if st.button("▶ Daily Brief", use_container_width=True, key="rd"):
                prompt = (f"Daily paid media brief for {td_str}.\n{live_ctx}\n"
                          "Format:\n🟢/🟡/🔴 HEALTH: one line\n"
                          "✅ WIN 1:\n✅ WIN 2:\n⚠️ RISK 1:\n⚠️ RISK 2:\n"
                          "🎯 TOP ACTION: single most important thing tomorrow\nUnder 200 words.")
                with st.spinner("Writing..."):
                    alrt(ai(prompt,eng,500),"ok",f"📋 Daily Brief · {td_str}")

    with t3:
        sh("Stored AI Insights","📜",BLU)
        ins = load_insights(30)
        if ins.empty:
            alrt("No stored insights yet. Run analyses above or use the FastAPI scheduler.", "info")
        else:
            for _, row in ins.iterrows():
                sev  = row.get("severity","info")
                kind = {"critical":"crit","warning":"warn","info":"ok"}.get(sev,"info")
                alrt(row.get("summary",""), kind,
                     f"{row.get('insight_type','').upper()} · {row.get('generated_at','')}")

# ═══════════════════════════════════════════════════════════
#  PAGE 6 — REAL-TIME MONITOR
# ═══════════════════════════════════════════════════════════
elif page == "⚡  Real-Time Monitor":
    banner("⚡","Real-Time Intraday Monitor",
           f"Live pacing · Hourly breakdown · UTC {datetime.utcnow().strftime('%H:%M')}")

    story("📖","What to Watch Intraday",
          "Real-time monitoring catches <b>same-day problems</b> before they compound. "
          "Key question: <i>Are we spending at the right pace?</i> "
          "Over-pacing 15%+ exhausts budget before peak hours. "
          "Under-pacing means missed impression share and leads.",
          issue="Budget can exhaust before peak conversion hours (12–3pm)",
          action="Monitor cumulative spend vs expected pace every 2 hours",
          benefit="Catching a pace issue at 10am saves 4+ hours of lost opportunity")

    auto_r = st.checkbox("🔁 Auto-refresh every 60s", value=False)
    if auto_r:
        import time; time.sleep(60); st.rerun()

    today_pm = pm[pm["date"]==pm["date"].max()].groupby(["hour","platform"]).agg(
        cost=("cost","sum"), clicks=("clicks","sum"),
        impressions=("impressions","sum"), conversions=("conversions","sum")).reset_index()
    today_cc = cc_raw[cc_raw["date"]==cc_raw["date"].max()]
    cur_h = datetime.utcnow().hour

    live_sp = today_pm["cost"].sum()
    live_cv = today_pm["conversions"].sum()
    live_rv = today_cc["total_revenue"].sum() if not today_cc.empty else 0
    live_mg = today_cc["margin_dollar"].sum() if not today_cc.empty else 0
    live_cpl  = live_sp/live_cv if live_cv > 0 else 0
    live_roas = live_rv/live_sp if live_sp > 0 else 0
    exp_pct = cur_h/24.0

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: st.markdown(kc("Live Spend",   live_sp,  None,"$","",",.0f","💸",RED,  f"Hour {cur_h:02d}:00 UTC"), unsafe_allow_html=True)
    with c2: st.markdown(kc("Live Revenue", live_rv,  None,"$","",",.0f","💰",GRN), unsafe_allow_html=True)
    with c3: st.markdown(kc("Live Margin",  live_mg,  None,"$","",",.0f","📊",BLU), unsafe_allow_html=True)
    with c4: st.markdown(kc("Conversions",  live_cv,  None,"","",",.0f", "🎯",AMB), unsafe_allow_html=True)
    with c5: st.markdown(kc("Live CPL",     live_cpl, None,"$","",".2f", "📉",PUR), unsafe_allow_html=True)
    with c6: st.markdown(kc("Live ROAS",    live_roas,None,"","x",".2f", "📈",TEA), unsafe_allow_html=True)

    mc, sc = st.columns([3,1])
    with mc:
        sh("Cumulative Spend vs Revenue — Today","📈",BLU)
        fig_rt = make_subplots(rows=2,cols=1,
            subplot_titles=["Cumulative Spend vs Revenue","Hourly Spend by Platform"],
            vertical_spacing=0.14)
        cum = today_pm.groupby("hour").agg(cost=("cost","sum")).cumsum()
        cc_cum = (today_cc.set_index("hour")[["total_revenue"]].cumsum()
                  if not today_cc.empty else pd.DataFrame())
        if not cum.empty:
            fig_rt.add_trace(go.Scatter(x=cum.index,y=cum["cost"],name="Cum. Spend",
                fill="tozeroy",fillcolor=hr(RED,0.1),line=dict(color=RED,width=2)),row=1,col=1)
        if not cc_cum.empty:
            fig_rt.add_trace(go.Scatter(x=cc_cum.index,y=cc_cum["total_revenue"],name="Cum. Revenue",
                fill="tozeroy",fillcolor=hr(GRN,0.1),line=dict(color=GRN,width=2)),row=1,col=1)
        for p,color in PC.items():
            pd_ = today_pm[today_pm["platform"]==p]
            if not pd_.empty:
                fig_rt.add_trace(go.Bar(x=pd_["hour"],y=pd_["cost"],name=p.title(),
                    marker=dict(color=color,line=dict(width=0))),row=2,col=1)
        fig_rt.update_layout(**{**PL,"height":520,"barmode":"group",
                                 "legend":dict(orientation="h",y=1.04,x=0)})
        fig_rt.update_xaxes(title_text="Hour (UTC)",row=2)
        st.plotly_chart(fig_rt,use_container_width=True)

    with sc:
        sh("Day Elapsed","⏱️",AMB)
        fig_p = go.Figure(go.Indicator(
            mode="gauge+number", value=round(exp_pct*100,1),
            number=dict(suffix="% of day",font=dict(size=18,color=TP)),
            title=dict(text=f"Hour {cur_h}/24",font=dict(size=11,color=TS)),
            gauge=dict(
                axis=dict(range=[0,100],tickfont=dict(size=9,color=TM),nticks=5),
                bar=dict(color=BLU,thickness=0.65),bgcolor=SURF2,bordercolor=BDR,
                steps=[dict(range=[0,33],color=hr(GRN,0.08)),
                       dict(range=[33,66],color=hr(AMB,0.07)),
                       dict(range=[66,100],color=hr(RED,0.07))])))
        fig_p.update_layout(height=200,margin=dict(l=12,r=12,t=40,b=8),
                             paper_bgcolor=SURF,font_color=TS)
        st.plotly_chart(fig_p,use_container_width=True)

        sh("Platform Split Today","🥧",BLU)
        ts_ = today_pm["cost"].sum()
        for plat,color in PC.items():
            sp = today_pm[today_pm["platform"]==plat]["cost"].sum()
            pct = sp/ts_*100 if ts_ > 0 else 0
            st.markdown(
                f'<div style="margin:8px 0">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:3px">'
                f'<span style="font-size:11px;font-weight:700;color:{color}">{plat.title()}</span>'
                f'<span style="font-size:11px;color:{TS}">${sp:,.0f} · {pct:.1f}%</span></div>'
                f'<div style="background:{BDR};border-radius:4px;height:7px">'
                f'<div style="background:{color};width:{pct}%;height:7px;border-radius:4px"></div>'
                f'</div></div>', unsafe_allow_html=True)

        # Pacing narrative
        if live_rv > 0 and live_sp > 0:
            ratio = live_sp / live_rv
            if ratio < 0.6:
                st.markdown(f'<div style="margin-top:12px;background:{G_BG};border-left:3px solid {GRN};'
                             f'border-radius:6px;padding:8px 10px;font-size:11px;color:{TS}">'
                             f'✅ Revenue well ahead of spend — healthy margin pacing</div>',
                             unsafe_allow_html=True)
            elif ratio > 0.9:
                st.markdown(f'<div style="margin-top:12px;background:{R_BG};border-left:3px solid {RED};'
                             f'border-radius:6px;padding:8px 10px;font-size:11px;color:{TS}">'
                             f'⚠️ Spend approaching revenue — watch margin closely</div>',
                             unsafe_allow_html=True)

    sh(f"Last 6 Hours — Through {cur_h:02d}:00 UTC","🕐",TEA)
    rh = today_pm[today_pm["hour"] >= max(0, cur_h-5)]
    if not rh.empty:
        rp = rh.pivot_table(index="hour",columns="platform",
                             values=["cost","clicks","conversions"],aggfunc="sum")
        rp.columns = [f"{m.title()} · {p.title()}" for m,p in rp.columns]
        rp.index = [f"{h:02d}:00" for h in rp.index]
        cc_ = [c for c in rp.columns if "Cost" in c]
        oc_ = [c for c in rp.columns if "Cost" not in c]
        fmt = {**{c:"${:,.0f}" for c in cc_}, **{c:"{:,.0f}" for c in oc_}}
        st.dataframe(rp.style.format(fmt)
            .background_gradient(subset=cc_,cmap="RdYlGn_r"),
            use_container_width=True)
        st.caption("💡 Redder Cost cells = higher spend at that hour. Use for dayparting decisions.")

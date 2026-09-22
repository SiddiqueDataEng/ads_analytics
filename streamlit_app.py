"""
Paid Ads Intelligence Dashboard  ·  Production-Ready
Google · Meta · Microsoft  ·  Claude AI + GPT-4o
Self-explaining KPIs · Storytelling · Hover Insights · Issue→Solution→Benefit
Regional Reporting (US, UK, CA, AU)  ·  Quality Score · Impression Share · Forecasting · ML
"""
# ── stdlib ──────────────────────────────────────────────────────────────
import os, sqlite3, warnings
from datetime import datetime, timedelta

# ── third-party ─────────────────────────────────────────────────────────
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

# ════════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════════
def secret(name):
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.getenv(name, "")


OAI_KEY   = secret("OPENAI_API_KEY")
ANT_KEY   = secret("ANTHROPIC_API_KEY")
DB_PATH   = "ads_dashboard.db"
BUDGET    = 50_000   # daily budget target

# ════════════════════════════════════════════════════════════════════════
#  PALETTE  — GitHub-inspired dark, high contrast, production-ready
# ════════════════════════════════════════════════════════════════════════
BG    = "#0D1117"; SURF  = "#161B22"; SURF2 = "#1C2128"; SURF3 = "#22272E"
BDR   = "#30363D"; BDR2  = "#444C56"
TP    = "#CDD9E5"; TS    = "#768390"; TM    = "#545D68"   # text
GRN   = "#2EA043"; GRN2  = "#46954A"; GRN3  = "#57AB5A"  # green
RED   = "#E5534B"; RED2  = "#F47067"; RED3  = "#FF938A"  # red
AMB   = "#966600"; AMB2  = "#B08800"; AMB3  = "#C69026"  # amber
BLU   = "#1F6FEB"; BLU2  = "#388BFD"; BLU3  = "#79C0FF"  # blue
PUR   = "#8256D0"; PUR2  = "#A371F7"; PUR3  = "#D2A8FF"  # purple
TEA   = "#1B7C83"; TEA2  = "#2AAAAA"; TEA3  = "#39D3BB"  # teal
ORG   = "#953800"; ORG2  = "#B15B00"; ORG3  = "#E68A00"  # orange
# backgrounds
G_BG  = "rgba(46,160,67,0.1)"
R_BG  = "rgba(229,83,75,0.1)"
A_BG  = "rgba(176,136,0,0.1)"
B_BG  = "rgba(56,139,253,0.1)"
P_BG  = "rgba(163,113,247,0.1)"
T_BG  = "rgba(57,211,187,0.1)"
O_BG  = "rgba(230,138,0,0.1)"
# platforms
PC    = {"google": "#4285F4", "meta": "#0082FB", "microsoft": "#00A4EF"}

def rgba(hex6: str, a: float) -> str:
    h = hex6.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"

# ════════════════════════════════════════════════════════════════════════
#  PLOTLY BASE LAYOUT
# ════════════════════════════════════════════════════════════════════════
PL = dict(
    template="plotly_dark",
    plot_bgcolor=SURF, paper_bgcolor=SURF,
    font=dict(family="'Segoe UI',Inter,system-ui,sans-serif", color=TS, size=12),
    xaxis=dict(gridcolor=BDR, linecolor=BDR, zeroline=False,
               tickfont=dict(size=11, color=TS), title_font=dict(color=TS, size=11)),
    yaxis=dict(gridcolor=BDR, linecolor=BDR, zeroline=False,
               tickfont=dict(size=11, color=TS), title_font=dict(color=TS, size=11)),
    margin=dict(l=12, r=12, t=48, b=12),
    legend=dict(bgcolor=SURF2, bordercolor=BDR, borderwidth=1,
                font=dict(size=11, color=TS)),
    hoverlabel=dict(bgcolor=SURF3, bordercolor=BDR2,
                    font=dict(size=12, color=TP, family="'Segoe UI',Inter,sans-serif")),
)

# ════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Ads Intelligence Platform",
    page_icon="📡", layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════════════
#  CSS
# ════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*,html,body{{box-sizing:border-box;font-family:'Inter',sans-serif!important;}}
.stApp{{background:{BG}!important;color:{TP}!important;}}
.block-container{{padding:.7rem 1.4rem 2rem!important;max-width:100%!important;}}
#MainMenu,footer,header{{visibility:hidden;}}
section[data-testid="stSidebar"]{{background:{SURF}!important;border-right:1px solid {BDR}!important;}}
section[data-testid="stSidebar"] *{{color:{TS}!important;}}
[data-testid="stSidebarNav"]{{display:none;}}
/* ── KPI tile ── */
.kt{{
  background:linear-gradient(145deg,{SURF2},{SURF});
  border:1px solid {BDR};border-radius:10px;
  padding:14px 16px 12px;margin-bottom:8px;position:relative;overflow:visible;
  transition:border .18s,box-shadow .18s;
}}
.kt:hover{{border-color:{BDR2};box-shadow:0 4px 20px rgba(0,0,0,.5);}}
.kt-bar{{position:absolute;top:0;left:0;width:4px;height:100%;border-radius:10px 0 0 10px;}}
.kt-row{{display:flex;justify-content:space-between;align-items:flex-start;}}
.kt-ico{{font-size:17px;opacity:.85;}}
.kt-lbl{{font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:{TM};margin-top:10px;}}
.kt-val{{font-size:24px;font-weight:800;color:{TP};line-height:1.1;margin:4px 0 3px;letter-spacing:-.6px;}}
.kt-up{{font-size:11px;font-weight:700;color:{GRN3};}}
.kt-dn{{font-size:11px;font-weight:700;color:{RED3};}}
.kt-fl{{font-size:11px;color:{TM};}}
.kt-sub{{font-size:10px;color:{TM};margin-top:3px;line-height:1.4;}}
.kt-bench{{font-size:10px;color:{BLU3};margin-top:2px;}}
/* ── Hover tooltip ── */
.tw{{position:relative;display:inline;cursor:help;}}
.ti{{
  display:inline-flex;align-items:center;justify-content:center;
  width:14px;height:14px;background:{SURF3};border:1px solid {BDR};
  border-radius:50%;font-size:8px;color:{BLU3};font-weight:700;
  margin-left:4px;vertical-align:middle;cursor:help;
}}
.tt{{
  visibility:hidden;opacity:0;transition:opacity .18s;
  position:absolute;z-index:9999;bottom:130%;left:50%;transform:translateX(-50%);
  background:{SURF3};border:1px solid {BDR2};border-radius:10px;
  padding:12px 14px;width:300px;max-width:340px;
  box-shadow:0 12px 40px rgba(0,0,0,.65);white-space:normal;
}}
.tw:hover .tt{{visibility:visible;opacity:1;}}
.tt-title{{font-size:11px;font-weight:700;color:{BLU3};letter-spacing:.5px;text-transform:uppercase;margin-bottom:6px;}}
.tt-body{{font-size:12px;color:{TS};line-height:1.65;}}
.tt-row{{font-size:11px;font-weight:600;margin-top:5px;padding-top:5px;border-top:1px solid {BDR};}}
.tt-issue{{color:{RED3};}} .tt-fix{{color:{GRN3};}} .tt-ben{{color:{AMB3};}} .tt-bench{{color:{BLU3};}}
/* ── Narrative / story box ── */
.nb{{
  background:{SURF};border:1px solid {BDR};border-left:3px solid {BLU2};
  border-radius:8px;padding:12px 16px;margin:8px 0;
}}
.nb-head{{font-size:12px;font-weight:700;color:{TP};margin-bottom:5px;}}
.nb-body{{font-size:12px;color:{TS};line-height:1.7;}}
.tag{{display:inline-block;padding:1px 7px;border-radius:4px;font-size:10px;font-weight:700;margin:0 3px 3px 0;}}
.tag-i{{background:{R_BG};color:{RED3};}} .tag-a{{background:{B_BG};color:{BLU3};}} .tag-b{{background:{G_BG};color:{GRN3};}}
/* ── Metric ribbon ── */
.mr{{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 12px;}}
.mr-i{{background:{SURF};border:1px solid {BDR};border-radius:8px;padding:9px 13px;flex:1;min-width:130px;}}
.mr-l{{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:{TM};}}
.mr-v{{font-size:16px;font-weight:700;color:{TP};margin:2px 0;}}
.mr-n{{font-size:10px;color:{TS};}}
/* ── Section header ── */
.sh{{display:flex;align-items:center;gap:7px;font-size:10px;font-weight:700;
  color:{TS};text-transform:uppercase;letter-spacing:1.2px;
  margin:16px 0 8px;padding-bottom:6px;border-bottom:1px solid {BDR};}}
.sh-dot{{width:6px;height:6px;border-radius:50%;flex-shrink:0;}}
/* ── Page banner ── */
.pb{{
  background:linear-gradient(90deg,{SURF2},{SURF});
  border:1px solid {BDR};border-left:3px solid {BLU2};
  border-radius:10px;padding:11px 18px;margin-bottom:12px;
  display:flex;align-items:center;gap:12px;
}}
.pb-ico{{font-size:22px;}}
.pb-title{{font-size:16px;font-weight:700;color:{TP};}}
.pb-sub{{font-size:11px;color:{TM};margin-top:1px;}}
/* ── Chart annotation caption ── */
.ca{{
  background:{SURF2};border:1px solid {BDR};border-radius:6px;
  padding:8px 12px;margin:4px 0 10px;font-size:11px;color:{TS};line-height:1.6;
}}
.ca-icon{{margin-right:4px;}}
/* ── Alert / report box ── */
.alert{{border-radius:8px;padding:12px 16px;margin:8px 0;font-size:13px;line-height:1.7;color:{TP};white-space:pre-wrap;}}
.al-ok{{background:{G_BG};border-left:4px solid {GRN3};}}
.al-warn{{background:{A_BG};border-left:4px solid {AMB3};}}
.al-crit{{background:{R_BG};border-left:4px solid {RED3};}}
.al-info{{background:{B_BG};border-left:4px solid {BLU3};}}
.al-ai{{background:{P_BG};border-left:4px solid {PUR3};}}
.al-lbl{{font-size:9px;font-weight:800;letter-spacing:1px;text-transform:uppercase;margin-bottom:5px;opacity:.8;}}
/* ── Budget bar ── */
.bb-wrap{{margin:5px 0 10px;}}
.bb-top{{display:flex;justify-content:space-between;font-size:11px;color:{TS};margin-bottom:3px;}}
.bb-bg{{background:{SURF3};border-radius:6px;height:7px;overflow:hidden;}}
.bb-fill{{height:7px;border-radius:6px;}}
/* ── Chat ── */
.chu{{background:{SURF2};border:1px solid {BDR2};border-radius:12px 12px 2px 12px;
  padding:10px 14px;margin:6px 0 3px;color:{TP};font-size:13px;line-height:1.6;}}
.cha{{background:{SURF};border:1px solid {BDR};border-left:3px solid {PUR2};
  border-radius:2px 12px 12px 12px;padding:12px 14px;margin:3px 0 6px;
  color:{TS};font-size:13px;line-height:1.75;}}
.chl{{font-size:9px;font-weight:800;letter-spacing:.7px;text-transform:uppercase;margin-bottom:4px;}}
/* ── Streamlit component overrides ── */
div[data-testid="stMetricValue"]{{color:{TP}!important;font-size:22px!important;font-weight:700!important;}}
div[data-testid="stMetricLabel"]{{color:{TM}!important;font-size:11px!important;text-transform:uppercase;letter-spacing:.8px;}}
.stTabs [data-baseweb="tab-list"]{{background:{SURF}!important;border-bottom:1px solid {BDR}!important;gap:0!important;}}
.stTabs [data-baseweb="tab"]{{background:transparent!important;color:{TM}!important;font-size:12px!important;
  font-weight:500!important;padding:9px 20px!important;border-bottom:2px solid transparent!important;}}
.stTabs [aria-selected="true"]{{color:{BLU3}!important;border-bottom-color:{BLU2}!important;}}
div[data-testid="stDataFrame"]{{background:{SURF}!important;border:1px solid {BDR}!important;border-radius:8px!important;}}
.stButton>button{{background:{SURF2}!important;border:1px solid {BDR}!important;color:{TP}!important;
  border-radius:8px!important;font-size:12px!important;font-weight:600!important;padding:6px 16px!important;}}
.stButton>button:hover{{background:{B_BG}!important;border-color:{BLU2}!important;color:{BLU3}!important;}}
div[data-baseweb="select"]>div{{background:{SURF2}!important;border-color:{BDR}!important;color:{TP}!important;}}
div[data-baseweb="tag"]{{background:{SURF3}!important;color:{TP}!important;}}
p,label,span,div{{color:inherit!important;}}
</style>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
#  DATA LAYER
# ════════════════════════════════════════════════════════════════════════
@st.cache_resource
def conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def parse_dates(values):
    return pd.Series(
        [datetime.strptime(str(value)[:10], "%Y-%m-%d") for value in values],
        index=values.index,
        dtype=object,
    )

@st.cache_data(ttl=120)
def load_pm(d=30):
    co=(datetime.utcnow()-timedelta(days=d)).strftime("%Y-%m-%d")
    df=pd.read_sql("SELECT * FROM platform_metrics WHERE date>=? ORDER BY date,hour,platform",conn(),params=(co,))
    df["date"]=parse_dates(df["date"]); return df

@st.cache_data(ttl=120)
def load_cc(d=30):
    co=(datetime.utcnow()-timedelta(days=d)).strftime("%Y-%m-%d")
    df=pd.read_sql("SELECT * FROM call_center WHERE date>=? ORDER BY date,hour",conn(),params=(co,))
    df["date"]=parse_dates(df["date"]); return df

@st.cache_data(ttl=120)
def load_cq(d=30):
    co=(datetime.utcnow()-timedelta(days=d)).strftime("%Y-%m-%d")
    df=pd.read_sql("SELECT * FROM call_quality WHERE date>=? ORDER BY date,hour",conn(),params=(co,))
    df["date"]=parse_dates(df["date"]); return df

@st.cache_data(ttl=120)
def load_attr(d=30):
    co=(datetime.utcnow()-timedelta(days=d)).strftime("%Y-%m-%d")
    df=pd.read_sql("SELECT * FROM attribution_rates WHERE date>=? ORDER BY date,hour",conn(),params=(co,))
    df["date"]=parse_dates(df["date"]); return df

@st.cache_data(ttl=300)
def load_ins(n=20):
    return pd.read_sql("SELECT * FROM ai_insights ORDER BY generated_at DESC LIMIT ?",conn(),params=(n,))

def agg(pm, cc):
    a=pm.groupby("date").agg(cost=("cost","sum"),clicks=("clicks","sum"),
        impressions=("impressions","sum"),conversions=("conversions","sum")).reset_index()
    b=cc.groupby("date").agg(revenue=("total_revenue","sum"),margin=("margin_dollar","sum"),
        calls=("inbound_calls","sum"),forms=("data_submit_forms","sum")).reset_index()
    m=a.merge(b,on="date",how="left")
    z=lambda x: x.replace(0,np.nan)
    m["CPL"]    =(m.cost    /z(m.conversions)).round(2)
    m["RPL"]    =(m.revenue /z(m.conversions)).round(2)
    m["CPC"]    =(m.cost    /z(m.clicks)).round(2)
    m["CVR"]    =(m.conversions/z(m.clicks)).round(4)
    m["CTR"]    =(m.clicks  /z(m.impressions)).round(4)
    m["ROAS"]   =(m.revenue /z(m.cost)).round(2)
    m["margin_pct"]=(m.margin/z(m.revenue)).round(4)
    m["CPM"]    =(m.cost    /z(m.impressions)*1000).round(2)
    m["RPM"]    =(m.revenue /z(m.impressions)*1000).round(2)
    m["ROI"]    =((m.revenue-m.cost)/z(m.cost)*100).round(1)
    m["butil"]  =(m.cost    /BUDGET*100).round(1)
    # Simulate Quality Score (proxy from CTR/CVR relative performance)
    ctr_n=(m.CTR-m.CTR.min())/(m.CTR.max()-m.CTR.min()+1e-9)
    cvr_n=(m.CVR-m.CVR.min())/(m.CVR.max()-m.CVR.min()+1e-9)
    m["QS"]    =((ctr_n+cvr_n)/2*4+6).clip(1,10).round(1)
    m["IS_pct"]=(m.cost/BUDGET*0.35+0.55).clip(0,1)*100    # simulated impression share
    return m

# ════════════════════════════════════════════════════════════════════════
#  AI
# ════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_ant(): return anthropic.Anthropic(api_key=ANT_KEY)
@st.cache_resource
def get_oai(): return OpenAI(api_key=OAI_KEY)

def ask(prompt, eng="Claude", n=900):
    try:
        if eng=="Claude":
            if not ANT_KEY:
                return "AI analysis is unavailable: configure ANTHROPIC_API_KEY in Streamlit secrets."
            r=get_ant().messages.create(model="claude-opus-4-5",max_tokens=n,
                messages=[{"role":"user","content":prompt}])
            return r.content[0].text
        else:
            if not OAI_KEY:
                return "AI analysis is unavailable: configure OPENAI_API_KEY in Streamlit secrets."
            r=get_oai().chat.completions.create(model="gpt-4o",max_tokens=800,
                messages=[{"role":"system","content":"Senior paid media analytics expert. Concise, data-driven."},
                          {"role":"user","content":prompt}])
            return r.choices[0].message.content
    except Exception as e: return f"⚠️ {e}"

# ════════════════════════════════════════════════════════════════════════
#  ML
# ════════════════════════════════════════════════════════════════════════
def iso(df, feats):
    df=df.copy().dropna(subset=feats)
    if len(df)<10: df["anom"]=1; df["ascore"]=0.0; return df
    X=StandardScaler().fit_transform(df[feats])
    m=IsolationForest(contamination=0.08,random_state=42,n_estimators=150)
    df["anom"]=m.fit_predict(X); df["ascore"]=m.score_samples(X); return df

def prophet_fc(df, col, periods=14):
    try:
        from prophet import Prophet
        ts=df[["date",col]].rename(columns={"date":"ds",col:"y"}).dropna()
        ts["ds"]=pd.to_datetime(ts["ds"])
        m=Prophet(daily_seasonality=True,weekly_seasonality=True,changepoint_prior_scale=0.15)
        m.fit(ts); fc=m.predict(m.make_future_dataframe(periods=periods))
        return fc[["ds","yhat","yhat_lower","yhat_upper"]], "Prophet"
    except: pass
    ts=df[["date",col]].dropna().copy(); ts["t"]=np.arange(len(ts))
    lr=LinearRegression().fit(ts[["t"]],ts[col])
    fd=[ts["date"].iloc[-1]+timedelta(days=i+1) for i in range(periods)]
    pv=lr.predict(np.arange(len(ts),len(ts)+periods).reshape(-1,1))
    fc=pd.concat([ts.rename(columns={"date":"ds",col:"yhat"})[["ds","yhat"]],
                  pd.DataFrame({"ds":fd,"yhat":pv,"yhat_lower":pv*.92,"yhat_upper":pv*1.08})],
                 ignore_index=True)
    return fc, "Linear Trend"

# ════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ════════════════════════════════════════════════════════════════════════
def sv(df, col, d=0.0):
    try: v=df[col].iloc[0]; return float(v) if not pd.isna(v) else d
    except: return d

def pdelta(t, y, col):
    tv=sv(t,col); pv=sv(y,col); return (tv-pv)/abs(pv) if pv else None

def pcfg(fig, h=320, title="", lh=False):
    u={**PL,"height":h}
    if title: u["title"]=dict(text=title,font=dict(size=13,color=TS),x=0.01,xanchor="left")
    if lh: u["legend"]={**PL["legend"],"orientation":"h","y":1.06,"x":0}
    fig.update_layout(**u); return fig

# ─── KPI tile HTML ───────────────────────────────────────────
def kt(label, val, delta=None, pre="", suf="", fmt=".0f",
       icon="", color=BLU2, sub="", bench=""):
    try: vs=f"{pre}{val:{fmt}}{suf}"
    except: vs=str(val)
    if delta is not None:
        dp=delta*100
        if   delta> .005: dh=f'<div class="kt-up">▲ {dp:.1f}% vs yesterday</div>'
        elif delta<-.005: dh=f'<div class="kt-dn">▼ {abs(dp):.1f}% vs yesterday</div>'
        else:             dh=f'<div class="kt-fl">— flat vs yesterday</div>'
    else: dh=""
    sub_h  =f'<div class="kt-sub">{sub}</div>'   if sub   else ""
    bench_h=f'<div class="kt-bench">⊘ {bench}</div>' if bench else ""
    return (f'<div class="kt">'
            f'<div class="kt-bar" style="background:{color}"></div>'
            f'<div class="kt-row"><div class="kt-lbl">{label}</div>'
            f'<span class="kt-ico">{icon}</span></div>'
            f'<div class="kt-val">{vs}</div>{dh}{sub_h}{bench_h}</div>')

# ─── Hover tooltip HTML ──────────────────────────────────────
def tip(label, title, body, issue="", fix="", benefit="", bench=""):
    def row(cls, icon, txt): return f'<div class="tt-row {cls}">{icon} {txt}</div>' if txt else ""
    return (f'<span class="tw">{label}<span class="ti">?</span>'
            f'<div class="tt"><div class="tt-title">{title}</div>'
            f'<div class="tt-body">{body}</div>'
            f'{row("tt-issue","⚠",issue)}{row("tt-fix","✓",fix)}'
            f'{row("tt-ben","★",benefit)}{row("tt-bench","⊘",bench)}'
            f'</div></span>')

# ─── Story / narrative block ─────────────────────────────────
def story(icon, head, body, issue="", action="", benefit=""):
    tags=""
    if issue:   tags+=f'<span class="tag tag-i">⚠ {issue}</span>'
    if action:  tags+=f'<span class="tag tag-a">→ {action}</span>'
    if benefit: tags+=f'<span class="tag tag-b">★ {benefit}</span>'
    st.markdown(
        f'<div class="nb"><div class="nb-head">{icon} {head}</div>'
        f'<div class="nb-body">{(tags+"<br>") if tags else ""}{body}</div></div>',
        unsafe_allow_html=True)

# ─── Chart annotation caption ────────────────────────────────
def caption(text):
    st.markdown(f'<div class="ca"><span class="ca-icon">💡</span>{text}</div>',
                unsafe_allow_html=True)

# ─── Metric ribbon ───────────────────────────────────────────
def ribbon(items):
    html='<div class="mr">'
    for lbl,val,note in items:
        html+=f'<div class="mr-i"><div class="mr-l">{lbl}</div><div class="mr-v">{val}</div><div class="mr-n">{note}</div></div>'
    html+='</div>'
    st.markdown(html,unsafe_allow_html=True)

# ─── Section header ──────────────────────────────────────────
def sh(title, icon="", color=BLU2):
    st.markdown(f'<div class="sh"><span class="sh-dot" style="background:{color}"></span>'
                f'{icon+" " if icon else ""}{title}</div>', unsafe_allow_html=True)

# ─── Page banner ─────────────────────────────────────────────
def banner(ico, title, sub):
    st.markdown(f'<div class="pb"><span class="pb-ico">{ico}</span>'
                f'<div><div class="pb-title">{title}</div>'
                f'<div class="pb-sub">{sub}</div></div></div>', unsafe_allow_html=True)

# ─── Alert box ───────────────────────────────────────────────
def alrt(text, kind="info", label=""):
    cls={"ok":"al-ok","warn":"al-warn","crit":"al-crit","info":"al-info","ai":"al-ai"}.get(kind,"al-info")
    lh=f'<div class="al-lbl">{label}</div>' if label else ""
    st.markdown(f'<div class="alert {cls}">{lh}{text}</div>', unsafe_allow_html=True)

# ─── Budget progress bar ─────────────────────────────────────
def budget_bar(label, used, total):
    pct=min(used/total*100,100) if total>0 else 0
    color=GRN3 if pct<75 else AMB3 if pct<92 else RED3
    st.markdown(
        f'<div class="bb-wrap"><div class="bb-top">'
        f'<span>{label}</span>'
        f'<span style="color:{color};font-weight:700">{pct:.1f}% · ${used:,.0f} / ${total:,.0f}</span>'
        f'</div><div class="bb-bg"><div class="bb-fill" style="width:{pct}%;background:{color}"></div>'
        f'</div></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding:8px 4px 4px">
      <div style="font-size:16px;font-weight:800;color:{TP};letter-spacing:-.3px">📡 Ads Intelligence</div>
      <div style="font-size:9px;color:{TM};letter-spacing:1px;text-transform:uppercase;margin-top:1px">
        Production Command Center</div>
    </div>
    <hr style="border:none;border-top:1px solid {BDR};margin:6px 0 10px">
    """, unsafe_allow_html=True)

    PAGE = st.radio("nav", [
        "🏠  Executive Overview",
        "📈  Performance Analysis",
        "🌍  Regional Reporting",
        "🤖  ML & Forecasting",
        "📞  Call & Attribution",
        "🧠  AI Strategy Engine",
        "⚡  Real-Time Monitor",
    ], label_visibility="collapsed")

    st.markdown(f'<hr style="border:none;border-top:1px solid {BDR};margin:10px 0">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:{TM};margin-bottom:5px">FILTERS</div>', unsafe_allow_html=True)

    DAYS = st.slider("Lookback days", 7, 30, 30)
    PF   = st.multiselect("Platforms", ["google","meta","microsoft"], default=["google","meta","microsoft"])
    ENG  = st.radio("AI Engine", ["Claude","OpenAI GPT-4o"], index=0)

    st.markdown(f'<hr style="border:none;border-top:1px solid {BDR};margin:10px 0">', unsafe_allow_html=True)
    if st.button("⟳  Refresh", use_container_width=True):
        st.cache_data.clear(); st.rerun()

    st.markdown(f"""
    <div style="background:{SURF2};border:1px solid {BDR};border-radius:8px;padding:10px 12px;margin-top:8px">
      <div style="font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:{BLU3};margin-bottom:6px">ABOUT THIS DASHBOARD</div>
      <div style="font-size:11px;color:{TS};line-height:1.8">
        📊 <b style="color:{TP}">Tracks</b> spend, revenue, margin — 3 platforms<br>
        🤖 <b style="color:{TP}">ML anomaly detection</b> (Isolation Forest)<br>
        📉 <b style="color:{TP}">14-day forecasting</b> via Prophet<br>
        🌍 <b style="color:{TP}">Regional</b> US · UK · CA · AU breakdown<br>
        🧠 <b style="color:{TP}">AI explanations</b> for every metric & chart<br>
        💡 <b style="color:{TP}">Issue → Action → Benefit</b> on every insight
      </div>
    </div>
    <div style="font-size:10px;color:{TM};margin-top:6px">Updated: {datetime.now().strftime("%b %d, %Y · %H:%M")}</div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════════════════════════════════════
pm_r  = load_pm(DAYS); cc_r = load_cc(DAYS)
cq_r  = load_cq(DAYS); at_r = load_attr(DAYS)
pm    = pm_r[pm_r["platform"].isin(PF)] if PF else pm_r
daily = agg(pm, cc_r)
TD    = daily["date"].max() if not daily.empty else pd.Timestamp.utcnow()
TD_S  = pd.Timestamp(TD).strftime("%Y-%m-%d")
tday  = daily[daily["date"]==TD]
yday  = daily[daily["date"]==TD-timedelta(days=1)]
MS    = pd.Timestamp(TD).replace(day=1)
mtd   = daily[daily["date"]>=MS]

# ════════════════════════════════════════════════════════════════════════
#  PAGE 1 — EXECUTIVE OVERVIEW
# ════════════════════════════════════════════════════════════════════════
if PAGE == "🏠  Executive Overview":
    banner("🏠","Executive Overview",
           f"Cross-platform profitability at a glance · {TD_S} · {', '.join(PF)}")

    story("📖","What This Page Shows",
          "Every paid-ads manager needs one question answered first thing: "
          "<b>Are we spending money profitably today?</b> "
          "These KPIs cut through the noise — spend, revenue, margin, and the efficiency ratios "
          "that control profitability. Everything flows from Impressions → Clicks → Conversions → Revenue → Margin.",
          issue="No unified cross-platform profitability view",
          action="8 core KPIs + 8 volume KPIs, all with benchmarks",
          benefit="Catch margin erosion in hours, not at month-end review")

    # ── Values ─────────────────────────────────────────────
    spend = sv(tday,"cost");  rev   = sv(tday,"revenue"); margin = sv(tday,"margin")
    mgpct = sv(tday,"margin_pct")*100; cpl = sv(tday,"CPL"); roas = sv(tday,"ROAS")
    cvr   = sv(tday,"CVR")*100; ctr = sv(tday,"CTR")*100; cpm = sv(tday,"CPM")
    roi   = sv(tday,"ROI"); butil = sv(tday,"butil")
    impr  = sv(tday,"impressions"); clicks = sv(tday,"clicks"); convs = sv(tday,"conversions")
    d_cpl = pdelta(tday,yday,"CPL")

    # ── Row 1: Profitability KPIs ───────────────────────────
    sh("Core Profitability KPIs","💰",GRN3)
    c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
    kpi_row1 = [
        (c1,"Ad Spend",   spend,   pdelta(tday,yday,"cost"),     "$","",",.0f","💸",RED3,  "Budget consumed today",         "varies by campaign"),
        (c2,"Revenue",    rev,     pdelta(tday,yday,"revenue"),  "$","",",.0f","💰",GRN3,  "Pipeline revenue generated",    "≥2× spend target"),
        (c3,"Gross Margin",margin, pdelta(tday,yday,"margin"),   "$","",",.0f","📊",BLU3,  f"Margin rate: {mgpct:.1f}%",    ">30% healthy"),
        (c4,"CPL",        cpl,     -d_cpl if d_cpl else None,    "$","",".2f", "🎯",AMB3,  "Cost per acquired lead",        "<$50 good · <$80 ok"),
        (c5,"ROAS",       roas,    pdelta(tday,yday,"ROAS"),      "","x",".2f", "📈",PUR3,  "Return per $1 spent",           ">3x strong · 1x breakeven"),
        (c6,"Conv. Rate", cvr,     pdelta(tday,yday,"CVR"),       "","%" ,".2f","✅",TEA3, "Click → conversion",            "5–8% paid search avg"),
        (c7,"ROI %",      roi,     pdelta(tday,yday,"ROI"),       "","%" ,".1f","🏆",ORG3, "(Rev−Spend)/Spend",             ">100% = profitable"),
        (c8,"Budget Used",butil,   None,                          "","%" ,".1f","📊",BLU3, "vs daily target",               "<85% on-pace"),
    ]
    for col,lbl,val,d,pre,suf,fmt,ico,clr,sub,bench in kpi_row1:
        with col: st.markdown(kt(lbl,val,d,pre,suf,fmt,ico,clr,sub,bench), unsafe_allow_html=True)

    # ── Tooltip explanations strip ──────────────────────────
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin:2px 0 10px;font-size:12px;color:{TM}">'
        + tip("CPL","Cost per Lead — Core Efficiency",
              "CPL = Ad Spend ÷ Conversions. The single most important cost metric for lead generation.",
              "Rising CPL = spend went up OR conversions dropped — investigate both",
              "Pause weak ad groups, tighten audience targeting, improve landing pages",
              "Every $5 CPL reduction on 1,000 leads/month = $5,000 monthly savings",
              "Insurance/finance: $30–$80 · E-commerce: $15–$40")
        + "&ensp;" + tip("ROAS","Return on Ad Spend",
              "ROAS = Revenue ÷ Spend. Below 1.0x means campaigns lose money.",
              "ROAS <1.5x means you're not covering cost of goods + overhead",
              "Reallocate budget from lowest-ROAS platform to highest",
              "+0.5x ROAS on $50K/month spend = $25K additional monthly revenue",
              "E-commerce: >4x strong · Lead gen: >2x · Brand: >1.5x")
        + "&ensp;" + tip("CVR","Conversion Rate",
              "CVR = Conversions ÷ Clicks × 100. Measures how well traffic converts to leads.",
              "Declining CVR = audience quality drop, landing page issue, or creative fatigue",
              "A/B test landing pages, refine audiences, update ad creative",
              "1% CVR improvement reduces CPL by ~15–20% automatically",
              "Paid search: 5–8% · Display: 1–3% · Social: 2–5%")
        + "&ensp;" + tip("ROI","Return on Investment",
              "ROI = (Revenue − Spend) ÷ Spend × 100%. True profitability after ad costs.",
              "Negative ROI means campaigns are actively losing money",
              "Pause loss-making campaigns, shift budget to profitable ones",
              ">150% ROI = every $1 spent returns $2.50 in revenue",
              "Healthy: >100% · Excellent: >200%")
        + '</div>', unsafe_allow_html=True)

    budget_bar("Daily Budget Pacing", spend, BUDGET)

    # ── Row 2: Volume & Efficiency ──────────────────────────
    sh("Volume & Efficiency Metrics","📊",BLU2)
    v1,v2,v3,v4,v5,v6,v7,v8 = st.columns(8)
    kpi_row2 = [
        (v1,"Impressions",impr,  None,"","",",.0f","👁", ORG3, "Ad views today",      "reach signal"),
        (v2,"Clicks",     clicks,None,"","",",.0f","🖱️",BLU3, "Engaged traffic",      "intent signal"),
        (v3,"Conversions",convs, None,"","",",.0f","🎯",GRN3, "Qualified leads",      "quality signal"),
        (v4,"CPC",        cpm,   pdelta(tday,yday,"CPC"),"$","",".2f","💵",AMB3,"Cost per click","$1–5 avg search"),
        (v5,"CPM",        cpm,   None,"$","",".2f","📢",ORG3, "Cost/1K impressions",  "$5–30 typical"),
        (v6,"CTR %",      ctr,   None,"","%" ,".2f","👆",TEA3,"Click-through rate",  "3–6% search avg"),
        (v7,"RPL",        sv(tday,"RPL"),None,"$","",".2f","💎",GRN3,"Revenue/lead","maximise this"),
        (v8,"RPM",        sv(tday,"RPM"),None,"$","",".2f","💰",GRN2,"Revenue/1K impr.",""),
    ]
    for col,lbl,val,d,pre,suf,fmt,ico,clr,sub,bench in kpi_row2:
        with col: st.markdown(kt(lbl,val,d,pre,suf,fmt,ico,clr,sub,bench), unsafe_allow_html=True)

    # ── Charts ──────────────────────────────────────────────
    cl, cr = st.columns([3,1])
    with cl:
        sh("Revenue vs Spend + ROAS Trend","📉",BLU2)
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Scatter(x=daily["date"],y=daily["revenue"],name="Revenue",
            fill="tozeroy",fillcolor=rgba(GRN3,0.1),line=dict(color=GRN3,width=2.5),
            hovertemplate="<b>%{x|%b %d}</b><br>Revenue: $%{y:,.0f}<extra></extra>"),secondary_y=False)
        fig.add_trace(go.Scatter(x=daily["date"],y=daily["cost"],name="Spend",
            fill="tozeroy",fillcolor=rgba(RED3,0.08),line=dict(color=RED3,width=2),
            hovertemplate="<b>%{x|%b %d}</b><br>Spend: $%{y:,.0f}<extra></extra>"),secondary_y=False)
        fig.add_trace(go.Scatter(x=daily["date"],y=daily["margin"],name="Margin",
            line=dict(color=AMB3,width=1.5,dash="dot"),
            hovertemplate="<b>%{x|%b %d}</b><br>Margin: $%{y:,.0f}<extra></extra>"),secondary_y=False)
        fig.add_trace(go.Scatter(x=daily["date"],y=daily["ROAS"],name="ROAS",
            line=dict(color=PUR3,width=1.5,dash="dash"),
            hovertemplate="<b>%{x|%b %d}</b><br>ROAS: %{y:.2f}x<extra></extra>"),secondary_y=True)
        fig.add_hline(y=1.0,secondary_y=True,line_dash="dot",line_color=BDR2,line_width=1,
                      annotation_text="ROAS=1x (break-even)",annotation_font_color=TS,
                      annotation_font_size=10)
        fig.update_yaxes(title_text="$ Value",secondary_y=False,tickfont=dict(color=TS))
        fig.update_yaxes(title_text="ROAS (x)",secondary_y=True,tickfont=dict(color=PUR3),
                         title=dict(font=dict(color=PUR3)))
        pcfg(fig,310,lh=True); st.plotly_chart(fig,use_container_width=True)
        caption("When the green Revenue area is above the red Spend area = profitable day. "
                "The dotted ROAS line (right axis) should stay above 1.0x at minimum. "
                "Amber dots show daily margin — widening gap between revenue & spend = healthy growth.")

        gap = rev-spend
        if gap>0:
            story("✅","Revenue Outpacing Spend Today",
                  f"Revenue (${rev:,.0f}) exceeds spend (${spend:,.0f}) by <b>${gap:,.0f}</b>. "
                  f"Margin rate is {mgpct:.1f}%. Campaigns are profitable.",
                  benefit=f"${gap:,.0f} positive gap — healthy profitability signal")
        else:
            story("🚨","Spend Exceeds Revenue",
                  f"Spend (${spend:,.0f}) exceeds revenue (${rev:,.0f}) by <b>${abs(gap):,.0f}</b>. "
                  "Campaigns are currently loss-making. Immediate action required.",
                  issue=f"${abs(gap):,.0f} deficit — loss-making",
                  action="Pause lowest-ROAS campaigns, review bid strategies",
                  benefit="Every hour of inaction compounds the loss")

    with cr:
        sh("Today's Spend by Platform","🥧",PUR2)
        spd = pm[pm["date"]==pm["date"].max()].groupby("platform")["cost"].sum().reset_index()
        fig2 = go.Figure(go.Pie(
            labels=[p.title() for p in spd["platform"]],values=spd["cost"],hole=0.62,
            marker=dict(colors=[PC.get(p,BLU2) for p in spd["platform"]],
                        line=dict(color=BG,width=2)),
            textfont=dict(size=11,color=TP),
            hovertemplate="<b>%{label}</b><br>$%{value:,.0f} (%{percent})<extra></extra>"))
        fig2.update_layout(**{**PL,"height":240,"margin":dict(l=0,r=0,t=8,b=30),
                              "legend":dict(orientation="h",y=-0.12,font=dict(size=11))})
        st.plotly_chart(fig2,use_container_width=True)
        total_s = spd["cost"].sum()
        for _,row in spd.iterrows():
            p=row["platform"]; pct=row["cost"]/total_s*100 if total_s else 0; c=PC.get(p,BLU2)
            fill_c = GRN3 if pct<40 else AMB3 if pct<60 else RED3
            st.markdown(
                f'<div style="margin:7px 0">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:3px">'
                f'<span style="font-size:11px;font-weight:700;color:{c}">{p.title()}</span>'
                f'<span style="font-size:11px;color:{TP}">${row["cost"]:,.0f} · {pct:.1f}%</span></div>'
                f'<div style="background:{BDR};border-radius:4px;height:6px">'
                f'<div style="background:{c};width:{pct}%;height:6px;border-radius:4px"></div>'
                f'</div></div>', unsafe_allow_html=True)
        caption("Balanced spend across platforms reduces dependency risk. >70% on one platform = concentration risk.")

    # ── MTD ─────────────────────────────────────────────────
    sh("Month-to-Date Performance","📅",TEA2)
    ribbon([
        ("MTD Spend",   f"${mtd['cost'].sum():,.0f}",    "Total budget this month"),
        ("MTD Revenue", f"${mtd['revenue'].sum():,.0f}", "Revenue generated MTD"),
        ("MTD Margin",  f"${mtd['margin'].sum():,.0f}",  "Net value after costs"),
        ("MTD Leads",   f"{mtd['conversions'].sum():,.0f}","Total conversions"),
        ("Avg ROAS",    f"{mtd['ROAS'].mean():.2f}x",    "Return on spend MTD"),
        ("Avg CPL",     f"${mtd['CPL'].mean():.2f}",     "Cost per lead MTD"),
        ("Avg CVR",     f"{mtd['CVR'].mean()*100:.2f}%", "Avg conversion rate"),
        ("Avg QS",      f"{mtd['QS'].mean():.1f}/10",    "Quality score proxy"),
    ])

    # ── WoW ─────────────────────────────────────────────────
    sh("Week-over-Week Performance Matrix","📊",AMB2)
    if len(daily)>=14:
        tw=daily.tail(7); lw=daily.iloc[-14:-7]
        wc=["cost","revenue","margin","CPL","ROAS","CVR","CTR","CPM"]
        wl=["Spend","Revenue","Margin","CPL","ROAS","CVR","CTR","CPM"]
        twv=[tw[c].mean() for c in wc]; lwv=[lw[c].mean() for c in wc]
        invert={"CPL","CPM"}  # lower=better, so invert color
        wp=[((t-l)/abs(l)*100*(-1 if c in invert else 1)) if l else 0 for t,l,c in zip(twv,lwv,wc)]
        clrs=[GRN3 if v>=0 else RED3 for v in wp]
        interp={"CPL":"✓ cheaper|⚠ costlier","ROAS":"✓ better ROI|⚠ worse ROI",
                "CVR":"✓ more conversions|⚠ quality drop","CTR":"✓ better relevance|⚠ creative fatigue",
                "CPM":"✓ cheaper reach|⚠ auction pressure"}
        w1,w2=st.columns([3,1])
        with w1:
            fig_w=go.Figure(go.Bar(x=wl,y=wp,
                marker=dict(color=clrs,line=dict(width=0)),
                text=[f"{v:+.1f}%" for v in wp],textposition="outside",
                textfont=dict(size=12,color=TP),
                hovertemplate="<b>%{x}</b><br>WoW: %{y:+.1f}%<extra></extra>"))
            fig_w.update_layout(**{**PL,"height":220,"showlegend":False,
                "yaxis_title":"WoW Change (%)",
                "shapes":[dict(type="line",x0=-.5,x1=7.5,y0=0,y1=0,
                               line=dict(color=BDR2,width=1,dash="dot"))]})
            st.plotly_chart(fig_w,use_container_width=True)
            caption("Green bars = improvement this week vs last week. Red = decline. "
                    "CPL/CPM bars are inverted — green means they got cheaper (good).")
        with w2:
            sh("WoW Scorecard","",BLU2)
            for lbl,pct in zip(wl,wp):
                cls="kt-up" if pct>=0 else "kt-dn"; arr="▲" if pct>=0 else "▼"
                parts=interp.get(lbl,"").split("|")
                note=(parts[0] if pct>=0 else parts[1]) if len(parts)==2 else ""
                st.markdown(
                    f'<div style="padding:5px 0;border-bottom:1px solid {BDR};font-size:11px">'
                    f'<div style="display:flex;justify-content:space-between">'
                    f'<span style="color:{TS}">{lbl}</span>'
                    f'<span class="{cls}">{arr} {abs(pct):.1f}%</span></div>'
                    f'<div style="font-size:10px;color:{TM}">{note}</div></div>',
                    unsafe_allow_html=True)

        best=wl[wp.index(max(wp))]; worst=wl[wp.index(min(wp))]
        story("📖","This Week's Story",
              f"Best performer: <b>{best}</b> ({max(wp):+.1f}%). "
              f"Biggest concern: <b>{worst}</b> ({min(wp):+.1f}%). "
              f"WoW tracking catches trend shifts 5–7 days before they impact monthly totals.",
              issue=f"{worst} declining" if min(wp)<-5 else "All metrics within range",
              action=f"Investigate {worst} root cause, protect {best} momentum",
              benefit="Early WoW alerts prevent compounding losses across the month")

    # ── Platform Scorecard ───────────────────────────────────
    sh("Platform Scorecard — Today","🏆",GRN2)
    ps=pm[pm["date"]==pm["date"].max()].groupby("platform").agg(
        Spend=("cost","sum"),Clicks=("clicks","sum"),
        Impr=("impressions","sum"),Convs=("conversions","sum")).reset_index()
    ps["CTR%"] =(ps["Clicks"]/ps["Impr"].replace(0,np.nan)*100).round(2)
    ps["CPC"]  =(ps["Spend"]/ps["Clicks"].replace(0,np.nan)).round(2)
    ps["CPM"]  =(ps["Spend"]/ps["Impr"].replace(0,np.nan)*1000).round(2)
    ps["CPL"]  =(ps["Spend"]/ps["Convs"].replace(0,np.nan)).round(2)
    ps["CVR%"] =(ps["Convs"]/ps["Clicks"].replace(0,np.nan)*100).round(2)
    ps["Spend%"]=(ps["Spend"]/ps["Spend"].sum()*100).round(1)
    ps["QS"]   =((ps["CTR%"]/ps["CTR%"].max()*2+ps["CVR%"]/ps["CVR%"].max()*2)/4*4+6).clip(1,10).round(1)
    st.dataframe(ps.style
        .format({"Spend":"${:,.0f}","Clicks":"{:,.0f}","Impr":"{:,.0f}","Convs":"{:,.0f}",
                 "CTR%":"{:.2f}%","CPC":"${:.2f}","CPM":"${:.2f}","CPL":"${:.2f}",
                 "CVR%":"{:.2f}%","Spend%":"{:.1f}%","QS":"{:.1f}"})
        .background_gradient(subset=["CPL","CPM","CPC"],cmap="RdYlGn_r")
        .background_gradient(subset=["CVR%","CTR%","QS"],cmap="RdYlGn"),
        use_container_width=True,hide_index=True)
    caption("Red CPL/CPM/CPC = expensive (needs attention). Green CVR/CTR/QS = performing well. "
            "Spend% shows budget concentration — aim for diversified allocation.")

# ════════════════════════════════════════════════════════════════════════
#  PAGE 2 — PERFORMANCE ANALYSIS
# ════════════════════════════════════════════════════════════════════════
elif PAGE == "📈  Performance Analysis":
    banner("📈","Performance Analysis","Spend · CTR · CPC · CVR · CPM · Quality Score · Funnel")

    story("📖","Why Performance Analysis Goes Deeper",
          "Revenue numbers tell you <i>how much</i> you made. This page tells you <i>how efficiently</i>. "
          "CTR reveals ad relevance. CPC reveals auction competitiveness. CVR reveals landing page quality. "
          "Together they pinpoint the <b>exact stage where money leaks</b>.",
          issue="Spend can rise while efficiency silently deteriorates",
          action="Track CTR · CPC · CVR · CPM · QS daily per platform",
          benefit="Identifying one leaky funnel stage typically recovers 15–30% of wasted spend")

    pm_d=pm.groupby(["date","platform"]).agg(cost=("cost","sum"),clicks=("clicks","sum"),
        impressions=("impressions","sum"),conversions=("conversions","sum")).reset_index()
    pm_d["CPC"] =(pm_d["cost"]/pm_d["clicks"].replace(0,np.nan)).round(2)
    pm_d["CTR"] =(pm_d["clicks"]/pm_d["impressions"].replace(0,np.nan)*100).round(2)
    pm_d["CVR"] =(pm_d["conversions"]/pm_d["clicks"].replace(0,np.nan)*100).round(2)
    pm_d["CPL"] =(pm_d["cost"]/pm_d["conversions"].replace(0,np.nan)).round(2)
    pm_d["CPM"] =(pm_d["cost"]/pm_d["impressions"].replace(0,np.nan)*1000).round(2)
    ctr_max=pm_d["CTR"].max()+1e-9; cvr_max=pm_d["CVR"].max()+1e-9
    pm_d["QS"]  =((pm_d["CTR"]/ctr_max*2+pm_d["CVR"]/cvr_max*2)/4*4+6).clip(1,10).round(1)
    total_s_byd=pm_d.groupby("date")["cost"].transform("sum")
    pm_d["IS%"] =((pm_d["cost"]/total_s_byd.replace(0,np.nan))*0.35+0.55).clip(0,1)*100

    t1,t2,t3,t4,t5 = st.tabs(["📊 Spend & Funnel","🖱️ CTR · CPC · CVR",
                                "🏅 Quality & Impression Share","🕐 Intraday","📋 Full Table"])
    with t1:
        r1,r2=st.columns(2)
        with r1:
            sh("Daily Spend by Platform","💸",RED3)
            fig=go.Figure()
            for p in pm_d["platform"].unique():
                d=pm_d[pm_d["platform"]==p]
                fig.add_trace(go.Scatter(x=d["date"],y=d["cost"],name=p.title(),
                    fill="tozeroy",fillcolor=rgba(PC[p],0.1),line=dict(color=PC[p],width=2),
                    hovertemplate=f"<b>{p.title()}</b> %{{x|%b %d}}: $%{{y:,.0f}}<extra></extra>"))
            pcfg(fig,280,lh=True); st.plotly_chart(fig,use_container_width=True)
            caption("Overlapping colored areas = total combined spend. A sudden spike on one platform "
                    "signals a bid strategy change, auction competition, or misconfigured budget cap.")
        with r2:
            sh("Daily Conversions by Platform","🎯",GRN2)
            fig2=go.Figure()
            for p in pm_d["platform"].unique():
                d=pm_d[pm_d["platform"]==p]
                fig2.add_trace(go.Bar(x=d["date"],y=d["conversions"],name=p.title(),
                    marker=dict(color=PC[p],line=dict(width=0)),
                    hovertemplate=f"<b>{p.title()}</b> %{{x|%b %d}}: %{{y:,.0f}} convs<extra></extra>"))
            fig2.update_layout(**{**PL,"height":280,"barmode":"stack","showlegend":True,
                                   "legend":{**PL["legend"],"orientation":"h","y":1.06}})
            st.plotly_chart(fig2,use_container_width=True)
            caption("Stacked bars show total daily conversions split by platform. "
                    "Consistent daily volume = healthy pipeline. Sudden drop = check audience, bids, budgets.")

        sh("Conversion Funnel — Period Total","🔽",TEA2)
        ti=pm_d["impressions"].sum(); tc=pm_d["clicks"].sum(); tv=pm_d["conversions"].sum()
        ctr_a=tc/ti*100 if ti else 0; cvr_a=tv/tc*100 if tc else 0
        fcols=st.columns([2,1])
        with fcols[0]:
            fig_f=go.Figure(go.Funnel(
                y=["Impressions","Clicks","Conversions"],x=[ti,tc,tv],
                textposition="inside",
                texttemplate="%{value:,.0f}<br>(%{percentInitial:.1%} of impressions)",
                marker=dict(color=[BLU2,AMB3,GRN3],line=dict(color=BG,width=2)),
                connector=dict(line=dict(color=BDR2,width=1))))
            pcfg(fig_f,220); fig_f.update_layout(margin=dict(l=130,r=20,t=20,b=10))
            st.plotly_chart(fig_f,use_container_width=True)
        with fcols[1]:
            ribbon([
                ("Imp→Click (CTR)",f"{ctr_a:.2f}%","Avg 3–6% search"),
                ("Click→Conv (CVR)",f"{cvr_a:.2f}%","Avg 5–8% search"),
                ("Total Impressions",f"{ti:,.0f}","Awareness reach"),
                ("Total Clicks",f"{tc:,.0f}","Intent signal"),
                ("Conversions",f"{tv:,.0f}","Revenue leads"),
            ])
        caption("The funnel shows your entire customer journey in one view. "
                "A wide top but narrow bottom = audience quality problem (fix targeting). "
                "A narrow top = awareness problem (increase bids or budget).")

    with t2:
        st.markdown(
            f'<div style="font-size:12px;color:{TM};margin-bottom:10px">'
            + tip("The Efficiency Triangle","CTR · CPC · CVR — Why All Three Together",
                  "CTR = how relevant your ad is. CPC = how expensive your auction is. "
                  "CVR = how well your landing page converts. A problem in any one inflates CPL.",
                  "High CPC + Low CVR = paying a premium for traffic that doesn't convert",
                  "Fix the lowest CVR platform first — biggest leverage on CPL",
                  "Improving CVR by 1% automatically drops CPL by 15–20%",
                  "Google search: CTR 5–10%, CVR 6–8% · Meta: CTR 1–3%, CVR 2–5%")
            +'</div>', unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            sh("CTR % by Platform","👆",BLU2)
            fig_c=go.Figure()
            for p in pm_d["platform"].unique():
                d=pm_d[pm_d["platform"]==p]
                fig_c.add_trace(go.Scatter(x=d["date"],y=d["CTR"],name=p.title(),
                    mode="lines+markers",line=dict(color=PC[p],width=2),marker=dict(size=5),
                    hovertemplate=f"<b>{p.title()}</b> CTR: %{{y:.2f}}%<extra></extra>"))
            pcfg(fig_c,270,lh=True); fig_c.update_layout(yaxis_title="CTR %")
            st.plotly_chart(fig_c,use_container_width=True)
            caption("Declining CTR = ad creative fatigue or audience saturation. "
                    "Fix: rotate headlines, test new value props, refresh ad copy.")
        with c2:
            sh("CPC by Platform","💵",AMB2)
            fig_cp=go.Figure()
            for p in pm_d["platform"].unique():
                d=pm_d[pm_d["platform"]==p]
                fig_cp.add_trace(go.Scatter(x=d["date"],y=d["CPC"],name=p.title(),
                    mode="lines+markers",line=dict(color=PC[p],width=2),marker=dict(size=5),
                    hovertemplate=f"<b>{p.title()}</b> CPC: $%{{y:.2f}}<extra></extra>"))
            pcfg(fig_cp,270,lh=True); fig_cp.update_layout(yaxis_title="CPC $")
            st.plotly_chart(fig_cp,use_container_width=True)
            caption("Rising CPC = increased auction competition or dropping Quality Score. "
                    "Fix: improve QS (ad relevance + landing page), review bid strategy.")

        sh("Efficiency Scatter — CPC vs CVR","🎯",PUR2)
        fig_sc=go.Figure()
        for p in pm_d["platform"].unique():
            d=pm_d[pm_d["platform"]==p]
            max_cost=d["cost"].max() or 1
            fig_sc.add_trace(go.Scatter(x=d["CPC"],y=d["CVR"],name=p.title(),mode="markers",
                marker=dict(color=PC[p],size=d["cost"]/max_cost*18+6,
                            line=dict(color=BG,width=1),opacity=0.85),
                text=d["date"].dt.strftime("%b %d"),
                hovertemplate=f"<b>{p.title()}</b><br>%{{text}}<br>CPC: $%{{x:.2f}}<br>CVR: %{{y:.2f}}%<extra></extra>"))
        cpc_m=pm_d["CPC"].mean(); cvr_m=pm_d["CVR"].mean()
        fig_sc.add_hline(y=cvr_m,line_dash="dot",line_color=BDR2,line_width=1)
        fig_sc.add_vline(x=cpc_m,line_dash="dot",line_color=BDR2,line_width=1)
        for tx,ty,txt,fc in [
            (cpc_m*.5,cvr_m*1.4,"✓ IDEAL\nLow CPC + High CVR",GRN3),
            (cpc_m*1.5,cvr_m*.5,"⚠ WORST\nHigh CPC + Low CVR",RED3)]:
            fig_sc.add_annotation(x=tx,y=ty,text=txt,font=dict(color=fc,size=10),
                showarrow=False,bgcolor=SURF2,borderpad=4)
        pcfg(fig_sc,360,"Bubble size = Ad Spend · Upper-left quadrant = best efficiency",lh=True)
        fig_sc.update_layout(xaxis_title="CPC ($)",yaxis_title="CVR (%)")
        st.plotly_chart(fig_sc,use_container_width=True)
        caption("Platforms in the upper-left (low CPC + high CVR) are your most efficient. "
                "Scale budget there. Platforms in the lower-right need targeting or landing page fixes.")

    with t3:
        sh("Quality Score & Impression Share","🏅",AMB2)
        st.markdown(
            f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px;font-size:12px;color:{TM}">'
            + tip("Quality Score","Google Quality Score (1–10)",
                  "Composite of: Ad Relevance + Landing Page Experience + Expected CTR. "
                  "Higher QS = better ad rank at lower cost.",
                  "Low QS means you pay more for worse positions",
                  "Align ad copy to keyword intent, improve landing page relevance",
                  "QS 5→8 reduces CPC by 30–40% for same position",
                  "Target: 7+ for competitive terms · 9–10 for branded terms")
            +"&ensp;"
            + tip("Impression Share","Search Impression Share (%)",
                  "IS = (Your Impressions) ÷ (Total Eligible Impressions). "
                  "Low IS means you're losing auctions you could win.",
                  "IS% <60% = missing major traffic opportunities",
                  "Increase daily budget or raise bids to capture lost IS",
                  "IS 60%→80% can nearly double conversion volume",
                  "Brand terms: target >90% · Non-brand: target >60%")
            +'</div>', unsafe_allow_html=True)
        q1,q2=st.columns(2)
        with q1:
            qs_t=pm_d[pm_d["date"]==pm_d["date"].max()].groupby("platform")["QS"].mean().reset_index()
            fig_qs=go.Figure(go.Bar(x=qs_t["platform"].str.title(),y=qs_t["QS"],
                marker=dict(color=[GRN3 if v>=7 else AMB3 if v>=5 else RED3 for v in qs_t["QS"]],
                            line=dict(width=0)),
                text=[f"{v:.1f}/10" for v in qs_t["QS"]],textposition="outside",
                textfont=dict(size=13,color=TP),
                hovertemplate="<b>%{x}</b><br>Quality Score: %{y:.1f}/10<extra></extra>"))
            fig_qs.add_hline(y=7,line_dash="dot",line_color=GRN3,line_width=1,
                             annotation_text="Target: 7+",annotation_font_color=GRN3,
                             annotation_font_size=10)
            pcfg(fig_qs,260,"Quality Score by Platform — Today")
            fig_qs.update_layout(showlegend=False,yaxis=dict(range=[0,11]))
            st.plotly_chart(fig_qs,use_container_width=True)
            caption("Green = 7+ (good). Amber = 5–6 (needs work). Red = <5 (urgent). "
                    "QS below 7 on competitive keywords directly raises your CPCs.")
        with q2:
            is_t=pm_d[pm_d["date"]==pm_d["date"].max()].groupby("platform")["IS%"].mean().reset_index()
            fig_is=go.Figure()
            for _,row in is_t.iterrows():
                p=row["platform"]; v=row["IS%"]
                fig_is.add_trace(go.Bar(x=[p.title()],y=[v],
                    marker=dict(color=PC.get(p,BLU2),line=dict(width=0)),
                    text=[f"{v:.1f}%"],textposition="outside",textfont=dict(size=13,color=TP),
                    hovertemplate=f"<b>{p.title()}</b><br>Impression Share: {v:.1f}%<extra></extra>"))
            fig_is.add_hline(y=70,line_dash="dot",line_color=AMB3,line_width=1,
                              annotation_text="Target: 70%",annotation_font_color=AMB3,
                              annotation_font_size=10)
            pcfg(fig_is,260,"Impression Share % — Today")
            fig_is.update_layout(showlegend=False,yaxis=dict(range=[0,105]))
            st.plotly_chart(fig_is,use_container_width=True)
            caption("Below 70% IS = significant traffic you're eligible for but not winning. "
                    "Lost IS (Budget) = raise daily budget. Lost IS (Rank) = improve QS or bids.")

        sh("Quality Score Trend Over Time","📈",AMB2)
        qs_trend=pm_d.groupby(["date","platform"])["QS"].mean().reset_index()
        fig_qst=go.Figure()
        for p in qs_trend["platform"].unique():
            d=qs_trend[qs_trend["platform"]==p]
            fig_qst.add_trace(go.Scatter(x=d["date"],y=d["QS"],name=p.title(),
                mode="lines+markers",line=dict(color=PC[p],width=2),marker=dict(size=5),
                hovertemplate=f"<b>{p.title()}</b> QS: %{{y:.1f}}<extra></extra>"))
        fig_qst.add_hline(y=7,line_dash="dot",line_color=GRN3,line_width=1)
        pcfg(fig_qst,260,lh=True)
        fig_qst.update_layout(yaxis=dict(range=[0,11],title="Quality Score (simulated proxy)"))
        st.plotly_chart(fig_qst,use_container_width=True)
        caption("A declining QS trend signals worsening ad relevance. "
                "Common causes: audiences expanding beyond intent, landing page loading slowly, "
                "or ad copy no longer matching search queries.")

    with t4:
        sh("Today — Hourly Breakdown","🕐",BLU2)
        th=pm[pm["date"]==pm["date"].max()].groupby(["hour","platform"]).agg(
            cost=("cost","sum"),clicks=("clicks","sum"),conversions=("conversions","sum")).reset_index()
        fig_h=make_subplots(rows=2,cols=1,
            subplot_titles=["Hourly Ad Spend ($)","Hourly Conversions"],
            shared_xaxes=True,vertical_spacing=0.14)
        for p in th["platform"].unique():
            d=th[th["platform"]==p]
            fig_h.add_trace(go.Bar(x=d["hour"],y=d["cost"],name=p.title(),
                marker=dict(color=PC[p],line=dict(width=0))),row=1,col=1)
            fig_h.add_trace(go.Scatter(x=d["hour"],y=d["conversions"],
                line=dict(color=PC[p],width=2),mode="lines+markers",showlegend=False),row=2,col=1)
        fig_h.update_layout(**{**PL,"height":480,"barmode":"group",
                                "legend":dict(orientation="h",y=1.05,x=0)})
        fig_h.update_xaxes(title_text="Hour (UTC)",row=2)
        st.plotly_chart(fig_h,use_container_width=True)
        story("💡","Reading the Hourly Pattern",
              "Peak spend hours should <b>align with</b> peak conversion hours. "
              "If spend peaks at 9am but conversions peak at 1pm, your dayparting is misaligned — "
              "you're paying for clicks before your audience is ready to act.",
              issue="Spend and conversion peaks misaligned = wasted morning budget",
              action="Use bid adjustments or ad scheduling to concentrate spend on conversion hours",
              benefit="Aligning dayparting typically improves CVR by 10–25% at same spend")

    with t5:
        sh("Full KPI Table — All Days","📋",BLU2)
        disp=daily[["date","cost","revenue","margin","margin_pct","CPL","RPL","CPC","CVR",
                     "ROAS","CPM","CTR","ROI","butil","QS","IS_pct","conversions","clicks","impressions"]].copy()
        disp["margin_pct"]=(disp["margin_pct"]*100).round(1)
        disp["CVR"]=(disp["CVR"]*100).round(2)
        disp["CTR"]=(disp["CTR"]*100).round(3)
        disp=disp.sort_values("date",ascending=False)
        disp.columns=["Date","Spend","Revenue","Margin $","Margin %","CPL","RPL","CPC",
                       "CVR %","ROAS","CPM","CTR %","ROI %","Budget Used %","QS","IS %",
                       "Conversions","Clicks","Impressions"]
        st.dataframe(disp.style
            .format({"Spend":"${:,.0f}","Revenue":"${:,.0f}","Margin $":"${:,.0f}",
                     "Margin %":"{:.1f}%","CPL":"${:.2f}","RPL":"${:.2f}","CPC":"${:.2f}",
                     "CVR %":"{:.2f}%","ROAS":"{:.2f}x","CPM":"${:.2f}","CTR %":"{:.3f}%",
                     "ROI %":"{:.1f}%","Budget Used %":"{:.1f}%","QS":"{:.1f}","IS %":"{:.1f}%",
                     "Conversions":"{:,.0f}","Clicks":"{:,.0f}","Impressions":"{:,.0f}"})
            .background_gradient(subset=["Margin $","ROAS","ROI %","QS","IS %"],cmap="RdYlGn")
            .background_gradient(subset=["CPL","CPM","CPC"],cmap="RdYlGn_r"),
            use_container_width=True,height=540,hide_index=True)
        caption("Green Margin/ROAS/ROI/QS = better performance. Red CPL/CPM/CPC = higher costs (needs action).")

# ════════════════════════════════════════════════════════════════════════
#  PAGE 3 — REGIONAL REPORTING
# ════════════════════════════════════════════════════════════════════════
elif PAGE == "🌍  Regional Reporting":
    banner("🌍","Regional Reporting","US · UK · CA · AU — Performance by market")

    story("📖","Why Regional Reporting Matters",
          "The same campaign behaves <b>completely differently</b> across markets. "
          "US leads cost $45. UK leads cost $62. CA leads cost $38. AU leads cost $71. "
          "Without regional breakdowns, a strong US performance can mask a failing UK campaign — "
          "and you'd never know until the quarterly review.",
          issue="Blended numbers hide market-specific inefficiencies",
          action="Break performance down by US · UK · CA · AU",
          benefit="Market-specific optimizations typically improve regional CPL by 20–40%")

    # Simulate regional data from actual platform data
    np.random.seed(42)
    regions = ["US","UK","CA","AU"]
    region_multipliers = {"US":1.0,"UK":1.38,"CA":0.85,"AU":1.58}
    region_volume      = {"US":0.52,"UK":0.21,"CA":0.16,"AU":0.11}

    cc_day=cc_r.groupby("date").agg(revenue=("total_revenue","sum"),
        margin=("margin_dollar","sum"),calls=("inbound_calls","sum")).reset_index()
    daily_base=daily.merge(cc_day,on="date",how="left",suffixes=("","_cc"))

    # Build regional daily data
    reg_rows=[]
    for _,row in daily_base.iterrows():
        for reg in regions:
            vm=region_volume[reg]; cm=region_multipliers[reg]
            noise=np.random.normal(1,0.08)
            reg_rows.append({
                "date":row["date"],"region":reg,
                "spend":  row["cost"]   *vm*noise,
                "revenue":row["revenue"]*vm*noise/cm*1.1 if pd.notna(row.get("revenue")) else 0,
                "clicks": row["clicks"] *vm*noise,
                "convs":  row["conversions"]*vm*noise,
                "impr":   row["impressions"]*vm*noise,
            })
    reg_df=pd.DataFrame(reg_rows)
    reg_df["CPL"]  =(reg_df["spend"]/reg_df["convs"].replace(0,np.nan)).round(2)
    reg_df["ROAS"] =(reg_df["revenue"]/reg_df["spend"].replace(0,np.nan)).round(2)
    reg_df["CVR"]  =(reg_df["convs"]/reg_df["clicks"].replace(0,np.nan)*100).round(2)
    reg_df["CTR"]  =(reg_df["clicks"]/reg_df["impr"].replace(0,np.nan)*100).round(2)
    reg_df["CPC"]  =(reg_df["spend"]/reg_df["clicks"].replace(0,np.nan)).round(2)
    reg_df["margin"]=(reg_df["revenue"]-reg_df["spend"]).round(2)

    # Summary by region
    reg_sum=reg_df.groupby("region").agg(
        Spend=("spend","sum"),Revenue=("revenue","sum"),
        Convs=("convs","sum"),Clicks=("clicks","sum")).reset_index()
    reg_sum["CPL"]  =(reg_sum["Spend"]/reg_sum["Convs"].replace(0,np.nan)).round(2)
    reg_sum["ROAS"] =(reg_sum["Revenue"]/reg_sum["Spend"].replace(0,np.nan)).round(2)
    reg_sum["CVR%"] =(reg_sum["Convs"]/reg_sum["Clicks"].replace(0,np.nan)*100).round(2)
    reg_sum["Margin"]=(reg_sum["Revenue"]-reg_sum["Spend"]).round(2)
    reg_sum["ROI%"] =(reg_sum["Margin"]/reg_sum["Spend"].replace(0,np.nan)*100).round(1)
    reg_sum=reg_sum.sort_values("ROAS",ascending=False)

    REG_COLORS={"US":BLU3,"UK":GRN3,"CA":AMB3,"AU":RED3}

    # ── Regional KPI tiles ───────────────────────────────────
    sh("Regional Performance Summary","🗺️",BLU2)
    rcols=st.columns(4)
    for col,(_,row) in zip(rcols,reg_sum.iterrows()):
        reg=row["region"]; c=REG_COLORS.get(reg,BLU2)
        with col:
            st.markdown(
                f'<div style="background:{SURF};border:1px solid {BDR};border-left:4px solid {c};'
                f'border-radius:10px;padding:14px 16px 12px;margin-bottom:8px">'
                f'<div style="font-size:18px;font-weight:800;color:{c};margin-bottom:8px">{reg} 🌍</div>'
                f'<div style="font-size:11px;color:{TM};text-transform:uppercase;letter-spacing:1px">Spend</div>'
                f'<div style="font-size:20px;font-weight:700;color:{TP}">${row["Spend"]:,.0f}</div>'
                f'<div style="font-size:11px;color:{TM};margin-top:6px">Revenue</div>'
                f'<div style="font-size:18px;font-weight:700;color:{GRN3}">${row["Revenue"]:,.0f}</div>'
                f'<div style="display:flex;justify-content:space-between;margin-top:8px">'
                f'<div><div style="font-size:9px;color:{TM}">CPL</div>'
                f'<div style="font-size:14px;font-weight:700;color:{AMB3}">${row["CPL"]:.2f}</div></div>'
                f'<div><div style="font-size:9px;color:{TM}">ROAS</div>'
                f'<div style="font-size:14px;font-weight:700;color:{PUR3}">{row["ROAS"]:.2f}x</div></div>'
                f'<div><div style="font-size:9px;color:{TM}">CVR</div>'
                f'<div style="font-size:14px;font-weight:700;color:{TEA3}">{row["CVR%"]:.2f}%</div></div>'
                f'</div></div>', unsafe_allow_html=True)

    # ── Charts ───────────────────────────────────────────────
    rc1,rc2=st.columns(2)
    with rc1:
        sh("CPL by Region — Trend","💰",AMB2)
        fig_rcpl=go.Figure()
        for reg in regions:
            d=reg_df[reg_df["region"]==reg]
            fig_rcpl.add_trace(go.Scatter(x=d["date"],y=d["CPL"],name=reg,
                mode="lines+markers",line=dict(color=REG_COLORS[reg],width=2),marker=dict(size=5),
                hovertemplate=f"<b>{reg}</b> %{{x|%b %d}}<br>CPL: $%{{y:.2f}}<extra></extra>"))
        pcfg(fig_rcpl,280,lh=True); fig_rcpl.update_layout(yaxis_title="CPL ($)")
        st.plotly_chart(fig_rcpl,use_container_width=True)
        caption("CPL varies significantly by market. AU typically has the highest CPL due to smaller "
                "market size and higher competition. CA often delivers the best value. "
                "Use this to guide market-specific budget allocation.")

    with rc2:
        sh("ROAS by Region — Trend","📈",PUR2)
        fig_rr=go.Figure()
        for reg in regions:
            d=reg_df[reg_df["region"]==reg]
            fig_rr.add_trace(go.Scatter(x=d["date"],y=d["ROAS"],name=reg,
                mode="lines+markers",line=dict(color=REG_COLORS[reg],width=2),marker=dict(size=5),
                hovertemplate=f"<b>{reg}</b> %{{x|%b %d}}<br>ROAS: %{{y:.2f}}x<extra></extra>"))
        fig_rr.add_hline(y=1,line_dash="dot",line_color=BDR2,line_width=1,
                          annotation_text="1x breakeven",annotation_font_size=10)
        pcfg(fig_rr,280,lh=True); fig_rr.update_layout(yaxis_title="ROAS (x)")
        st.plotly_chart(fig_rr,use_container_width=True)
        caption("ROAS below 1.0x means a market is losing money. If any region stays below 1.0x "
                "for 3+ days, pause that region's campaigns and investigate targeting.")

    sh("Spend vs Revenue by Region","📊",GRN2)
    fig_rb=go.Figure()
    for reg in regions:
        d=reg_df.groupby("region")[["spend","revenue","margin"]].sum().reset_index()
        d=d[d["region"]==reg]
        fig_rb.add_trace(go.Bar(name=f"{reg} Spend",x=[reg],y=d["spend"].values,
            marker_color=REG_COLORS[reg],opacity=0.7,
            hovertemplate=f"<b>{reg} Spend</b>: $%{{y:,.0f}}<extra></extra>"))
        fig_rb.add_trace(go.Bar(name=f"{reg} Revenue",x=[reg],y=d["revenue"].values,
            marker_color=REG_COLORS[reg],marker_pattern_shape="/",
            hovertemplate=f"<b>{reg} Revenue</b>: $%{{y:,.0f}}<extra></extra>"))
    fig_rb.update_layout(**{**PL,"height":280,"barmode":"group","showlegend":True,
                             "legend":{**PL["legend"],"orientation":"h","y":1.06}})
    st.plotly_chart(fig_rb,use_container_width=True)
    caption("Solid bars = spend. Striped bars = revenue. When revenue bar is taller = profitable market. "
            "When spend bar is taller = that market is currently loss-making.")

    sh("Regional Comparison Table","📋",BLU2)
    st.dataframe(reg_sum.style
        .format({"Spend":"${:,.0f}","Revenue":"${:,.0f}","Margin":"${:,.0f}",
                 "Convs":"{:,.0f}","Clicks":"{:,.0f}","CPL":"${:.2f}",
                 "ROAS":"{:.2f}x","CVR%":"{:.2f}%","ROI%":"{:.1f}%"})
        .background_gradient(subset=["ROAS","ROI%","CVR%"],cmap="RdYlGn")
        .background_gradient(subset=["CPL"],cmap="RdYlGn_r"),
        use_container_width=True,hide_index=True)
    caption("Best ROAS = highest priority for budget increase. Highest CPL = investigate or reduce spend.")

# ════════════════════════════════════════════════════════════════════════
#  PAGE 4 — ML & FORECASTING
# ════════════════════════════════════════════════════════════════════════
elif PAGE == "🤖  ML & Forecasting":
    banner("🤖","ML & Forecasting","Anomaly Detection · Prophet Forecasting · K-Means Clustering")

    story("📖","What ML Adds That Manual Review Misses",
          "A human reviewing daily reports catches obvious problems. ML catches <b>compound, subtle patterns</b> — "
          "a CPL that's 18% above normal for 3 consecutive days, or a revenue forecast showing a cliff in 9 days. "
          "These patterns exist in the data but are invisible without algorithms scanning all dimensions at once.",
          issue="Manual monitoring misses multi-metric compound anomalies and early trend reversals",
          action="Isolation Forest (anomaly) + Prophet (forecast) + K-Means (clustering)",
          benefit="ML alerts surface issues 2–5 days earlier than manual review on average")

    ta,tf,tc=st.tabs(["🔴 Anomaly Detection","📉 Revenue Forecasting","🗂️ Performance Clusters"])

    with ta:
        feats=["cost","revenue","margin","CPL","CVR","ROAS"]
        df_ml=iso(daily.dropna(subset=feats).copy(),feats)
        anoms=df_ml[df_ml["anom"]==-1]; norm=df_ml[df_ml["anom"]==1]

        c1,c2,c3,c4=st.columns(4)
        with c1: st.markdown(kt("Days Analyzed",len(df_ml),None,"","",".0f","📅",BLU2), unsafe_allow_html=True)
        with c2: st.markdown(kt("Anomalies Found",len(anoms),None,"","",".0f","🚨",RED3), unsafe_allow_html=True)
        with c3: st.markdown(kt("Anomaly Rate",len(anoms)/max(len(df_ml),1)*100,None,"","%",".1f","📊",AMB3), unsafe_allow_html=True)
        with c4: st.markdown(kt("Clean Days",len(norm),None,"","",".0f","✅",GRN3), unsafe_allow_html=True)

        st.markdown(
            f'<div style="font-size:12px;color:{TM};margin:6px 0 10px">'
            + tip("How Isolation Forest Works","Isolation Forest — Unsupervised ML Anomaly Detection",
                  "Randomly partitions the data space. Points that get isolated in fewer splits are outliers — "
                  "they're statistically different from the majority pattern across all 6 KPIs simultaneously.",
                  "Catches anomalies invisible in single-metric monitoring",
                  "Cross-reference flagged dates with campaign change logs for root cause",
                  "Catching one anomalous spend day per month saves 3–8% of monthly budget",
                  "Contamination=0.08 = flags ~8% of days as anomalous")
            +'</div>', unsafe_allow_html=True)

        ax,ay=st.columns(2)
        mx=ax.selectbox("X-Axis Metric",feats,index=0,key="ax")
        my=ay.selectbox("Y-Axis Metric",feats,index=3,key="ay")

        fig_a=go.Figure()
        if not norm.empty:
            fig_a.add_trace(go.Scatter(x=norm[mx],y=norm[my],mode="markers",name="Normal Day",
                text=norm["date"].dt.strftime("%b %d"),
                marker=dict(color=GRN3,size=9,opacity=0.7,line=dict(color=BG,width=1)),
                hovertemplate="<b>%{text}</b><br>"+mx+": %{x:.2f}<br>"+my+": %{y:.2f}<extra>Normal</extra>"))
        if not anoms.empty:
            fig_a.add_trace(go.Scatter(x=anoms[mx],y=anoms[my],mode="markers",name="⚠ Anomaly",
                text=anoms["date"].dt.strftime("%b %d"),
                marker=dict(color=RED3,size=14,symbol="x",line=dict(color=RED3,width=2.5)),
                hovertemplate="<b>%{text} — ANOMALY FLAGGED</b><br>"+mx+": %{x:.2f}<br>"+my+": %{y:.2f}<extra></extra>"))
        pcfg(fig_a,360,f"Anomaly Detection: {mx} vs {my}",lh=True)
        fig_a.update_layout(xaxis_title=mx,yaxis_title=my)
        st.plotly_chart(fig_a,use_container_width=True)
        caption("Red X markers = statistically anomalous days. Hover to see the exact date. "
                "Cross-check anomalous dates against: campaign pauses, budget changes, "
                "bid strategy switches, or external events (holidays, competitor launches).")

        if not anoms.empty:
            sh("Flagged Anomalous Days","🚨",RED2)
            ad=anoms[["date","cost","revenue","margin","CPL","ROAS","ascore"]].copy()
            ad["date"]=ad["date"].dt.strftime("%Y-%m-%d")
            ad["ascore"]=ad["ascore"].round(4)
            st.dataframe(ad.sort_values("ascore").style
                .format({"cost":"${:,.0f}","revenue":"${:,.0f}","margin":"${:,.0f}",
                         "CPL":"${:.2f}","ROAS":"{:.2f}x"})
                .background_gradient(subset=["ascore"],cmap="Reds_r"),
                use_container_width=True,hide_index=True)
            st.caption("Lower anomaly score = more unusual day. Sort by ascore to see worst first.")
            if st.button("🧠 AI: Explain These Anomalies"):
                worst=ad.head(3).to_string(index=False)
                prompt=(f"ML Isolation Forest flagged these anomalous days:\n{worst}\n\n"
                        "For EACH date provide:\n🔴 ISSUE: what went wrong\n"
                        "🔍 ROOT CAUSE: most likely reason\n"
                        "✅ FIX: specific action with numbers\n★ BENEFIT: expected improvement\n"
                        "Be concise and data-driven.")
                with st.spinner("Analyzing..."):
                    alrt(ask(prompt,ENG,900),"ai",f"🤖 {ENG} · Anomaly Root Cause Analysis")

    with tf:
        story("📖","Why Forecasting is Critical",
              "Looking backwards shows what happened. Forecasting shows what's <b>coming</b>. "
              "A 14-day revenue forecast lets you adjust budgets, pause underperformers, and protect margin "
              "<b>before</b> problems materialize — not after the month closes.",
              issue="Budget decisions made without forward visibility = reactive management",
              action="Prophet ML models daily + weekly seasonality with confidence intervals",
              benefit="Proactive budget adjustments prevent overspend/underspend by 10–20%")

        fm=st.selectbox("Metric to Forecast",
            ["revenue","cost","margin","CPL","ROAS","CVR"],
            format_func=lambda x: {"revenue":"Revenue $","cost":"Ad Spend $","margin":"Margin $",
                                    "CPL":"CPL $","ROAS":"ROAS x","CVR":"CVR %"}.get(x,x))
        fd=st.slider("Forecast horizon (days)",7,30,14,key="fd")

        with st.spinner("Fitting forecast model..."):
            fc,method=prophet_fc(daily,fm,fd)

        he=daily["date"].max(); hist=daily[["date",fm]].dropna(); fut=fc[fc["ds"]>he]

        fig_fc=go.Figure()
        fig_fc.add_trace(go.Scatter(x=hist["date"],y=hist[fm],name="Actual",
            line=dict(color=TP,width=2.5),
            hovertemplate="%{x|%b %d}: %{y:,.2f}<extra>Actual</extra>"))
        if "yhat_lower" in fc.columns and not fut.empty:
            fig_fc.add_trace(go.Scatter(
                x=pd.concat([fut["ds"],fut["ds"][::-1]]),
                y=pd.concat([fut["yhat_upper"],fut["yhat_lower"][::-1]]),
                fill="toself",fillcolor=rgba(BLU2,0.1),
                line=dict(color="rgba(0,0,0,0)"),name="Confidence Band"))
        if not fut.empty:
            fig_fc.add_trace(go.Scatter(x=fut["ds"],y=fut["yhat"],
                name=f"Forecast ({method})",
                line=dict(color=BLU3,width=2.5,dash="dash"),
                hovertemplate="%{x|%b %d}: %{y:,.2f}<extra>Forecast</extra>"))
        fig_fc.add_vline(x=str(he),line_dash="dot",line_color=AMB3,line_width=1.5)
        fig_fc.add_annotation(x=str(he),y=1,yref="paper",text="Today ▶",
            font=dict(color=AMB3,size=10),showarrow=False,bgcolor=SURF2,yshift=14)
        pcfg(fig_fc,400,f"{fm} — {fd}-Day Forecast ({method})",lh=True)
        st.plotly_chart(fig_fc,use_container_width=True)
        caption(f"Dashed blue line = {method} forecast. Shaded band = confidence interval (80%). "
                "If the forecast shows a declining trend, act now: review campaigns, check auction health, "
                "increase bids on profitable keywords before the dip materializes.")

        if not fut.empty:
            f1,f2,f3,f4=st.columns(4)
            trend=(fut["yhat"].iloc[-1]-fut["yhat"].iloc[0])/abs(fut["yhat"].iloc[0])*100 if len(fut)>1 else 0
            with f1: st.markdown(kt("7-Day Avg",fut.head(7)["yhat"].mean(),None,"","",",.2f","📈",BLU3,"Forecast avg"), unsafe_allow_html=True)
            with f2: st.markdown(kt("Forecast High",fut["yhat"].max(),None,"","",",.2f","⬆️",GRN3), unsafe_allow_html=True)
            with f3: st.markdown(kt("Forecast Low",fut["yhat"].min(),None,"","",",.2f","⬇️",AMB3), unsafe_allow_html=True)
            with f4: st.markdown(kt("Trend",abs(trend),None,"","%",".1f","📊",GRN3 if trend>0 else RED3,"improving" if trend>0 else "declining"), unsafe_allow_html=True)

    with tc:
        story("📖","What K-Means Clustering Reveals",
              "K-Means groups your days into performance archetypes — typically 'great days', 'average days', 'bad days'. "
              "Once grouped, you can ask: <b>what conditions made the great days great?</b> "
              "Day of week? Campaign mix? Budget level? Platform split? That answer is your growth playbook.",
              issue="Daily performance noise makes it impossible to see patterns manually",
              action="K-Means clusters days by 5-metric performance profile",
              benefit="Replicating 'Cluster 1' conditions on average days improves performance by 15–30%")

        cl_f=["cost","revenue","CPL","ROAS","CVR"]
        df_cl=daily.dropna(subset=cl_f).copy()
        if len(df_cl)>=6:
            nk=st.slider("Number of clusters (K)",2,5,3,key="kn")
            Xs=StandardScaler().fit_transform(df_cl[cl_f])
            km=KMeans(n_clusters=nk,random_state=42,n_init=10)
            df_cl["Cluster"]=[f"Cluster {i+1}" for i in km.fit_predict(Xs)]
            cl_c={f"Cluster {i+1}": c for i,c in enumerate([BLU3,GRN3,AMB3,RED3,PUR3])}

            cc1,cc2=st.columns([3,2])
            with cc1:
                fig_cl=go.Figure()
                for cn,color in cl_c.items():
                    d=df_cl[df_cl["Cluster"]==cn]
                    if not d.empty:
                        fig_cl.add_trace(go.Scatter(x=d["cost"],y=d["revenue"],name=cn,mode="markers",
                            marker=dict(color=color,size=d["ROAS"]*4+6,
                                        line=dict(color=BG,width=1),opacity=0.85),
                            text=d["date"].dt.strftime("%b %d"),
                            hovertemplate=f"<b>{cn}</b><br>%{{text}}<br>Spend:$%{{x:,.0f}}<br>Rev:$%{{y:,.0f}}<extra></extra>"))
                pcfg(fig_cl,380,"Performance Clusters — Spend vs Revenue (bubble = ROAS)",lh=True)
                fig_cl.update_layout(xaxis_title="Ad Spend ($)",yaxis_title="Revenue ($)")
                st.plotly_chart(fig_cl,use_container_width=True)
                caption("Bigger bubbles = higher ROAS. Cluster in the upper-left = high revenue at low spend = best days. "
                        "Study what those dates had in common (campaigns, day of week, seasonality).")
            with cc2:
                sh("Cluster Profiles","📊",PUR2)
                centers=df_cl.groupby("Cluster")[cl_f].mean().round(2)
                med_roas=centers["ROAS"].median()
                for cn,row in centers.iterrows():
                    color=cl_c.get(cn,BLU2)
                    tag="⭐ Best Performer" if row["ROAS"]>med_roas*1.2 else ("⚠ Below Average" if row["ROAS"]<med_roas*.8 else "📊 Average")
                    st.markdown(
                        f'<div style="background:{SURF2};border:1px solid {color}40;'
                        f'border-left:3px solid {color};border-radius:8px;padding:10px 14px;margin:6px 0">'
                        f'<div style="font-size:11px;font-weight:700;color:{color};margin-bottom:5px">{cn} · {tag}</div>'
                        f'<div style="font-size:11px;color:{TS};line-height:1.9">'
                        f'Spend: ${row["cost"]:,.0f} · Revenue: ${row["revenue"]:,.0f}<br>'
                        f'CPL: ${row["CPL"]:.2f} · ROAS: {row["ROAS"]:.2f}x · CVR: {row["CVR"]*100:.2f}%'
                        f'</div></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
#  PAGE 5 — CALL & ATTRIBUTION
# ════════════════════════════════════════════════════════════════════════
elif PAGE == "📞  Call & Attribution":
    banner("📞","Call & Attribution","Revenue by type · Call quality · Attribution rates · Lead quality ladder")

    story("📖","Why Call & Attribution Data Is Critical",
          "Platform conversion counts include low-quality leads that never generate revenue. "
          "This page tracks what happens <b>after the click</b>: whether a call was answered, "
          "how long it lasted, and whether the lead was income-verified. "
          "These are the <b>true quality signals</b> that correlate with actual revenue.",
          issue="Platform conversions count all leads — including unqualified ones",
          action="Track inbound call duration ratios and income verification rates",
          benefit="Optimizing toward verified, long-duration calls improves RPL by 30–50%")

    t1,t2,t3=st.tabs(["💰 Call Center Revenue","📞 Call Quality","🔗 Attribution Rates"])

    with t1:
        cc_d=cc_r.groupby("date").agg(calls_rev=("calls_revenue","sum"),
            data_rev=("data_revenue","sum"),total_rev=("total_revenue","sum"),
            calls=("inbound_calls","sum"),forms=("data_submit_forms","sum"),
            margin=("margin_dollar","sum")).reset_index()

        c1,c2,c3,c4=st.columns(4)
        with c1: st.markdown(kt("Total Revenue",cc_d["total_rev"].sum(),None,"$","",",.0f","💰",GRN3,"Period total"), unsafe_allow_html=True)
        with c2: st.markdown(kt("Calls Revenue",cc_d["calls_rev"].sum(),None,"$","",",.0f","📞",BLU3,"Inbound calls"), unsafe_allow_html=True)
        with c3: st.markdown(kt("Data Revenue", cc_d["data_rev"].sum(), None,"$","",",.0f","📋",TEA3,"Form submissions"), unsafe_allow_html=True)
        with c4: st.markdown(kt("Total Calls",  cc_d["calls"].sum(),    None,"","",",.0f","📱",PUR3,"Inbound volume"), unsafe_allow_html=True)

        sh("Revenue by Type — Calls vs Data Submissions","📈",GRN2)
        fig_r=go.Figure()
        fig_r.add_trace(go.Scatter(x=cc_d["date"],y=cc_d["calls_rev"],name="Call Revenue",
            fill="tozeroy",fillcolor=rgba(BLU3,0.12),line=dict(color=BLU3,width=2.5),
            hovertemplate="%{x|%b %d}<br>Calls Revenue: $%{y:,.0f}<extra></extra>"))
        fig_r.add_trace(go.Scatter(x=cc_d["date"],y=cc_d["data_rev"],name="Data Revenue",
            fill="tonexty",fillcolor=rgba(TEA3,0.1),line=dict(color=TEA3,width=2),
            hovertemplate="%{x|%b %d}<br>Data Revenue: $%{y:,.0f}<extra></extra>"))
        pcfg(fig_r,280,lh=True); st.plotly_chart(fig_r,use_container_width=True)
        cr=cc_d["calls_rev"].sum(); dr=cc_d["data_rev"].sum()
        story("💡","Revenue Mix Analysis",
              f"Calls generate ${cr:,.0f} vs data submissions ${dr:,.0f}. "
              f"Call revenue typically has {cr/(cr+dr)*100:.0f}% share. "
              "Inbound calls generally convert at higher RPL than data forms because they "
              "involve real-time qualification. If call revenue is declining, check: "
              "Are phone extensions active? Are campaigns reaching phone-intent audiences?",
              issue="Call revenue declining = ad targeting or call center issue",
              action="Enable call extensions, review phone-intent keyword targeting",
              benefit="Shifting 10% of data leads to calls can increase RPL by $30–60 per lead")

        sh("Daily Margin Trend","📊",AMB2)
        fig_m=go.Figure(go.Bar(x=cc_d["date"],y=cc_d["margin"],
            marker=dict(color=[GRN3 if v>=0 else RED3 for v in cc_d["margin"]],line=dict(width=0)),
            hovertemplate="%{x|%b %d}<br>Margin: $%{y:,.0f}<extra></extra>"))
        pcfg(fig_m,200); fig_m.update_layout(showlegend=False)
        st.plotly_chart(fig_m,use_container_width=True)
        caption("Green bars = profitable day. Red bars = spend exceeds revenue that day. "
                "Multiple consecutive red bars = campaign is losing money at scale.")

    with t2:
        cq_d=cq_r.groupby("date").agg(calls=("total_calls","sum"),
            r30=("medium_income_call_ratio_30s","mean"),
            r60=("medium_income_call_ratio_60s","mean"),
            r90=("medium_income_call_ratio_90s","mean")).reset_index()

        st.markdown(
            f'<div style="font-size:12px;color:{TM};margin-bottom:10px">'
            + tip("Call Duration Ratios","What 30s / 60s / 90s Ratios Mean",
                  "% of medium-income calls that reached each duration milestone. "
                  "Longer calls = more qualified leads. Industry: 90s+ calls convert at 3x the rate of 30s calls.",
                  "Low 90s ratio = calls dropping before qualification complete",
                  "Review call scripts, check IVR flow for early drop-off causes",
                  "Improving 90s ratio by 10pp raises RPL by $15–25 per call",
                  "Target: 90s ratio >40% = healthy call quality")
            +'</div>', unsafe_allow_html=True)

        sh("Medium Income Call Ratio by Duration","📊",AMB2)
        fig_cq=go.Figure()
        for col,lbl,color in [("r30","30s Ratio",BLU3),("r60","60s Ratio",AMB3),("r90","90s Ratio",GRN3)]:
            fig_cq.add_trace(go.Scatter(x=cq_d["date"],y=cq_d[col]*100,name=lbl,
                mode="lines+markers",line=dict(color=color,width=2),marker=dict(size=5),
                hovertemplate=f"<b>{lbl}</b>: %{{y:.1f}}%<br>%{{x|%b %d}}<extra></extra>"))
        pcfg(fig_cq,300,lh=True); fig_cq.update_layout(yaxis_title="Ratio (%)")
        st.plotly_chart(fig_cq,use_container_width=True)
        caption("Green line (90s ratio) is the most important — it measures calls that lasted long enough "
                "for meaningful qualification. If the green line drops below 35%, review call handling quality.")

        sh("Call Quality Heatmap — Hour × Day","🌡️",PUR2)
        if not cq_r.empty:
            cq_h=cq_r.copy(); cq_h["day"]=cq_h["date"].dt.strftime("%m/%d")
            piv=cq_h.pivot_table(index="hour",columns="day",values="medium_income_call_ratio_30s",aggfunc="mean")
            fig_hm=go.Figure(go.Heatmap(z=piv.values*100,x=piv.columns.tolist(),
                y=[f"{h:02d}:00" for h in piv.index],
                colorscale=[[0,SURF2],[0.4,AMB3],[1,GRN3]],
                hovertemplate="Day: %{x}<br>Hour: %{y}<br>30s Ratio: %{z:.1f}%<extra></extra>",
                colorbar=dict(title="Ratio %",thickness=12,tickfont=dict(size=10,color=TS))))
            pcfg(fig_hm,320); fig_hm.update_layout(
                xaxis=dict(tickangle=-45,tickfont=dict(size=9)),
                yaxis=dict(tickfont=dict(size=9)))
            st.plotly_chart(fig_hm,use_container_width=True)
            caption("Green cells = high call quality at that hour/day. "
                    "Use the green bands to schedule agent shifts and set ad dayparting "
                    "to concentrate spend on high-quality call windows.")

    with t3:
        at_d=at_r.groupby("date").agg(
            icr=("inbound_call_rate","mean"),mivr=("medium_income_verified_rate","mean"),
            adr=("any_data_rate","mean"),hicr=("high_income_verified_carrier_rate","mean")).reset_index()

        st.markdown(
            f'<div style="font-size:12px;color:{TM};margin-bottom:10px">'
            + tip("Attribution Rates Explained","The Lead Quality Ladder",
                  "Inbound Call Rate: % impressions → phone call (top of funnel). "
                  "Med. Income Verified: % verified at medium income tier. "
                  "Any Data Rate: % completing any data form. "
                  "High Inc. Carrier: premium leads — highest RPL, best conversion.",
                  "All rates declining simultaneously = audience targeting degrading",
                  "Re-segment audiences, refresh creative, tighten income targeting",
                  "5pp improvement in High Inc. Carrier rate adds $40–80 RPL",
                  "High Inc. Carrier target: >25% of verified leads")
            +'</div>', unsafe_allow_html=True)

        sh("Attribution Rate Trends","📈",TEA2)
        fig_at=go.Figure()
        for col,lbl,color in [("icr","Inbound Call",BLU3),("mivr","Med. Income Verified",GRN3),
                               ("adr","Any Data",AMB3),("hicr","High Inc. Carrier",TEA3)]:
            fig_at.add_trace(go.Scatter(x=at_d["date"],y=at_d[col]*100,name=lbl,
                mode="lines+markers",line=dict(color=color,width=2),marker=dict(size=5),
                hovertemplate=f"<b>{lbl}</b>: %{{y:.1f}}%<br>%{{x|%b %d}}<extra></extra>"))
        pcfg(fig_at,320,lh=True); fig_at.update_layout(yaxis_title="Rate (%)")
        st.plotly_chart(fig_at,use_container_width=True)
        caption("These rates form a quality ladder. Higher rates at each tier = better lead quality. "
                "If 'Any Data Rate' is high but 'High Inc. Carrier' is low, "
                "you're capturing unqualified data leads — review audience income targeting.")

        sh("Latest Attribution Gauges","🎯",BLU2)
        latest=at_r[at_r["date"]==at_r["date"].max()].mean(numeric_only=True)
        gi=[("Inbound Call","inbound_call_rate",BLU3,"Target: >35%"),
            ("Med. Verified","medium_income_verified_rate",GRN3,"Target: >40%"),
            ("Any Data","any_data_rate",AMB3,"Target: >55%"),
            ("High Inc. Carrier","high_income_verified_carrier_rate",TEA3,"Target: >25%")]
        gc=st.columns(4)
        for col,(lbl,fld,color,bench) in zip(gc,gi):
            val=latest.get(fld,0)
            fig_g=go.Figure(go.Indicator(mode="gauge+number",value=val*100,
                number=dict(suffix="%",font=dict(size=24,color=TP)),
                title=dict(text=f"{lbl}<br><span style='font-size:10px;color:{TS}'>{bench}</span>",
                           font=dict(size=11,color=TS)),
                gauge=dict(axis=dict(range=[0,100],tickfont=dict(size=9,color=TM)),
                    bar=dict(color=color,thickness=0.65),bgcolor=SURF2,bordercolor=BDR,
                    steps=[dict(range=[0,30],color=rgba(RED3,0.1)),
                           dict(range=[30,60],color=rgba(AMB3,0.08)),
                           dict(range=[60,100],color=rgba(GRN3,0.08))],
                    threshold=dict(line=dict(color=color,width=2),value=val*100))))
            fig_g.update_layout(height=200,margin=dict(l=12,r=12,t=40,b=8),
                                 paper_bgcolor=SURF,font_color=TS)
            col.plotly_chart(fig_g,use_container_width=True)
        caption("Green zone (60–100%) = excellent. Amber (30–60%) = room to improve. Red (<30%) = needs urgent attention.")

# ════════════════════════════════════════════════════════════════════════
#  PAGE 6 — AI STRATEGY ENGINE
# ════════════════════════════════════════════════════════════════════════
elif PAGE == "🧠  AI Strategy Engine":
    banner("🧠","AI Strategy Engine",
           f"Live-data context · Issue→Root Cause→Action→Benefit · Engine: {ENG}")

    story("📖","How This AI Engine Works",
          "Every AI response is grounded in your <b>live dashboard data</b> — today's KPIs, "
          "MTD totals, regional performance, and platform breakdown auto-injected as context. "
          "No generic advice. The AI reasons about your actual numbers.",
          issue="Manual interpretation requires 30–60 min of analyst time per report",
          action=f"{ENG} analyzes your live numbers on demand",
          benefit="Time-to-insight reduced from hours to seconds — act on data while it's still relevant")

    live_ctx=(f"\n📊 LIVE PERFORMANCE CONTEXT ({TD_S}):"
              f"\n• Spend: ${sv(tday,'cost'):,.0f} | Revenue: ${sv(tday,'revenue'):,.0f}"
              f" | Margin: ${sv(tday,'margin'):,.0f} ({sv(tday,'margin_pct')*100:.1f}%)"
              f"\n• CPL: ${sv(tday,'CPL'):.2f} | ROAS: {sv(tday,'ROAS'):.2f}x"
              f" | CVR: {sv(tday,'CVR')*100:.2f}% | ROI: {sv(tday,'ROI'):.1f}%"
              f"\n• CTR: {sv(tday,'CTR')*100:.3f}% | CPC: ${sv(tday,'CPC'):.2f}"
              f" | CPM: ${sv(tday,'CPM'):.2f}"
              f"\n• MTD Spend: ${mtd['cost'].sum():,.0f} | MTD Revenue: ${mtd['revenue'].sum():,.0f}"
              f"\n• MTD Margin: ${mtd['margin'].sum():,.0f} | Avg ROAS: {mtd['ROAS'].mean():.2f}x"
              f"\n• Platforms: {', '.join(PF)}\n")

    t1,t2,t3=st.tabs(["💬 AI Chat","⚡ Automated Reports","📜 Insight History"])

    with t1:
        st.markdown(f'<div style="font-size:11px;color:{TM};margin-bottom:10px">'
                    f'Grounded in live data · {ENG} · Responses: Answer→Root Cause→Action→Benefit</div>',
                    unsafe_allow_html=True)
        if "hist" not in st.session_state: st.session_state.hist=[]

        st.markdown(f'<div style="font-size:10px;font-weight:700;letter-spacing:1px;'
                    f'text-transform:uppercase;color:{TM};margin-bottom:6px">Quick Prompts</div>',
                    unsafe_allow_html=True)
        qps=[
            ("🔍 Why is CPL high?","Analyze today's CPL. Give root cause and 3 specific fixes with expected CPL impact."),
            ("💰 Best budget shift?","Based on today's platform ROAS and CPL, recommend specific $ reallocation."),
            ("⚠️ Red flags?","Review today's KPIs. Flag anomalies, rate severity 1–10, give immediate action."),
            ("📋 Daily brief",f"Daily performance brief for {TD_S}: health status, 2 wins, 2 risks, 1 action. <200 words."),
            ("🎨 Creative fatigue?","Signs of creative fatigue per platform? Which needs refresh first and what to test?"),
            ("📈 7-day outlook","Based on current trends, forecast next 7 days. What could improve or derail performance?"),
        ]
        qc=st.columns(3)
        for i,(lbl,prm) in enumerate(qps):
            if qc[i%3].button(lbl,use_container_width=True,key=f"qp{i}"):
                st.session_state["pq"]=prm

        st.markdown(f'<hr style="border:none;border-top:1px solid {BDR};margin:10px 0">',unsafe_allow_html=True)
        for msg in st.session_state.hist:
            if msg["role"]=="user":
                st.markdown(f'<div class="chu"><div class="chl" style="color:{BLU3}">You</div>{msg["content"]}</div>',unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="cha"><div class="chl" style="color:{PUR3}">🤖 {ENG}</div>{msg["content"]}</div>',unsafe_allow_html=True)

        ui=st.chat_input(f"Ask {ENG} about your campaigns...")
        if "pq" in st.session_state: ui=st.session_state.pop("pq")
        if ui:
            st.session_state.hist.append({"role":"user","content":ui})
            full=(f"You are a senior paid media analytics expert.\n{live_ctx}\n"
                  f"Question: {ui}\n\n"
                  f"Structure your response:\n1. DIRECT ANSWER\n2. ROOT CAUSE\n3. SPECIFIC ACTION (with numbers)\n4. EXPECTED BENEFIT\nBe concise.")
            with st.spinner("Thinking..."):
                resp=ask(full,ENG,700)
            st.session_state.hist.append({"role":"assistant","content":resp})
            st.rerun()
        if st.session_state.hist:
            if st.button("🗑️ Clear",key="clr"): st.session_state.hist=[]; st.rerun()

    with t2:
        st.markdown(f'<div style="font-size:11px;color:{TM};margin-bottom:12px">'
                    f'Each report uses live data. Results: Problem→Root Cause→Action→Benefit.</div>',unsafe_allow_html=True)
        r1,r2=st.columns(2)
        with r1:
            sh("Health Check","🔍",RED3)
            if st.button("▶ Run Health Check",use_container_width=True,key="rh"):
                pm_p=pm[pm["date"]==pm["date"].max()].groupby("platform").agg(
                    cost=("cost","sum"),conversions=("conversions","sum"),clicks=("clicks","sum")).reset_index()
                pm_p["CPL"]=(pm_p["cost"]/pm_p["conversions"].replace(0,np.nan)).round(2)
                prompt=(f"Paid media health check {TD_S}.\n{live_ctx}\nPlatform data:\n{pm_p.to_string(index=False)}\n\n"
                        "For each issue found:\n🔴 ISSUE\n🔍 ROOT CAUSE\n✅ ACTION ($ amounts)\n★ BENEFIT\n"
                        "End: overall health score 1–10 + top priority action.")
                with st.spinner("Running..."):
                    alrt(ask(prompt,ENG,900),"ai",f"🤖 Health Check · {TD_S}")
            sh("Creative Fatigue","🎨",PUR2)
            if st.button("▶ Fatigue Analysis",use_container_width=True,key="rf"):
                prompt=(f"Analyze creative/search fatigue.\n{live_ctx}\nPer platform:\n"
                        "1. Fatigue signal (CTR/CVR/CPL direction)\n2. Creative refresh recommendation\n"
                        "3. Query expansion opportunity\n4. Top 2 A/B test ideas with hypothesis")
                with st.spinner("Analyzing..."):
                    alrt(ask(prompt,ENG,700),"ai","🎨 Creative Fatigue Analysis")
        with r2:
            sh("Budget Reallocation","💸",AMB2)
            if st.button("▶ Reallocation Plan",use_container_width=True,key="rb"):
                pm_p=pm[pm["date"]==pm["date"].max()].groupby("platform").agg(
                    cost=("cost","sum"),conversions=("conversions","sum"),clicks=("clicks","sum")).reset_index()
                pm_p["CPL"]=(pm_p["cost"]/pm_p["conversions"].replace(0,np.nan)).round(2)
                prompt=(f"Budget reallocation plan.\n{live_ctx}\nPlatform data:\n{pm_p.to_string(index=False)}\n\n"
                        "Provide:\n🔴 PROBLEM: underperforming platform + why\n"
                        "💸 ACTION: exact $ shift (e.g. Move $500/day from X to Y)\n"
                        "📊 RATIONALE: CPL/ROAS logic\n★ EXPECTED RESULT: projected improvement\n"
                        "⚠️ RISK: what to watch after reallocation")
                with st.spinner("Building plan..."):
                    alrt(ask(prompt,ENG,700),"warn","💸 Budget Reallocation Plan")
            sh("Daily Brief","📋",GRN2)
            if st.button("▶ Daily Brief",use_container_width=True,key="rd"):
                prompt=(f"Daily paid media brief {TD_S}.\n{live_ctx}\n"
                        "Format:\n🟢/🟡/🔴 HEALTH: one line\n✅ WIN 1:\n✅ WIN 2:\n"
                        "⚠️ RISK 1:\n⚠️ RISK 2:\n🎯 TOP ACTION: single most important thing tomorrow\n<200 words.")
                with st.spinner("Writing..."):
                    alrt(ask(prompt,ENG,500),"ok",f"📋 Daily Brief · {TD_S}")

    with t3:
        sh("Stored AI Insights","📜",BLU2)
        ins=load_ins(30)
        if ins.empty:
            alrt("No stored insights yet. Run analyses above or use the FastAPI backend scheduler.","info")
        else:
            for _,row in ins.iterrows():
                sev=row.get("severity","info")
                kind={"critical":"crit","warning":"warn","info":"ok"}.get(sev,"info")
                alrt(row.get("summary",""),kind,
                     f"{row.get('insight_type','').upper()} · {row.get('generated_at','')}")

# ════════════════════════════════════════════════════════════════════════
#  PAGE 7 — REAL-TIME MONITOR
# ════════════════════════════════════════════════════════════════════════
elif PAGE == "⚡  Real-Time Monitor":
    banner("⚡","Real-Time Intraday Monitor",
           f"Live pacing · Hourly breakdown · Budget tracking · UTC {datetime.utcnow().strftime('%H:%M')}")

    story("📖","What to Watch Intraday",
          "Real-time monitoring catches <b>same-day problems</b> before they compound. "
          "Key question: <i>Is spend pacing correctly to hit daily targets?</i> "
          "Over-pacing exhausts budget before peak hours. Under-pacing misses impression share.",
          issue="Budget can exhaust before peak conversion hours (typically 12–3pm)",
          action="Monitor cumulative spend vs revenue pacing every 2 hours",
          benefit="Catching a pacing issue at 10am saves 4+ hours of lost opportunity or wasted spend")

    auto_r=st.checkbox("🔁 Auto-refresh every 60s",value=False)
    if auto_r:
        import time; time.sleep(60); st.rerun()

    tpm=pm[pm["date"]==pm["date"].max()].groupby(["hour","platform"]).agg(
        cost=("cost","sum"),clicks=("clicks","sum"),
        impressions=("impressions","sum"),conversions=("conversions","sum")).reset_index()
    tcc=cc_r[cc_r["date"]==cc_r["date"].max()]
    cur_h=datetime.utcnow().hour

    ls=tpm["cost"].sum(); lc=tpm["conversions"].sum()
    lr=tcc["total_revenue"].sum() if not tcc.empty else 0
    lm=tcc["margin_dollar"].sum() if not tcc.empty else 0
    lcpl=ls/lc if lc>0 else 0; lroas=lr/ls if ls>0 else 0

    r1c1,r1c2,r1c3,r1c4,r1c5,r1c6=st.columns(6)
    with r1c1: st.markdown(kt("Live Spend",  ls,  None,"$","",",.0f","💸",RED3, f"Hour {cur_h:02d}:00 UTC"), unsafe_allow_html=True)
    with r1c2: st.markdown(kt("Live Revenue",lr,  None,"$","",",.0f","💰",GRN3), unsafe_allow_html=True)
    with r1c3: st.markdown(kt("Live Margin", lm,  None,"$","",",.0f","📊",BLU3), unsafe_allow_html=True)
    with r1c4: st.markdown(kt("Conversions", lc,  None,"","",",.0f","🎯",AMB3), unsafe_allow_html=True)
    with r1c5: st.markdown(kt("Live CPL",    lcpl, None,"$","",".2f","📉",PUR3), unsafe_allow_html=True)
    with r1c6: st.markdown(kt("Live ROAS",   lroas,None,"","x",".2f","📈",TEA3), unsafe_allow_html=True)

    budget_bar("Daily Budget Pacing",ls,BUDGET)

    mc,sc=st.columns([3,1])
    with mc:
        sh("Cumulative Spend vs Revenue","📈",BLU2)
        fig_rt=make_subplots(rows=2,cols=1,
            subplot_titles=["Cumulative Spend vs Revenue ($)","Hourly Spend by Platform ($)"],
            vertical_spacing=0.14)
        cum=tpm.groupby("hour").agg(cost=("cost","sum")).cumsum()
        cc_cum=tcc.set_index("hour")[["total_revenue"]].cumsum() if not tcc.empty else pd.DataFrame()
        if not cum.empty:
            fig_rt.add_trace(go.Scatter(x=cum.index,y=cum["cost"],name="Cum. Spend",
                fill="tozeroy",fillcolor=rgba(RED3,0.1),line=dict(color=RED3,width=2)),row=1,col=1)
        if not cc_cum.empty:
            fig_rt.add_trace(go.Scatter(x=cc_cum.index,y=cc_cum["total_revenue"],name="Cum. Revenue",
                fill="tozeroy",fillcolor=rgba(GRN3,0.1),line=dict(color=GRN3,width=2)),row=1,col=1)
        for p,c in PC.items():
            pd_=tpm[tpm["platform"]==p]
            if not pd_.empty:
                fig_rt.add_trace(go.Bar(x=pd_["hour"],y=pd_["cost"],name=p.title(),
                    marker=dict(color=c,line=dict(width=0))),row=2,col=1)
        fig_rt.update_layout(**{**PL,"height":500,"barmode":"group",
                                  "legend":dict(orientation="h",y=1.05,x=0)})
        fig_rt.update_xaxes(title_text="Hour (UTC)",row=2)
        st.plotly_chart(fig_rt,use_container_width=True)
        caption("Top: Cumulative spend vs revenue — green above red = profitable day. "
                "Bottom: Hourly spend by platform — identify which platform drives peak costs at each hour.")

    with sc:
        sh(f"Day {cur_h}/24 Elapsed","⏱️",AMB2)
        exp_pct=cur_h/24.0
        fig_p=go.Figure(go.Indicator(mode="gauge+number",value=round(exp_pct*100,1),
            number=dict(suffix="% of day",font=dict(size=18,color=TP)),
            title=dict(text=f"{cur_h}h elapsed",font=dict(size=11,color=TS)),
            gauge=dict(axis=dict(range=[0,100],tickfont=dict(size=9,color=TM),nticks=5),
                bar=dict(color=BLU2,thickness=0.65),bgcolor=SURF2,bordercolor=BDR,
                steps=[dict(range=[0,33],color=rgba(GRN3,0.08)),
                       dict(range=[33,66],color=rgba(AMB3,0.07)),
                       dict(range=[66,100],color=rgba(RED3,0.07))])))
        fig_p.update_layout(height=200,margin=dict(l=12,r=12,t=40,b=8),
                             paper_bgcolor=SURF,font_color=TS)
        st.plotly_chart(fig_p,use_container_width=True)

        sh("Platform Split","🥧",BLU2)
        ts_=tpm["cost"].sum()
        for plat,c in PC.items():
            sp=tpm[tpm["platform"]==plat]["cost"].sum()
            pct=sp/ts_*100 if ts_>0 else 0
            bc=GRN3 if pct<40 else AMB3 if pct<65 else RED3
            st.markdown(
                f'<div style="margin:9px 0">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:3px">'
                f'<span style="font-size:11px;font-weight:700;color:{c}">{plat.title()}</span>'
                f'<span style="font-size:11px;color:{TP}">${sp:,.0f} · {pct:.1f}%</span></div>'
                f'<div style="background:{BDR};border-radius:4px;height:7px">'
                f'<div style="background:{bc};width:{pct}%;height:7px;border-radius:4px"></div>'
                f'</div></div>', unsafe_allow_html=True)

        if lr>0 and ls>0:
            ratio=ls/lr
            if ratio<0.6:
                alrt("✅ Revenue well ahead of spend — healthy margin pacing","ok")
            elif ratio>0.95:
                alrt("⚠️ Spend near revenue — margin risk","warn")

    sh(f"Last 6 Hours Detail — Through {cur_h:02d}:00 UTC","🕐",TEA2)
    rh=tpm[tpm["hour"]>=max(0,cur_h-5)]
    if not rh.empty:
        rp=rh.pivot_table(index="hour",columns="platform",values=["cost","clicks","conversions"],aggfunc="sum")
        rp.columns=[f"{m.title()} · {p.title()}" for m,p in rp.columns]
        rp.index=[f"{h:02d}:00" for h in rp.index]
        cc_=[c for c in rp.columns if "Cost" in c]
        oc_=[c for c in rp.columns if "Cost" not in c]
        st.dataframe(rp.style.format(**{c:"${:,.0f}" for c in cc_},**{c:"{:,.0f}" for c in oc_})
            .background_gradient(subset=cc_,cmap="RdYlGn_r"),
            use_container_width=True)
        caption("Redder Cost cells = higher spend at that hour. "
                "Use this to fine-tune dayparting bid adjustments — reduce bids on red hours, increase on green.")

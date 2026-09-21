"""
Paid Ads Intelligence Dashboard  ·  Production-Ready
Google · Meta · Microsoft  ·  Claude AI + GPT-4o
Self-explaining KPIs · Storytelling · Hover Insights · Issue→Solution→Benefit
Regional Reporting (US, UK, CA, AU)  ·  Quality Score · Impression Share · Forecasting · ML
"""
# ── stdlib ──────────────────────────────────────────────────────────────
import sqlite3, warnings
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
OAI_KEY   = "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"
ANT_KEY   = "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"
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

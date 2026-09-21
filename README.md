# 📡 Ads Intelligence Dashboard

A production-ready, self-explaining multi-platform paid ads analytics system with AI-powered insights, ML anomaly detection, and time-series forecasting.

---

## What It Does

| Capability | Detail |
|---|---|
| **Multi-platform tracking** | Google Ads · Meta Ads · Microsoft Advertising |
| **Real-time KPIs** | Spend, Revenue, Margin, CPL, ROAS, CVR with day-over-day deltas |
| **ML Anomaly Detection** | Isolation Forest scans 6 KPIs simultaneously for outlier days |
| **Forecasting** | Prophet + Linear Trend — 7–30 day forward projections with confidence bands |
| **Performance Clustering** | K-Means groups days into performance archetypes |
| **AI Strategy Engine** | Claude + GPT-4o — live-data context, root cause analysis, budget reallocation |
| **Call Center Analytics** | Revenue by type, call quality ratios (30s/60s/90s), attribution rates |
| **Storytelling** | Every chart explains itself: Issue → Analysis → Benefit |
| **Hover Tooltips** | `?` icons reveal metric definitions, root causes, and fix recommendations |

---

## Project Structure

```
ads_dashboard/
├── streamlit_app.py          # Main dashboard (self-explaining, all 6 pages)
├── main.py                   # FastAPI backend entry point
├── scheduler.py              # APScheduler — hourly data ingestion
├── requirements.txt
│
├── data_generators/
│   └── ads_generator.py      # Realistic synthetic data for Google/Meta/Microsoft
│
├── db/
│   ├── models.py             # SQLAlchemy ORM models
│   └── database.py           # Async SQLite engine + session factory
│
├── pipeline/
│   ├── ingest.py             # Bulk insert from generator → DB
│   └── transforms.py         # KPI calculations: CPL, ROAS, CVR, WoW, MTD
│
├── ai/
│   └── claude_engine.py      # Claude anomaly detection + daily brief engine
│
├── api/
│   └── routes.py             # FastAPI endpoints for all KPI data
│
└── frontend/
    └── index.html            # Static HTML dashboard (alternative frontend)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
```

### 3. Run the Streamlit dashboard

```bash
streamlit run streamlit_app.py
```

Opens at **http://localhost:8501**

### 4. (Optional) Run the FastAPI backend

```bash
uvicorn main:app --reload --port 8000
```

On first run, automatically seeds **30 days of realistic synthetic data** across all platforms and starts the hourly scheduler.

---

## Dashboard Pages

| Page | What It Shows |
|---|---|
| 🏠 **Executive Overview** | Today's KPIs, revenue vs spend trend, MTD summary, WoW performance matrix, platform scorecard |
| 📈 **Performance Analysis** | Spend/conversion by platform, CTR/CPC/CVR trends, efficiency scatter, hourly intraday, full KPI table |
| 🤖 **ML & Forecasting** | Isolation Forest anomaly scatter + flagged days, Prophet forecast with confidence bands, K-Means clustering |
| 📞 **Call & Attribution** | Call vs data revenue, margin trend, call quality duration ratios, attribution rate gauges |
| 🧠 **AI Strategy Engine** | Live-context AI chat (Claude/GPT-4o), automated health checks, budget reallocation plans, daily briefs |
| ⚡ **Real-Time Monitor** | Intraday cumulative spend vs revenue, hourly breakdown, pacing gauge, platform split bars |

---

## Self-Explaining Design

Every section follows the **Issue → Analysis → Benefit** pattern:

- **Story blocks** — narrative cards under charts explaining what happened and why it matters
- **Hover tooltips** (`?` icons) — pop up with metric definition, root cause, fix, and expected benefit
- **Dynamic narratives** — e.g. if spend > revenue, an alert fires with specific next steps
- **WoW scorecard** — each metric shows human interpretation: "✓ cheaper leads", "⚠ traffic quality drop"

---

## Data Model

| Table | Description |
|---|---|
| `platform_metrics` | Hourly impressions, clicks, cost, conversions per platform |
| `call_center` | Hourly call/data revenue, inbound calls, margin |
| `call_quality` | Medium income call ratios at 30s/60s/90s duration |
| `data_quality` | High income verified carrier ratios by income tier |
| `attribution_rates` | Inbound call, medium income verified, any data, high income carrier rates |
| `ai_insights` | Stored Claude/GPT-4o analysis results with severity levels |

---

## Key Metrics Explained

| Metric | Formula | Good Direction |
|---|---|---|
| **CPL** | Spend ÷ Conversions | ↓ Lower is better |
| **RPL** | Revenue ÷ Conversions | ↑ Higher is better |
| **ROAS** | Revenue ÷ Spend | ↑ Higher is better (>1.0x = profitable) |
| **CVR** | Conversions ÷ Clicks | ↑ Higher is better |
| **CPC** | Spend ÷ Clicks | ↓ Lower is better |
| **CTR** | Clicks ÷ Impressions | ↑ Higher is better |
| **Margin $** | Revenue − Spend | ↑ Positive means profitable |
| **Margin %** | Margin ÷ Revenue | ↑ Higher protects profitability |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Dashboard UI | Streamlit + Plotly |
| AI / LLM | Anthropic Claude + OpenAI GPT-4o |
| ML | scikit-learn (Isolation Forest, K-Means, Linear Regression) |
| Forecasting | Facebook Prophet + statsmodels |
| Backend API | FastAPI + uvicorn |
| Database | SQLite via SQLAlchemy async ORM |
| Scheduling | APScheduler (hourly data refresh) |
| Data Generation | Custom synthetic generator with realistic seasonality |

---

## API Endpoints (FastAPI)

```
GET  /api/topline-kpis?date=YYYY-MM-DD
GET  /api/realtime-hourly?date=YYYY-MM-DD
GET  /api/wow-trend?date=YYYY-MM-DD
GET  /api/mtd-summary?date=YYYY-MM-DD
GET  /api/call-quality?date=YYYY-MM-DD
GET  /api/attribution?date=YYYY-MM-DD
POST /api/run-anomaly-detection
POST /api/run-daily-brief
```

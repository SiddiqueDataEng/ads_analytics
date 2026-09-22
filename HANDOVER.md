# Handover & Technical Documentation

## 1. What changed in this pass

| Area | Before | After |
|---|---|---|
| Secrets | Hardcoded API keys in `streamlit_app.py`, committed to a public repo | Loaded from `.env` / Streamlit secrets, `.gitignore`d |
| Data source | Only `data_generators/ads_generator.py` (synthetic) | Real connectors for Google Ads, Meta, Microsoft Ads (`connectors/`), toggled via `DATA_SOURCE` |
| Conversion tracking | One aggregate `conversions` count per platform/hour | New `conversion_actions` table + `pipeline/conversion_mapping.py` harmonizes every platform's native conversion action names into canonical types (`phone_call`, `lead_form`, `purchase`, `data_submit`) |
| Profitability | Blended CPA/ROAS/margin only at the platform level | Added `get_conversion_type_profitability()` — blended CPA and net margin **per conversion type**, across all three platforms |
| AI recommendations | Claude free-formed dollar amounts in budget reallocation prompts | Reallocation $ amounts are now computed in Python from real 7-day CPL data first (`_compute_budget_signals`); Claude explains/validates rather than invents the number |
| Fatigue detection | Not implemented | `run_fatigue_scan()` — campaign-level CTR trend (5-day, early vs late window) flags real decliners; Claude turns flagged campaigns into next-step recommendations |
| Weekly strategy | Not implemented | `run_weekly_strategy()` — cross-channel weekly note grounded in WoW + conversion-type profitability |
| Data integrity | No reconciliation path | `GET /api/reconciliation` re-pulls a day live and diffs it against what's stored, flagging >2% mismatches per platform/metric |
| Reliability | N/A | Live ingestion isolates per-platform failures — if Microsoft's API is down, Google + Meta still ingest; failures are logged, not silently papered over with fake numbers |

## 2. Architecture

```
scheduler.py (APScheduler, hourly + daily + weekly cron)
     │
     ├── DATA_SOURCE=synthetic → data_generators/ads_generator.py
     └── DATA_SOURCE=live      → connectors/live.py
                                     ├── connectors/google_ads.py   (GAQL reporting API)
                                     ├── connectors/meta_ads.py     (Marketing API Insights)
                                     └── connectors/microsoft_ads.py (Bing Ads Reporting API, async)
                                          │
                                          ▼
                              pipeline/ingest.py → SQLite (db/models.py)
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
        pipeline/transforms.py    ai/claude_engine.py     pipeline/reconciliation.py
        (KPIs, WoW, MTD,          (anomaly, budget,        (live vs. stored diff)
         conversion profitability) fatigue, weekly)
                    │                     │
                    └──────────┬──────────┘
                               ▼
              api/routes.py (FastAPI)  ──or──  streamlit_app.py (reads DB directly)
```

Both frontends (FastAPI+`frontend/index.html` and the Streamlit app) read from the same SQLite database, so whichever one you run, they stay in sync.

## 3. Getting real credentials (one-time setup per platform)

**Google Ads**
1. Apply for a developer token in the Google Ads API Center (Tools → API Center on the MCC account). Test-account tokens are instant; production approval can take a few days.
2. Create an OAuth 2.0 Client ID (Desktop app type) in Google Cloud Console, generate a refresh token via the standard OAuth installed-app flow.
3. Fill `GOOGLE_ADS_*` in `.env`.

**Meta Ads**
1. Create a Meta App in developers.facebook.com, add the Marketing API product.
2. Generate a **long-lived** System User access token (not a short-lived user token — those expire in ~1-2 hours and will break the hourly scheduler).
3. Fill `META_*` in `.env`, including the ad account ID in `act_XXXXXXXXXX` format.

**Microsoft Advertising**
1. Register an app in Azure AD, request Bing Ads API access (developer token via the Microsoft Advertising UI → User Settings → Developer Settings).
2. Generate a refresh token via the OAuth 2.0 authorization code flow.
3. Fill `MSADS_*` in `.env`.

Once all three are set, flip `DATA_SOURCE=live` in `.env` and restart. On first run with an empty DB, `main.py` will backfill ~30 days of real history before the scheduler takes over hourly ingestion.

## 4. Conversion action mapping — you need to do this once

`pipeline/conversion_mapping.py` ships with placeholder native action names guessed from common naming patterns. **Before going live, replace `CONVERSION_MAP` with your actual account's exact conversion action names**:

- Google Ads: Tools & Settings → Conversions → copy the exact "Action name" for each goal
- Meta: Events Manager → note each `action_type` string (visible in Ads Manager column customization or via a test API pull)
- Microsoft: Tools → Conversion Goals → copy the exact goal name

Anything not in the map falls into `"unmapped"` rather than being dropped — check `/api/dashboard/conversion-profitability` periodically for an `unmapped` bucket and extend the map as needed.

## 5. Reconciliation (client handover requirement)

`GET /api/reconciliation?date=YYYY-MM-DD` re-fetches that day directly from all three platforms and diffs it against what's stored in the DB, flagging any metric that differs by more than 2%. Expected sources of small, non-alarming drift:

- **Attribution lag** — conversions can be attributed to a day up to 24-72h after the fact on some platforms; a same-day pull will look artificially low until the platform's data settles.
- **Time zone mismatch** — confirm all three platforms and the DB are reporting in the same time zone (`segments.date` in Google Ads uses the account's time zone, not UTC, unless configured otherwise).

Run this as a spot-check before the handover call, and re-run it on any day a client flags "my numbers don't match."

## 6. Known limitations / next steps

- **Regional (US/UK/CA/AU) breakdown in the Streamlit dashboard is still simulated** (`np.random` noise applied to totals) — pulling real geo data requires adding geo-target dimensions to each connector (Google: `segments.geo_target_region`; Meta: `region` breakdown; Microsoft: `LocationId`). Scoped out of this pass; flag if it's a priority.
- **Quality Score and Impression Share are proxy-calculated**, not pulled from the Google Ads API directly (both are available via GAQL — `metrics.search_impression_share`, `ad_group_criterion.quality_info.quality_score` — but require ad-group-level queries not yet wired in).
- **Package versions in `requirements.txt`** for `google-ads`, `facebook-business`, and `bingads` are best-effort — pin/verify against the current PyPI releases before deploying, since I can't reach the network from this environment to confirm exact current versions.
- **These connectors have not been run against live credentials** — I don't have network access in this sandbox to test real API calls. Test each connector individually (`python -c "from connectors import google_ads; print(google_ads.fetch_hourly('2026-09-21', 10))"`) before relying on the scheduler.
- **Bing Ads reporting is async** (submit → poll → download) and meaningfully slower than the other two connectors per call — don't schedule it more often than hourly, and expect the hourly job to take longer in live mode than in synthetic mode.

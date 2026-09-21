"""
Claude AI Optimization & Action Engine.
Detects anomalies, generates budget reallocation recommendations,
flags creative/search fatigue — per the project spec.
"""
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime

from anthropic import Anthropic
from sqlalchemy.ext.asyncio import AsyncSession
from db.models import AIInsight
from pipeline.transforms import get_topline_kpis, get_wow_trend, get_mtd_summary


ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Thresholds for anomaly detection (can be tuned)
ANOMALY_THRESHOLDS = {
    "cpl_spike_pct": 0.25,       # CPL up >25% vs prior day = anomaly
    "cvr_drop_pct": -0.20,       # CVR drop >20% vs prior day
    "margin_drop_pct": -0.15,    # Margin drop >15%
    "pacing_over_pct": 0.15,     # Spending >15% over expected MTD pace
}


def _build_anomaly_prompt(kpis: Dict, wow: Dict, mtd: Dict) -> str:
    return f"""You are a senior paid media analytics expert. Analyze the following performance data and identify any anomalies or urgent action items.

TODAY'S KPIs:
- CPL: ${kpis['CPL']:.2f}
- RPL: ${kpis['RPL']:.2f}
- CPC: ${kpis['CPC']:.2f}
- CVR: {kpis['CVR']*100:.2f}%
- Margin $: ${kpis['margin_dollar']:.2f}
- Margin %: {kpis['margin_pct']*100:.2f}%
- Total Cost: ${kpis['total_cost']:.2f}
- Total Revenue: ${kpis['total_revenue']:.2f}

WEEK-OVER-WEEK CHANGES:
- Cost WoW: {wow['wow_cost_pct']*100:.1f}%
- Revenue WoW: {wow['wow_revenue_pct']*100:.1f}%
- Margin WoW: {wow['wow_margin_pct']*100:.1f}%
- CVR WoW: {wow['wow_cvr_pct']*100:.1f}%

MTD SUMMARY:
- MTD Cost: ${mtd['mtd_cost']:.2f}
- MTD Revenue: ${mtd['mtd_revenue']:.2f}
- MTD Margin: ${mtd['mtd_margin_dollar']:.2f} ({mtd['mtd_margin_pct']*100:.1f}%)
- MTD CPL: ${mtd['mtd_cpl']:.2f}

Respond in JSON with this exact structure:
{{
  "anomalies": [
    {{"type": "cpa_spike|cvr_drop|budget_pacing|margin_decline", "severity": "critical|warning|info", "description": "...", "action": "..."}}
  ],
  "budget_recommendations": [
    {{"platform": "google|meta|microsoft|all", "action": "increase|decrease|pause|reallocate", "amount_pct": 10, "rationale": "..."}}
  ],
  "fatigue_flags": [
    {{"channel": "...", "type": "search_query|creative", "description": "...", "next_step": "..."}}
  ],
  "summary": "2-3 sentence executive summary of overall performance and top priority action."
}}"""


def _build_daily_brief_prompt(kpis: Dict, mtd: Dict, date: str) -> str:
    return f"""You are a paid media performance analyst. Write a concise daily performance brief for {date}.

KPIs: CPL ${kpis['CPL']:.2f} | RPL ${kpis['RPL']:.2f} | CVR {kpis['CVR']*100:.2f}% | Margin {kpis['margin_pct']*100:.1f}%
MTD: Cost ${mtd['mtd_cost']:.2f} | Revenue ${mtd['mtd_revenue']:.2f} | Margin ${mtd['mtd_margin_dollar']:.2f}

Provide:
1. One-line health status (green/yellow/red)
2. Top 2 wins today
3. Top 2 risks or areas needing attention
4. One concrete optimization action for tomorrow

Keep it under 200 words, plain text, actionable."""


class ClaudeEngine:
    def __init__(self):
        if ANTHROPIC_API_KEY:
            self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        else:
            self.client = None

    def _call_claude(self, prompt: str, max_tokens: int = 1024) -> str:
        """Call Claude API — falls back to mock if no API key."""
        if not self.client:
            return self._mock_response(prompt)
        message = self.client.messages.create(
            model="claude-opus-4-5",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def _mock_response(self, prompt: str) -> str:
        """Return realistic mock AI response when no API key is set."""
        if "anomalies" in prompt:
            return json.dumps({
                "anomalies": [
                    {
                        "type": "cpa_spike",
                        "severity": "warning",
                        "description": "Google CPL increased 18% above 7-day average, primarily driven by non-brand campaigns.",
                        "action": "Reduce bids on non-brand exact match by 10-15%. Review search terms for irrelevant queries."
                    }
                ],
                "budget_recommendations": [
                    {
                        "platform": "meta",
                        "action": "increase",
                        "amount_pct": 12,
                        "rationale": "Meta showing strongest RPL at $94 with improving CVR trend. Headroom available before saturation."
                    },
                    {
                        "platform": "microsoft",
                        "action": "decrease",
                        "amount_pct": 8,
                        "rationale": "Microsoft margin below target at 18%. Reallocate budget to higher-performing channels."
                    }
                ],
                "fatigue_flags": [
                    {
                        "channel": "google",
                        "type": "search_query",
                        "description": "CTR declining on top 3 ad groups over last 5 days — creative fatigue signal.",
                        "next_step": "Rotate in 2 new RSA variants. Test new value prop headlines focused on speed/ease."
                    }
                ],
                "summary": "Overall performance is within acceptable range but Google efficiency is slipping. Prioritize Meta budget shift and Google creative refresh this week to protect margin targets."
            })
        return "Performance is on track. CPL trending within budget targets. Monitor Meta CVR closely as it shows early signs of improvement."

    async def run_anomaly_detection(self, session: AsyncSession, date: str) -> AIInsight:
        kpis = await get_topline_kpis(session, date)
        wow = await get_wow_trend(session, date)
        mtd = await get_mtd_summary(session, date)

        prompt = _build_anomaly_prompt(kpis, wow, mtd)
        raw = self._call_claude(prompt, max_tokens=1500)

        # Parse JSON response
        try:
            parsed = json.loads(raw)
            summary = parsed.get("summary", raw[:300])
            anomalies = parsed.get("anomalies", [])
            severity = "critical" if any(a.get("severity") == "critical" for a in anomalies) else \
                       "warning" if any(a.get("severity") == "warning" for a in anomalies) else "info"
        except Exception:
            summary = raw[:300]
            severity = "info"

        insight = AIInsight(
            insight_type="anomaly",
            platform="all",
            summary=summary,
            full_response=raw[:5000],
            severity=severity,
        )
        session.add(insight)
        await session.commit()
        await session.refresh(insight)
        return insight

    async def run_daily_brief(self, session: AsyncSession, date: str) -> AIInsight:
        kpis = await get_topline_kpis(session, date)
        mtd = await get_mtd_summary(session, date)

        prompt = _build_daily_brief_prompt(kpis, mtd, date)
        raw = self._call_claude(prompt, max_tokens=512)

        insight = AIInsight(
            insight_type="daily_summary",
            platform="all",
            summary=raw[:500],
            full_response=raw[:5000],
            severity="info",
        )
        session.add(insight)
        await session.commit()
        await session.refresh(insight)
        return insight

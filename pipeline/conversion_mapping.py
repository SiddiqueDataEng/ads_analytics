"""
Conversion-action harmonization layer.

Google, Meta, and Microsoft each report conversions under their own native
action names, and those names vary per account/campaign setup. This module
maps every native action name to a small set of CANONICAL conversion types
so profitability (blended CPA, margin per conversion type) can be computed
consistently across platforms — this is the "Map and harmonize multiple
conversion actions across platforms" deliverable.

Edit CONVERSION_MAP to match your actual account's conversion action names —
pull the exact strings from:
  - Google Ads:      Tools > Conversions > Action name
  - Meta Ads:         Events Manager > Custom Conversions / standard events
  - Microsoft Ads:    Tools > Conversion Goals > Goal name

Anything not found in the map falls through to "unmapped" rather than being
silently dropped, so you can see in the dashboard what still needs mapping.
"""
from typing import Optional

CANONICAL_TYPES = ["phone_call", "lead_form", "purchase", "data_submit", "other"]

# native_action_name (lowercased, exact match) -> canonical_type
CONVERSION_MAP = {
    # ── Google Ads ──────────────────────────────────────────────
    "phone call leads": "phone_call",
    "calls from ads": "phone_call",
    "phone_call_leads": "phone_call",
    "lead form submission": "lead_form",
    "submit lead form": "lead_form",
    "purchase": "purchase",
    "form submit": "data_submit",

    # ── Meta Ads ────────────────────────────────────────────────
    "onsite_conversion.lead_grouped": "lead_form",
    "onsite_conversion.call_confirm": "phone_call",
    "offsite_conversion.fb_pixel_purchase": "purchase",
    "onsite_conversion.messaging_conversation_started_7d": "lead_form",
    "lead": "lead_form",

    # ── Microsoft Ads ───────────────────────────────────────────
    "phone_call": "phone_call",
    "phonecall": "phone_call",
    "submit_lead_form": "lead_form",
    "purchaseconversiongoal": "purchase",
}


def map_conversion_action(native_action_name: str, platform: Optional[str] = None) -> str:
    """
    Return the canonical_type for a native conversion action name.
    Falls back to 'unmapped' (not silently dropped) if not found, so the
    dashboard can surface unmapped actions for someone to classify.
    """
    if not native_action_name:
        return "unmapped"
    key = native_action_name.strip().lower()
    return CONVERSION_MAP.get(key, "unmapped")


def unmapped_actions_seen(native_action_names: list[str]) -> list[str]:
    """Utility for an audit pass: which native action names have no mapping yet."""
    return sorted({
        name for name in native_action_names
        if map_conversion_action(name) == "unmapped"
    })

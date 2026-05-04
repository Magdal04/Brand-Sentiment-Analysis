"""
AIJudge — calls an AI model API to obtain a decision and rationale for a Scenario.

Environment variables
---------------------
OPENAI_API_KEY  : required for live AI calls.
                  If absent the app falls back to a rule-based default so the
                  demo always runs without a key.
OPENAI_BASE_URL : optional; override for alternative OpenAI-compatible endpoints.
OPENAI_MODEL    : optional; defaults to "gpt-4o-mini".
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass

# Allow this module to be imported both when Streamlit adds moral_machine/ to
# sys.path (direct import) and when used as part of the moral_machine package.
try:
    from scenarios import Scenario  # type: ignore[import]  # direct / Streamlit path
except ImportError:
    from moral_machine.scenarios import Scenario  # package path


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AIJudgement:
    """Result of asking the AI model to judge a scenario."""

    choice_key: str      # e.g. "A", "B", or "C"
    motive: str          # AI's plain-text explanation
    raw_response: str    # raw model output (or an error/fallback note)
    used_fallback: bool  # True when the deterministic fallback was used


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _build_prompt(scenario: Scenario) -> str:
    options_block = "\n".join(
        f"  {opt.key}: {opt.label} — {opt.details}" for opt in scenario.options
    )
    valid_keys = [opt.key for opt in scenario.options]
    return (
        "You are an ethical AI advisor participating in a Moral Machine demo.\n"
        "Read the following scenario carefully, then choose the most ethically "
        "defensible option.\n\n"
        f"## Scenario: {scenario.title}\n\n"
        f"{scenario.description}\n\n"
        f"## Options\n{options_block}\n\n"
        "## Response format\n"
        "Respond with ONLY valid JSON — no markdown fences, no extra text:\n"
        f'{{"choice": "<one of {valid_keys}>", "motive": "<2-5 sentence explanation>"}}'
    )


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def _parse_response(raw: str, scenario: Scenario) -> tuple[str | None, str | None]:
    """Return (choice_key, motive) or (None, None) on failure."""
    try:
        # Strip optional code fences
        text = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        data = json.loads(text)
        choice = str(data.get("choice", "")).strip().upper()
        motive = str(data.get("motive", "")).strip()
        valid = {opt.key.upper() for opt in scenario.options}
        if choice in valid and motive:
            return choice, motive
    except Exception:
        pass
    return None, None


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------


def _rule_based_fallback(scenario: Scenario, error: str = "") -> AIJudgement:
    """
    Deterministic fallback that selects the first option.

    Guarantees the demo is always runnable — even without an API key or when
    the API call fails.
    """
    first = scenario.options[0]
    note = f" [API error: {error}]" if error else " [No API key configured]"
    motive = (
        f"{note} Falling back to a rule-based heuristic: choosing option {first.key} "
        f"('{first.label}') as the default safe choice — "
        "prioritising immediate, certain outcomes over uncertain future ones."
    )
    return AIJudgement(
        choice_key=first.key,
        motive=motive,
        raw_response=f"<fallback{note}>",
        used_fallback=True,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def judge(scenario: Scenario) -> AIJudgement:
    """
    Ask the configured AI model to judge *scenario*.

    Returns an :class:`AIJudgement`.  If ``OPENAI_API_KEY`` is not set or the
    API call fails, returns a deterministic fallback judgement so the demo
    always runs without errors.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return _rule_based_fallback(scenario)

    prompt = _build_prompt(scenario)
    raw_response: str | None = None

    # --- Attempt 1: openai SDK ------------------------------------------------
    try:
        import openai  # type: ignore

        client = openai.OpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=350,
        )
        raw_response = (completion.choices[0].message.content or "").strip()
    except ImportError:
        # openai SDK not installed — fall through to requests
        pass
    except Exception as exc:
        return _rule_based_fallback(scenario, str(exc))

    # --- Attempt 2: raw requests (SDK not installed) --------------------------
    if raw_response is None:
        try:
            import requests  # type: ignore

            base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            resp = requests.post(
                f"{base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 350,
                },
                timeout=30,
            )
            resp.raise_for_status()
            raw_response = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            return _rule_based_fallback(scenario, str(exc))

    # --- Parse response -------------------------------------------------------
    choice, motive = _parse_response(raw_response, scenario)
    if choice is None:
        fb = _rule_based_fallback(scenario, "model returned invalid JSON")
        fb.raw_response = raw_response
        return fb

    return AIJudgement(
        choice_key=choice,
        motive=motive,
        raw_response=raw_response,
        used_fallback=False,
    )

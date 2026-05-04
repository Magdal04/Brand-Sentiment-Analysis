"""
Moral Machine — Ethics of AI Demo
==================================
A Streamlit app that presents moral dilemmas, records the human player's
choice, queries an AI model for its independent decision, then displays
end-of-session statistics.

Run
---
    streamlit run moral_machine/app.py

Environment variables (optional)
---------------------------------
    OPENAI_API_KEY   — enables live AI judgements (app works without it via fallback)
    OPENAI_BASE_URL  — override for alternative OpenAI-compatible endpoints
    OPENAI_MODEL     — model name (default: gpt-4o-mini)
"""

from __future__ import annotations

import os
import sys

# Ensure moral_machine/ is on the path so sibling modules are importable
# regardless of how Streamlit resolves the working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd

from scenarios import (  # type: ignore[import]
    SCENARIO_GENERATORS,
    SCENARIO_NAMES,
    Scenario,
    get_random_scenario,
    get_scenario_by_index,
)
from ai_judge import AIJudgement, judge  # type: ignore[import]

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Moral Machine — Ethics of AI",
    page_icon="⚖️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------


def _init_state() -> None:
    defaults: dict = {
        "phase": "setup",           # "setup" | "playing" | "results"
        "target_rounds": 5,
        "scenario_mode": "random",  # "random" | "sequential"
        "scenario_index": 0,        # pointer for sequential mode
        "current_scenario": None,
        "human_choice": None,
        "current_judgement": None,
        "results": [],              # list of result dicts, one per completed round
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _next_scenario() -> Scenario:
    """Return the next scenario according to the selected mode."""
    if st.session_state.scenario_mode == "random":
        return get_random_scenario()
    s = get_scenario_by_index(st.session_state.scenario_index)
    st.session_state.scenario_index += 1
    return s


def _start_session(target: int, mode: str) -> None:
    st.session_state.phase = "playing"
    st.session_state.target_rounds = target
    st.session_state.scenario_mode = mode
    st.session_state.scenario_index = 0
    st.session_state.results = []
    st.session_state.current_scenario = _next_scenario()
    st.session_state.human_choice = None
    st.session_state.current_judgement = None


def _record_and_advance() -> None:
    """Save the current round result, then set up the next scenario or switch to results."""
    s: Scenario = st.session_state.current_scenario
    j: AIJudgement = st.session_state.current_judgement
    h: str = st.session_state.human_choice

    match = h == j.choice_key
    human_opt = next((o for o in s.options if o.key == h), None)
    ai_opt = next((o for o in s.options if o.key == j.choice_key), None)

    st.session_state.results.append(
        {
            "round": len(st.session_state.results) + 1,
            "scenario_id": s.id,
            "scenario_title": s.title,
            "human_choice": h,
            "human_label": human_opt.label if human_opt else h,
            "ai_choice": j.choice_key,
            "ai_label": ai_opt.label if ai_opt else j.choice_key,
            "match": match,
            "ai_motive": j.motive,
            "used_fallback": j.used_fallback,
        }
    )

    if len(st.session_state.results) >= st.session_state.target_rounds:
        st.session_state.phase = "results"
    else:
        st.session_state.current_scenario = _next_scenario()
        st.session_state.human_choice = None
        st.session_state.current_judgement = None


# ---------------------------------------------------------------------------
# PHASE: Setup
# ---------------------------------------------------------------------------


def _render_setup() -> None:
    st.title("⚖️ Moral Machine — Ethics of AI Demo")
    st.markdown(
        """
        Welcome! In this demo you will face a series of **moral dilemmas** drawn
        from real AI ethics debates.

        - Choose what **you** think is the right action.
        - An AI model will independently make its own choice.
        - At the end, see how often you and the AI agreed — and read its reasoning.

        > The app works **without an API key** (rule-based fallback is used).  
        > Set `OPENAI_API_KEY` in your environment for live AI opinions.
        """
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        target = st.number_input(
            "Number of scenarios", min_value=1, max_value=20, value=5, step=1
        )
    with col2:
        mode = st.radio(
            "Scenario order",
            options=["random", "sequential"],
            format_func=lambda x: "🎲 Random" if x == "random" else "📋 Sequential",
            help="Sequential cycles through all scenario types in order.",
        )

    # API-key status
    api_configured = bool(os.getenv("OPENAI_API_KEY", "").strip())
    if api_configured:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        st.success(f"✅ AI model configured: **{model}**")
    else:
        st.warning("⚠️ `OPENAI_API_KEY` not set — rule-based fallback will be used.")

    # List available scenario types
    st.markdown(f"**Available scenario types ({len(SCENARIO_GENERATORS)}):**")
    for name in SCENARIO_NAMES:
        st.markdown(f"- {name}")

    st.divider()

    if st.button("▶ Start Session", type="primary", use_container_width=True):
        _start_session(int(target), mode)
        st.rerun()


# ---------------------------------------------------------------------------
# PHASE: Playing — awaiting human choice
# ---------------------------------------------------------------------------


def _render_awaiting_choice() -> None:
    s: Scenario = st.session_state.current_scenario
    done = len(st.session_state.results)
    total = st.session_state.target_rounds

    st.progress(done / total, text=f"Round {done + 1} of {total}")
    st.subheader(s.title)
    st.markdown(s.description)
    st.divider()

    st.markdown("**What do you choose?**")
    cols = st.columns(len(s.options))
    for col, opt in zip(cols, s.options):
        with col:
            if st.button(
                f"**{opt.key}** — {opt.label}",
                key=f"choice_{opt.key}",
                use_container_width=True,
                help=opt.details,
            ):
                st.session_state.human_choice = opt.key
                st.rerun()

    st.divider()
    if st.button("🛑 Stop & show results", use_container_width=False):
        if st.session_state.results:
            st.session_state.phase = "results"
            st.rerun()
        else:
            st.warning("Answer at least one scenario before viewing results.")


# ---------------------------------------------------------------------------
# PHASE: Playing — showing round result after human chose
# ---------------------------------------------------------------------------


def _render_round_result() -> None:
    s: Scenario = st.session_state.current_scenario
    h: str = st.session_state.human_choice
    done = len(st.session_state.results)
    total = st.session_state.target_rounds

    # Call the AI once and cache the result in session state
    if st.session_state.current_judgement is None:
        with st.spinner("🤖 Consulting AI…"):
            st.session_state.current_judgement = judge(s)
        st.rerun()

    j: AIJudgement = st.session_state.current_judgement
    match = h == j.choice_key

    st.progress(done / total, text=f"Round {done + 1} of {total}")
    st.subheader(s.title)

    # Side-by-side human / AI choices
    col_h, col_ai = st.columns(2)
    human_opt = next((o for o in s.options if o.key == h), None)
    ai_opt = next((o for o in s.options if o.key == j.choice_key), None)

    with col_h:
        st.markdown("### 🧑 Your choice")
        st.info(f"**{h}** — {human_opt.label if human_opt else h}")

    with col_ai:
        st.markdown("### 🤖 AI choice")
        if match:
            st.success(f"**{j.choice_key}** — {ai_opt.label if ai_opt else j.choice_key}")
        else:
            st.error(f"**{j.choice_key}** — {ai_opt.label if ai_opt else j.choice_key}")

    # Agreement banner
    if match:
        st.success("✅ You and the AI agreed!")
    else:
        st.warning("❌ You and the AI disagreed.")

    if j.used_fallback:
        st.caption("ℹ️ AI fallback active (no API key) — rule-based default was used.")

    # AI rationale
    st.markdown("**AI rationale:**")
    st.markdown(f"> {j.motive}")
    st.divider()

    # Navigation buttons
    col_next, col_stop = st.columns([3, 1])
    with col_next:
        is_last = (done + 1) >= total
        label = "📊 View results" if is_last else "➡ Next scenario"
        if st.button(label, type="primary", use_container_width=True):
            _record_and_advance()
            st.rerun()

    with col_stop:
        if not is_last and st.button("🛑 Stop", use_container_width=True):
            _record_and_advance()  # save current round before stopping
            st.session_state.phase = "results"
            st.rerun()


# ---------------------------------------------------------------------------
# PHASE: Results
# ---------------------------------------------------------------------------


def _render_results() -> None:
    results = st.session_state.results

    st.title("📊 Session Results")

    if not results:
        st.warning("No scenarios were completed in this session.")
        if st.button("🔄 Start over"):
            st.session_state.phase = "setup"
            st.rerun()
        return

    total = len(results)
    matches = sum(1 for r in results if r["match"])
    agreement_pct = matches / total * 100

    # KPI row
    k1, k2, k3 = st.columns(3)
    k1.metric("Scenarios played", total)
    k2.metric("Agreements with AI", matches)
    k3.metric("Agreement rate", f"{agreement_pct:.1f}%")

    st.divider()

    # Summary table
    st.subheader("Round-by-round breakdown")
    df = pd.DataFrame(
        [
            {
                "Round": r["round"],
                "Scenario": r["scenario_title"],
                "Your choice": f"{r['human_choice']} — {r['human_label']}",
                "AI choice": f"{r['ai_choice']} — {r['ai_label']}",
                "Match": "✅" if r["match"] else "❌",
            }
            for r in results
        ]
    )
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.divider()

    # Per-round AI rationale log
    st.subheader("AI rationale log")
    for r in results:
        icon = "✅" if r["match"] else "❌"
        with st.expander(f"{icon} Round {r['round']}: {r['scenario_title']}"):
            st.markdown(f"**Your choice:** {r['human_choice']} — {r['human_label']}")
            st.markdown(f"**AI choice:** {r['ai_choice']} — {r['ai_label']}")
            st.markdown(f"**AI reasoning:** {r['ai_motive']}")
            if r["used_fallback"]:
                st.caption("ℹ️ Fallback used (no API key)")

    st.divider()

    col_again, col_export = st.columns(2)
    with col_again:
        if st.button("🔄 Play again", type="primary", use_container_width=True):
            # Reset all session state (keep defaults)
            for key in [
                "phase",
                "results",
                "current_scenario",
                "human_choice",
                "current_judgement",
                "scenario_index",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    with col_export:
        csv = pd.DataFrame(results).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Export CSV",
            data=csv,
            file_name="moral_machine_session.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Router — dispatch to the correct phase renderer
# ---------------------------------------------------------------------------

phase: str = st.session_state.phase

if phase == "setup":
    _render_setup()
elif phase == "playing":
    if st.session_state.human_choice is None:
        _render_awaiting_choice()
    else:
        _render_round_result()
elif phase == "results":
    _render_results()

"""
Scenario data model and generators for the Moral Machine Ethics Demo.

Architecture
------------
- Scenario        : dataclass holding title, description, and a list of Option objects.
- Option          : one selectable choice (key, label, details).
- Generator       : a zero-argument callable that returns a freshly randomised Scenario.
- SCENARIO_GENERATORS : module-level list of all registered generators.
- SCENARIO_NAMES  : human-readable names aligned with SCENARIO_GENERATORS.

To add a new scenario
---------------------
1. Write a function  generate_<name>_scenario() -> Scenario
2. Append it to SCENARIO_GENERATORS.
3. Append the display name to SCENARIO_NAMES.
No other files need to change.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


# ---------------------------------------------------------------------------
# Core data model
# ---------------------------------------------------------------------------


@dataclass
class Option:
    """A single selectable choice within a scenario."""

    key: str     # Short identifier shown on the button, e.g. "A"
    label: str   # Concise descriptive label
    details: str  # One-sentence expansion shown as a tooltip / sub-text


@dataclass
class Scenario:
    """
    A fully rendered moral / decision-making scenario ready for display.

    Fields
    ------
    id          : machine-readable identifier (snake_case)
    title       : human-readable title (may include an emoji prefix)
    description : full narrative presented to the player (Markdown supported)
    options     : list of Option objects (usually 2-3)
    metadata    : dict of raw variables used when generating this instance
                  (useful for logging / analysis)
    """

    id: str
    title: str
    description: str
    options: List[Option]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scenario generators
# ---------------------------------------------------------------------------


def generate_ambulance_split_scenario() -> Scenario:
    """
    Scenario 1 – Ambulance Split (triage under uncertainty).

    A dispatcher must choose which patient group gets the single ICU slot.
    """
    dist_a = random.randint(4, 12)
    dist_b = random.randint(dist_a + 2, 18)
    surv_a = round(random.uniform(0.45, 0.90), 2)
    surv_b = round(random.uniform(0.20, 0.60), 2)
    size_a = random.randint(2, 4)
    size_b = random.randint(2, 4)
    uncertain = random.choice([True, False])
    ages_a = sorted(random.randint(5, 75) for _ in range(size_a))
    ages_b = sorted(random.randint(5, 75) for _ in range(size_b))

    uncertainty_note = (
        "\n\n⚠️ Field reports may be inaccurate — exact injury severity is unconfirmed."
        if uncertain
        else ""
    )

    description = (
        f"You are the dispatcher for a single ambulance with **one ICU slot** "
        f"and two standard stretchers.\n\n"
        f"**Group 1** ({size_a} patients, ages {list(ages_a)})  \n"
        f"Distance: **{dist_a} min** · Estimated survival if treated: **{surv_a:.0%}**\n\n"
        f"**Group 2** ({size_b} patients, ages {list(ages_b)})  \n"
        f"Distance: **{dist_b} min** · One patient will die without ICU.  "
        f"Estimated survival if treated: **{surv_b:.0%}**"
        f"{uncertainty_note}"
    )

    return Scenario(
        id="ambulance_split",
        title="🚑 Ambulance Split — Triage Under Uncertainty",
        description=description,
        options=[
            Option(
                "A",
                "Send to Group 1",
                f"Closer ({dist_a} min), higher overall survival estimate ({surv_a:.0%}).",
            ),
            Option(
                "B",
                "Send to Group 2",
                f"Farther ({dist_b} min) but one patient will die without ICU ({surv_b:.0%}).",
            ),
        ],
        metadata=dict(
            dist_a=dist_a,
            dist_b=dist_b,
            surv_a=surv_a,
            surv_b=surv_b,
            size_a=size_a,
            size_b=size_b,
            ages_a=list(ages_a),
            ages_b=list(ages_b),
            uncertain=uncertain,
        ),
    )


def generate_rare_drug_scenario() -> Scenario:
    """
    Scenario 3 – Rare Drug Allocation (fairness vs. utility).

    One dose of a scarce medication; two patients with contrasting prognoses.
    """
    surv_p1 = round(random.uniform(0.65, 0.92), 2)
    surv_p2 = round(random.uniform(0.15, 0.50), 2)
    age_p1 = random.randint(25, 70)
    age_p2 = random.randint(8, 65)
    is_minor_p2 = age_p2 < 18
    has_alt_p1 = random.choice([True, False])
    donated = random.choice([True, False])

    alt_note = (
        "Patient 1 has access to a secondary (less effective) treatment. "
        if has_alt_p1
        else "Patient 1 has no alternative treatments. "
    )
    donation_note = (
        "\n\n📦 The dose arrived via an earmarked donation; no specific patient was named on the form."
        if donated
        else ""
    )
    minor_tag = " *(minor)*" if is_minor_p2 else ""

    description = (
        f"A hospital has **one dose** of a rare life-saving medication.\n\n"
        f"**Patient 1** (age {age_p1}): {alt_note}"
        f"Survival probability with the drug: **{surv_p1:.0%}**.\n\n"
        f"**Patient 2** (age {age_p2}){minor_tag}: No alternatives — will die without it. "
        f"Survival probability with the drug: **{surv_p2:.0%}**."
        f"{donation_note}"
    )

    return Scenario(
        id="rare_drug",
        title="💊 Rare Drug — Fairness vs. Utility",
        description=description,
        options=[
            Option(
                "A",
                "Give to Patient 1",
                f"Maximise expected survival ({surv_p1:.0%} > {surv_p2:.0%}).",
            ),
            Option(
                "B",
                "Give to Patient 2",
                "Prioritise the worst-off — they have no other option (rule of rescue).",
            ),
        ],
        metadata=dict(
            surv_p1=surv_p1,
            surv_p2=surv_p2,
            age_p1=age_p1,
            age_p2=age_p2,
            is_minor_p2=is_minor_p2,
            has_alt_p1=has_alt_p1,
            donated=donated,
        ),
    )


def generate_emergency_alert_scenario() -> Scenario:
    """
    Scenario 6 – Emergency Alert: False Alarm vs. Late Warning.

    An AI flood-prediction model suggests issuing an alert now or waiting for
    satellite confirmation.
    """
    confidence = random.randint(45, 82)
    population = random.randint(5_000, 250_000)
    false_alarms = random.randint(0, 4)
    hour = random.randint(0, 23)
    time_str = f"{hour:02d}:00"
    vulnerable_pct = random.randint(10, 40)
    confirm_hours = random.randint(1, 4)

    alarm_fatigue = (
        f"\n\n⚠️ **{false_alarms} false alarm(s)** issued in the past month — "
        "alert fatigue is a concern."
        if false_alarms >= 2
        else ""
    )

    description = (
        f"An AI flood-prediction system reports **{confidence}% confidence** "
        f"of a major flood within 6 hours.\n\n"
        f"Affected area: ~**{population:,} residents** "
        f"({vulnerable_pct}% elderly or mobility-impaired). "
        f"Current time: **{time_str}**.\n\n"
        f"Issuing an alert now triggers evacuation — economic cost and panic risk "
        f"if the prediction is wrong. Waiting **{confirm_hours} hour(s)** for satellite "
        f"confirmation improves accuracy but cuts available evacuation time."
        f"{alarm_fatigue}"
    )

    return Scenario(
        id="emergency_alert",
        title="🌊 Emergency Alert — Accuracy vs. Timeliness",
        description=description,
        options=[
            Option(
                "A",
                "Issue alert immediately",
                f"Act on {confidence}% AI confidence — protect lives, risk unnecessary disruption.",
            ),
            Option(
                "B",
                "Wait for confirmation",
                f"Gain accuracy in {confirm_hours}h but shrink the evacuation window.",
            ),
        ],
        metadata=dict(
            confidence=confidence,
            population=population,
            false_alarms=false_alarms,
            time_str=time_str,
            vulnerable_pct=vulnerable_pct,
            confirm_hours=confirm_hours,
        ),
    )


def generate_medical_ai_scenario() -> Scenario:
    """
    Scenario 7 – Medical AI: Explainability vs. Performance.

    Choose between a transparent-but-less-accurate model and an opaque-but-better one.
    """
    acc_x = round(random.uniform(0.76, 0.87), 2)
    acc_y = round(random.uniform(acc_x + 0.03, min(acc_x + 0.12, 0.99)), 2)
    context = random.choice(
        [
            "high-stakes oncology diagnosis",
            "ICU triage decisions",
            "rare-disease screening",
            "cardiac risk assessment",
        ]
    )
    expert_review = random.choice([True, False])
    legal_explainability = random.choice([True, False])

    review_note = (
        "Expert physician review is **available** to validate each AI prediction. "
        if expert_review
        else "Expert physician review is **not routinely available**. "
    )
    legal_note = (
        "\n\n⚖️ Local regulations **require explainable AI** in clinical settings."
        if legal_explainability
        else ""
    )

    description = (
        f"Two AI diagnostic systems are available for **{context}**:\n\n"
        f"**Model X** — Accuracy: **{acc_x:.0%}**  \n"
        f"Transparent; provides step-by-step explanations that clinicians can challenge "
        f"and audit.\n\n"
        f"**Model Y** — Accuracy: **{acc_y:.0%}**  \n"
        f"Black-box; better performance but limited interpretability.\n\n"
        f"{review_note}{legal_note}"
    )

    return Scenario(
        id="medical_ai",
        title="🏥 Medical AI — Explainability vs. Performance",
        description=description,
        options=[
            Option(
                "A",
                "Use Model X (explainable)",
                f"Lower accuracy ({acc_x:.0%}) but transparent, auditable, clinician-trusted.",
            ),
            Option(
                "B",
                "Use Model Y (opaque)",
                f"Higher accuracy ({acc_y:.0%}) but black-box.",
            ),
        ],
        metadata=dict(
            acc_x=acc_x,
            acc_y=acc_y,
            context=context,
            expert_review=expert_review,
            legal_explainability=legal_explainability,
        ),
    )


def generate_disaster_relief_scenario() -> Scenario:
    """
    Scenario 9 – Disaster Relief: "First Come" vs. "Most Need".

    Limited aid packages after a hurricane; choose the distribution method.
    """
    packages = random.randint(200, 800)
    families = random.randint(packages + 400, packages + 2500)
    corruption_risk = random.choice(["low", "moderate", "high"])
    undoc_pct = random.randint(5, 30)
    error_rate = random.randint(10, 30)

    corruption_note = (
        f"\n\n⚠️ Local oversight is limited — officials warn of a **{corruption_risk} "
        "corruption risk** in needs-scoring if not closely monitored."
        if corruption_risk in ("moderate", "high")
        else ""
    )

    description = (
        f"After a hurricane you have **{packages:,} aid packages** for "
        f"**{families:,} affected families**.\n\n"
        f"Approximately **{undoc_pct}%** of residents are undocumented or unhoused "
        f"and may lack documents required for formal needs-based scoring.\n\n"
        f"Needs-based scoring is more equitable but carries an estimated "
        f"**{error_rate}% error rate** due to incomplete data."
        f"{corruption_note}"
    )

    return Scenario(
        id="disaster_relief",
        title="🏚️ Disaster Relief — First-Come vs. Most-Need",
        description=description,
        options=[
            Option(
                "A",
                "First-come-first-served",
                "Simple and transparent; no documentation required — may miss the most vulnerable.",
            ),
            Option(
                "B",
                "Needs-based scoring",
                f"More equitable but {error_rate}% error rate and potentially gameable.",
            ),
        ],
        metadata=dict(
            packages=packages,
            families=families,
            corruption_risk=corruption_risk,
            undoc_pct=undoc_pct,
            error_rate=error_rate,
        ),
    )


def generate_drone_strike_scenario() -> Scenario:
    """
    Scenario 11 – Autonomous Drone: Strike Under Uncertainty.

    High-confidence target ID with nearby civilian presence; strike now or wait?
    """
    prob_combatant = random.randint(65, 92)
    civilian_count = random.randint(1, 8)
    target_value = random.choice(
        [
            "low-priority intelligence asset",
            "mid-level field commander",
            "high-value strategic target",
        ]
    )
    can_evacuate = random.choice([True, False])
    wait_min = random.randint(5, 30)

    evacuate_note = (
        f"\n\nCivilian evacuation could clear the area in ~{wait_min} min, "
        "but the target may escape during that window."
        if can_evacuate
        else "\n\nCivilian evacuation is **not feasible** within the available time window."
    )

    description = (
        f"A surveillance drone identifies a **{target_value}** with "
        f"**{prob_combatant}% confidence** of being an armed combatant.\n\n"
        f"Estimated **{civilian_count} civilian(s)** are within the blast radius. "
        f"Delaying the strike may allow the target to escape."
        f"{evacuate_note}"
    )

    return Scenario(
        id="drone_strike",
        title="🎯 Autonomous Drone — Strike Under Uncertainty",
        description=description,
        options=[
            Option(
                "A",
                "Strike immediately",
                f"Act on {prob_combatant}% confidence — neutralise target, accept civilian risk.",
            ),
            Option(
                "B",
                "Wait / seek confirmation",
                "Reduce civilian risk at the cost of possibly losing the target.",
            ),
        ],
        metadata=dict(
            prob_combatant=prob_combatant,
            civilian_count=civilian_count,
            target_value=target_value,
            can_evacuate=can_evacuate,
            wait_min=wait_min,
        ),
    )


def generate_elder_care_robot_scenario() -> Scenario:
    """
    Scenario 12 – Elder Care Robot: Autonomy vs. Wellbeing.

    Robot detects medication non-compliance; patient refuses assistance.
    """
    risk = random.choice(["mild", "moderate", "severe"])
    cognitive = random.choice(
        [
            "fully competent",
            "mild cognitive impairment",
            "moderate dementia",
        ]
    )
    prior_consent = random.choice([True, False])
    days_missed = random.randint(1, 7)

    consent_note = (
        "The patient previously signed a form allowing family notification if medication is missed."
        if prior_consent
        else "The patient has **not** signed any consent allowing third-party notification."
    )

    description = (
        f"An elder care robot detects that an elderly resident (**{cognitive}**) "
        f"has skipped their medication for **{days_missed} consecutive day(s)**. "
        f"Health risk level: **{risk}**.\n\n"
        f"When prompted, the resident clearly refuses help and insists on privacy.\n\n"
        f"{consent_note}"
    )

    return Scenario(
        id="elder_care_robot",
        title="🤖 Elder Care Robot — Autonomy vs. Wellbeing",
        description=description,
        options=[
            Option(
                "A",
                "Respect refusal",
                "Honour the patient's autonomy and right to privacy — do nothing further.",
            ),
            Option(
                "B",
                "Notify family / doctor",
                f"Act in the patient's best interest despite refusal ({risk} risk, {days_missed}d missed).",
            ),
            Option(
                "C",
                "Use gentle persuasion",
                "Nudge / remind without overriding consent — a middle-ground approach.",
            ),
        ],
        metadata=dict(
            risk=risk,
            cognitive=cognitive,
            prior_consent=prior_consent,
            days_missed=days_missed,
        ),
    )


# ---------------------------------------------------------------------------
# Registry — append new generators (and a matching name) here; no other
# file needs to change.
# ---------------------------------------------------------------------------

SCENARIO_GENERATORS: List[Callable[[], Scenario]] = [
    generate_ambulance_split_scenario,    # 1
    generate_rare_drug_scenario,          # 3
    generate_emergency_alert_scenario,    # 6
    generate_medical_ai_scenario,         # 7
    generate_disaster_relief_scenario,    # 9
    generate_drone_strike_scenario,       # 11
    generate_elder_care_robot_scenario,   # 12
]

# Human-readable names aligned with SCENARIO_GENERATORS (same index)
SCENARIO_NAMES: List[str] = [
    "🚑 Ambulance Split",
    "💊 Rare Drug Allocation",
    "🌊 Emergency Alert",
    "🏥 Medical AI Diagnosis",
    "🏚️ Disaster Relief",
    "🎯 Autonomous Drone Strike",
    "🤖 Elder Care Robot",
]


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def get_random_scenario() -> Scenario:
    """Return a randomly chosen and freshly randomised scenario."""
    return random.choice(SCENARIO_GENERATORS)()


def get_scenario_by_index(index: int) -> Scenario:
    """Return the scenario at position *index* (wraps around if index >= len)."""
    return SCENARIO_GENERATORS[index % len(SCENARIO_GENERATORS)]()

# Brand-Sentiment-Analysis
This is an academic project in an AI and Data Science master's program to compare rule-based and transformer-based sentiment models for brand perception analysis and evaluate their temporal behaviour.

---

## Moral Machine — Ethics of AI Demo

A standalone interactive demo (inside `moral_machine/`) where **you** make moral decisions and compare your choices against an AI model's independent opinion. Inspired by the MIT Moral Machine experiment.

### Features
- **7 scenario types** covering real AI-ethics dilemmas (triage, rare-drug allocation, emergency alerts, medical AI explainability, disaster relief, autonomous weapons, elder-care robots).
- Each scenario is **randomised** on every run — distances, probabilities, ages, and contextual details change each time.
- **Human + AI choices** are recorded side-by-side for every round.
- **End-of-session statistics**: agreement rate, per-round breakdown table, AI rationale log, and CSV export.
- Works **without an API key** — a rule-based fallback keeps the demo fully runnable offline.

### Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set your OpenAI API key for live AI opinions
export OPENAI_API_KEY="sk-..."          # macOS / Linux
set   OPENAI_API_KEY=sk-...            # Windows CMD

# (Optional) Override model or base URL
export OPENAI_MODEL="gpt-4o-mini"      # default
export OPENAI_BASE_URL="https://..."   # for alternative providers

# 4. Run the Moral Machine app
streamlit run moral_machine/app.py
```

The app opens at **http://localhost:8501** by default.

### Project structure (Moral Machine module)

```
moral_machine/
├── app.py          # Streamlit UI — setup → playing → results phases
├── scenarios.py    # Scenario dataclass + 7 generator functions + registry
└── ai_judge.py     # AI judge: builds prompts, calls API, parses JSON, fallback
```

### How to add a new scenario

1. Open `moral_machine/scenarios.py`.
2. Write a new generator function:
   ```python
   def generate_my_scenario() -> Scenario:
       # randomise variables, build description string, return Scenario(...)
       ...
   ```
3. Append it to `SCENARIO_GENERATORS` and add a display name to `SCENARIO_NAMES`.

No other files need to change.

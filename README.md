# SourceWise — Supplier Evaluation & Sourcing Decision Tool

SourceWise helps a small apparel brand choose a garment manufacturer. It goes beyond "who is cheapest" and asks
**what a high minimum order quantity (MOQ) really costs you if a style doesn't sell.**

## The problem

When I ran my women's fashion brand, **Clozet**, I had to choose between three garment manufacturers. I compared
them by hand on quality, cost, MOQ, lead time and reliability. The hardest call was whether to pay **₹330 a piece at
an MOQ of 50** (Manufacturer B: flexible, high quality) instead of **₹300 at an MOQ of 100** (Manufacturer A). A third
option, Manufacturer C, was cheaper still at an MOQ of 30 but couldn't make the garment to spec. SourceWise turns that
spreadsheet-and-gut-feel process into a reusable, data-driven tool.

## What it does

| Tab | What you get |
|---|---|
| **Suppliers** | Load the sample file or upload your own CSV. The file is validated with clear error messages. You can edit cells in the app, and filter by state, city and specialization. |
| **Scorecard** | A weighted ranking from sidebar sliders (quality, capability, cost, MOQ, lead time, reliability, costing transparency, advance payment). Suppliers that fail the capability requirement are listed separately with the reason. A stacked bar chart shows what drives each score. |
| **MOQ Risk** | Profit and **capital at risk** (money stuck in unsold stock) per supplier under low, expected and high demand, plus probability-weighted expected profit. It flags cases where **the cheaper supplier is riskier**. |
| **Scaling** | Green, Amber or Red: can each supplier's capacity and reorder speed keep up if demand grows? |
| **Recommendation** | The recommended supplier and the key trade-off in plain English, with warnings (capacity, lead time, advance payment, downside loss), why the runner-up lost, and an **Excel summary download**. |

## How it works

### 1. Weighted scorecard (`sourcewise/scoring.py`)

1. **Hard filter.** Suppliers with `meets_required_capability = FALSE` are excluded from ranking.
2. **Min-max normalisation** to 0–1 across the *eligible* suppliers, so 1 is always best:

   ```
   higher is better:  norm = (x − min) / (max − min)       quality, capability, reliability, transparency
   lower is better:   norm = (max − x) / (max − min)       unit cost, MOQ, lead time, advance payment
   ```

   If every supplier has the same value, everyone gets 1.0. The factor then can't separate anyone, and there's no
   divide-by-zero.
3. **Weights** come from the sliders (0–10) and are rescaled to sum to 100%:

   ```
   score = Σ (norm_factor × weight_factor) × 100            → a score out of 100
   ```

Ties go to the lower unit cost. The city, state and specialization columns are descriptive only and never scored.

### 2. MOQ vs unit-cost risk model (`sourcewise/moq_risk.py`)

For each supplier × demand scenario:

```
units_bought    = max(moq_per_style, planned_quantity)
units_sold      = min(units_bought, demand)
revenue         = units_sold × selling_price
cost            = units_bought × unit_cost
profit          = revenue − cost
unsold_units    = units_bought − units_sold
capital_at_risk = unsold_units × unit_cost

expected_profit = Σ profit_scenario × probability_scenario      (probabilities must sum to 100%)
```

A supplier is flagged **cheaper but riskier** when its unit cost is lower than another supplier's, but its capital at
risk in the lowest-demand scenario is higher. The cause is usually a bigger MOQ.

**Worked Clozet example** (selling price ₹899, planned 50 units, demand 30 / 60 / 120 at 30 / 50 / 20%):

| | Buys | Profit (low) | Capital at risk (low) | Expected profit |
|---|---|---|---|---|
| Manufacturer A: ₹300, MOQ 100 | 100 | −₹3,030 | ₹21,000 | ₹23,041 |
| Manufacturer B: ₹330, MOQ 50 | 50 | ₹10,470 | ₹6,600 | ₹23,056 |

B costs ₹30 more per unit but puts **₹14,400 less capital at risk** if demand is low, for about the same expected profit.
That was exactly the trade-off at Clozet. The test suite checks these hand-calculated numbers.

### 3. Scaling check (`sourcewise/capacity.py`)

```
required_units = current_monthly_units × growth_multiplier
capacity_ok    = monthly_capacity × usable_share ≥ required_units
reorder_ok     = reorder_time_days ≤ reorder_deadline_days
status         = Green (both OK) · Amber (one fails) · Red (both fail)
```

### 4. Recommendation (`sourcewise/recommend.py`)

Only eligible suppliers selected on the MOQ Risk tab are considered:

```
risk_score     = 100 × (½ · norm(expected_profit) + ½ · norm_inverted(capital_at_risk_low))
decision_score = 60% × scorecard_score + 40% × risk_score − scaling_penalty   (Amber −10, Red −25)
```

The 60/40 split, the penalties and the warning thresholds (lead time > 30 days, advance ≥ 50%) can all be changed in
**Decision settings**. The key trade-off compares the recommendation with the cheapest supplier compared. If the
recommendation already is the cheapest, it compares with the runner-up.

## Screenshots

> _Placeholder: add screenshots to `docs/screenshots/` and uncomment the lines below._

<!--
![Scorecard](docs/screenshots/scorecard.png)
![MOQ risk](docs/screenshots/moq-risk.png)
![Recommendation](docs/screenshots/recommendation.png)
-->

## Run locally

Requires Python 3.11+.

```bash
git clone <your-repo-url> sourcewise
cd sourcewise
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501 with the sample data loaded.

Run the tests:

```bash
pytest
```

The suite covers validation, normalisation edge cases, hard filters, the hand-calculated risk maths, scaling status
logic, recommendation ranking, the Excel export, and a headless smoke test of the Streamlit app.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository. `app.py` and `requirements.txt` must be at the repository root.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and click **Create app**.
3. Pick the repository and branch, and set **Main file path** to `app.py`.
4. Under **Advanced settings**, choose Python 3.11 or 3.12.
5. Click **Deploy**. Dependencies are installed from `requirements.txt`. The app needs no secrets or database.

Every push to the branch redeploys the app automatically.

## Using your own data

Click **Download CSV template** in the sidebar, then fill in one row per supplier.

| Column | Meaning | Allowed values |
|---|---|---|
| `supplier_name` | Supplier name | text, unique |
| `unit_cost` | Price per piece (₹) | > 0 |
| `moq_per_style` | Minimum order per style (units) | whole number ≥ 1 |
| `sample_quality` | Sample quality | 1–5 |
| `capability_score` | Stitching / finishing / silhouette capability | 1–5 |
| `meets_required_capability` | Hard requirement | TRUE / FALSE (also yes/no, 1/0) |
| `lead_time_days` | Days to first production | whole number ≥ 0 |
| `reorder_time_days` | Days to restock | whole number ≥ 0 |
| `monthly_capacity` | Units per month | whole number ≥ 0 |
| `reliability_score` | Communication, responsiveness, deadlines | 1–5 |
| `costing_transparency` | Openness of costing | 1–5 |
| `advance_payment_pct` | Upfront payment (% of order) | 0–100 |
| `city`, `state`, `specialization` | *Optional* descriptive fields used for filters | text |

Headers are matched case-insensitively. Every problem in a file is reported at once, with spreadsheet row numbers.

## Project structure

```
sourcewise/
├── app.py                    # Streamlit UI (no business logic)
├── sourcewise/
│   ├── models.py             # Supplier dataclass + CSV validation
│   ├── scoring.py            # normalisation, weighted scorecard, hard filters
│   ├── moq_risk.py           # demand-scenario profit / capital-at-risk model
│   ├── capacity.py           # scaling / reorder feasibility check
│   ├── recommend.py          # recommendation, trade-off text, Excel export
│   └── formatting.py         # ₹ formatting with Indian digit grouping (₹1,02,300)
├── data/sample_suppliers.csv
├── tests/                    # pytest suite, one file per module + app smoke test
├── requirements.txt
└── README.md
```

## Limitations

- The risk model looks at a single order per style. It ignores reorders, markdowns and salvage value of unsold stock,
  so capital at risk is a worst case.
- Scores are relative to the suppliers being compared: adding or filtering suppliers changes the 0–1 scale.
- Ratings (quality, reliability and so on) are your own judgement, and the output is only as good as those inputs.

## About the sample data

`data/sample_suppliers.csv` is **synthetic**. It contains 30 fictional Indian clothing manufacturers modelled on real
apparel clusters: Surat, Ahmedabad, Jaipur, Tiruppur, Ludhiana, Delhi NCR (Okhla), Bengaluru, Mumbai and Kolkata.
Manufacturers A, B and C are anonymised versions of the three suppliers from my Clozet experience. No row describes a
real company.

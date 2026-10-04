"""SourceWise — supplier evaluation and sourcing decision tool.

The package holds all business logic; ``app.py`` is a thin Streamlit UI on top.

Modules
-------
models      Supplier dataclass, CSV loading and validation.
scoring     Min-max normalisation, weighted scorecard and hard filters.
moq_risk    Demand-scenario profit and capital-at-risk model (MOQ vs unit cost).
capacity    Scaling / reorder feasibility check (Green / Amber / Red).
recommend   Final recommendation, trade-off explanation and Excel export.
formatting  Small helpers for Indian-rupee formatting.
"""

__version__ = "1.0.0"

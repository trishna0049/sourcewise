"""Supplier data model, CSV loading and validation.

Every analysis in SourceWise starts from a validated supplier table with one
row per supplier. :func:`validate_suppliers` checks required columns, types and
ranges and collects *all* problems into one :class:`SupplierDataError`, so a
manager uploading a spreadsheet sees every issue at once instead of fixing them
one by one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any

import pandas as pd

@dataclass(frozen=True)
class ColumnSpec:
    """Validation rule for one required column.

    Attributes:
        kind: One of ``"str"``, ``"int"``, ``"float"``, ``"bool"``.
        label: Human-readable name used in error messages.
        minimum: Lowest allowed value (numeric columns), or None.
        maximum: Highest allowed value (numeric columns), or None.
        exclusive_min: If True the value must be strictly greater than ``minimum``.
    """

    kind: str
    label: str
    minimum: float | None = None
    maximum: float | None = None
    exclusive_min: bool = False


REQUIRED_COLUMNS: dict[str, ColumnSpec] = {
    "supplier_name": ColumnSpec("str", "Supplier name"),
    "unit_cost": ColumnSpec("float", "Unit cost (₹)", 0, exclusive_min=True),
    "moq_per_style": ColumnSpec("int", "Minimum order quantity per style (units)", 1),
    "sample_quality": ColumnSpec("float", "Sample quality (1–5)", 1, 5),
    "capability_score": ColumnSpec("float", "Capability score (1–5)", 1, 5),
    "meets_required_capability": ColumnSpec("bool", "Meets required capability (TRUE/FALSE)"),
    "lead_time_days": ColumnSpec("int", "Lead time for first production (days)", 0),
    "reorder_time_days": ColumnSpec("int", "Reorder time (days)", 0),
    "monthly_capacity": ColumnSpec("int", "Monthly capacity (units)", 0),
    "reliability_score": ColumnSpec("float", "Reliability score (1–5)", 1, 5),
    "costing_transparency": ColumnSpec("float", "Costing transparency (1–5)", 1, 5),
    "advance_payment_pct": ColumnSpec("float", "Advance payment (% of order value)", 0, 100),
}

#: Descriptive columns shown in tables and filters but never used in scoring.
OPTIONAL_COLUMNS: tuple[str, ...] = ("city", "state", "specialization")

_TRUE_VALUES = {"true", "t", "yes", "y", "1", "1.0"}
_FALSE_VALUES = {"false", "f", "no", "n", "0", "0.0"}

#: Friendly column names for tables and exports.
DISPLAY_LABELS: dict[str, str] = {
    "supplier_name": "Supplier",
    "city": "City",
    "state": "State",
    "specialization": "Specialization",
    "unit_cost": "Unit cost (₹)",
    "moq_per_style": "MOQ / style",
    "sample_quality": "Sample quality",
    "capability_score": "Capability",
    "meets_required_capability": "Meets capability",
    "lead_time_days": "Lead time (days)",
    "reorder_time_days": "Reorder time (days)",
    "monthly_capacity": "Monthly capacity",
    "reliability_score": "Reliability",
    "costing_transparency": "Costing transparency",
    "advance_payment_pct": "Advance (%)",
}

#: Maximum number of example rows listed per error message.
_MAX_ROWS_IN_MESSAGE = 5


class SupplierDataError(ValueError):
    """Raised when supplier data fails validation.

    Attributes:
        errors: Human-readable messages, one per problem found.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("Supplier data is invalid:\n- " + "\n- ".join(errors))


@dataclass(frozen=True)
class Supplier:
    """One garment manufacturer and the attributes used to evaluate it."""

    supplier_name: str
    unit_cost: float
    moq_per_style: int
    sample_quality: float
    capability_score: float
    meets_required_capability: bool
    lead_time_days: int
    reorder_time_days: int
    monthly_capacity: int
    reliability_score: float
    costing_transparency: float
    advance_payment_pct: float
    city: str | None = None
    state: str | None = None
    specialization: str | None = None

    @classmethod
    def from_row(cls, row: pd.Series | dict[str, Any]) -> "Supplier":
        """Build a supplier from a row of a *validated* supplier DataFrame."""
        data = dict(row)
        kwargs: dict[str, Any] = {}
        for column, spec in REQUIRED_COLUMNS.items():
            kind, value = spec.kind, data[column]
            if kind == "int":
                value = int(value)
            elif kind == "float":
                value = float(value)
            elif kind == "bool":
                value = bool(value)
            else:
                value = str(value)
            kwargs[column] = value
        for column in OPTIONAL_COLUMNS:
            value = data.get(column)
            kwargs[column] = None if value is None or pd.isna(value) else str(value)
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Return the supplier as a plain dictionary."""
        return asdict(self)


def _clean_header(name: object) -> str:
    """Normalise a column header: trim, lower-case, spaces/dashes to underscores."""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _row_list(mask: pd.Series) -> str:
    """Describe failing rows using spreadsheet row numbers (the header is row 1)."""
    failing = mask[mask].index
    shown = [str(int(i) + 2) for i in failing[:_MAX_ROWS_IN_MESSAGE]]
    text = ("row " if len(failing) == 1 else "rows ") + ", ".join(shown)
    extra = len(failing) - len(shown)
    return text + (f" and {extra} more" if extra else "")


def _parse_bool(value: object) -> bool | None:
    """Parse common spreadsheet boolean spellings; return None if unrecognised."""
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return None


def validate_suppliers(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a supplier table and return a cleaned, correctly-typed copy.

    Checks performed:
        * all required columns are present (headers are matched case-insensitively);
        * at least one supplier row exists;
        * supplier names are non-blank and unique;
        * numeric columns are numbers within their allowed range, whole numbers
          where counts are expected (MOQ, days, capacity);
        * ``meets_required_capability`` is a recognisable TRUE/FALSE value.

    Optional descriptive columns (city, state, specialization) are kept when
    present and ignored when missing. Unknown extra columns are kept as-is.

    Args:
        df: Raw supplier data, e.g. straight from :func:`pandas.read_csv`.

    Returns:
        A new DataFrame with a fresh 0..n-1 index and typed columns.

    Raises:
        SupplierDataError: listing every problem found.
    """
    data = df.copy()
    data.columns = [_clean_header(c) for c in data.columns]
    # Rows that are entirely blank (common at the end of spreadsheets) are dropped.
    data = data.dropna(how="all").reset_index(drop=True)

    errors: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        errors.append("Missing required column(s): " + ", ".join(missing) + ".")
    duplicated_headers = sorted({c for c in data.columns if list(data.columns).count(c) > 1})
    if duplicated_headers:
        errors.append("Duplicate column header(s): " + ", ".join(duplicated_headers) + ".")
        raise SupplierDataError(errors)
    if data.empty:
        raise SupplierDataError(errors + ["The file has no supplier rows."])

    # Keep checking the columns that are present so every problem is reported at once.
    for column, spec in REQUIRED_COLUMNS.items():
        if column in missing:
            continue
        kind, label, low, high = spec.kind, spec.label, spec.minimum, spec.maximum
        raw = data[column]
        if kind == "str":
            text = raw.astype("string").str.strip()
            blank = text.isna() | (text == "")
            if blank.any():
                errors.append(f"{label}: blank in {_row_list(blank)}.")
            dupes = text.duplicated(keep=False) & ~blank
            if dupes.any():
                names = ", ".join(sorted(set(text[dupes].dropna())))
                errors.append(f"{label}: duplicate names ({names}) — each supplier must be unique.")
            data[column] = text.astype(object)
            continue

        if kind == "bool":
            parsed = raw.map(_parse_bool)
            bad = parsed.isna()
            if bad.any():
                errors.append(f"{label}: use TRUE or FALSE (problem in {_row_list(bad)}).")
            data[column] = parsed.astype(object)
            continue

        numeric = pd.to_numeric(raw, errors="coerce")
        not_number = numeric.isna()
        if not_number.any():
            errors.append(f"{label}: must be a number (problem in {_row_list(not_number)}).")
        valid = numeric[~not_number]
        if kind == "int":
            fractional = (valid % 1 != 0).reindex(numeric.index, fill_value=False)
            if fractional.any():
                errors.append(f"{label}: must be a whole number (problem in {_row_list(fractional)}).")
        if low is not None:
            below = valid <= low if spec.exclusive_min else valid < low
            too_low = below.reindex(numeric.index, fill_value=False)
            if too_low.any():
                bound = f"greater than {low:g}" if spec.exclusive_min else f"at least {low:g}"
                errors.append(f"{label}: must be {bound} (problem in {_row_list(too_low)}).")
        if high is not None:
            too_high = (valid > high).reindex(numeric.index, fill_value=False)
            if too_high.any():
                errors.append(f"{label}: must be at most {high:g} (problem in {_row_list(too_high)}).")
        data[column] = numeric

    if errors:
        raise SupplierDataError(errors)

    for column, spec in REQUIRED_COLUMNS.items():
        if spec.kind == "int":
            data[column] = data[column].astype("int64")
        elif spec.kind == "float":
            data[column] = data[column].astype("float64")
        elif spec.kind == "bool":
            data[column] = data[column].astype(bool)
    for column in OPTIONAL_COLUMNS:
        if column in data.columns:
            data[column] = data[column].astype("string").str.strip().astype(object)

    ordered = ["supplier_name"] + [c for c in OPTIONAL_COLUMNS if c in data.columns]
    ordered += [c for c in REQUIRED_COLUMNS if c != "supplier_name"]
    ordered += [c for c in data.columns if c not in ordered]
    return data[ordered].reset_index(drop=True)


def load_suppliers(source: str | Path | IO[Any]) -> pd.DataFrame:
    """Read a supplier CSV from a path or file-like object and validate it.

    Raises:
        SupplierDataError: if the file cannot be parsed or fails validation.
    """
    try:
        raw = pd.read_csv(source, skipinitialspace=True)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise SupplierDataError([f"Could not read the CSV file: {exc}"]) from exc
    return validate_suppliers(raw)


def to_suppliers(df: pd.DataFrame) -> list[Supplier]:
    """Convert a validated supplier DataFrame into :class:`Supplier` objects."""
    return [Supplier.from_row(row) for _, row in df.iterrows()]

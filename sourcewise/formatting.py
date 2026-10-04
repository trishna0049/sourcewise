"""Formatting helpers for rupee amounts and unit counts.

Indian number grouping is used (``12,34,567``): the last three digits form one
group and every earlier group has two digits.
"""

from __future__ import annotations

import math

RUPEE = "₹"


def indian_group(value: float, decimals: int = 0) -> str:
    """Format a number with Indian digit grouping, e.g. ``1234567 -> '12,34,567'``.

    Args:
        value: The number to format. NaN / None returns an em dash.
        decimals: Digits after the decimal point.

    Returns:
        The grouped number as a string, with a leading ``-`` for negatives.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    sign = "-" if value < 0 else ""
    rounded = f"{abs(value):.{decimals}f}"
    whole, _, frac = rounded.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return f"{sign}{whole}" + (f".{frac}" if frac else "")


def inr(value: float, decimals: int = 0) -> str:
    """Format a rupee amount, e.g. ``-21000 -> '-₹21,000'``."""
    text = indian_group(value, decimals)
    if text == "—":
        return text
    if text.startswith("-"):
        return f"-{RUPEE}{text[1:]}"
    return f"{RUPEE}{text}"


def units(value: float) -> str:
    """Format a unit count with Indian grouping, e.g. ``3200 -> '3,200 units'``."""
    return f"{indian_group(value)} units"

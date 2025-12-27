from __future__ import annotations

from datetime import date
from typing import List, Sequence


def select_training_weeks(
    available_weeks: Sequence[date],
    current_week: date,
    train_window_years: float,
    min_train_weeks: int,
) -> List[date]:
    if train_window_years <= 0:
        raise ValueError("train_window_years must be positive")
    if min_train_weeks <= 0:
        raise ValueError("min_train_weeks must be positive")

    if not available_weeks:
        return []

    weeks_sorted = sorted(available_weeks)
    eligible = [week for week in weeks_sorted if week < current_week]
    if not eligible:
        return []

    max_weeks = max(int(train_window_years * 52), 1)
    selected = eligible[-max_weeks:]

    if len(selected) < min_train_weeks:
        return []

    return selected

from __future__ import annotations

from datetime import date, timedelta

from mlstock.model.train import select_training_weeks


def _make_weeks(start: date, count: int) -> list[date]:
    return [start + timedelta(weeks=offset) for offset in range(count)]


def test_select_training_weeks_limits_window() -> None:
    weeks = _make_weeks(date(2020, 1, 3), 60)
    current = weeks[-1]

    selected = select_training_weeks(
        weeks,
        current,
        train_window_years=1,
        min_train_weeks=10,
    )

    assert current not in selected
    assert len(selected) == 52
    assert selected[0] == weeks[-53]
    assert selected[-1] == weeks[-2]


def test_select_training_weeks_minimum() -> None:
    weeks = _make_weeks(date(2024, 1, 5), 10)
    current = weeks[-1]

    selected = select_training_weeks(
        weeks,
        current,
        train_window_years=1,
        min_train_weeks=20,
    )

    assert selected == []


def test_select_training_weeks_sorts_input() -> None:
    weeks = [date(2024, 1, 19), date(2024, 1, 5), date(2024, 1, 12)]
    current = date(2024, 1, 26)

    selected = select_training_weeks(
        weeks,
        current,
        train_window_years=1,
        min_train_weeks=1,
    )

    assert selected == sorted(weeks)

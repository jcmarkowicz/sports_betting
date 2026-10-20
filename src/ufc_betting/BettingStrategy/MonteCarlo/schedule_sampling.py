"""Shuffle intact fight cards across a monthly UFC event schedule."""

import numpy as np
import pandas as pd
from .event_types import event_types


def sample_schedule(
    fights,
    *,
    date_col="date",
    title_col="title_fight",
    event_name_col="event_name",
    event_type_col="event_type",
    random_state=None,
    replace=False,
):
    """Shuffle whole events and balance numbered UFC cards across months.
    Without replacement every event is used exactly once. Aim for one numbered
    card per occupied month; with a surplus/shortage, spread them as evenly as
    date-slot capacity permits. A title fight never determines event type.
    With replace=True, draw whole cards uniformly and independently for each
    date slot, retaining the legacy unconstrained event-bootstrap behavior.
    title_col is accepted for compatibility but is not used.
    """
    rng = np.random.default_rng(random_state)
    data = fights.copy().reset_index(drop=True)
    if data.empty:raise ValueError('Cannot sample an empty schedule')
    data[date_col] = pd.to_datetime(data[date_col])
    if data[date_col].isna().any():
        raise ValueError('Event dates must be nonmissing')
    events = [card for _, card in data.groupby(date_col, sort=True)]
    dates = pd.DatetimeIndex([card[date_col].iloc[0] for card in events])
    months = dates.to_period("M")
    full_months = months.unique()
    
    if replace:
        assignment = rng.choice(len(events), size=len(events), replace=True)
    else:
        types = event_types(data, event_name_col=event_name_col, event_type_col=event_type_col)
        typed = pd.DataFrame({'date': data[date_col], 'type': types})
        if typed.groupby('date')['type'].nunique().gt(1).any():
            raise ValueError('Each event date must identify one event type')
        if event_name_col in data and data.groupby(date_col)[event_name_col].nunique().gt(1).any():
            raise ValueError('Multiple events on one date require separate schedule slots')
        numbered_events = np.flatnonzero(typed.groupby('date', sort=True)['type'].first().eq('numbered'))
        assignment = np.full(len(events), -1, dtype=int)
        counts = np.zeros(len(full_months), dtype=int)
        capacity = np.array([(months == month).sum() for month in full_months])
        for source_event_id in rng.permutation(numbered_events):
            available = np.flatnonzero(counts < capacity)
            least = available[counts[available] == counts[available].min()]
            month_id = rng.choice(least)
            slot = rng.choice(np.flatnonzero((months == full_months[month_id]) & (assignment == -1)))
            assignment[slot] = source_event_id
            counts[month_id] += 1
        remaining_events = np.setdiff1d(np.arange(len(events)), numbered_events)
        assignment[assignment == -1] = rng.permutation(remaining_events)

    # Move each complete card to its assigned date, preserving fight order.
    sampled_cards = []
    for event_id, source_event_id in enumerate(assignment):
        card = events[source_event_id].copy()
        card["source_date"] = card[date_col]
        card["source_row_position"] = card.index
        card["sampled_event_id"] = event_id
        card[date_col] = dates[event_id]
        sampled_cards.append(card)
    return pd.concat(sampled_cards, ignore_index=True)

"""Deterministic chronological order for stateful pre-fight features."""
import pandas as pd


def chronological_fights(frame):
    result = frame.copy()
    result['date'] = pd.to_datetime(result['date'])
    # UFCStats lists the main event first. Reverse card positions so early
    # tournament bouts precede later bouts involving the same fighter.
    columns, ascending = ['date'], [True]
    if 'bout_position' in result:
        result['bout_position'] = pd.to_numeric(result['bout_position'], errors='raise')
        columns.append('bout_position')
        ascending.append(False)
    for name in ('fighter_red', 'fighter_blue'):
        columns.append(name)
        ascending.append(True)
    return result.sort_values(columns, ascending=ascending, kind='stable').reset_index(drop=True)

"""Event families used by schedule and position-stratified fight sampling."""

import re

import pandas as pd


def event_types(frame, *, event_name_col='event_name', event_type_col='event_type'):
    """Use explicit types, or infer families from UFC event names.

    Monthly cards (numbered UFC, Noche UFC, UFC Freedom 250) use 'numbered';
    Fight Nights use 'fight_night'. Other
    named UFC series remain separate (on ESPN, on ABC, on FOX, TUF Finale,
    etc.). Unrecognized names require an explicit event_type instead of
    silently mixing all unknown events into one donor population.
    """
    if event_type_col in frame:
        values = frame[event_type_col].astype('string').str.strip().str.lower()
        values = values.str.replace(r'[\s-]+', '_', regex=True)
        if values.isna().any() or values.eq('').any():
            raise ValueError(f'{event_type_col} must be nonempty for every fight')
        # Normalize the former Noche family on previously classified inputs.
        return values.replace({'noche_ufc': 'numbered', 'ufc_noche': 'numbered'})
    if event_name_col not in frame:
        raise ValueError(f'Supply {event_name_col} or {event_type_col}; title_fight is not an event type')

    def classify(value):
        name = str(value).strip()
        if re.match(r'^UFC\s+\d+\b', name, re.I):
            return 'numbered'
        if re.match(r'^UFC\s+Fight Night\b', name, re.I):
            return 'fight_night'
        if re.match(r'^(?:Noche\s+UFC|UFC\s+Noche|UFC\s+Freedom\s+250)\b', name, re.I):
            return 'numbered'
        match = re.match(r'^UFC\s+on\s+(ESPN\+?|ABC|FOX|FX|Fuel TV|Versus)\b', name, re.I)
        if match:
            return 'ufc_on_' + re.sub(r'\s+', '_', match[1].lower()).replace('+', '_plus')
        if re.match(r'^(?:The )?Ultimate Fighter\b', name, re.I) and 'finale' in name.lower():
            return 'tuf_finale'
        raise ValueError(f'Unrecognized event name {value!r}; supply explicit {event_type_col}')

    return frame[event_name_col].map(classify).astype('string')

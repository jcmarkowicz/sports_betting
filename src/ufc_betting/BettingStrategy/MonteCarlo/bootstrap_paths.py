"""Persist sampled event indices with validation-dataset provenance."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from ufc_betting.config import settings
from .event_types import event_types


def new_bootstrap_directory():
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
    return settings.data_dir / 'bootstrapped_events' / name


def dataset_metadata(frame, odds_type, replace):
    """Fingerprint the actual ordered inputs used for settlement, excluding features."""
    columns = ['date', 'fighter_red', 'fighter_blue', 'winner', 'proba_red',
               'risk_red_proba', 'prediction_se']
    columns += [f'{prefix}_{odds_type}_{side}'
                for prefix in ('dec', 'dec_fair') for side in ('blue', 'red')]
    if 'title_fight' in frame:
        columns.append('title_fight')
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f'Saving/loading paths requires columns: {sorted(missing)}')
    data = frame[columns].copy().reset_index(drop=True)
    if data[['fighter_red', 'fighter_blue']].isna().any().any():
        raise ValueError('Saved paths require nonmissing fighter identities')
    if data.duplicated(['date', 'fighter_red', 'fighter_blue']).any():
        raise ValueError('Saved paths require unique fights within each event')
    data['date'] = pd.to_datetime(data.date).dt.strftime('%Y-%m-%d')
    for col in columns:
        if col not in ('date', 'fighter_red', 'fighter_blue'):
            data[col] = data[col].astype(float)
    if not replace:
        data['event_type'] = event_types(frame).to_numpy()
        columns.append('event_type')
    payload = data.to_json(orient='split', index=False, double_precision=15)
    counts = data.groupby('date', sort=True).size()
    return dict(
        schema_version=1 if replace else 2, odds_type=odds_type, replace=bool(replace),
        dataset_sha256=hashlib.sha256(payload.encode()).hexdigest(),
        fingerprint_columns=columns, n_fights=len(data), n_events=len(counts),
        event_dates=counts.index.tolist(), fights_per_event=counts.tolist(),
        fights=data[['date', 'fighter_red', 'fighter_blue']].to_dict('records'),
    )


def save_bootstrap_paths(directory, indices, metadata, seed):
    """Save a new path set; never silently overwrite an existing directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / 'paths.npy'
    np.save(path, indices, allow_pickle=False)
    metadata = dict(metadata, n_samples=len(indices), seed=None if seed is None else int(seed),
                    created_at=datetime.now(timezone.utc).isoformat(),
                    paths_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def load_bootstrap_paths(directory, expected, k):
    """Load the first k saved samples only when the dataset and sampling mode match."""
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('Saved bootstrap paths do not match the validation dataset or sampling mode')
    path = directory / 'paths.npy'
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get('paths_sha256'):
        raise ValueError('Saved bootstrap paths checksum mismatch')
    indices = np.load(path, allow_pickle=False)
    n_events = expected['n_events']
    if (indices.ndim != 2 or indices.shape != (metadata.get('n_samples'), n_events)
            or not np.issubdtype(indices.dtype, np.integer)
            or (indices < 0).any() or (indices >= n_events).any()):
        raise ValueError('Invalid saved bootstrap event indices')
    if not expected['replace'] and not np.all(np.sort(indices, axis=1) == np.arange(n_events)):
        raise ValueError('Saved schedules contain repeated events without replacement')
    if k > len(indices):
        raise ValueError(f'Saved set has {len(indices)} samples; requested {k}')
    return indices[:k].copy(), metadata

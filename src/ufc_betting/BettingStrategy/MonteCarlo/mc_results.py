"""Save Monte Carlo profits and bankroll paths independently of plotting."""
from pathlib import Path

import numpy as np


def save_simulation_paths(results, output_dir):
    """Save {label: simulate_result} as label_paths.npz; return label -> path.

    Each archive contains moneyline_profit, parlay_profit, and bankroll arrays.
    Optional simulation details are omitted. Existing files are overwritten.
    """
    output_dir = Path(output_dir)
    for label in results:
        if not isinstance(label, str) or not label or '/' in label or '\\' in label:
            raise ValueError('Result labels must be nonempty names without path separators')
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for label, result in results.items():
        moneylines, parlays, bankroll = result[:3]
        path = output_dir / f'{label}_paths.npz'
        np.savez_compressed(path, moneyline_profit=np.asarray(moneylines),
                            parlay_profit=np.asarray(parlays), bankroll=np.asarray(bankroll))
        saved[label] = path
    return saved

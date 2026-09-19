import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from ufc_betting.Models.RiskManagement.odds_classification import (
    ODDS_TYPES, evaluate_returns, load_split,
)
from ufc_betting.config import config


class OddsClassificationTests(unittest.TestCase):
    def test_returns_include_empty_cells_and_exclude_unsettled(self):
        rows = pd.DataFrame({
            "actual_odds_type": ["close1"] * 3,
            "predicted_odds_type": ["open"] * 3,
            "stake": [1.0, 1.0, 0.0],
            "net_profit": [2.0, -1.0, np.nan],
        })
        result = evaluate_returns(rows).set_index(
            ["actual_odds_type", "predicted_odds_type"])
        self.assertEqual(len(result), 9)
        cell = result.loc[("close1", "open")]
        self.assertEqual(cell.percent_return, 50)
        self.assertEqual(cell.n_bets, 2)
        self.assertEqual(cell.n_predictions, 3)
        self.assertTrue(np.isnan(result.loc[("open", "open"), "percent_return"]))

    def test_stacking_uses_source_odds_and_selected_fighter(self):
        with TemporaryDirectory() as directory:
            for i, odds_type in enumerate(ODDS_TYPES):
                frame = pd.DataFrame({name: [0.1, 0.2] for name in
                                      getattr(config, f"{odds_type}_feats")})
                frame["proba_fair_open_diff"] = [0.3, 0.4]
                frame["pred_winner"] = [1, 0]
                frame["winner"] = [1, 1]
                frame["date"] = "2020-01-01"
                frame["fighter_red"] = "red"
                frame["fighter_blue"] = "blue"
                frame[f"dec_{odds_type}_red"] = 2.0 + i
                frame[f"dec_{odds_type}_blue"] = 5.0 + i
                frame.to_csv(Path(directory) / f"train_logit_{odds_type}.csv", index=False)
            X, y, meta = load_split(directory, "train")
        self.assertEqual(y.tolist(), [0, 0, 1, 1, 2, 2])
        self.assertFalse(X.isna().any().any())
        self.assertNotIn("proba_fair_close1_diff", X)
        self.assertNotIn("winner", X)
        self.assertNotIn("pred_winner", X)
        self.assertEqual(meta.choice_fighter.tolist(), ["red", "blue"] * 3)
        self.assertEqual(meta.net_profit.tolist(), [1, -1, 2, -1, 3, -1])
        self.assertEqual(meta.choice_decimal_odds.tolist(), [2, 5, 3, 6, 4, 7])


if __name__ == "__main__":
    unittest.main()

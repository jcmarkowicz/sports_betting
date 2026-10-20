import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import test_joint_kelly_decay_search as fixtures
from Notebooks.Search.run_search import load_search_paths, run_search
from ufc_betting.BettingStrategy.MonteCarlo.ordered_sampling import generate_ordered_paths


class SearchPathLoadingTests(unittest.TestCase):
    def frame(self):
        return fixtures.JointKellyDecayTests().frame().assign(
            bout_position=[1, 2] * 3, event_name="UFC 300")

    def test_compact_selection_and_end_to_end_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions.csv"
            self.frame().to_csv(predictions, index=False)
            saved = generate_ordered_paths(self.frame(), n_paths=4, odds_type=None,
                output_path=root / "paths", storage="compact", progress_every=0)
            saved.close()
            chosen = load_search_paths(root / "paths", predictions, search_market="open", n_paths=2)
            again = load_search_paths(root / "paths", predictions, search_market="open", n_paths=2)
            for a, b in zip(chosen, again):
                pd.testing.assert_frame_equal(a, b)
            result = run_search("open", n_trials=1, n_paths=2, paths_path=root / "paths",
                                predictions_path=predictions, params_dir=root / "params",
                                score_type="event_log_growth", ulcer_penalty=.02)
            self.assertEqual(result["ulcer_penalty"], .02)
            np.testing.assert_allclose(result["sampled_path_results"]["ulcer_index"],
                                       result["path_ulcer_indices"])
            saved_result = pd.read_pickle(root / "params/search_results/open_search_results.pkl")
            self.assertEqual(saved_result["ulcer_penalty"], .02)
            self.assertAlmostEqual(result["best_value"],
                result["path_mean_event_log_growth"].mean() - .02 * result["path_ulcer_indices"].mean())
            self.assertEqual(result["path_selection"]["n_paths"], 2)
            self.assertTrue((root / "params/search_results/open_search_results.pkl").is_file())
            # Windows permits renaming only after the reader has released its file handles.
            file = root / "paths/donors.npy"
            file.rename(root / "paths/renamed.npy")

    def test_sampled_prices_infer_and_reject_open(self):
        frame = self.frame().assign(sampled_event_id=np.repeat([0, 1, 2], 2))
        frame.attrs["generation"] = {"sample_prices": True}
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "legacy.pkl"
            pd.to_pickle([frame, frame], file)
            with patch("Notebooks.Search.run_search.predict_intermediate_paths", return_value=[frame]) as infer:
                actual = load_search_paths(file, "unused.csv", search_market="close1", n_paths=2)
                self.assertEqual(len(actual), 1)
                self.assertEqual(infer.call_args.kwargs["search_market"], "close1")
            with self.assertRaisesRegex(ValueError, "Sampled-price paths require"):
                load_search_paths(file, "unused.csv", search_market="open", n_paths=2)

    def test_missing_and_incomplete_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "output_path"):
                load_search_paths(Path(tmp) / "missing", "unused", search_market="open")
            with self.assertRaisesRegex(ValueError, "metadata.json"):
                load_search_paths(tmp, "unused", search_market="open")


if __name__ == "__main__":
    unittest.main()

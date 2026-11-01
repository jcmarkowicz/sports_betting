import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock
import pandas as pd


class ComparisonHelperTests(unittest.TestCase):
    def test_routes_selected_predictions_saves_both_csvs_and_calls_all_plots(self):
        module = Path(__file__).resolve().parents[1] / 'Notebooks/Bets/plot_comparison_results.py'
        import ast
        tree = ast.parse(module.read_text(encoding='utf-8'))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        moneylines = pd.DataFrame({'stake': [.1], 'profit': [1.]})
        parlays = pd.DataFrame({'ticket': [1], 'profit': [-1.]})
        simulator = Mock(return_value=(moneylines, parlays))
        distribution, accuracy = Mock(), Mock()
        env = dict(simulate_and_plot=simulator,
                   plot_bankroll_distributions=Mock(return_value=(distribution, None)),
                   plot_event_accuracy_distributions=Mock(return_value=(accuracy, None, pd.DataFrame())))
        exec(compile(tree, str(module), 'exec'), env)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            env['settings'] = SimpleNamespace(data_dir=root)
            prediction = root/'predictions/trainval/open/inference/test_with_close1.csv'
            prediction.parent.mkdir(parents=True)
            pd.DataFrame({'probability_model':['inference']}).to_csv(prediction,index=False)
            a, b = env['plot_comparison_results']('trainval','open','inference','test','close1',
                predictions_root=root/'predictions', output_dir=root/'outputs')
            self.assertIs(a,moneylines)
            self.assertIs(b,parlays)
            self.assertEqual(simulator.call_args.kwargs['predictions_path'],prediction.resolve())
            self.assertTrue(simulator.call_args.kwargs['use_decay'])
            tag='trainval_test_open_with_close1_inference_decay'
            pd.testing.assert_frame_equal(pd.read_csv(root/'outputs'/f'{tag}_moneylines.csv'),moneylines)
            pd.testing.assert_frame_equal(pd.read_csv(root/'outputs'/f'{tag}_parlays.csv'),parlays)
            distribution.savefig.assert_called_once()
            accuracy.savefig.assert_called_once()
            with self.assertRaisesRegex(ValueError,'supports splits'):
                env['plot_comparison_results']('trainval','open','inference','val','close1')
            with self.assertRaisesRegex(ValueError,'own odds'):
                env['plot_comparison_results']('train','close1','inference','val','close2')
            self.assertEqual(simulator.call_count,1)
            params_dir = root/'kelly_params/inference_market_models'
            params_dir.mkdir(parents=True)
            (params_dir/'close1.json').write_text('{}')
            env['plot_comparison_results']('trainval','open','inference','test','close1',
                predictions_root=root/'predictions', params_dir=params_dir)
            self.assertEqual(simulator.call_args.kwargs['params_dir'], params_dir.resolve())
            self.assertTrue((root/'backtest_results/model_comparisons/inference_market_models'/
                             f'{tag}_moneylines.csv').is_file())
            with self.assertRaisesRegex(FileNotFoundError, 'Missing betting parameters'):
                env['plot_comparison_results']('trainval','open','inference','test','close1',
                    predictions_root=root/'predictions', params_dir=root/'missing')


if __name__ == '__main__':
    unittest.main()

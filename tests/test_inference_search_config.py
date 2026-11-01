import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from Notebooks.Search.run_search import load_search_paths
from Notebooks.Search.run_inference_search import run_inference_search
from ufc_betting.config import VAL_START_DATE


class InferenceSearchTests(unittest.TestCase):
    def test_market_specific_validation_sources_and_sequential_opening_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for market in ('open','close1','close2'):
                folder=root/'train'/market
                (folder/'inference').mkdir(parents=True)
                pd.DataFrame(dict(date=[str(VAL_START_DATE)],probability_model=['inference'])).to_csv(
                    folder/'inference'/f'val_with_{market}.csv',index=False)
                for name in ('model_se.pkl','scaler.pkl','encoder.pkl'):
                    (folder/name).touch()
            paths=root/'paths';paths.mkdir()
            with patch('Notebooks.Search.run_inference_search.run_search',return_value={}) as search:
                run_inference_search(comparison_dir=root, paths_path=paths,
                                     params_dir=root/'params',n_trials=2,n_paths=3,
                                     score_kwargs={'drawdown_penalty': .12, 'ulcer_penalty': .03})
            self.assertEqual([c.kwargs['search_market'] for c in search.call_args_list],['open','close1','close2'])
            for call in search.call_args_list:
                kw=call.kwargs;market=kw['search_market']
                self.assertEqual(kw['prediction_model_type'],market)
                self.assertEqual(kw['probability_model'],'inference')
                self.assertEqual(kw['score_type'], 'event_log_growth')
                self.assertEqual(kw['drawdown_penalty'], .12)
                self.assertEqual(kw['ulcer_penalty'], .03)
                self.assertEqual(kw['predictions_path'],root/'train'/market/'inference'/f'val_with_{market}.csv')
                self.assertEqual(kw['opening_params_path'],root/'params/open.json')

    def test_sampled_prices_use_inference_artifact_not_regularized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            frame=pd.DataFrame(dict(date=['2025-01-01'],bout_position=[1],sampled_event_id=[0]))
            frame.attrs['generation']={'sample_prices':True}
            path=root/'paths.pkl';pd.to_pickle([frame],path)
            ref=root/'predictions.csv'
            pd.DataFrame({'probability_model':['inference']}).to_csv(ref,index=False)
            for name in ('model_se.pkl','scaler.pkl','encoder.pkl'):(root/name).touch()
            with patch('Notebooks.Search.run_search.predict_intermediate_paths',return_value=[frame]) as infer:
                load_search_paths(path,ref,search_market='close2',prediction_model_type='close2',
                                  probability_model='inference',prediction_artifacts_dir=root,n_paths=1)
            self.assertEqual(infer.call_args.kwargs['prediction_model'],root/'model_se.pkl')
            self.assertEqual(infer.call_args.kwargs['prediction_se_model'],root/'model_se.pkl')
            pd.DataFrame({'probability_model':['regularized']}).to_csv(ref,index=False)
            with self.assertRaisesRegex(ValueError,'probability_model'):
                load_search_paths(path,ref,search_market='close2',probability_model='inference')


if __name__=='__main__':unittest.main()

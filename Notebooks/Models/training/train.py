import numpy as np 
import pandas as pd

from ufc_betting.Models.LogisticRegression.train_test_builder import TrainTestBuilder
from ufc_betting.Models.LogisticRegression.statistical_functions import run_logit_model
from ufc_betting.Models.LogisticRegression.visual_functions import plot_model_metrics, expected_calibration_error
from ufc_betting.Models.LogisticRegression.utils import save_results
from ufc_betting.config import config
import joblib


def workflow(
    feats, 
    odds_type, 
    scalar_path,
    save_model,
    alpha
):
    builder = TrainTestBuilder(
        df=df_model,
        feats=feats,
        target_col=winner_col,
        date_col=date_col,
        odds_type=odds_type,
        year=2010, 
        month=2, 
        day=26, 
    )
    pkt = builder.prepare_train_test(
        train_size=0.85, scale=True
    )
    X_train = pkt['X_train']
    y_train = pkt['y_train']
    print(y_train.value_counts())

    X_test = pkt['X_test']
    y_test = pkt['y_test']

    results = run_logit_model(
        X_train, 
        y_train,
        X_test, 
        y_test,
        cov_type="HC3",
        reg=True,
        alpha=alpha,
    )

    df_train = pkt['train_df']
    df_test = pkt['test_df']
    df_train['proba_se'] = results['proba_train_se']
    df_test['proba_se'] = results['proba_test_se']

    if save_model:
        model_open = results['model']
        model_open.save(config.model_open_path)
        joblib.dump(pkt['scaler'], scalar_path)

    png_path = fr'C:\Users\jcmar\my_files\SportsBetting\Data\plot_pngs\train_{odds_type}_model_metrics.png'
    plot_model_metrics(
        y_train, 
        results["train_pred_class"], 
        results["train_pred_proba"], 
        suptitle=f'Train Results for {odds_type} Odds',
        path=png_path
    )
    
    png_path = fr'C:\Users\jcmar\my_files\SportsBetting\Data\plot_pngs\test_{odds_type}_model_metrics.png'
    plot_model_metrics(
        y_test, 
        results['test_pred_class'], 
        results["test_pred_proba"], 
        suptitle=f'Test Results for {odds_type} Odds', path=png_path
    )

    df_results_train = save_results(
        df_train, 
        results['train_pred_proba'], 
        fr'C:\Users\jcmar\my_files\SportsBetting\data\model_results\train_logit_{odds_type}.csv'
    )
    df_results_test = save_results(
        df_test, 
        results['test_pred_proba'], 
        fr'C:\Users\jcmar\my_files\SportsBetting\data\model_results\test_logit_{odds_type}.csv'
    )

    return df_results_train, df_results_test


if __name__ == "__main__":
    fp = r'C:\Users\jcmar\my_files\SportsBetting\data\training_data\entire_odds_stats_2026-03-07.csv'
    df_model = pd.read_csv(fp)
    winner_col = 'winner'
    date_col = 'date'

    train_results, test_results = workflow(
        feats=config.open_feats, 
        odds_type='open',
        scalar_path=None,
        save_model=False,
        alpha=3,
    )

    print(train_results['proba_se'])

from pathlib import Path


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def plot_lasso_path(X_train, y_train, C_min=1e-4, C_max=1e2, num_C=100, figsize=(12,6), *, return_figure=False):
    """Return coefficient paths; optionally also return the figure for saving."""

    # Generate C values across the logarithmic range
    C_values = np.logspace(np.log10(C_max), np.log10(C_min), num_C)

    # Store coefficients
    coefs = []

    for C in C_values:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                l1_ratio=1,
                solver='liblinear',
                C=C,
                max_iter=1000
            )
        )
        model.fit(X_train, y_train)
        
        coef = model.named_steps['logisticregression'].coef_[0]
        coefs.append(coef)

    # Convert to DataFrame
    coefs = np.array(coefs)
    coef_df = pd.DataFrame(coefs, columns=X_train.columns)

    # Plot
    fig = plt.figure(figsize=figsize)
    for feature in coef_df.columns:
        plt.plot(np.log10(1 / C_values), coef_df[feature], label=feature)

    plt.xlabel('log10(1 / C) - Penalty Strength')
    plt.ylabel('Coefficient Value (Feature Importance)')
    plt.title('Lasso Path: Feature Importance vs Regularization')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.show()

    coef_df.index = pd.Index(C_values, name="C")
    return (coef_df, fig) if return_figure else coef_df


def rank_features_by_lasso_path(coef_df, C_values):

    ranks = {}

    for feature in coef_df.columns:
        coefs = coef_df[feature].values
        
        # Indices where coefficient is zero
        zero_indices = np.where(coefs == 0)[0]

        if len(zero_indices) == 0:
            # Coefficient never hits zero â†’ maximum importance
            vanish_index = len(C_values)
        else:
            vanish_index = None
            # Find index where it becomes zero *and stays zero*
            for idx in zero_indices:
                if np.all(coefs[idx:] == 0):
                    vanish_index = idx
                    break
            
            if vanish_index is None:
                # Never permanently zero
                vanish_index = len(C_values)

        ranks[feature] = vanish_index

    # Convert to DataFrame
    rank_df = pd.DataFrame.from_dict(ranks, orient='index', columns=['vanish_index'])

    # Higher vanish_index â†’ more important â†’ higher rank
    rank_df['rank'] = rank_df['vanish_index'].rank(method='min', ascending=False).astype(int)

    # Sort most â†’ least important
    rank_df = rank_df.sort_values('rank', kind='stable')
    rank_df.index.name = 'feature'

    with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.width', None):
        print(rank_df)

    return rank_df

    # Generate C values across the logarithmic range


def plot_lasso_importance(rank_df, top_n=30, figsize=(12, 10)):
    """Plot persistence under increasing L1 penalty, not causal importance."""
    selected = rank_df.sort_values('rank', kind='stable').head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(selected.index, selected['vanish_index'])
    ax.set(xlabel='Regularization steps before coefficient stays zero',
           title=f'Lasso Feature Importance — Top {len(selected)} Features')
    fig.tight_layout()
    plt.show()
    return fig


def save_lasso_outputs(path_fig, importance_fig, rank_df, coef_df, *,
                       path_plot_path, importance_plot_path,
                       ranked_features_path, coefficient_paths_path):
    """Save figures and tables to caller-supplied paths; return those paths."""
    paths = {name: Path(value) for name, value in dict(
        path_plot=path_plot_path, importance_plot=importance_plot_path,
        ranked_features=ranked_features_path,
        coefficient_paths=coefficient_paths_path,
    ).items()}
    if len({p.resolve() for p in paths.values()}) != len(paths):
        raise ValueError('Each output must have a different path')
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    path_fig.savefig(paths['path_plot'], dpi=180, bbox_inches='tight')
    importance_fig.savefig(paths['importance_plot'], dpi=180, bbox_inches='tight')
    rank_df.to_csv(paths['ranked_features'], index_label='feature')
    coef_df.to_csv(paths['coefficient_paths'], index_label='C')
    return paths

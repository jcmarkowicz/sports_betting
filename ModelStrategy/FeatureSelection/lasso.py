

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def plot_lasso_path(X_train, y_train, C_min=1e-4, C_max=1e2, num_C=100, figsize=(12,6)):

    # Generate C values across the logarithmic range
    C_values = np.logspace(np.log10(C_max), np.log10(C_min), num_C)

    # Store coefficients
    coefs = []

    for C in C_values:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty='l1',
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
    plt.figure(figsize=figsize)
    for feature in coef_df.columns:
        plt.plot(np.log10(1 / C_values), coef_df[feature], label=feature)

    plt.xlabel('log10(1 / C) - Penalty Strength')
    plt.ylabel('Coefficient Value (Feature Importance)')
    plt.title('Lasso Path: Feature Importance vs Regularization')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.show()

    return coef_df


def rank_features_by_lasso_path(coef_df, C_values):

    ranks = {}

    for feature in coef_df.columns:
        coefs = coef_df[feature].values
        
        # Indices where coefficient is zero
        zero_indices = np.where(coefs == 0)[0]

        if len(zero_indices) == 0:
            # Coefficient never hits zero → maximum importance
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

    # Higher vanish_index → more important → higher rank
    rank_df['rank'] = rank_df['vanish_index'].rank(method='min', ascending=False).astype(int)

    # Sort most → least important
    rank_df = rank_df.sort_values('rank')

    with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.width', None):
        print(rank_df)

    return rank_df

    # Generate C values across the logarithmic range


C_min=1e-4
C_max=1e2
num_C=100
C_values = np.logspace(np.log10(C_max), np.log10(C_min), num_C)
import numpy as np 
import pandas as pd 

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import brier_score_loss, accuracy_score, f1_score

def poisson_binomial_pmf(probs, k):
    """
    probs : list or array of success probabilities p_i
    k     : number of total successes in the sum
    """
    # Start distribution at P(X=0) = 1
    pmf = np.array([1.0])
    
    # Convolution process
    for p in probs:
        pmf = np.convolve(pmf, [1-p, p])
    
    return pmf[k]


def check_perfect_multicollinearity(df: pd.DataFrame, tol: float = 1e-12):
    """
    Checks for perfectly collinear columns in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input features.
        tol (float): Tolerance for detecting linear dependence.
        
    Returns:
        collinear_pairs (list of tuples): List of column pairs or sets that are perfectly collinear.
    """
    X = df.values
    n_cols = X.shape[1]
    rank = np.linalg.matrix_rank(X, tol=tol)
    
    if rank == n_cols:
        print("No perfect multicollinearity detected.")
        return []
    else:
        print(f"Perfect multicollinearity detected! Matrix rank = {rank}, columns = {n_cols}")
        collinear_cols = []
        for i in range(n_cols):
            # Check if column i can be expressed as linear combination of the others
            others = np.delete(X, i, axis=1)
            coef, residuals, _, _ = np.linalg.lstsq(others, X[:, i], rcond=None)
            if np.allclose(others @ coef, X[:, i], atol=tol):
                collinear_cols.append(df.columns[i])
        print("Perfectly collinear columns:", collinear_cols)
        return collinear_cols
    

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE).

    Parameters
    ----------
    y_true : array-like, shape (n,)
        True binary labels.
    y_prob : array-like, shape (n,)
        Predicted probabilities.
    n_bins : int

    Returns
    -------
    ece : float
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    ece = 0.0
    n = len(y_true)

    for b in range(n_bins):
        mask = bin_ids == b
        if np.any(mask):
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            ece += np.abs(acc - conf) * np.sum(mask) / n

    return ece



def run_logit_model(
        X_train, y_train,
        X_test, y_test, 
        cov_type="HC3",
        reg= False,
        alpha=0, #
        method="l1"
    ):
    """
    https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.fit_regularized.html  
    """

    # --- add constant ---
    X_train_sm = sm.add_constant(X_train.copy())
    X_test_sm = sm.add_constant(X_test.copy())

    # --- predictions ---
    if reg is False: 
        model = sm.Logit(y_train, X_train_sm).fit(cov_type=cov_type, disp=False)

        train_pred = model.predict(X_train_sm)
        train_class = (train_pred >= 0.5).astype(int)

        test_pred = model.predict(X_test_sm)
        test_class = (test_pred >= 0.5).astype(int)
        
    else:
        model = sm.Logit(y_train, X_train_sm).fit_regularized(
        method=method,     
        alpha=alpha # to multiply l1 penalty term     
        )

        train_pred = model.predict(X_train_sm)
        train_class = (train_pred >= 0.5).astype(int)

        test_pred = model.predict(X_test_sm)
        test_class = (test_pred >= 0.5).astype(int)

    # --- metrics ---
    accuracy = accuracy_score(y_test, test_class)
    brier = brier_score_loss(y_test, test_pred)
    
    accuracy_train = accuracy_score(y_train, train_class)
    brier_train = brier_score_loss(y_train, train_pred)
    f1_train = f1_score(y_train, train_class)
    f1_test = f1_score(y_test, test_class)

    check_vif(X_train)
    
    results = {
        "train_pred_proba": train_pred,
        "test_pred_proba": test_pred,
        "test_pred_class": test_class,
        "train_pred_class": train_class,
        'model': model
    }
    
    print(model.summary())
    print(f'TEST STATS: Accuracy: {accuracy}, Brier: {brier}, F1: {f1_test}')
    print(f'TRAIN STATS: Accuracy: {accuracy_train}, Brier: {brier_train}, F1: {f1_train}')

    return results


def check_vif(X_train):
    vif_values = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

    # Create a DataFrame
    vif_df = pd.DataFrame({
        'feature': X_train.columns,
        'VIF': vif_values
    })

    # Sort by VIF descending
    vif_df = vif_df.sort_values('VIF', ascending=False).reset_index(drop=True)

    print(vif_df)

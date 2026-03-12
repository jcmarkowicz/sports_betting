import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, f1_score

def plot_line_movement(df_, odds_type):
    df = df_.copy()

    # Determine if the predicted winner is favorite
    df['is_fav'] = np.where(
        ((df['pred_winner']==1) & (df[f'{odds_type}_red'] <= df[f'{odds_type}_blue'])) |
        ((df['pred_winner']==0) & (df[f'{odds_type}_blue'] < df[f'{odds_type}_red'])),
        1, 0
    )

    # Get juice of the chosen fighter
    df['choice_close_line'] = np.where(
        df['pred_winner'] == 1,
        df[f'{odds_type}_red'],
        df[f'{odds_type}_blue']
    )

    df['choice_open_line'] = np.where(
        df['pred_winner'] == 1,
        df[f'open_red'],
        df[f'open_blue']
    )

    df['choice_line_movement'] = df['choice_close_line'] - df['choice_open_line'] 
    # Compute stats
    mean, std = df['choice_line_movement'].mean(), df['choice_line_movement'].std()

    # Plot histograms with KDE

    sns.histplot( df['choice_line_movement'].values, bins=60, kde=True, color='blue', alpha=0.6)
    plt.title(f'Line Movement {odds_type} \nMean={mean:.2f}, Std={std:.2f}')
    plt.xlabel('Line Movement')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

def plot_juice_histogram(df_, odds_type):
    df = df_.copy()

    # Determine if the predicted winner is favorite
    df['is_fav'] = np.where(
        ((df['pred_winner']==1) & (df[f'{odds_type}_red'] <= df[f'{odds_type}_blue'])) |
        ((df['pred_winner']==0) & (df[f'{odds_type}_blue'] < df[f'{odds_type}_red'])),
        1, 0
    )

    # Get juice of the chosen fighter
    df['choice_juice'] = np.where(
        df['pred_winner'] == 1,
        df[f'juice_{odds_type}_red'],
        df[f'juice_{odds_type}_blue']
    )

    juice_fav = df.loc[df['is_fav']==1, 'choice_juice']
    juice_dog = df.loc[df['is_fav']==0, 'choice_juice']

    # Compute stats
    fav_mean, fav_std = juice_fav.mean(), juice_fav.std()
    dog_mean, dog_std = juice_dog.mean(), juice_dog.std()

    # Plot histograms with KDE
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sns.histplot(juice_fav, bins=60, kde=True, color='blue', alpha=0.6, ax=axes[0])
    axes[0].set_title(f'Favorite {odds_type} Juice\nMean={fav_mean:.2f}, Std={fav_std:.2f}')
    axes[0].set_xlabel('Juice')
    axes[0].set_ylabel('Count')
    sns.despine(ax=axes[0])

    sns.histplot(juice_dog, bins=60, kde=True, color='red', alpha=0.6, ax=axes[1])
    axes[1].set_title(f'Underdog {odds_type} Juice\nMean={dog_mean:.2f}, Std={dog_std:.2f}')
    axes[1].set_xlabel('Juice')
    sns.despine(ax=axes[1])

    plt.tight_layout()
    plt.show()


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    mce = 0

    for i in range(n_bins):
        start, end = bins[i], bins[i+1]
        idx = (y_prob >= start) & (y_prob < end)

        if np.sum(idx) == 0:
            continue

        prob_avg = y_prob[idx].mean()
        true_avg = y_true[idx].mean()

        gap = abs(prob_avg - true_avg)
        weight = np.sum(idx) / len(y_prob)

        ece += weight * gap
        mce = max(mce, gap)

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

    plt.figure(figsize=(7, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model Calibration')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')

    plt.title(f'Calibration Curve\nECE={ece:.3f}  MCE={mce:.3f}')
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_model_metrics(y_test, y_pred, probs, suptitle='', path=None):

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_test, y_pred)

    # ---- ROC ----
    fpr, tpr, thresholds_roc = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    # ---- F1 vs threshold ----
    thresholds = np.linspace(0, 1, 200)
    f1_scores = [f1_score(y_test, (probs >= t).astype(int)) for t in thresholds]

    # ---- Calibration + ECE/MCE ----
    bins = np.linspace(0, 1, 11)
    ece = 0
    mce = 0

    for i in range(len(bins)-1):
        start, end = bins[i], bins[i+1]
        idx = (probs >= start) & (probs < end)

        if np.sum(idx) == 0:
            continue

        prob_avg = probs[idx].mean()
        true_avg = y_test[idx].mean()

        gap = abs(prob_avg - true_avg)
        weight = np.sum(idx) / len(probs)

        ece += weight * gap
        mce = max(mce, gap)

    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
    fig, axes = plt.subplots(1, 4, figsize=(26, 5))

    # ---- 1. Calibration Curve ----
    axes[0].plot(prob_pred, prob_true, marker='o', label='Model')
    axes[0].plot([0,1], [0,1], linestyle='--', label='Perfect')

    axes[0].set_title(f'Calibration Curve\nECE={ece:.3f}  MCE={mce:.3f}')
    axes[0].set_xlabel("Predicted Probability")
    axes[0].set_ylabel("True Probability")
    axes[0].legend()
    axes[0].grid(True)

    # ---- 2. Confusion Matrix ----
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axes[1], cmap="Blues", colorbar=False)

    axes[1].set_title("Confusion Matrix")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    axes[1].set_xticks([0,1])
    axes[1].set_yticks([0,1])
    axes[1].set_xticklabels(["Blue Fighter (0)", "Red Fighter (1)"])
    axes[1].set_yticklabels(["Blue Fighter (0)", "Red Fighter (1)"])

    # ---- 3. ROC Curve ----
    axes[2].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    axes[2].plot([0,1], [0,1], linestyle="--")

    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].set_title("ROC Curve")
    axes[2].legend()
    axes[2].grid(True)

    # ---- 4. F1 vs Threshold ----
    axes[3].plot(thresholds, f1_scores)

    axes[3].set_xlabel("Threshold")
    axes[3].set_ylabel("F1 Score")
    axes[3].set_title("F1 Score vs Threshold")
    axes[3].grid(True)

    fig.suptitle(suptitle)
    plt.tight_layout()

    if path is not None:
        fig.savefig(path,
                    dpi=300,
                    bbox_inches="tight")

    plt.show()

def plot_calibration_curve(y_true, y_hat, n_bins, title_suffix=""):
    prob_true, prob_pred = calibration_curve(y_true, y_hat, n_bins=n_bins, strategy="quantile")

    bin_counts = np.histogram(y_hat, bins=n_bins)[0]
    weights = bin_counts / len(y_hat)
    ece = np.sum(np.abs(prob_true - prob_pred) * weights)
    ece2 = expected_calibration_error(y_true, y_hat, n_bins=n_bins)
    plt.title(f'{title_suffix}: Calibration Curve\nECE1={ece:.6f}, ECE2={ece2:.6f}')
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.legend()
    plt.grid(True)
    plt.show()


def accuracy_by_line_movement(df_results_open, df_results_close, close_col):
    open_red = df_results_open['open_red']
    close_red = df_results_close[close_col]
    line_movement = close_red - open_red

    open_preds = df_results_open['pred_winner']
    close_preds = df_results_close['pred_winner']
    y_true = df_results_open['winner']

    df_results_open['line_movement'] = line_movement
    
    df_results_open['pred_winner_close'] = df_results_close['pred_winner']
    df = df_results_open[df_results_open['line_movement'] > -1000]
    print(df.shape)
    df['movement_bins'] = pd.cut(df['line_movement'], bins=45)

    accuarcy_by_bin = df.groupby('movement_bins').apply(
        lambda x: accuracy_score(x['winner'], x['pred_winner'])
    )
    movement_by_bin = df.groupby('movement_bins').apply(
        lambda x: np.mean(x['line_movement'])
    )

    plt.plot(movement_by_bin, accuarcy_by_bin, marker='o')
    plt.xlabel('Line Movement (Close - Open)') 
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Line Movement')  
    plt.show()

    return df


def analyze_line_movement_perf(df, odds_type='close', filter_ev=True):
    df = df.copy()

    # -----------------------------
    # 0. Extract probability of chosen side
    # -----------------------------
    df['proba_choice'] = np.where(
        df['pred_winner'] == 1,
        df['proba_red'],
        df['proba_blue']
    )

    # -----------------------------
    # 1. Compute decimal odds
    # -----------------------------
    def american_to_decimal(x):
        if x < 0:
            return 100 / abs(x) + 1
        else:
            return x / 100 + 1

    df['red_decimal']  = df[f'{odds_type}_red'].apply(american_to_decimal)
    df['blue_decimal'] = df[f'{odds_type}_blue'].apply(american_to_decimal)

    df['choice_decimal_odds'] = np.where(
        df['pred_winner'] == 1,
        df['red_decimal'],
        df['blue_decimal']
    )

    # Convert to NET odds
    df['choice_net_odds'] = df['choice_decimal_odds'] - 1.0

    # -----------------------------
    # 2. Expected Value (EV)
    # -----------------------------
    df['EV'] = df['proba_choice'] * df['choice_net_odds'] - (1 - df['proba_choice'])

    # Filter only EV > 0 bets?
    if filter_ev:
        df = df[df['EV'] > 0].reset_index(drop=True)

    # -----------------------------
    # 3. Determine fav/dog at open & close
    # -----------------------------
    df['fav_open'] = np.where(df['open_red'] <= df['open_blue'], 1, 0)
    df['fav_close'] = np.where(df[f'{odds_type}_red'] <= df[f'{odds_type}_blue'], 1, 0)

    df['choice_open_fav'] = np.where(
        ((df['pred_winner']==1) & (df['fav_open']==1)) |
        ((df['pred_winner']==0) & (df['fav_open']==0)),
        1, 0
    )

    df['choice_close_fav'] = np.where(
        ((df['pred_winner']==1) & (df['fav_close']==1)) |
        ((df['pred_winner']==0) & (df['fav_close']==0)),
        1, 0
    )

    # -----------------------------
    # 4. Movement category
    # -----------------------------
    def movement_label(row):
        if row['choice_open_fav']==1 and row['choice_close_fav']==0:
            return "Fav → Dog"
        if row['choice_open_fav']==0 and row['choice_close_fav']==1:
            return "Dog → Fav"
        if row['choice_open_fav']==1 and row['choice_close_fav']==1:
            return "Stayed Fav"
        return "Stayed Dog"

    df['movement_category'] = df.apply(movement_label, axis=1)

    # -----------------------------
    # 5. Profit
    # -----------------------------
    df['correct'] = (df['pred_winner'] == df['winner']).astype(int)
    df['profit'] = np.where(df['correct'] == 1, df['choice_net_odds'], -1)

    # -----------------------------
    # 6. Stats by movement category
    # -----------------------------
    stats = df.groupby('movement_category').agg(
        accuracy=('correct', 'mean'),
        n=('correct', 'count'),
        total_profit=('profit', 'sum'),
        mean_EV=('EV', 'mean')
    ).reset_index()

    # -----------------------------
    # 7. PLOTS
    # -----------------------------
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy plot
    sns.barplot(data=stats, x='movement_category', y='accuracy', ax=axs[0])
    axs[0].set_title(f"Accuracy by Line Movement Category\nEV filter = {filter_ev}")
    axs[0].set_xlabel("Category")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_ylim(0, 1)

    # Profit plot
    sns.barplot(data=stats, x='movement_category', y='total_profit', ax=axs[1])
    axs[1].set_title("Total Net Odds by Category")
    axs[1].set_xlabel("Category")
    axs[1].set_ylabel("Total Net Odds")

    plt.tight_layout()
    plt.show()

    return stats


def plot_accuracy_by_variable(df, variable, pred_col="correct_pred"):
    grouped = df.groupby(variable)[pred_col].agg(["mean", "count"])

    plt.figure(figsize=(10, 6))
    bars = plt.bar(grouped.index, grouped["mean"], color="skyblue")

    plt.title(f"Prediction Accuracy by {variable}")
    plt.ylabel("Accuracy")
    plt.xlabel(variable)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")

    # Annotate with counts AND accuracy %
    for bar, (acc, count) in zip(bars, zip(grouped["mean"], grouped["count"])):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={count}\n{acc*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.show()

def plot_accuracy_by_bins(df_, variable, bins=10, pred_col="correct_pred"):
    df = df_.copy()
    # Create bins using pandas.cut
    df["_bins"] = pd.cut(df[variable], bins=bins)

    # Group by bins
    grouped = df.groupby("_bins")[pred_col].agg(["mean", "count"])

    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(grouped.index.astype(str), grouped["mean"], color="skyblue")

    plt.title(f"Prediction Accuracy by Binned {variable}")
    plt.ylabel("Accuracy")
    plt.xlabel(variable)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")

    # Annotate counts + accuracy %
    for bar, (acc, count) in zip(bars, zip(grouped["mean"], grouped["count"])):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={count}\n{acc*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.show()

    # Clean temporary column
    df.drop(columns=["_bins"], inplace=True)



def InverseProbabilityNC(predicted_score, y):
    """
    Computes nonconformity scores based on inverse probability.
    """
    prob = np.zeros(y.size, dtype=np.float32)
    for i, y_ in enumerate(y):
        if y_ >= predicted_score.shape[1]:
            prob[i] = 0
        else:
            prob[i] = predicted_score[i, int(y_)]
    return 1 - prob


def compute_tcp_pvalues(X_train, y_train, X_test, n_classes=2):
    """
    Computes Transductive Conformal Prediction p-values for a test set.

    Returns:
        p_values_tcp : np.ndarray of shape (n_test, n_classes)
    """
    # ------------------------------
    # Base training model
    # ------------------------------
    base_model = sm.Logit(y_train, X_train).fit(disp=False)

    p_train = base_model.predict(X_train)
    pred_train = np.column_stack([1 - p_train, p_train])

    # training nonconformity
    nc_train = InverseProbabilityNC(pred_train, y_train.values)

    # ------------------------------
    # Transductive Conformal Prediction
    # ------------------------------
    n_test = len(X_test)
    p_values_tcp = np.zeros((n_test, n_classes))

    for i in range(n_test):

        x_i = X_test.iloc[[i]]  # shape (1,d)

        for c in range(n_classes):

            # Augmented dataset (train + test point)
            X_aug = sm.add_constant(np.vstack([X_train, x_i]))
            y_aug = np.concatenate([y_train.values, np.array([c])])

            # Fit new model
            model_aug = sm.Logit(y_aug, X_aug).fit(disp=False)

            # Predicted probabilities for augmented dataset
            p_aug = model_aug.predict(X_aug)
            pred_aug = np.column_stack([1 - p_aug, p_aug])

            # Compute nonconformity scores
            nc_aug = InverseProbabilityNC(pred_aug, y_aug)

            # TCP p-value for test point
            test_nc = nc_aug[-1]
            calib_nc = nc_aug[:-1]
            p_values_tcp[i, c] = (np.sum(calib_nc >= test_nc) + 1) / (len(calib_nc) + 1)

    return p_values_tcp
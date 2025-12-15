import streamlit as st

import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score, confusion_matrix, classification_report, roc_curve, auc, f1_score, confusion_matrix, ConfusionMatrixDisplay, silhouette_score
from sklearn.calibration import calibration_curve
import statsmodels.api as sm

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from models.get_train_test import TrainTestBuilder

class DataUtils:
    @staticmethod
    def filter_columns(df, cols):
        return df[cols]

    @staticmethod
    def rename_column(df, old, new):
        df = df.copy()
        df.rename(columns={old: new}, inplace=True)
        return df

    @staticmethod
    def sort_df(df, col, ascending=True):
        return df.sort_values(by=col, ascending=ascending)

    @staticmethod
    def paginate(df, page, page_size=50):
        start = (page - 1) * page_size
        end = start + page_size
        return df.iloc[start:end]

def display_paginated_df(df, page_size=50, title="Data Viewer"):
    """
    Displays a paginated DataFrame in Streamlit with navigation buttons.
    
    Args:
        df (pd.DataFrame): The DataFrame to display.
        page_size (int): Number of rows per page.
        title (str): Optional title for the section.
    """
    st.subheader(title)
    
    total_rows = len(df)
    total_pages = (total_rows + page_size - 1) // page_size  # ceiling division

    # --- Initialize page state ---
    if "page" not in st.session_state:
        st.session_state.page = 1

    # --- Navigation buttons ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("⏮ First Page"):
            st.session_state.page = 1
    with col2:
        if st.button("◀ Prev"):
            if st.session_state.page > 1:
                st.session_state.page -= 1
    with col3:
        if st.button("Next ▶"):
            if st.session_state.page < total_pages:
                st.session_state.page += 1
    with col4:
        if st.button("Last Page ⏭"):
            st.session_state.page = total_pages

    # --- Manual page input ---
    page_input = st.number_input(
        "Page",
        value=st.session_state.page,
        min_value=1,
        max_value=total_pages
    )
    st.session_state.page = page_input

    # --- Paginate DataFrame ---
    paged_df = DataUtils.paginate(df, st.session_state.page, page_size)

    # --- Display ---
    start_row = (st.session_state.page - 1) * page_size
    end_row = min(start_row + page_size, total_rows)

    st.dataframe(paged_df, use_container_width=True)
    st.write(f"Showing rows {start_row + 1} to {end_row} of {total_rows}")

st.title("Models Page")
def plot_accuracy_by_variable(df, variable, pred_col="correct_pred"):
    grouped = df.groupby(variable)[pred_col].agg(["mean", "count"])

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(grouped.index, grouped["mean"], color="skyblue")

    ax.set_title(f"Prediction Accuracy by {variable}")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel(variable)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")

    # Annotate counts AND accuracy %
    for bar, (acc, count) in zip(bars, zip(grouped["mean"], grouped["count"])):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={count}\n{acc*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    st.pyplot(fig,use_container_width=False)  # Streamlit display

def plot_accuracy_by_bins(df, variable, bins=10, pred_col="correct_pred"):

    # Create bins using pandas.cut
    df["_bins"] = pd.cut(df[variable], bins=bins)

    # Group by bins
    grouped = df.groupby("_bins")[pred_col].agg(["mean", "count"])

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(grouped.index.astype(str), grouped["mean"], color="skyblue")

    ax.set_title(f"Prediction Accuracy by Binned {variable}")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel(variable)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")

    # Annotate counts + accuracy %
    for bar, (acc, count) in zip(bars, zip(grouped["mean"], grouped["count"])):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={count}\n{acc*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    st.pyplot(fig,use_container_width=False)  # Streamlit display

    # Clean temporary column
    df.drop(columns=["_bins"], inplace=True)
    df.drop(columns=["_bins"], inplace=True)


# --- Split red/blue features ---

def build_train_test(X_train, X_test, y_train, y_test, open_close=True):
    red_train, blue_train, red_cols, blue_cols = split_red_blue_features(X_train)
    red_test, blue_test, _, _ = split_red_blue_features(X_test)

    # --- Split fav/dog features ---
    fav_train, dog_train, y_favdog_train = split_fav_dog(pd.concat([X_train, y_train], axis=1))
    fav_test, dog_test, y_favdog_test = split_fav_dog(pd.concat([X_test, y_test], axis=1))

    # --- Identify leftover columns not in red/blue/fav/dog ---
    assigned_train_cols = red_cols + blue_cols + list(fav_train.columns) + list(dog_train.columns)
    leftover_train = X_train[[c for c in X_train.columns if c not in assigned_train_cols]]
    leftover_test = X_test[[c for c in X_test.columns if c not in assigned_train_cols]]

    # --- Combine everything ---
    X_train_lr = pd.concat([
        red_train.reset_index(drop=True),
        blue_train.reset_index(drop=True),
        fav_train.reset_index(drop=True),
        dog_train.reset_index(drop=True),
        leftover_train.reset_index(drop=True)
    ], axis=1).reset_index(drop=True)

    X_test_lr = pd.concat([
        red_test.reset_index(drop=True),
        blue_test.reset_index(drop=True),
        fav_test.reset_index(drop=True),
        dog_test.reset_index(drop=True),
        leftover_test.reset_index(drop=True)
    ], axis=1).reset_index(drop=True)


    X_train_fav_dog = pd.concat([
        fav_train.reset_index(drop=True),
        dog_train.reset_index(drop=True),
        leftover_train.reset_index(drop=True)
    ], axis=1).reset_index(drop=True)

    X_test_fav_dog = pd.concat([
        fav_test.reset_index(drop=True),
        dog_test.reset_index(drop=True),
        leftover_test.reset_index(drop=True)
    ], axis=1).reset_index(drop=True)


    X_train_red_blue = pd.concat([
        red_train.reset_index(drop=True),
        blue_train.reset_index(drop=True),
        leftover_train.reset_index(drop=True)
    ], axis=1).reset_index(drop=True)


    X_test_red_blue = pd.concat([
        red_test.reset_index(drop=True),
        blue_test.reset_index(drop=True),
        leftover_test.reset_index(drop=True)
    ], axis=1).reset_index(drop=True)

    if open_close is False:
        X_train_red_blue = X_train_red_blue.drop(columns=['open_red', 'open_blue'])
        X_test_red_blue = X_test_red_blue.drop(columns=['open_red', 'open_blue'])
        X_train_fav_dog = X_train_fav_dog.drop(columns=['open_fav', 'open_dog'])
        X_test_fav_dog = X_test_fav_dog.drop(columns=['open_fav', 'open_dog'])

    return X_train_lr, X_test_lr, X_train_fav_dog, X_test_fav_dog, X_train_red_blue, X_test_red_blue, y_favdog_train, y_favdog_test

# _, _, _, _, X_train_red_blue_w, X_test_red_blue_w, _, _ = build_train_test(X_train_women, X_test_women, y_train_women, y_test_women,  open_close=False)
# X_train_lr, X_test_lr, _, _, X_train_red_blue_m, X_test_red_blue_m, y_favdog_train, y_favdog_test = build_train_test(X_train_men, X_test_men, y_train_men, y_test_men,  open_close=False)


def split_red_blue_features(df: pd.DataFrame):
    """
    Splits the dataframe into red and blue sides with properly prefixed columns.
    Diff columns are given a red_ or blue_ prefix and negated for blue side.
    
    Only columns with 'red', 'blue', or 'diff' postfixes (plus allowed globals) 
    are included in the final output.

    Args:
        df (pd.DataFrame): Input dataframe with red, blue, and diff columns.

    Returns:
        red_df (pd.DataFrame)
        blue_df (pd.DataFrame)
        red_cols_final (list): final red columns
        blue_cols_final (list): final blue columns
    """
    
    allowed_globals = []  # include any global columns you want
    
    red_cols = [c for c in df.columns if "red" in c]
    blue_cols = [c for c in df.columns if "blue" in c]
    diff_cols = [c for c in df.columns if "diff" in c]

    # ---- Red DataFrame ----
    red_df = df[red_cols].copy()
    for c in diff_cols:
        base = c.replace("_diff", "")
        red_df[f"red_{base}_diff"] = df[c]
    # Include allowed global columns
    for col in allowed_globals:
        if col in df.columns:
            red_df[col] = df[col]

    # ---- Blue DataFrame ----
    blue_df = df[blue_cols].copy()
    for c in diff_cols:
        base = c.replace("_diff", "")
        blue_df[f"blue_{base}_diff"] = -df[c]  # negate diffs for blue
    
    for col in allowed_globals:
        if col in df.columns:
            blue_df[col] = df[col]

    # ---- Final column lists ----
    red_cols_final = [c for c in red_df.columns if c.endswith('_red')] + allowed_globals
    blue_cols_final = [c for c in blue_df.columns if c.endswith('_blue')] + allowed_globals

    # ---- Ensure only allowed columns exist ----
    red_df = red_df[red_cols_final]
    blue_df = blue_df[blue_cols_final]

    return red_df, blue_df, red_cols_final, blue_cols_final

def split_fav_dog(df: pd.DataFrame):

    fav_rows = []
    dog_rows = []
    df = df.reset_index(drop=True).copy()

    for _, row in df.iterrows():

        # -----------------------------------------
        # Identify favorite
        # -----------------------------------------
        red_fav = row["open_red"] <= row["open_blue"]   # ties → red fav
        blue_fav = row["open_blue"] < row["open_red"]

        # -----------------------------------------
        # Base fighter features
        # -----------------------------------------
        if red_fav:
            fav_row = {col.replace("red", "fav"): row[col] for col in row.index if "red" in col}
            dog_row = {col.replace("blue", "dog"): row[col] for col in row.index if "blue" in col}
        else:
            fav_row = {col.replace("blue", "fav"): row[col] for col in row.index if "blue" in col}
            dog_row = {col.replace("red", "dog"): row[col] for col in row.index if "red" in col}

        # -----------------------------------------
        # Fix diff_* properly
        # -----------------------------------------
        diff_row_fav = {}
        diff_row_dog = {}

        for col in row.index:
            if col.endswith("_diff"):
                base = col.replace("_diff", "")
                value = row[col]

                if red_fav:
                    diff_row_fav[f"fav_{base}_diff"] = value
                    diff_row_dog[f"dog_{base}_diff"] = -value
                else:
                    diff_row_fav[f"fav_{base}_diff"] = -value
                    diff_row_dog[f"dog_{base}_diff"] = value

        fav_row.update(diff_row_fav)
        dog_row.update(diff_row_dog)

        fav_rows.append(fav_row)
        dog_rows.append(dog_row)

    # ---------- Build DataFrames ----------
    fav_df = pd.DataFrame(fav_rows)
    dog_df = pd.DataFrame(dog_rows)

    # ---------- FINAL SAFETY CLEANUP ----------
    allowed_prefixes = ("fav_", "dog_")
    allowed_globals = []

    fav_df = fav_df[[c for c in fav_df.columns if c.endswith("_fav")]]
    dog_df = dog_df[[c for c in dog_df.columns if c.endswith("_dog")]]

    # -----------------------------------------
    # Winner label (fav = 1 if favorite wins)
    # -----------------------------------------
    y_fav_dog = []

    for _, row in df.iterrows():
        if row["open_red"] <= row["open_blue"]:  # red fav
            y_fav_dog.append(1 if row["winner"] == 1 else 0)
        else:
            y_fav_dog.append(1 if row["winner"] == 0 else 0)

    print(np.array(y_fav_dog).shape, df.shape)
    return fav_df, dog_df, np.array(y_fav_dog)


def build_train_test(X_train, X_test, y_train, y_test, open_close=True):
    red_train, blue_train, red_cols, blue_cols = split_red_blue_features(X_train)
    red_test, blue_test, _, _ = split_red_blue_features(X_test)

    # --- Split fav/dog features ---
    fav_train, dog_train, y_favdog_train = split_fav_dog(pd.concat([X_train, y_train], axis=1))
    fav_test, dog_test, y_favdog_test = split_fav_dog(pd.concat([X_test, y_test], axis=1))

    # --- Identify leftover columns not in red/blue/fav/dog ---
    assigned_train_cols = red_cols + blue_cols + list(fav_train.columns) + list(dog_train.columns)
    leftover_train = X_train[[c for c in X_train.columns if c not in assigned_train_cols]]
    leftover_test = X_test[[c for c in X_test.columns if c not in assigned_train_cols]]

    # --- Combine everything ---
    X_train_lr = pd.concat([
        red_train.reset_index(drop=True),
        blue_train.reset_index(drop=True),
        fav_train.reset_index(drop=True),
        dog_train.reset_index(drop=True),
        leftover_train.reset_index(drop=True)
    ], axis=1).reset_index(drop=True)

    X_test_lr = pd.concat([
        red_test.reset_index(drop=True),
        blue_test.reset_index(drop=True),
        fav_test.reset_index(drop=True),
        dog_test.reset_index(drop=True),
        leftover_test.reset_index(drop=True)
    ], axis=1).reset_index(drop=True)


    X_train_fav_dog = pd.concat([
        fav_train.reset_index(drop=True),
        dog_train.reset_index(drop=True),
        leftover_train.reset_index(drop=True)
    ], axis=1).reset_index(drop=True)

    X_test_fav_dog = pd.concat([
        fav_test.reset_index(drop=True),
        dog_test.reset_index(drop=True),
        leftover_test.reset_index(drop=True)
    ], axis=1).reset_index(drop=True)


    X_train_red_blue = pd.concat([
        red_train.reset_index(drop=True),
        blue_train.reset_index(drop=True),
        leftover_train.reset_index(drop=True)
    ], axis=1).reset_index(drop=True)


    X_test_red_blue = pd.concat([
        red_test.reset_index(drop=True),
        blue_test.reset_index(drop=True),
        leftover_test.reset_index(drop=True)
    ], axis=1).reset_index(drop=True)

    if open_close is False:
        X_train_red_blue = X_train_red_blue.drop(columns=['open_red', 'open_blue'])
        X_test_red_blue = X_test_red_blue.drop(columns=['open_red', 'open_blue'])
        X_train_fav_dog = X_train_fav_dog.drop(columns=['open_fav', 'open_dog'])
        X_test_fav_dog = X_test_fav_dog.drop(columns=['open_fav', 'open_dog'])

    return X_train_lr, X_test_lr, X_train_fav_dog, X_test_fav_dog, X_train_red_blue, X_test_red_blue, y_favdog_train, y_favdog_test


def plot_model_metrics(y_test, y_pred, probs):
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # ROC
    fpr, tpr, thresholds_roc = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    # F1 vs threshold
    thresholds = np.linspace(0, 1, 200)
    f1_scores = [f1_score(y_test, (probs >= t).astype(int)) for t in thresholds]

    # ------------------------------
    # Create side-by-side figure
    # ------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    # ---- 1. Confusion Matrix ----
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axes[0], cmap="Blues", colorbar=False)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Plot on our subplot axis, not a new figure
    disp.plot(cmap='Blues', ax=axes[0], colorbar=False)
    axes[0].set_title("Confusion Matrix - Logistic Regression")

    # Explicit axis labels
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["Blue Fighter (0)", "Red Fighter (1)"])
    axes[0].set_yticklabels(["Blue Fighter (0)", "Red Fighter (1)"])

    # ---- 2. ROC Curve ----
    axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].grid(True)
    axes[1].legend()

    # ---- 3. F1 vs Threshold ----
    axes[2].plot(thresholds, f1_scores)
    axes[2].set_xlabel("Threshold")
    axes[2].set_ylabel("F1 Score")
    axes[2].set_title("F1 Score vs Threshold")
    axes[2].grid(True)

    # ---- 4. Accuracy + Brier ----
    # axes[3].axis("off")
    # text = (
    #     f"Model Performance\n\n"
    #     f"Accuracy: {accuracy:.4f}\n"
    #     f"Brier Score: {brier:.4f}"
    # )
    # axes[3].text(0.1, 0.5, text, fontsize=14, verticalalignment="center")

    plt.tight_layout()
    st.pyplot(fig,use_container_width=False)


def run_logit_model(
        X_train, y_train,
        X_test, y_test, 
        cov_type="HC3",
        n_bins=20,
        vif_func=None
    ):

    # --- add constant ---
    X_train_sm = sm.add_constant(X_train.copy())
    X_test_sm = sm.add_constant(X_test.copy())

    # --- fit model ---
    model = sm.Logit(y_train, X_train_sm).fit(cov_type=cov_type, disp=False)

    # --- predictions ---
    train_pred = model.predict(X_train_sm)
    train_class = (train_pred >= 0.5).astype(int)
    test_pred = model.predict(X_test_sm)
    test_class = (test_pred >= 0.5).astype(int)

    # --- metrics ---
    accuracy = accuracy_score(y_test, test_class)
    brier = brier_score_loss(y_test, test_pred)
    
    accuracy_train = accuracy_score(y_train, train_class)
    brier_train = brier_score_loss(y_train, train_pred)

    # --- compute ECE ---
    def plot_calibration_curve(y_true, y_hat, n_bins, title_suffix=""):

        # Compute calibration
        prob_true, prob_pred = calibration_curve(y_true, y_hat, n_bins=n_bins, strategy="quantile")

        # Compute Expected Calibration Error (ECE)
        bin_counts = np.histogram(y_hat, bins=n_bins)[0]
        weights = bin_counts / len(y_hat)
        ece = np.sum(np.abs(prob_true - prob_pred) * weights)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot calibration curve
        ax.plot(prob_pred, prob_true, marker='o', label='Model Calibration')
        ax.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')

        ax.set_title(f'{title_suffix}: Calibration Curve\nECE={ece:.6f}')
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("True Probability")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        st.pyplot(fig,use_container_width=False)  # Pass the figure object to Streamlit

    plot_calibration_curve(y_test, test_pred, n_bins, 'Test Set')
    plot_calibration_curve(y_train, train_pred, n_bins, 'Train Set')
    # --- run VIF if provided ---
    if vif_func is not None:
        vif_func(X_train)
    
    results = {
        "train_pred_proba": train_pred,
        "test_pred_proba": test_pred,
        "test_pred_class": test_class,
        "train_pred_class": train_class,

        'model': model
    }
    # Model summary — if it's text
    st.markdown(f"```\n{model.summary()}\n```")  # triple backticks for code block

    # Test stats
    st.markdown(f"**TEST STATS:**\n- Accuracy: {accuracy:.3f}\n- Brier: {brier:.3f}")

    # Train stats
    st.markdown(f"**TRAIN STATS:**\n- Accuracy: {accuracy_train:.3f}\n- Brier: {brier_train:.3f}")

    return results


non_feats = [
    'date','event_date','event_location','fighter_blue','fighter_red',
    'method','og_blue_name','og_red_fighter', 'red_fighter_stats', 'blue_fighter_stats',
    'pimp_close1_blue','pimp_close1_red','pimp_close2_blue','pimp_close2_red',
    'juice_close1_blue','juice_close1_red','juice_close2_blue','juice_close2_red',
    'line_movement_close1_blue','line_movement_close1_red','line_movement_close2_blue','line_movement_close2_red',
    'winner_name',
    
    'red_fighter_odds','blue_fighter_odds',
    'dec_close1_blue','dec_close1_red','dec_close2_blue','dec_close2_red',
    'dec_fair_close1_blue','dec_fair_close1_red','dec_fair_close2_blue','dec_fair_close2_red',
    'red_ud_to_fav_close1','red_ud_to_fav_close2','blue_ud_to_fav_close1','blue_ud_to_fav_close2',
    'red_stayed_fav_close1','red_stayed_fav_close2','blue_stayed_fav_close1','blue_stayed_fav_close2',
    'red_fav_to_ud_close1','red_fav_to_ud_close2','blue_fav_to_ud_close1','blue_fav_to_ud_close2',
    'red_stayed_dog_close1','red_stayed_dog_close2','blue_stayed_dog_close1','blue_stayed_dog_close2',
    'proba_fair_close1_red','proba_fair_close1_blue','proba_fair_close2_red','proba_fair_close2_blue',
    'performance_bonus_winner', 'fight_otn_bonus', 'close1_blue','close1_red', 'close2_blue', 'close2_red'
]

selected_feats = ['proba_fair_open_diff', 'age_diff', 'open_red', 'open_blue', 'reach_diff', 'sig_str_absorbed_total_diff', 
                  'td_attempted_pm_diff', 'elo_diff', 'sig_str_landed_total_diff',
                    'ko_losses_diff', 'win_pct_diff', 'kd_total_diff','avg_fight_min_diff', 'height_diff', 'head_str_total_diff', 'losses_diff', 'control_total_diff'
                  ]

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
fp = BASE_DIR / "data" / "entire_odds_stats_2025-12-04.csv"


df_model = None
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    df_model = pd.read_csv(uploaded_file)
    st.write(df_model.head()) 

    
y = 'winner'
builder = TrainTestBuilder(df=df_model, target_col=y, non_features=non_feats, train_size=0.8, random_state=42)
builder.filter_by_year(2010, date_col='event_date')
X_train, X_test, y_train, y_test, df_train, df_test = builder.prepare_train_test(selected_feats, clustering=False)

y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

_, _, X_train_fav_dog, X_test_fav_dog, X_train_red_blue, X_test_red_blue, y_favdog_train, y_favdog_test = build_train_test(X_train, X_test, y_train, y_test, open_close=False)

results_open = run_logit_model(
    X_train_red_blue, y_train,
    X_test_red_blue, y_test,
    cov_type="HC3",
    n_bins=30,
    vif_func=None
)

st.markdown("""
            <p style="font-size:18px; line-height:1.6;">
            Results with Open Odds data 
""")

y_pred = results_open["test_pred_class"]
red_probs = results_open["test_pred_proba"]
plot_model_metrics(y_test, y_pred, red_probs)

selected_feats = ['proba_fair_close1_diff', 'proba_fair_close2_diff', 'age_diff', 'open_red', 'open_blue', 'reach_diff', 'sig_str_absorbed_total_diff', 
                  'td_attempted_pm_diff', 'elo_diff', 'sig_str_landed_total_diff',
                    'ko_losses_diff', 'win_pct_diff', 'kd_total_diff','avg_fight_min_diff', 'height_diff', 'head_str_total_diff', 'losses_diff', 'control_total_diff'
                   ]

builder = TrainTestBuilder(df=df_model, target_col=y, non_features=non_feats, train_size=0.8, random_state=42)
builder.filter_by_year(2011, date_col='event_date')
X_train, X_test, y_train, y_test, df_train, df_test = builder.prepare_train_test(selected_feats, clustering=None)

y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

X_train['open_pred'] = results_open['train_pred_class']
X_test['open_pred'] = results_open["test_pred_class"]

st.markdown("""
            <p style="font-size:18px; line-height:1.6;">
            Results with Close1 and Close 2 Odds data 
""")

_, _, X_train_fav_dog, X_test_fav_dog, X_train_red_blue, X_test_red_blue, _, _ = build_train_test(X_train, X_test, y_train, y_test, open_close=False)
results = run_logit_model(
    X_train_red_blue, y_train,
    X_test_red_blue, y_test,
    cov_type="HC3",
    n_bins=20,
    vif_func=None
)

y_pred = results["test_pred_class"]
red_probs = results["test_pred_proba"]
plot_model_metrics(y_test, y_pred, red_probs)
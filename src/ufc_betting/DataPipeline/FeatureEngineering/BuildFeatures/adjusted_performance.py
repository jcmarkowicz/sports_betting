import numpy as np 
import pandas as pd 

from collections import defaultdict

def time_decay_average(arr, decay_lambda=0.13):
    arr = np.asarray(arr)
    
    # 0 = most recent, larger = older
    age = np.arange(len(arr)-1, -1, -1)
    
    weights = np.exp(-decay_lambda * age)
    
    return np.sum(weights * arr) / np.sum(weights)

def mean_absolute_deviation(series):
    return np.mean(np.abs(series - np.mean(series)))


class AdjustedPerformanceHistory:
    """Maintain prefight histories used by adjusted-performance features."""

    def __init__(self, performance_col, opponent_performance_col, decay_lambda=0.13):
        self.performance_col = performance_col
        self.opponent_performance_col = opponent_performance_col
        self.decay_lambda = decay_lambda

        # Only completed prior dates are visible to prefight calculations.
        self.weight_class_history = defaultdict(list)
        self.fighter_allowed_history = defaultdict(list)
        self.fighter_adjusted_history = defaultdict(list)

        # Values from the current date are committed when the date changes.
        self.event_weight_class_history = defaultdict(list)
        self.event_fighter_allowed_history = defaultdict(list)
        self.event_fighter_adjusted_history = defaultdict(list)
        self.current_date = None

    def _commit_event(self):
        for weight_class, values in self.event_weight_class_history.items():
            self.weight_class_history[weight_class].extend(values)

        for fighter, values in self.event_fighter_allowed_history.items():
            self.fighter_allowed_history[fighter].extend(values)

        for fighter, values in self.event_fighter_adjusted_history.items():
            self.fighter_adjusted_history[fighter].extend(values)

        self.event_weight_class_history.clear()
        self.event_fighter_allowed_history.clear()
        self.event_fighter_adjusted_history.clear()

    def start_date(self, date):
        if self.current_date is not None and date != self.current_date:
            self._commit_event()
        self.current_date = date

    def historical_adjusted_performance(self, fighter, time_decay=False):
        history = self.fighter_adjusted_history[fighter]
        if not history:
            return None
        if time_decay:
            return time_decay_average(history, self.decay_lambda)
        return history[-1]

    def compute_current_performance(self, hero_performance, villain, weight_class, k=3):
        """Calculate the current fight's score using completed prior dates only."""
        if pd.isna(hero_performance) or pd.isna(weight_class):
            return None

        villain_history = self.fighter_allowed_history[villain]
        weight_class_history = self.weight_class_history[weight_class]
        if not villain_history or not weight_class_history:
            return None

        villain_values = np.asarray(villain_history, dtype=float)
        weight_class_values = np.asarray(weight_class_history, dtype=float)

        n = len(villain_values)
        bayes_weight = n / (n + k)

        villain_allowed_mean = time_decay_average(
            villain_values,
            self.decay_lambda,
        )
        weight_class_allowed_mean = np.mean(weight_class_values)
        allowed_mean = (
            bayes_weight * villain_allowed_mean
            + (1 - bayes_weight) * weight_class_allowed_mean
        )

        villain_mad = mean_absolute_deviation(villain_values)
        weight_class_mad = mean_absolute_deviation(weight_class_values)
        shrunk_mad = (
            bayes_weight * villain_mad
            + (1 - bayes_weight) * weight_class_mad
        )
        shrunk_mad = max(shrunk_mad, 1e-3)

        adjusted = (hero_performance - allowed_mean) / shrunk_mad
        return np.clip(adjusted, -7, 7)

    def buffer_fight(self, row, red_adjusted, blue_adjusted):
        """Buffer raw and adjusted values until the current date is complete."""
        weight_class = row["weight_class"]
        if pd.isna(weight_class):
            return

        for color, adjusted in (
            ("red", red_adjusted),
            ("blue", blue_adjusted),
        ):
            fighter = row[f"fighter_{color}"]
            allowed = row[f"{self.opponent_performance_col}_{color}"]

            if pd.notna(allowed):
                self.event_weight_class_history[weight_class].append(allowed)
                self.event_fighter_allowed_history[fighter].append(allowed)

            if pd.notna(adjusted):
                self.event_fighter_adjusted_history[fighter].append(adjusted)

def adjusted_performance(df, row, hero, villain, weight_class, fight_date, performance_col, opponent_performance_col, hero_color, k=3):
    """
    Usage: performance col=sig strikes landed, opponent performance col=sig strikes absorbed

    Computes performance of hero fighter for SINGLE FIGHT 
    """

    # must include fight date stats, this stat is for next fight 
    df_history = df[(df['weight_class'] == weight_class) & (df['date'] < fight_date)].dropna()
    df_villain = df_history[(df_history['fighter_red'] == villain) | (df_history['fighter_blue'] == villain)]

    if df_history.shape[0] == 0 or df_villain.shape[0] == 0: 
         return None

    # bayesian shrinkage
    n = df_villain.shape[0]
    villain_allowed = np.where(
         df_villain['fighter_red'] == villain, 
         df_villain[opponent_performance_col + '_red'], 
         df_villain[opponent_performance_col + '_blue']
    )
    villain_allowed_mean = time_decay_average(villain_allowed)
    
    if n == 0:
         print('wrong df villain ')

    # total weight class average 
    weight_class_allowed_mean = df_history[[opponent_performance_col + '_red', opponent_performance_col + '_blue']].mean().mean()

    wc_total_allowed = pd.concat([
        df_history[opponent_performance_col + '_red'],
        df_history[opponent_performance_col + '_blue']
    ])

    # Z score formulation
  
    # average allowed by villain, wc weighted
    w_bayes = n / (n+k)
    mu_shrunk = w_bayes * villain_allowed_mean + (1-w_bayes) * weight_class_allowed_mean

    # get sigma, wc weighted 
    MAD_villian = mean_absolute_deviation(villain_allowed)
    MAD_wc = mean_absolute_deviation(wc_total_allowed)

    mad_shrunk = w_bayes * MAD_villian + (1-w_bayes) * MAD_wc
    mad_shrunk = max(mad_shrunk, 1e-3) # avoid division by zero
    
    # get performance of current fight 
    hero_performance = row[performance_col + f'_{hero_color}']

    if hero_performance is np.isnan(hero_performance):
         print(f'Hero performance nan')

    # adjusted performance for current fight 
    adjusted_perf = (hero_performance - mu_shrunk) / mad_shrunk 
    adjusted_perf = np.clip(adjusted_perf, -7, 7) # cap extreme values

    return adjusted_perf 


def compute_adjusted_performance(df_, performance_col, opponent_performance_col, time_decay=False):
    """
    Build prefight adjusted-performance history in chronological order.

    Fights on the same date are buffered together so they cannot influence one
    another's prefight features.
    """
    df = df_.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", kind="stable").reset_index(drop=True)

    history = AdjustedPerformanceHistory(
        performance_col=performance_col,
        opponent_performance_col=opponent_performance_col,
    )
    red_arr = []
    blue_arr = []

    for row in df.itertuples(index=False):
        row = row._asdict()
        history.start_date(row["date"])

        red_fighter = row["fighter_red"]
        blue_fighter = row["fighter_blue"]
        red_arr.append(
            history.historical_adjusted_performance(
                red_fighter,
                time_decay=time_decay,
            )
        )
        blue_arr.append(
            history.historical_adjusted_performance(
                blue_fighter,
                time_decay=time_decay,
            )
        )

        red_adjusted = history.compute_current_performance(
            hero_performance=row[f"{performance_col}_red"],
            villain=blue_fighter,
            weight_class=row["weight_class"],
        )
        blue_adjusted = history.compute_current_performance(
            hero_performance=row[f"{performance_col}_blue"],
            villain=red_fighter,
            weight_class=row["weight_class"],
        )

        history.buffer_fight(row, red_adjusted, blue_adjusted)

    return np.column_stack([red_arr, blue_arr])

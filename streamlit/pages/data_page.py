import streamlit as st

import numpy as np 
import pandas as pd 
st.title("Data Page")

import matplotlib.pyplot as plt


class DataLoader:
    def __init__(self, path: str):
        self.path = path

    def load_csv(self):
        return pd.read_csv(self.path)

    def load_parquet(self):
        return pd.read_parquet(self.path)

    def load_excel(self):
        return pd.read_excel(self.path)
    

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
    
def display_paginated_df(df, page_size=50, title="Data Viewer", key_prefix=""):
    """
    Displays a paginated DataFrame in Streamlit with navigation buttons.
    key_prefix: string to make widget keys unique
    """
    import streamlit as st
    import pandas as pd

    st.subheader(title)
    total_rows = len(df)
    total_pages = (total_rows + page_size - 1) // page_size  # ceiling division

    # Initialize page state
    if f"page_{key_prefix}" not in st.session_state:
        st.session_state[f"page_{key_prefix}"] = 1

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("⏮ First Page", key=f"first_{key_prefix}"):
            st.session_state[f"page_{key_prefix}"] = 1
    with col2:
        if st.button("◀ Prev", key=f"prev_{key_prefix}"):
            if st.session_state[f"page_{key_prefix}"] > 1:
                st.session_state[f"page_{key_prefix}"] -= 1
    with col3:
        if st.button("Next ▶", key=f"next_{key_prefix}"):
            if st.session_state[f"page_{key_prefix}"] < total_pages:
                st.session_state[f"page_{key_prefix}"] += 1
    with col4:
        if st.button("Last Page ⏭", key=f"last_{key_prefix}"):
            st.session_state[f"page_{key_prefix}"] = total_pages

    # Manual page input
    page_input = st.number_input(
        "Page",
        value=st.session_state[f"page_{key_prefix}"],
        min_value=1,
        max_value=total_pages,
        key=f"numinput_{key_prefix}"
    )
    st.session_state[f"page_{key_prefix}"] = page_input

    # Paginate
    paged_df = DataUtils.paginate(df, st.session_state[f"page_{key_prefix}"], page_size)

    # Display
    start_row = (st.session_state[f"page_{key_prefix}"] - 1) * page_size
    end_row = min(start_row + page_size, total_rows)

    st.dataframe(paged_df, use_container_width=True)
    st.write(f"Showing rows {start_row + 1} to {end_row} of {total_rows}")
# Load data using class
loader_history = DataLoader(r'C:\Users\jcmar\my_files\SportsBetting\data\entire_odds_stats_2025-11-19.csv')
df_history = loader_history.load_csv()
df_history = df_history.sort_values(by='date', ascending=True)
df_display = df_history[['date', 'fighter_red', 'fighter_blue', 'elo_red', 'elo_blue', 'glicko_red', 'glicko_blue']]

display_paginated_df(df_display, page_size=50, title="Data", key_prefix="history")

fighters_red = df_history[['fighter_red', 'elo_red']].rename(columns={'fighter_red': 'fighter', 'elo_red': 'elo'})
fighters_blue = df_history[['fighter_blue', 'elo_blue']].rename(columns={'fighter_blue': 'fighter', 'elo_blue': 'elo'})

all_fighters = pd.concat([fighters_red, fighters_blue], ignore_index=True)
top_fighters = all_fighters.groupby('fighter')['elo'].max().sort_values(ascending=False)
top_10_fighters = top_fighters.head(50)
display_paginated_df(top_10_fighters, page_size=50, title="Fighters Ranked by Elo", key_prefix='elo')



fighters_red = df_history[['fighter_red', 'glicko_red']].rename(columns={'fighter_red': 'fighter', 'glicko_red': 'glicko'})
fighters_blue = df_history[['fighter_blue', 'glicko_blue']].rename(columns={'fighter_blue': 'fighter', 'glicko_blue': 'glicko'})

all_fighters = pd.concat([fighters_red, fighters_blue], ignore_index=True)
top_fighters = all_fighters.groupby('fighter')['glicko'].max().sort_values(ascending=False)
top_10_fighters = top_fighters.head(50)
display_paginated_df(top_10_fighters, page_size=50, title="Fighters Ranked by Glicko", key_prefix='glicko')

import streamlit as st

import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


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

def show_image(fp, title=None, figsize=(6,6)):
    """
    Loads an image from file_path, plots it with Matplotlib, and displays in Streamlit.

    Args:
        file_path (str or Path): path to the image
        title (str, optional): title for the figure
        figsize (tuple, optional): figure size in inches
    """
    # Load image
    img = Image.open(fp)
    st.image(img, caption=title, use_container_width =True)

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
from BettingStrategy.ModelStrategy.LogisticRegression.get_train_test import TrainTestBuilder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from utils import show_image

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]


st.markdown("""
            <p style="font-size:18px; line-height:1.6;">
            Results with Open Odds data 
""")

train_open_fp = BASE_DIR / "Data" / "plot_pngs" / "train_open_model_metrics.png" 
test_open_fp = BASE_DIR / "Data" / "plot_pngs" / "test_open_model_metrics.png" 

show_image(train_open_fp)
show_image(test_open_fp)

st.markdown("""
            <p style="font-size:18px; line-height:1.6;">
            Results with Close1 and Close 2 Odds data 
""")

train_close1_fp = BASE_DIR / "Data" / "plot_pngs" / "train_close1_model_metrics.png" 
test_close1_fp = BASE_DIR / "Data" / "plot_pngs" / "test_close1_model_metrics.png" 

show_image(train_close1_fp)
show_image(test_close1_fp)

train_close2_fp = BASE_DIR / "Data" / "plot_pngs" / "train_close2_model_metrics.png" 
test_close2_fp = BASE_DIR / "Data" / "plot_pngs" / "test_close2_model_metrics.png" 

show_image(train_close2_fp)
show_image(test_close2_fp)